import pytest
import torch
import torch.nn.functional as F

from franken.config import DistillConfig
from franken.tasks import lm

H, V, B, S = 8, 17, 3, 5


def _outs(seed=0, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    s_h = torch.randn(B, S, H, generator=g, requires_grad=requires_grad)
    t_h = torch.randn(B, S, H, generator=g)
    s_w = torch.randn(V, H, generator=g, requires_grad=requires_grad)
    t_w = torch.randn(V, H, generator=g)
    student = {"hidden_states": [None, s_h], "lm_head_weight": s_w}
    teacher = {"hidden_states": [None, t_h], "lm_head_weight": t_w}
    return student, teacher, s_h, s_w


def _mask():
    m = torch.ones(B, S, dtype=torch.long)
    m[0, 3:] = 0  # right padding, as the collator produces
    m[2, 1:] = 0
    return m


def _naive(student, teacher, mask, temperature):
    """One reduction over every kept position -- what unbounded memory would compute."""
    s_h, t_h = student["hidden_states"][-1], teacher["hidden_states"][-1]
    keep = mask.reshape(-1).bool()
    s = (s_h.reshape(-1, H)[keep] @ student["lm_head_weight"].T) / temperature
    t = (t_h.reshape(-1, H)[keep] @ teacher["lm_head_weight"].T) / temperature
    total = F.kl_div(
        F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1), reduction="sum", log_target=True
    )
    return total / int(keep.sum())


@pytest.mark.parametrize("chunk", [1, 3, 4, 10_000])
def test_chunked_kl_matches_one_unchunked_reduction(monkeypatch, chunk):
    # Positions are independent (softmax is over the vocab axis), so chunking is associativity.
    # Tolerance, not equality: k reductions plus k-1 adds round differently from one reduction.
    monkeypatch.setattr(lm, "_CHUNK", chunk)
    student, teacher, _, _ = _outs()
    mask = _mask()
    assert float(lm.logit_kl(student, teacher, mask)) == pytest.approx(
        float(_naive(student, teacher, mask, 1.0)), rel=1e-6
    )


@pytest.mark.parametrize("chunk", [1, 3, 10_000])
def test_chunked_kl_matches_unchunked_gradients(monkeypatch, chunk):
    # The whole memory strategy is only sound if `checkpoint`'s recompute reproduces the gradient,
    # so the value agreeing is not enough.
    mask = _mask()

    monkeypatch.setattr(lm, "_CHUNK", 10_000)
    student, teacher, s_h_ref, s_w_ref = _outs()
    _naive(student, teacher, mask, 1.0).backward()

    monkeypatch.setattr(lm, "_CHUNK", chunk)
    student, teacher, s_h, s_w = _outs()
    lm.logit_kl(student, teacher, mask).backward()

    assert torch.allclose(s_h.grad, s_h_ref.grad, rtol=1e-5, atol=1e-7)
    assert torch.allclose(s_w.grad, s_w_ref.grad, rtol=1e-5, atol=1e-7)


def test_padded_positions_do_not_reach_the_loss(monkeypatch):
    # Dropped before the projection, so a padded state cannot dilute the mean -- and under
    # right padding with sdpa_causal its hidden state is meaningless.
    monkeypatch.setattr(lm, "_CHUNK", 3)
    mask = _mask()
    student, teacher, _, _ = _outs(requires_grad=False)
    before = lm.logit_kl(student, teacher, mask)
    student["hidden_states"][-1][mask == 0] += 1000.0
    assert lm.logit_kl(student, teacher, mask) == pytest.approx(float(before), rel=1e-9)


def test_temperature_is_applied_inside_the_projection():
    student, teacher, _, _ = _outs(requires_grad=False)
    mask = _mask()
    hot, cold = 1.0, 4.0
    for t in (hot, cold):
        assert float(lm.logit_kl(student, teacher, mask, t)) == pytest.approx(
            float(_naive(student, teacher, mask, t)), rel=1e-6
        )
    # Softening both sides shrinks the divergence; the caller restores scale with T^2.
    assert float(lm.logit_kl(student, teacher, mask, cold)) < float(
        lm.logit_kl(student, teacher, mask, hot)
    )


def test_an_all_padding_batch_is_zero_not_nan():
    # `_mix` never emits one, but a batch sampler edge case dividing by zero would poison the run.
    student, teacher, _, _ = _outs()
    out = lm.logit_kl(student, teacher, torch.zeros(B, S, dtype=torch.long))
    assert float(out) == 0.0


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_hidden_term_is_skipped_when_beta_is_zero(monkeypatch, beta):
    # Scaling by zero would still allocate one (B, S, H) difference per layer and retain it until
    # backward, so a beta ablation measures nothing unless the call itself is skipped.
    calls = []

    def spy(*args):
        calls.append(args)
        return torch.zeros(())

    monkeypatch.setattr(lm, "layerwise_hidden_loss", spy)
    student, teacher, _, _ = _outs(requires_grad=False)
    loss = lm.LogitDistillLoss(DistillConfig(alpha=1.0, beta=beta, temperature=1.0))

    total, kl, hidden = loss(student, teacher, _mask())
    assert len(calls) == (0 if beta == 0.0 else 1)
    # `compute_loss` detaches it either way, so the skipped branch still owes a tensor.
    assert float(hidden) == 0.0
    assert float(total) == pytest.approx(float(kl))


@pytest.mark.parametrize("missing", ["student", "teacher"])
def test_a_backend_with_no_output_projection_fails_loudly(missing):
    # bert supplies no `lm_head_weight`; silently skipping the KL would train on hidden MSE alone.
    student, teacher, _, _ = _outs()
    del {"student": student, "teacher": teacher}[missing]["lm_head_weight"]
    with pytest.raises(ValueError, match=f"{missing} backend"):
        lm.logit_kl(student, teacher, _mask())
