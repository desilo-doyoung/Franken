import datasets
import pytest
import torch
from torch import nn

from franken.config import Config
from franken.distill.dist import DistEnv
from franken.distill.trainer import BatchLoader, BestCheckpoint, RangePenalty, resolve_lr


class Opt:
    def __init__(self, lr):
        self.lr = lr


def test_explicit_lr_is_used_verbatim():
    assert resolve_lr(Opt(3e-5), global_batch=999.0, log=print) == 3e-5


def test_null_lr_scales_as_sqrt_of_batch():
    # 4x the batch is 2x the rate, from the tuned bs32/lr2e-5 reference.
    assert resolve_lr(Opt(None), 32.0, lambda *a: None) == pytest.approx(2e-5)
    assert resolve_lr(Opt(None), 128.0, lambda *a: None) == pytest.approx(4e-5)


# --------------------------------------------------------------------- BestCheckpoint


def tiny(value: float) -> nn.Module:
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


def test_higher_is_better_keeps_the_maximum():
    best = BestCheckpoint("recall@10", True)
    assert best.consider({"recall@10": 0.5}, tiny(1.0))
    assert not best.consider({"recall@10": 0.4}, tiny(2.0))
    assert best.consider({"recall@10": 0.7}, tiny(3.0))
    assert best.best == 0.7


def test_lower_is_better_keeps_the_minimum():
    best = BestCheckpoint("loss", False)
    assert best.consider({"loss": 1.0}, tiny(1.0))
    assert not best.consider({"loss": 2.0}, tiny(2.0))
    assert best.consider({"loss": 0.5}, tiny(3.0))
    assert best.best == 0.5


def test_saved_state_is_cloned_off_the_live_weights():
    # state_dict() hands back live references; a later optimizer.step() must not rewrite the
    # checkpoint that was already selected.
    best = BestCheckpoint("m", True)
    model = tiny(1.0)
    best.consider({"m": 1.0}, model)
    with torch.no_grad():
        model.weight.fill_(99.0)
    best.restore(model)
    assert model.weight.item() == 1.0


def test_restore_without_a_candidate_is_a_no_op():
    model = tiny(5.0)
    BestCheckpoint("m", True).restore(model)
    assert model.weight.item() == 5.0


# --------------------------------------------------------------------- RangePenalty


class FakeAct(nn.Module):
    def __init__(self, domain=None):
        super().__init__()
        self.domain = domain


class FakeBackend:
    def __init__(self, n_layers=3, domain=None):
        self.preacts = [nn.Identity() for _ in range(n_layers)]
        self.acts = [FakeAct(domain) for _ in range(n_layers)]

    def activation_ops(self, model):
        return self.acts

    def ffn_preact_modules(self, model):
        return self.preacts


def cfg_with(range_penalty=1.0, layers=None, domain=32):
    return Config.from_dict(
        {
            "model": {
                "num_hidden_layers": 3,
                "activation": "quad_silu",
                "activation_kwargs": {"domain": domain},
            },
            "distill": {"range_penalty": range_penalty, "range_penalty_layers": layers},
        }
    )


def test_penalty_is_inactive_without_a_weight():
    p = RangePenalty(FakeBackend(domain=32.0), None, cfg_with(range_penalty=0.0), print)
    assert not p


def test_penalty_is_inactive_when_the_op_has_no_domain():
    p = RangePenalty(FakeBackend(domain=None), None, cfg_with(), print)
    assert not p


def test_penalty_hooks_only_the_named_layers():
    backend = FakeBackend(domain=2.0)
    p = RangePenalty(backend, None, cfg_with(layers=[0, 2]), lambda *a: None)
    assert p and p._targets == [backend.preacts[0], backend.preacts[2]]


def test_hooks_are_removed_when_the_block_exits():
    backend = FakeBackend(domain=2.0)
    p = RangePenalty(backend, None, cfg_with(), lambda *a: None)
    with p:
        backend.preacts[0](torch.tensor([9.0]))
        assert p.measure() is not None
    p.clear()
    backend.preacts[0](torch.tensor([9.0]))
    assert p.measure() is None


def test_in_range_activations_produce_no_penalty():
    backend = FakeBackend(domain=2.0)
    with RangePenalty(backend, None, cfg_with(), lambda *a: None) as p:
        backend.preacts[0](torch.tensor([1.0, -1.0]))
        assert p.measure() is None


def test_penalty_is_the_squared_distance_past_the_domain():
    # Meaned over the OUT-OF-RANGE elements only: the in-range bulk must not dilute it.
    backend = FakeBackend(domain=2.0)
    with RangePenalty(backend, None, cfg_with(), lambda *a: None) as p:
        backend.preacts[0](torch.tensor([3.0, 0.0, 0.0, 0.0]))
        assert p.measure().item() == pytest.approx(1.0)


def test_eval_mode_activations_are_not_captured():
    backend = FakeBackend(domain=2.0)
    with RangePenalty(backend, None, cfg_with(), lambda *a: None) as p:
        backend.preacts[0].eval()
        backend.preacts[0](torch.tensor([9.0]))
        assert p.measure() is None


def test_epoch_mean_averages_then_resets():
    backend = FakeBackend(domain=2.0)
    with RangePenalty(backend, None, cfg_with(), lambda *a: None) as p:
        for x in (3.0, 5.0):  # penalties 1.0 and 9.0
            p.clear()
            backend.preacts[0](torch.tensor([x]))
            p.measure()
        assert p.epoch_mean() == pytest.approx(5.0)
        assert p.epoch_mean() is None


# --------------------------------------------------------------------- BatchLoader


def dataset(lengths):
    ds = datasets.Dataset.from_dict({"input_ids": [list(range(n)) for n in lengths]})
    return ds.with_format("torch", columns=["input_ids"])


def loader_for(train: dict, lengths) -> BatchLoader:
    cfg = Config.from_dict({"train": {"distill": train}})
    return BatchLoader(cfg, DistEnv(), dataset(lengths), lambda x: x, lambda *a: None)


def test_fixed_batch_size_gives_ceiling_steps_per_epoch():
    b = loader_for({"batch_size": 4}, [3] * 10)
    assert len(b) == 3
    assert b.global_batch == 4.0


LENGTHS = list(range(1, 61))


def test_token_budget_replaces_the_sequence_count():
    b = loader_for({"tokens_per_step": 64}, LENGTHS)
    assert b.plan is not None and len(b) == len(b.plan)
    assert b.accum_steps == 1  # fits one rank, so no accumulation
    for batch in b.plan:
        if len(batch) > 1:
            assert len(batch) * max(LENGTHS[i] for i in batch) <= 64
    assert b.global_batch == pytest.approx(len(LENGTHS) / len(b.plan))


def test_replanning_never_changes_steps_per_epoch():
    # steps/epoch drives the LR schedule; only the order may move between epochs.
    b = loader_for({"tokens_per_step": 64}, LENGTHS)
    first = list(b.loader(0).batch_sampler)
    steps = len(b)
    second = list(b.loader(1).batch_sampler)
    assert len(b) == steps
    assert sorted(map(sorted, first)) == sorted(map(sorted, second))
    assert first != second


SHORT = [4, 5, 6, 7, 8] * 40  # well under the budget, so packing is in the linear regime


def test_the_machine_ceiling_only_changes_how_the_step_is_chopped(monkeypatch):
    # The whole point: a smaller card costs speed, it does not train a different model.
    seen = {}
    for ceiling in (32, 64, 128):
        monkeypatch.setenv("FRANKEN_MAX_TOKENS_PER_RANK", str(ceiling))
        b = loader_for({"tokens_per_step": 128}, SHORT)
        assert b.micro_tokens * b.accum_steps == 128  # world_size 1; exact by construction
        seen[ceiling] = (b.accum_steps, b.global_batch)
    assert [a for a, _g in seen.values()] == [4, 2, 1]
    batches = {g for _a, g in seen.values()}
    assert max(batches) - min(batches) < 0.5, seen


def test_an_indivisible_step_is_rejected(monkeypatch):
    monkeypatch.setenv("FRANKEN_MAX_TOKENS_PER_RANK", "30")
    with pytest.raises(ValueError, match="not divisible"):
        loader_for({"tokens_per_step": 64}, LENGTHS)
