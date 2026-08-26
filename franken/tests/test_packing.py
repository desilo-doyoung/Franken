import pytest
import torch

from franken.distill.packing import doc_ids, doc_positions

EOS = 99


def test_positions_restart_after_every_eos():
    ids = torch.tensor([[1, 2, EOS, 3, 4, 5, EOS, 6]])
    assert doc_positions(ids, EOS).tolist() == [[0, 1, 2, 0, 1, 2, 3, 0]]


def test_unpacked_row_is_a_plain_arange():
    # The embed track never packs, so the derivation must be a no-op there.
    ids = torch.tensor([[7, 8, 9, 10]])
    assert doc_positions(ids, EOS).tolist() == [[0, 1, 2, 3]]


def test_a_block_starting_mid_document_restarts():
    # A long document chopped across blocks: the continuation has no prefix here.
    ids = torch.tensor([[4, 5, 6, EOS, 7]])
    assert doc_positions(ids, EOS)[0, 0].item() == 0


def test_consecutive_eos_does_not_go_negative():
    ids = torch.tensor([[1, EOS, EOS, 2]])
    assert doc_positions(ids, EOS).tolist() == [[0, 1, 0, 0]]


def test_doc_ids_group_tokens_by_document():
    ids = torch.tensor([[1, 2, EOS, 3, 4, 5, EOS, 6]])
    assert doc_ids(doc_positions(ids, EOS)).tolist() == [[0, 0, 0, 1, 1, 1, 1, 2]]


def test_doc_ids_match_transformers_own_derivation():
    """The one place student and teacher could silently disagree: HF builds the teacher's
    block-diagonal mask from position_ids using its own rule, so ours must equal it."""
    from transformers.masking_utils import find_packed_sequence_indices

    ids = torch.tensor([[1, 2, EOS, 3, 4, 5, EOS, 6], [1, EOS, 2, 3, EOS, 4, 5, 6]])
    pos = doc_positions(ids, EOS)
    assert torch.equal(doc_ids(pos), find_packed_sequence_indices(pos))


def test_batch_rows_are_independent():
    ids = torch.tensor([[1, EOS, 2, 3], [4, 5, 6, 7]])
    assert doc_positions(ids, EOS).tolist() == [[0, 1, 0, 1], [0, 1, 2, 3]]


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex compiles a Triton kernel"
)


def _tiny_llama(attn_impl, dtype=torch.float64, device="cpu"):
    import torch as _t

    from franken.models.llama.config import LlamaModelConfig
    from franken.models.llama.model import LlamaModel

    cfg = LlamaModelConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=100,
        attn_impl=attn_impl,
    )
    cfg.validate()
    _t.manual_seed(0)
    return LlamaModel(cfg).to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa_causal"])
def test_isolated_block_equals_separate_forwards(attn_impl):
    """A packed block must equal running each document alone: no op normalizes across tokens."""
    model = _tiny_llama(attn_impl)
    doc_a = [11, 12, 13, EOS]
    doc_b = [21, 22, 23, 24, EOS]
    block = torch.tensor([doc_a + doc_b])
    pos = doc_positions(block, EOS)
    assert pos.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 4]]

    with torch.no_grad():
        packed = model(block, position_ids=pos)["last_hidden_state"][0]
        alone = torch.cat(
            [model(torch.tensor([d]))["last_hidden_state"][0] for d in (doc_a, doc_b)]
        )
    assert torch.allclose(packed, alone, atol=1e-9), (packed - alone).abs().max()


def test_without_isolation_the_block_leaks_across_documents():
    """Calibrates the test above: it passes because of isolation, not because context is ignored."""
    model = _tiny_llama("manual")
    doc_a, doc_b = [11, 12, 13, EOS], [21, 22, 23, 24, EOS]
    block = torch.tensor([doc_a + doc_b])
    with torch.no_grad():
        leaky = model(block)["last_hidden_state"][0]
        alone = torch.cat(
            [model(torch.tensor([d]))["last_hidden_state"][0] for d in (doc_a, doc_b)]
        )
    assert not torch.allclose(leaky, alone, atol=1e-6)


def test_packed_run_refuses_to_derive_the_lr():
    """max_seq_len must not become an lr knob: the sequence count moves with the block size."""
    from franken.config import OptimConfig
    from franken.distill.trainer import resolve_lr

    opt = OptimConfig(lr=None)
    with pytest.raises(ValueError, match="sqrt-batch scaling reads SEQUENCES"):
        resolve_lr(opt, 64.0, lambda _m: None, packed=True)
    assert resolve_lr(OptimConfig(lr=3e-5), 64.0, lambda _m: None, packed=True) == 3e-5


def _hf_llama():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel as HFLlama

    torch.manual_seed(0)
    return (
        HFLlama(
            LlamaConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                intermediate_size=128,
                vocab_size=100,
            )
        )
        .double()
        .eval()
    )


def test_hf_teacher_isolates_only_with_no_cache_and_no_2d_mask():
    """Both conditions are load-bearing and both fail SILENTLY: the teacher keeps cross-document
    attention while the student isolates, and only the identity self-test notices."""
    hf = _hf_llama()
    a, b = [11, 12, 13, EOS], [21, 22, 23, 24, EOS]
    block = torch.tensor([a + b])
    pos = doc_positions(block, EOS)

    with torch.no_grad():
        sep = torch.cat(
            [hf(input_ids=torch.tensor([d]), use_cache=False).last_hidden_state[0] for d in (a, b)]
        )
        isolated = hf(input_ids=block, position_ids=pos, use_cache=False).last_hidden_state[0]
        cached = hf(input_ids=block, position_ids=pos, use_cache=True).last_hidden_state[0]
        masked = hf(
            input_ids=block,
            position_ids=pos,
            use_cache=False,
            attention_mask=torch.ones_like(block),
        ).last_hidden_state[0]

    assert torch.allclose(isolated, sep, atol=1e-12), (isolated - sep).abs().max()
    assert not torch.allclose(cached, sep, atol=1e-6), "use_cache no longer defeats isolation"
    assert not torch.allclose(masked, sep, atol=1e-6), "2D mask no longer defeats isolation"


def test_load_teacher_disables_the_cache():
    """A config default flip upstream would otherwise re-break packed distillation in silence."""
    import inspect

    from franken.models.llama import backend as llama_backend
    from franken.models.qwen3 import backend as qwen3_backend

    for mod in (llama_backend, qwen3_backend):
        src = inspect.getsource(mod)
        assert "model.config.use_cache = False" in src, mod.__name__


def test_packed_model_inputs_omit_the_attention_mask():
    from franken.tasks.lm import LMDistillTask

    task = LMDistillTask()
    task._pack, task._eos_id = True, EOS
    batch = {
        "input_ids": torch.tensor([[1, 2, EOS, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    inputs = task.model_inputs(batch)
    assert set(inputs) == {"input_ids", "position_ids"}
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 0]]

    task._pack = False
    assert set(task.model_inputs(batch)) == {"input_ids", "attention_mask"}


@requires_cuda
def test_flex_matches_manual_on_the_same_packed_block():
    """Same documents both ways, so a difference is flex itself. fp32: Triton has no f64."""
    a, b, c = [11, 12, 13, EOS], [21, 22, 23, 24, EOS], [31, 32, EOS]
    block = torch.tensor([a + b + c], device="cuda")
    pos = doc_positions(block, EOS)
    out = {}
    for impl in ("manual", "flex"):
        model = _tiny_llama(impl, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out[impl] = model(block, position_ids=pos)["last_hidden_state"]
    assert torch.allclose(out["manual"], out["flex"], atol=1e-5), (
        (out["manual"] - out["flex"]).abs().max()
    )


@requires_cuda
def test_flex_isolated_block_equals_separate_forwards():
    model = _tiny_llama("flex", dtype=torch.float32, device="cuda")
    doc_a, doc_b = [11, 12, 13, EOS], [21, 22, 23, 24, EOS]
    block = torch.tensor([doc_a + doc_b], device="cuda")
    with torch.no_grad():
        packed = model(block, position_ids=doc_positions(block, EOS))["last_hidden_state"][0]
        alone = torch.cat(
            [
                model(
                    torch.tensor([d], device="cuda"),
                    position_ids=doc_positions(torch.tensor([d], device="cuda"), EOS),
                )["last_hidden_state"][0]
                for d in (doc_a, doc_b)
            ]
        )
    assert torch.allclose(packed, alone, atol=1e-5), (packed - alone).abs().max()


def test_mask_mod_composition_equals_the_additive_construction():
    """The packed mask comes from a mask_mod, the padded one stays additive; pin them together."""
    from torch.nn.attention.flex_attention import and_masks, create_mask

    from franken.models.llama.model import _causal_mod, _same_document_mod

    ids = torch.tensor([[11, 12, EOS, 21, 22, 23, EOS], [1, EOS, 2, 3, 4, EOS, 5]])
    pos = doc_positions(ids, EOS)
    d = doc_ids(pos)
    B, S = ids.shape

    minv = torch.finfo(torch.float32).min
    additive = torch.full((S, S), minv).triu(1)[None, None]
    additive = additive + (d[:, :, None] != d[:, None, :])[:, None].float() * minv

    keep = create_mask(and_masks(_causal_mod, _same_document_mod(d)), B, None, S, S, device="cpu")
    assert torch.equal(keep, additive == 0)


def test_flex_requires_pack_and_exact_softmax():
    from franken.config import Config

    cfg = Config.from_yaml("configs/llama/smoke.yaml")
    cfg.validate()
    assert cfg.model.attn_impl == "flex" and cfg.train.pack

    cfg.train.pack = False
    with pytest.raises(ValueError, match="needs train.pack"):
        cfg.validate()

    cfg = Config.from_yaml("configs/llama/smoke.yaml")
    cfg.model.softmax = "cgf"
    with pytest.raises(ValueError, match="fuses the softmax"):
        cfg.validate()


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa_causal"])
def test_eos_padding_is_invisible_to_real_tokens(attn_impl):
    """A padded bin must give its real tokens exactly the states they would have had unpadded.
    Right padding plus causal masking is what guarantees it, so this holds for any pad token; EOS
    is chosen so the pads also segment, keeping that region cheap under flex."""
    model = _tiny_llama(attn_impl)
    docs = [11, 12, 13, EOS, 21, 22, EOS]
    padded = torch.tensor([docs + [EOS] * 5])
    bare = torch.tensor([docs])

    with torch.no_grad():
        got = model(padded, position_ids=doc_positions(padded, EOS))["last_hidden_state"][0]
        want = model(bare, position_ids=doc_positions(bare, EOS))["last_hidden_state"][0]
    assert torch.allclose(got[: len(docs)], want, atol=1e-9), (got[: len(docs)] - want).abs().max()


def test_each_eos_pad_is_its_own_segment():
    ids = torch.tensor([[1, 2, EOS, EOS, EOS, EOS]])
    pos = doc_positions(ids, EOS)
    assert pos.tolist() == [[0, 1, 2, 0, 0, 0]]
    # distinct segment ids for every pad, so no pad can be attended to by the real document
    assert doc_ids(pos).tolist() == [[0, 0, 0, 1, 2, 3]]


def test_checkpointing_changes_neither_loss_nor_gradients():
    """Checkpointing is exact recomputation, not an approximation. If gradients move at all then
    use_reentrant=False is misused, and the student trains against a subtly wrong signal."""
    ids = torch.tensor([[11, 12, 13, EOS, 21, 22, 23, EOS]])
    pos = doc_positions(ids, EOS)
    runs = {}
    for flag in (False, True):
        model = _tiny_llama("manual").train()  # same seed, so identical weights
        model.grad_checkpoint = flag
        loss = model(ids, position_ids=pos)["last_hidden_state"].pow(2).mean()
        loss.backward()
        runs[flag] = (loss.detach().item(), [p.grad.clone() for p in model.parameters()])

    assert runs[False][0] == pytest.approx(runs[True][0], rel=1e-12)
    for plain, ckpt in zip(runs[False][1], runs[True][1], strict=True):
        assert torch.allclose(plain, ckpt, atol=1e-12), (plain - ckpt).abs().max()


def test_eval_never_pays_for_recompute(monkeypatch):
    from franken.models.llama import model as llama_model

    calls = []
    real = llama_model.checkpoint
    monkeypatch.setattr(
        llama_model, "checkpoint", lambda fn, *a, **kw: (calls.append(1), real(fn, *a, **kw))[1]
    )
    model = _tiny_llama("manual")  # already in eval mode
    model.grad_checkpoint = True
    probe = torch.tensor([[11, 12, EOS]])

    with torch.no_grad():
        model(probe)
    assert calls == [], "eval recomputed activations for gradients it never takes"

    model.train()
    model(probe)
    assert len(calls) == len(model.layers)
