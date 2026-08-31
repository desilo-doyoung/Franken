"""The backend must read weights off the module and CALL the wrapper. DDP has no attribute
passthrough, so `lm_head_weight` used to crash every multi-GPU `task: lm` run on its first forward.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from franken.models.llama.backend import LlamaBackend
from franken.models.llama.config import LlamaModelConfig
from franken.models.llama.model import LlamaModel


def _tiny():
    cfg = LlamaModelConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=50,
        attn_impl="manual",
    )
    cfg.validate()
    torch.manual_seed(0)
    return LlamaModel(cfg)


@pytest.fixture
def gloo():
    """A one-rank CPU process group: enough to build a real DDP, no GPU needed."""
    env = dict(MASTER_ADDR="127.0.0.1", MASTER_PORT="29613", RANK="0", WORLD_SIZE="1")
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    dist.init_process_group("gloo")
    try:
        yield
    finally:
        dist.destroy_process_group()
        for k, v in old.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)


def test_ddp_hides_the_attribute_the_backend_needs(gloo):
    """Calibrates the test below: without unwrapping this is the reported AttributeError."""
    wrapped = DistributedDataParallel(_tiny())
    with pytest.raises(AttributeError, match="embed_tokens"):
        _ = wrapped.embed_tokens


@pytest.mark.parametrize("compiled", [False, True])
def test_lm_head_weight_survives_the_training_wrappers(gloo, compiled):
    model = _tiny()
    wrapped = DistributedDataParallel(model)
    if compiled:
        # The trainer's order: DDP first, so compile's `_orig_mod` IS the DDP wrapper.
        wrapped = torch.compile(wrapped)

    out = LlamaBackend().forward(wrapped, {"input_ids": torch.tensor([[1, 2, 3]])})
    assert out["lm_head_weight"] is model.embed_tokens.weight


def test_the_forward_goes_through_ddp_not_around_it(gloo):
    """The allreduce hooks live in DDP.forward; unwrapping for the CALL too would silently leave
    every rank training its own replica."""
    wrapped = DistributedDataParallel(_tiny())
    seen = []
    wrapped.register_forward_hook(lambda *a: seen.append(1))

    LlamaBackend().forward(wrapped, {"input_ids": torch.tensor([[1, 2, 3]])})
    assert seen, "backend bypassed the DDP wrapper"
