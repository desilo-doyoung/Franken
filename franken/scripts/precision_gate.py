"""Check that `precision: bf16` is safe here, before spending GPU-hours on it.

It is safe only because the residual stream (= hidden_states, the hidden-loss input) stays fp32
under autocast. That is load-bearing, so assert it, plus the two things that would break it
silently: RoPE staying fp32, and the teacher staying out of the autocast region.

    uv run python -m franken.scripts.precision_gate --config configs/llama/gate_precision.yaml
"""

import argparse

import torch

from franken.config import Config
from franken.encode import embed_batches
from franken.metrics import recall_at_k
from franken.models import build_backend
from franken.tasks import build_task

# A bf16 teacher must move the targets by MORE than the band, or the rule is decoration.
BAND = 0.004


def _fail(msg):
    print(f"FAIL  {msg}")
    return False


def check_rope(model, device):
    """cos/sin must stay fp32 under autocast: `torch.einsum` is intercepted by name and returns
    bf16 despite a .float() input, which is why rope.py uses a broadcast multiply."""
    pos = torch.arange(16, device=device).unsqueeze(0)
    hidden = torch.zeros(1, 16, model.config.hidden_size, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cos, sin = model.rotary_emb(hidden, pos)
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        return _fail(f"RoPE cos/sin are {cos.dtype} under autocast, expected float32")
    print(f"ok    RoPE cos/sin stay {cos.dtype} under autocast")
    return True


def check_hidden_states(model, device):
    """The residual stream carries the hidden-loss targets; it must not be bf16."""
    ids = torch.randint(0, 1000, (2, 32), device=device)
    mask = torch.ones_like(ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=mask)
    dtypes = {h.dtype for h in out["hidden_states"]}
    if dtypes != {torch.float32}:
        return _fail(f"hidden_states dtypes under autocast: {dtypes}, expected float32 only")
    print(f"ok    all {len(out['hidden_states'])} hidden_states stay float32 under autocast")
    return True


@torch.no_grad()
def check_teacher_exclusion(backend, teacher, task, cfg, device):
    """fp32 vs bf16-autocast teacher embeddings. If they agree within the band, excluding the
    teacher buys nothing -- so a PASS here means they DISAGREE."""
    from torch.utils.data import DataLoader

    tokenizer = task.build_tokenizer(cfg)
    data = task.datasets(tokenizer, cfg)
    ds = data["validation"].with_format("torch", columns=task.torch_columns())
    batches = list(DataLoader(ds, batch_size=16, collate_fn=data["collator"]))

    def embed(autocast: bool):
        def ctx():
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast)

        return embed_batches(backend, task, batches, device, teacher, ctx=ctx)[0]

    fp32, bf16 = embed(False), embed(True)
    recall = recall_at_k(bf16, fp32)
    drift = 1.0 - recall
    if drift <= BAND:
        return _fail(
            f"bf16 teacher drifts only {drift:.4f} (recall {recall:.4f}) <= band {BAND} "
            "-- excluding the teacher from autocast is not load-bearing; re-check the claim"
        )
    print(
        f"ok    bf16 teacher drifts {drift:.4f} (recall {recall:.4f}) > band {BAND} "
        "-- teacher exclusion is load-bearing"
    )
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # Required, not defaulted: this gate is backend-agnostic, so a default would silently
    # score whichever model the default names.
    p.add_argument("--config", required=True, help="path to the experiment YAML")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device)
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)

    student = backend.build_student(cfg).to(device).eval()
    ok = check_rope(student, device) & check_hidden_states(student, device)

    teacher = backend.load_teacher(cfg).to(device)
    ok &= check_teacher_exclusion(backend, teacher, task, cfg, device)

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
