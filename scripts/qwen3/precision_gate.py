"""Check that `precision: bf16` is safe for this architecture, before spending GPU-hours on it.

bf16 is safe here only because autocast never stores the big activations in bf16: RMSNorm is
promoted to fp32 and `residual + hidden` promotes fp32+bf16 -> fp32, so the residual stream
(= hidden_states, the hidden-loss input) stays fp32. That is load-bearing, so assert it, along
with the two things that would silently break it: RoPE staying fp32, and the teacher staying
out of the autocast region.

    uv run python scripts/qwen3/precision_gate.py --config configs/qwen3/depth28_control.yaml
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from franken.config import Config  # noqa: E402
from franken.models import build_backend  # noqa: E402
from franken.tasks import build_task  # noqa: E402
from franken.tasks.embed import recall_at_k  # noqa: E402

# A bf16 teacher must move the targets by MORE than the comparison band, otherwise the
# "teacher stays fp32" rule is decoration rather than a real constraint.
BAND = 0.004


def _fail(msg):
    print(f"FAIL  {msg}")
    return False


def check_rope(model, device):
    """cos/sin must stay fp32 inside an autocast region. torch.einsum is intercepted by
    autocast on name and returns bf16 despite a .float() input, which corrupts the angles;
    rope.py uses a broadcast multiply for exactly this reason."""
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
    """Embed the validation pool with the teacher in fp32 and again under bf16 autocast. If
    those agree within the band, excluding the teacher from autocast buys nothing and this
    check is worthless -- so a PASS here means they DISAGREE."""
    tokenizer = task.build_tokenizer(cfg)
    data = task.datasets(tokenizer, cfg)
    ds = data["validation"].with_format("torch", columns=task.torch_columns())
    from torch.utils.data import DataLoader

    batches = list(DataLoader(ds, batch_size=16, collate_fn=data["collator"]))

    def embed(autocast):
        out = []
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if autocast
                else torch.autocast("cuda", enabled=False)
            )
            with ctx:
                out.append(backend.forward(teacher, task.model_inputs(batch))["output"].float())
        return torch.cat(out).cpu()

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
    p.add_argument("--config", default="configs/qwen3/depth28_control.yaml")
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
