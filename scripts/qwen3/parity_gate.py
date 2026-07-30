"""Parity gate: with exact ops + full depth + teacher weights, the from-scratch
Qwen3 student must BE the teacher. Any gap is a module bug (RoPE/QK-norm order,
repeat_kv, causal+pad mask, hidden_states bookkeeping), not float noise — and every
later FHE measurement assumes that gap is zero when the ops are exact.

Fails by design on FHE configs (cgf / polynomial activation); it's an exact-op gate.

Usage:
    uv run python scripts/qwen3/parity_gate.py --config configs/qwen3/exact.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from franken.config import Config
from franken.models import build_backend
from transformers import AutoTokenizer

COS_THRESHOLD = 0.9999
MAX_ULP = 16  # accumulated summation-order noise over 28 layers; a bug is orders larger

# Mixed lengths on purpose: real padding exercises the additive pad mask and the
# last-non-pad-token pooling index instead of reducing to "the last column".
PROBE_TEXTS = [
    "short",
    "The capital of France is Paris.",
    "Fully homomorphic encryption lets you compute directly on ciphertexts.",
    "Knowledge distillation transfers a teacher model's behaviour into a smaller "
    "student network trained to match its outputs rather than the ground-truth labels.",
]


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/exact.yaml")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend = build_backend(cfg.model.backend)

    teacher = backend.load_teacher(cfg).to(device)
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    student = student.to(device).eval()

    # The embed task will own this; until it exists, build it the same way it will.
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
    enc = tokenizer(
        PROBE_TEXTS,
        padding=True,
        truncation=True,
        max_length=cfg.train.max_seq_len,
        return_tensors="pt",
    )
    inputs = {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}

    t_out = backend.forward(teacher, inputs)
    s_out = backend.forward(student, inputs)

    cos = F.cosine_similarity(s_out["output"], t_out["output"], dim=-1)
    th, sh = t_out["hidden_states"], s_out["hidden_states"]

    print(
        f"\ndepth={cfg.model.num_hidden_layers}/{teacher.config.num_hidden_layers} "
        f"softmax={cfg.model.softmax} activation={cfg.model.activation}\n"
        f"padded={tuple(inputs['input_ids'].shape)} pad={tokenizer.padding_side} "
        f"lengths={inputs['attention_mask'].sum(-1).tolist()}"
    )
    print(f"pooled cosine: min={cos.min().item():.8f} all={[round(c, 6) for c in cos.tolist()]}")

    if len(th) != len(sh):
        print(f"FAIL: hidden_states {len(sh)} != {len(th)}; the gate needs full teacher depth.")
        return 1

    # Judge on real tokens only: pad positions are read by nothing (masked in the loss,
    # excluded by last-token pooling), and attn_impl "sdpa_causal" leaves garbage there by
    # design. The raw max is still printed so a broken pad mask stays visible.
    keep = inputs["attention_mask"].bool().unsqueeze(-1)
    n = keep.sum().item() * sh[0].shape[-1]
    mses = [(((a - b) * keep) ** 2).sum().item() / n for a, b in zip(sh, th, strict=True)]
    dmaxes = [((a - b).abs() * keep).max().item() for a, b in zip(sh, th, strict=True)]
    raw = max((a - b).abs().max().item() for a, b in zip(sh, th, strict=True))
    print(
        f"hidden ({len(sh)} entries): MSE max={max(mses):.3e} |Δ|max={max(dmaxes):.3e} "
        f"on real tokens (raw incl. pad {raw:.3e})"
    )

    # |Δ|max lands on Qwen3's massive-activation channels (|h| in the thousands), where one
    # fp32 ULP is already ~5e-4 absolute. So measure the gap in ULPs of the value it sits on:
    # a few ULPs is accumulated summation-order noise, orders more is a logic bug.
    worst = max(range(len(th)), key=lambda i: dmaxes[i])
    d = (sh[worst] - th[worst]).abs() * keep
    at = th[worst].flatten()[d.argmax()].abs().item()
    ulp = math.ldexp(1.0, math.frexp(at)[1] - 24) if at else float("inf")
    print(
        f"  worst: entry {worst} on |teacher|={at:.6g} -> {d.max().item() / ulp:.1f} ULP "
        f"({'rounding' if d.max().item() / ulp <= MAX_ULP else 'CHECK THIS'})"
    )

    acts = backend.activation_ops(student)
    print(
        f"ops: preact={len(backend.ffn_preact_modules(student))} act={len(acts)} "
        f"{type(acts[0]).__name__} domain={getattr(acts[0], 'domain', None)}"
    )

    ok = cos.min().item() > COS_THRESHOLD
    print(f"PARITY GATE {'PASSED' if ok else 'FAILED'} (threshold {COS_THRESHOLD})\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
