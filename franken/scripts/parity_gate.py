"""With exact ops, full depth and teacher weights, the from-scratch student must BE the teacher.
Any gap is a module bug, not float noise, and every later FHE measurement assumes it is zero.

Pooled cosine alone does NOT settle it -- a dropped llama3 rope scaling and a wrong rms_norm_eps
both scored 0.99998, i.e. inside COS_THRESHOLD -- so the hidden-state relative error gates too.

Fails by design on FHE configs -- it is an exact-op gate.

Usage:
    uv run python -m franken.scripts.parity_gate --config configs/llama/gate_parity.yaml
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from franken.config import Config
from franken.models import build_backend
from franken.tasks import build_task

COS_THRESHOLD = 0.9999
# Per-entry relative RMS error against the teacher tensor's OWN scale, the shape
# `distill.loss.masked_relative_mse_loss` uses. Calibrated at both ends: correct ports read
# 1.5e-6 (llama 16L) and 1.7e-6 (qwen3 28L), while a dropped llama3 rope scaling reads 4.1e-3
# and rms_norm_eps 1e-6-instead-of-1e-5 reads 2.5e-2. 1e-4 is ~60x above the noise, ~40x under
# the weakest injected bug.
MAX_REL = 1e-4

# Mixed lengths on purpose: real padding exercises the pad mask and the pooling index.
PROBE_TEXTS = [
    "short",
    "The capital of France is Paris.",
    "Fully homomorphic encryption lets you compute directly on ciphertexts.",
    "Knowledge distillation transfers a teacher model's behaviour into a smaller "
    "student network trained to match its outputs rather than the ground-truth labels.",
]


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Required, not defaulted: this gate is backend-agnostic, so a default would silently
    # score whichever model the default names.
    p.add_argument("--config", required=True, help="path to the experiment YAML")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend = build_backend(cfg.model.backend)

    teacher = backend.load_teacher(cfg).to(device)
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    student = student.to(device).eval()

    # The task's own tokenizer: it pins padding_side, which this gate's pad-mask check depends on.
    tokenizer = build_task(cfg.train.task).build_tokenizer(cfg)
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

    # Real tokens only: nothing reads pad positions, and "sdpa_causal" leaves garbage there. The
    # raw max is still printed so a broken pad mask stays visible.
    keep = inputs["attention_mask"].bool().unsqueeze(-1)
    n = keep.sum().item() * sh[0].shape[-1]
    mses = [(((a - b) * keep) ** 2).sum().item() / n for a, b in zip(sh, th, strict=True)]
    dmaxes = [((a - b).abs() * keep).max().item() for a, b in zip(sh, th, strict=True)]
    raw = max((a - b).abs().max().item() for a, b in zip(sh, th, strict=True))
    print(
        f"hidden ({len(sh)} entries): MSE max={max(mses):.3e} |Δ|max={max(dmaxes):.3e} "
        f"on real tokens (raw incl. pad {raw:.3e})"
    )

    # Relative to each entry's own scale. |Δ|max cannot separate a bug from rounding (it carries
    # the stream's magnitude), and normalizing it by the ULP at its argmax normalizes by whatever
    # value happens to sit there -- qwen3's 41x LARGER |Δ|max scored 2 ULP where llama scored 100.
    rels = []
    for a, b in zip(sh, th, strict=True):
        den = ((b * keep) ** 2).sum().item()
        rels.append(((((a - b) * keep) ** 2).sum().item() / den) ** 0.5 if den else 0.0)
    worst = max(range(len(rels)), key=lambda i: rels[i])
    rel_ok = rels[worst] <= MAX_REL
    print(
        f"  worst relative RMS: {rels[worst]:.3e} at entry {worst} of {len(rels) - 1} "
        f"({'rounding' if rel_ok else 'CHECK THIS'}, threshold {MAX_REL:g})"
    )
    # Only on failure: a module bug concentrates in the entries after it, fp32 noise does not.
    if not rel_ok:
        print("  profile: " + " ".join(f"{r:.1e}" for r in rels))

    acts = backend.activation_ops(student)
    print(
        f"ops: preact={len(backend.ffn_preact_modules(student))} act={len(acts)} "
        f"{type(acts[0]).__name__} domain={getattr(acts[0], 'domain', None)}"
    )

    ok = cos.min().item() > COS_THRESHOLD and rel_ok
    print(
        f"PARITY GATE {'PASSED' if ok else 'FAILED'} "
        f"(cosine > {COS_THRESHOLD}, relative RMS <= {MAX_REL:g})\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
