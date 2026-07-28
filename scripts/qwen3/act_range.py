"""What the FHE-approximated operators actually see, per layer.

Polynomial activations explode outside their domain and there is no clamp at inference,
so the domain must be picked from the *operator's input* — `gate_proj` output for the
activation, attention scores for the softmax — not from hidden states: RMSNorm strips
Qwen3's massive activations before `gate_proj`, so hidden-state stats overestimate the
required domain by ~20x, and domain costs multiplicative depth.

Usage:
    uv run python scripts/qwen3/act_range.py --config configs/qwen3/exact.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from franken.config import Config
from franken.models import build_backend
from franken.tasks import build_task
from torch.utils.data import DataLoader

SAMPLE = 20_000  # values kept per layer per batch, for quantiles only


def _record(store, i, x, domain):
    """min/max/over-domain counts are EXACT over every value — with no clamp at inference the
    single largest value decides safety, so that statistic must never be subsampled. Only the
    quantile uses a subsample."""
    x = x.flatten().float()
    s = store.setdefault(i, {"min": float("inf"), "max": -float("inf"), "over": 0, "n": 0, "q": []})
    s["min"] = min(s["min"], x.min().item())
    s["max"] = max(s["max"], x.max().item())
    s["over"] += int((x.abs() > domain).sum())
    s["n"] += x.numel()
    s["q"].append(x[torch.randint(0, x.numel(), (SAMPLE,), device=x.device)] if x.numel() > SAMPLE else x)


def _report(title, store, show_over):
    print(f"\n{title}")
    print(f"{'layer':>5} {'min':>10} {'max':>10} {'p99.9|x|':>10}" + (f" {'%|x|>D':>9}" if show_over else ""))
    worst = 0.0
    for i in sorted(store):
        s = store[i]
        worst = max(worst, abs(s["min"]), abs(s["max"]))
        row = f"{i:>5} {s['min']:>10.2f} {s['max']:>10.2f} {torch.cat(s['q']).abs().quantile(0.999):>10.2f}"
        if show_over:
            row += f" {100 * s['over'] / s['n']:>8.3f}%"
        print(row)
    print(f"  max|x| over all layers: {worst:.1f}  (exact, not subsampled)")
    return worst


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/exact.yaml")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device)
    model = backend.build_student(cfg)
    backend.seed_student(model, teacher, cfg)
    if args.student_ckpt:
        model.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    model = model.to(device).eval()
    del teacher

    acts = backend.activation_ops(model)
    domain = getattr(acts[0], "domain", None) or 32.0
    print(f"\nstudent: {args.student_ckpt or 'IDENTITY (seeded from teacher)'}")
    print(f"act={cfg.model.activation} softmax={cfg.model.softmax} | domain reference D={domain}")

    preact, scores, mask_holder = {}, {}, {}
    hooks = []
    for i, module in enumerate(backend.ffn_preact_modules(model)):
        hooks.append(
            module.register_forward_hook(
                lambda m, inp, out, i=i: _record(preact, i, out[mask_holder["m"]], domain)
            )
        )
    for i, layer in enumerate(model.layers):
        # Scores are the softmax's first argument, i.e. pre-mask. Keep only entries that are
        # actually visible: both tokens real, and key <= query (causal).
        hooks.append(
            layer.self_attn.softmax.register_forward_pre_hook(
                lambda m, a, i=i: _record(
                    scores, i, a[0][mask_holder["vis"].expand_as(a[0])], domain
                )
            )
        )

    data = task.datasets(tokenizer, cfg)
    ds = data["validation"].with_format("torch", columns=task.torch_columns())
    n_tok = 0
    for batch in DataLoader(ds, batch_size=16, collate_fn=data["collator"]):
        batch = {k: v.to(device) for k, v in batch.items()}
        am = batch["attention_mask"].bool()
        S = am.size(1)
        causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
        mask_holder["m"] = am
        mask_holder["vis"] = am[:, None, :, None] & am[:, None, None, :] & causal
        n_tok += int(am.sum())
        backend.forward(model, task.model_inputs(batch))
    for h in hooks:
        h.remove()

    print(f"\n{len(ds)} texts, {n_tok} real tokens")
    worst = _report("FFN pre-activations — input to the polynomial activation", preact, True)
    _report("attention scores — input to the softmax (visible entries only)", scores, False)

    over = [i for i in sorted(preact) if max(abs(preact[i]["min"]), preact[i]["max"]) > domain]
    print(f"\nlayers exceeding D={domain}: {over or 'none'}")
    print(f"a single domain covering every layer would need D >= {worst:.0f}\n")


if __name__ == "__main__":
    main()
