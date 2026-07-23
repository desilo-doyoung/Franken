"""Activation-range / FHE self-containment check for a trained student.

Runs the student over MRPC splits and records, per FFN layer:
  * pre-activation range  (intermediate.dense output -> input to the poly GELU)
  * activation output range (intermediate_act_fn output)

For the quad GELU with domain D, self-containment means every pre-activation
stays within [-D, D] (so the deployed bare poly is never fed out-of-domain) and
the output magnitude stays <= ~0.125*D^2 (the FHE dynamic-range budget). Prints
a numeric per-layer table and writes a pre-activation histogram PNG.

Usage:
    python scripts/bert/act_range.py --config configs/bert/quad_cgf_fhe.yaml --out preact.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from datasets import load_dataset
from franken.config import Config
from franken.models import build_backend
from franken.paths import RunPaths
from franken.tasks import build_task
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/bert/quad_cgf_fhe.yaml")
    p.add_argument("--student-ckpt", default=None)
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="preact.png", help="histogram PNG path")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backend = build_backend(cfg.model.backend)
    task = build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    student = backend.build_student(cfg)
    sc = args.student_ckpt or RunPaths(cfg).student_bin()
    student.load_state_dict(torch.load(sc, map_location=device))
    student = student.to(device).eval()

    # Model-agnostic access to the FFN pre-activation modules and activation ops.
    pre_modules = backend.ffn_preact_modules(student)
    act_modules = backend.activation_ops(student)
    n_layers = len(pre_modules)
    domain = getattr(act_modules[0], "domain", None) if act_modules else None

    # Capture pre-activations (FFN dense out) and post-activations (act op out) per layer.
    preacts = {i: [] for i in range(n_layers)}
    postacts = {i: [] for i in range(n_layers)}
    hooks = []
    for i, (pm, am) in enumerate(zip(pre_modules, act_modules, strict=True)):
        hooks.append(
            pm.register_forward_hook(
                lambda m, inp, out, i=i: preacts[i].append(out.detach().float().flatten().cpu())
            )
        )
        hooks.append(
            am.register_forward_hook(
                lambda m, inp, out, i=i: postacts[i].append(out.detach().float().flatten().cpu())
            )
        )

    ds = load_dataset("nyu-mll/glue", "mrpc")
    ds = ds.map(
        lambda b: tokenizer(
            b["sentence1"], b["sentence2"], truncation=True, max_length=cfg.train.max_seq_len
        ),
        batched=True,
    )
    collator = DataCollatorWithPadding(tokenizer)
    cols = task.torch_columns()

    for split in args.splits:
        if set(ds[split].unique("label")) == {-1}:
            continue
        d = ds[split].with_format("torch", columns=cols)
        dl = DataLoader(d, batch_size=args.batch_size, collate_fn=collator)
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            backend.forward(student, task.model_inputs(batch))

    for h in hooks:
        h.remove()

    pre = {i: torch.cat(v) for i, v in preacts.items()}
    post = {i: torch.cat(v) for i, v in postacts.items()}

    print(f"\nsplits={args.splits}  domain={domain}")
    print(
        f"{'layer':>5} {'pre min':>9} {'pre max':>9} {'pre p99.9':>10} "
        f"{'post min':>9} {'post max':>9} {'>|D| frac':>10}"
    )
    print("-" * 66)
    all_pre_max = 0.0
    all_post_max = 0.0
    for i in range(n_layers):
        x = pre[i]
        y = post[i]
        pmin, pmax = x.min().item(), x.max().item()
        xa = x.abs()
        # torch.quantile caps at ~16M elems; subsample for the percentile.
        if xa.numel() > 5_000_000:
            idx = torch.randint(0, xa.numel(), (5_000_000,))
            p999 = torch.quantile(xa[idx], 0.999).item()
        else:
            p999 = torch.quantile(xa, 0.999).item()
        ymin, ymax = y.min().item(), y.max().item()
        over = (x.abs() > domain).float().mean().item() if domain else float("nan")
        all_pre_max = max(all_pre_max, abs(pmin), abs(pmax))
        all_post_max = max(all_post_max, abs(ymin), abs(ymax))
        print(
            f"{i:>5} {pmin:>9.2f} {pmax:>9.2f} {p999:>10.2f} "
            f"{ymin:>9.2f} {ymax:>9.2f} {over:>10.5f}"
        )

    print("-" * 66)
    print(f"overall  pre |max| = {all_pre_max:.2f}   post |max| = {all_post_max:.2f}")
    if domain is not None:
        contained = all_pre_max <= domain
        verdict = "SELF-CONTAINED" if contained else "OUT OF DOMAIN"
        rel = "<=" if contained else ">"
        print(f"domain = {domain}  ->  pre-acts {verdict} (|max| {all_pre_max:.2f} {rel} {domain})")
        print(f"quad output budget 0.125*D^2 = {0.125 * domain**2:.1f}  (teacher GELU |max| = 143)")

    # Histogram of all pre-activations pooled across layers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = torch.cat([pre[i] for i in range(n_layers)]).numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pooled, bins=300, color="#4C72B0", log=True)
    if domain is not None:
        ax.axvline(domain, color="crimson", ls="--", lw=1.5, label=f"±domain ({domain:g})")
        ax.axvline(-domain, color="crimson", ls="--", lw=1.5)
    ax.set_xlabel("FFN pre-activation")
    ax.set_ylabel("count (log)")
    ax.set_title(f"quad+cgf student — FFN pre-activations ({'+'.join(args.splits)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nhistogram -> {args.out}")


if __name__ == "__main__":
    main()
