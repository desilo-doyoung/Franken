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

SAMPLE = 20_000  # values kept per layer per batch, so quantiles fit in memory


def _sample(x):
    x = x.flatten().float()
    if x.numel() > SAMPLE:
        x = x[torch.randint(0, x.numel(), (SAMPLE,), device=x.device)]
    return x.cpu()


def _report(title, per_layer, domain=None):
    print(f"\n{title}")
    head = f"{'layer':>5} {'min':>10} {'max':>10} {'p99.9|x|':>10}"
    print(head + (f" {'%|x|>D':>9}" if domain else ""))
    worst = 0.0
    for i in sorted(per_layer):
        x = torch.cat(per_layer[i])
        worst = max(worst, x.abs().max().item())
        row = f"{i:>5} {x.min():>10.2f} {x.max():>10.2f} {x.abs().quantile(0.999):>10.2f}"
        if domain:
            row += f" {(x.abs() > domain).float().mean() * 100:>8.3f}%"
        print(row)
    print(f"  max|x| over all layers: {worst:.1f}")
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
                lambda m, inp, out, i=i: preact.setdefault(i, []).append(
                    _sample(out[mask_holder["m"]])
                )
            )
        )
    for i, layer in enumerate(model.layers):
        # Scores are the softmax's first argument, i.e. pre-mask. Keep only entries that are
        # actually visible: both tokens real, and key <= query (causal).
        hooks.append(
            layer.self_attn.softmax.register_forward_pre_hook(
                lambda m, a, i=i: scores.setdefault(i, []).append(
                    _sample(a[0][mask_holder["vis"].expand_as(a[0])])
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
    worst = _report("FFN pre-activations — input to the polynomial activation", preact, domain)
    _report("attention scores — input to the softmax (visible entries only)", scores)

    over = [i for i in sorted(preact) if torch.cat(preact[i]).abs().max() > domain]
    print(f"\nlayers exceeding D={domain}: {over or 'none'}")
    print(f"a single domain covering every layer would need D >= {worst:.0f}\n")


if __name__ == "__main__":
    main()
