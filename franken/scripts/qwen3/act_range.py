"""What the FHE-approximated operators actually see, per layer.

The domain must be picked from the OPERATOR's input, not from hidden states: RMSNorm strips
Qwen3's massive activations first, so hidden-state stats overestimate it ~20x, and domain costs
multiplicative depth.

Usage:
    uv run python -m franken.scripts.qwen3.act_range --config configs/qwen3/gate_parity.yaml
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from franken.scripts import common

SAMPLE = 20_000  # values kept per layer per batch, for quantiles only


def _record(store, i, x, domain):
    """min/max and the over-domain count are EXACT: with no clamp at inference the single largest
    value decides safety. Only the quantile is subsampled."""
    x = x.flatten().float()
    s = store.setdefault(i, {"min": float("inf"), "max": -float("inf"), "over": 0, "n": 0, "q": []})
    s["min"] = min(s["min"], x.min().item())
    s["max"] = max(s["max"], x.max().item())
    s["over"] += int((x.abs() > domain).sum())
    s["n"] += x.numel()
    s["q"].append(
        x[torch.randint(0, x.numel(), (SAMPLE,), device=x.device)] if x.numel() > SAMPLE else x
    )


def _report(title, store, show_over):
    print(f"\n{title}")
    print(
        f"{'layer':>5} {'min':>10} {'max':>10} {'p99.9|x|':>10}"
        + (f" {'%|x|>D':>9}" if show_over else "")
    )
    worst = 0.0
    for i in sorted(store):
        s = store[i]
        worst = max(worst, abs(s["min"]), abs(s["max"]))
        p999 = torch.cat(s["q"]).abs().quantile(0.999)
        row = f"{i:>5} {s['min']:>10.2f} {s['max']:>10.2f} {p999:>10.2f}"
        if show_over:
            row += f" {100 * s['over'] / s['n']:>8.3f}%"
        print(row)
    print(f"  max|x| over all layers: {worst:.1f}  (exact, not subsampled)")
    return worst


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__, json=False)
    args = p.parse_args(argv)

    m = common.load(args)
    cfg, backend, task, tokenizer, device = m.cfg, m.backend, m.task, m.tokenizer, m.device
    torch.manual_seed(cfg.train.seed)
    model = m.student
    del m.teacher  # only needed to seed the student; the ranges are the student's

    acts = backend.activation_ops(model)
    domain = getattr(acts[0], "domain", None) or 32.0
    print(f"domain reference D={domain}")

    preact, scores, mask_holder = {}, {}, {}
    hooks = []
    for i, module in enumerate(backend.ffn_preact_modules(model)):
        hooks.append(
            module.register_forward_hook(
                lambda m, inp, out, i=i: _record(preact, i, out[mask_holder["m"]], domain)
            )
        )
    for i, softmax in enumerate(backend.softmax_ops(model)):
        # Pre-mask, so keep only visible entries: both tokens real and key <= query.
        hooks.append(
            softmax.register_forward_pre_hook(
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
    if scores:
        _report("attention scores — input to the softmax (visible entries only)", scores, False)
    else:
        # An empty store would print "max 0.0", which reads as a measurement.
        print(
            f"\nattention scores: NOT MEASURED — attn_impl={cfg.model.attn_impl!r} fuses the "
            f"softmax so its hook never fires. Re-run with 'manual' when softmax is not 'exact' "
            f"(here: {cfg.model.softmax!r})."
        )

    over = [i for i in sorted(preact) if max(abs(preact[i]["min"]), preact[i]["max"]) > domain]
    print(f"\nlayers exceeding D={domain}: {over or 'none'}")
    print(f"a single domain covering every layer would need D >= {worst:.0f}\n")


if __name__ == "__main__":
    main()
