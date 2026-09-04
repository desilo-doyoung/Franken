"""Score an LM student against its teacher: perplexity, top-1 agreement, KL.

Needs no pools, no judgements and no external benchmarks -- the teacher's own distribution is the
reference. Reported per registry source as well as overall, because a pointwise loss makes the
training distribution the region of fidelity: one aggregate hides which slice a depth cut cost.

Run with no --student-ckpt on a full-depth exact-op config for the identity self-test: agreement
must read 1.0000 and ppl must equal teacher_ppl.

    uv run python -m franken.scripts.llama.lm_eval --config configs/llama/smoke.yaml
"""

from __future__ import annotations

import json

import numpy as np
from torch.utils.data import DataLoader

from franken.data import corpus_sources
from franken.distill.batching import row_plan
from franken.distill.dist import max_tokens_per_rank
from franken.scripts import common
from franken.tasks import lm


def _loader(m, ds, collator, rows):
    # `select` leaves the underlying table at full width and `row_plan` reads it, so materialize.
    sub = ds.select(rows).flatten_indices()
    opt = m.cfg.train.distill
    if not opt.tokens_per_step:
        return DataLoader(sub, batch_size=opt.batch_size, collate_fn=collator)
    plan = row_plan(
        sub,
        min(opt.tokens_per_step, max_tokens_per_rank()),
        m.cfg.train.seed,
        m.cfg.train.max_seq_len if m.cfg.train.pack else None,
    )
    # The training planner drops its trailing partial so every step carries identical tokens. Per
    # source that is up to 3 rows of 15, so score the remainder as its own short batch.
    seen = {i for b in plan for i in b}
    if rest := [i for i in range(len(sub)) if i not in seen]:
        plan.append(rest)
    return DataLoader(sub, batch_sampler=plan, collate_fn=collator)


def by_source(m, split: str) -> dict:
    """One loader per source, the way the embed scorer builds one pool per source. The metrics are
    sums over positions, so scoring sources separately and adding the parts is the same arithmetic
    as one pass -- no second traversal, and the overall row cannot disagree with its own breakdown.
    """
    data = m.task.datasets(m.tokenizer, m.cfg, splits=(split,))
    ds = data[split].with_format("torch", columns=m.task.torch_columns())
    col = ds.data.column("source").to_numpy(zero_copy_only=False)

    out = {}
    for i, src in enumerate(corpus_sources(m.cfg.train.corpus)):
        rows = np.flatnonzero(col == i).tolist()
        if not rows:
            continue  # a fixed-size split need not reach every source
        loader = _loader(m, ds, data["collator"], rows)
        totals = lm.score_totals(m.backend, m.task, m.student, m.teacher, loader, m.device)
        out[src.name] = totals
    return out


def _table(rows: dict) -> list[str]:
    head = (
        f"  {'source':<16}{'agree':>8}{'kl':>10}{'ppl':>10}{'teacher':>10}{'delta':>9}{'tok':>11}"
    )
    out = [head, "  " + "-" * (len(head) - 2)]
    for name, totals in sorted(rows.items(), key=lambda kv: -kv[1]["positions"]):
        s = lm.metrics(totals)
        out.append(
            f"  {name:<16}{s['agreement']:>8.4f}{s['kl']:>10.4f}{s['ppl']:>10.3f}"
            f"{s['teacher_ppl']:>10.3f}{s['ppl'] / s['teacher_ppl'] - 1:>+9.2%}"
            f"{int(totals['positions']):>11,}"
        )
    return out


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    # Validation selects the checkpoint, test reports it; never score the model on the split that
    # picked it.
    p.add_argument("--split", default="validation", choices=("validation", "test"))
    args = p.parse_args(argv)

    m = common.load(args)
    rows = by_source(m, args.split)
    metrics = lm.metrics(lm.sum_totals(rows.values()))

    print(f"\n{args.split}  ({m.cfg.train.corpus})")
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.4f}")
    print(f"  {'ppl delta':<14} {metrics['ppl'] / metrics['teacher_ppl'] - 1:+.2%}")
    print("\nby source:")
    print("\n".join(_table(rows)))

    if args.json:
        metrics["by_source"] = {
            n: lm.metrics(t) | {"positions": int(t["positions"])} for n, t in rows.items()
        }
        with open(args.json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nwrote {args.json}")
