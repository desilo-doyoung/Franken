"""Rank layer-cut choices WITHOUT training: seed a student under each candidate map and score it.

`seed_student` copies teacher block `map[i]` into student block i, so a map fully determines the
untrained student. Scoring that student against the teacher asks which blocks the residual stream
can lose -- cheaply, before committing a day of distillation to one map.

Two passes: `loo` drops one block at a time to rank per-block importance, then `cut` scores whole
candidate maps at the target depth, including the greedy map built from `loo`.

🚨 A FILTER, NEVER A RANKER. This scores INITIALIZATION, and the qwen3 track already measured what
that is worth: init-vs-trained rank correlation +0.77, inverting every close call -- init called
stride 4-6x worse where trained said 2%. Training loss ranked maps correctly; init did not. So use
this only to eliminate the shapes that are off by 10-100x, then rank the survivors by a short
training run. Reading the fine order off this table is the mistake it exists to prevent.

⚠️ qwen3 also found CONTIGUITY IRRELEVANT once trained (`spread` landed inside the family), so a
contiguity gap here is an init artifact until a trained run says otherwise. What did replicate
across both models is END ASYMMETRY: early blocks build the representation and are the expensive
ones to remove.

    uv run python -m franken.scripts.llama.layer_search --config configs/llama/smoke.yaml --depth 12
"""

from __future__ import annotations

import copy
import json
import time

import torch
from torch.utils.data import DataLoader

from franken.distill.batching import row_plan
from franken.distill.dist import max_tokens_per_rank
from franken.scripts import common
from franken.tasks import lm


def _spread(lo: int, hi: int, n: int) -> set[int]:
    """`n` blocks spaced evenly across [lo, hi] inclusive."""
    if n == 1:
        return {(lo + hi) // 2}
    step = (hi - lo) / (n - 1)
    return {lo + round(i * step) for i in range(n)}


def patterns(teacher_depth: int, depth: int) -> dict[str, set[int]]:
    """Named cut shapes -> which teacher blocks to DROP."""
    n = teacher_depth - depth
    mid = (teacher_depth - n) // 2
    uniform = {round((i + 1) * teacher_depth / depth) - 1 for i in range(depth)}
    return {
        "stride_default": set(range(teacher_depth)) - uniform,
        "early": set(range(n)),
        "early_keep_first": set(range(1, n + 1)),
        "quarter_2": set(range(teacher_depth // 4, teacher_depth // 4 + n)),
        "middle": set(range(mid, mid + n)),
        "quarter_3": set(range(teacher_depth // 2, teacher_depth // 2 + n)),
        "late_keep_last": set(range(teacher_depth - n - 1, teacher_depth - 1)),
        "late": set(range(teacher_depth - n, teacher_depth)),
        # Both ends protected: the embedding-adjacent block and the one feeding the LM head.
        "spread_interior": _spread(1, teacher_depth - 2, n),
    }


def strat_stride(td: int, depth: int, loo=None) -> list[int]:
    """`resolve_layer_map`'s default -- the baseline to beat."""
    return sorted({round((i + 1) * td / depth) - 1 for i in range(depth)})


def strat_protect_ends_stride(td: int, depth: int, loo=None) -> list[int]:
    """Stride, but blocks 0, 1 and the last are never candidates."""
    fixed = [0, 1, td - 1]
    n = depth - len(fixed)
    pool = list(range(2, td - 1))
    return sorted(fixed + [pool[round(i * (len(pool) - 1) / max(n - 1, 1))] for i in range(n)])


def strat_interior_window(td: int, depth: int, loo=None, center: float = 0.45) -> list[int]:
    """Drop one contiguous window from the interior, centered at `center` of depth. Never touches
    block 0, 1 or the last -- the only asymmetry that replicated across models."""
    n = td - depth
    start = min(max(round(center * td - n / 2), 2), td - 1 - n)
    return sorted(set(range(td)) - set(range(start, start + n)))


def strat_greedy_loo(td: int, depth: int, loo=None) -> list[int]:
    """Drop the `n` cheapest single blocks. Ignores interactions by construction, which is why it
    is scored head-to-head rather than trusted."""
    rank = sorted(range(td), key=lambda i: -loo[f"drop_{i}"]["agreement"])
    return sorted(set(range(td)) - set(rank[: td - depth]))


STRATEGIES = {
    "stride": strat_stride,
    "protect_ends": strat_protect_ends_stride,
    "interior_window": strat_interior_window,
    "greedy_loo": strat_greedy_loo,
}


def fixed_loader(m, split: str, rows: int):
    """One fixed batch list reused by every candidate, so the comparison is paired -- differences
    are the map's, not the sample's."""
    data = m.task.datasets(m.tokenizer, m.cfg, splits=(split,))
    ds = data[split].with_format("torch", columns=m.task.torch_columns())
    ds = ds.select(range(min(rows, len(ds)))).flatten_indices()
    opt = m.cfg.train.distill
    plan = row_plan(
        ds,
        min(opt.tokens_per_step, max_tokens_per_rank()),
        m.cfg.train.seed,
        m.cfg.train.max_seq_len if m.cfg.train.pack else None,
    )
    seen = {i for b in plan for i in b}
    if rest := [i for i in range(len(ds)) if i not in seen]:
        plan.append(rest)
    return DataLoader(ds, batch_sampler=plan, collate_fn=data["collator"])


def score_map(m, student, keep: list[int], loader) -> dict:
    cfg = copy.deepcopy(m.cfg)
    cfg.model.num_hidden_layers = len(keep)
    cfg.distill.hidden_layer_map = list(keep)
    m.backend.seed_student(student, m.teacher, cfg)
    student.eval().requires_grad_(False)
    totals = lm.score_totals(m.backend, m.task, student, m.teacher, loader, m.device)
    return lm.metrics(totals)


def build_student(m, depth: int):
    cfg = copy.deepcopy(m.cfg)
    cfg.model.num_hidden_layers = depth
    return m.backend.build_student(cfg).to(m.device).eval().requires_grad_(False)


def run_pass(m, depth: int, maps: dict[str, list[int]], loader, teacher_depth: int) -> dict:
    """All candidates at one depth share a student module; only the weights are reseeded."""
    student = build_student(m, depth)
    out = {}
    for name, keep in maps.items():
        t0 = time.monotonic()
        out[name] = score_map(m, student, keep, loader) | {"keep": keep}
        r = out[name]
        drop = sorted(set(range(teacher_depth)) - set(keep))
        print(
            f"  {name:<18} agree {r['agreement']:.4f}  kl {r['kl']:>7.3f}  ppl {r['ppl']:>9.2f}"
            f"  ({time.monotonic() - t0:.0f}s)  drop={drop}",
            flush=True,
        )
    del student
    torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--depth", type=int, default=12, help="student depth for the cut pass")
    p.add_argument("--rows", type=int, default=48, help="corpus rows scored per candidate")
    p.add_argument("--split", default="validation", choices=("validation", "test"))
    p.add_argument("--skip-loo", action="store_true", help="cut pass only")
    p.add_argument("--sweep", help="score every STRATEGY across a depth range, e.g. 8-15")
    p.add_argument("--loo-json", help="reuse a previous run's leave-one-out instead of rescoring")
    args = p.parse_args(argv)

    m = common.load(args)
    td = m.teacher.config.num_hidden_layers
    loader = fixed_loader(m, args.split, args.rows)
    tokens = sum(int(b["attention_mask"].sum()) for b in loader)
    print(f"\nteacher depth {td}, scoring {tokens:,} tokens per candidate ({args.split})")

    result = {"teacher_depth": td, "depth": args.depth, "tokens": tokens}

    loo = {}
    if args.loo_json:
        with open(args.loo_json) as f:
            loo = json.load(f)["loo"]
        print(f"reusing leave-one-out from {args.loo_json}")
    elif not args.skip_loo:
        print(f"\nleave-one-out (depth {td - 1}) -- which single block does the stream miss most:")
        maps = {f"drop_{i}": [j for j in range(td) if j != i] for i in range(td)}
        loo = run_pass(m, td - 1, maps, loader, td)
        result["loo"] = loo
        rank = sorted(range(td), key=lambda i: -loo[f"drop_{i}"]["agreement"])
        print(f"\n  cheapest to drop -> costliest: {rank}")

    if args.sweep:
        lo, hi = (int(x) for x in args.sweep.split("-"))
        for depth in range(hi, lo - 1, -1):
            maps = {}
            for name, fn in STRATEGIES.items():
                if name == "greedy_loo" and not loo:
                    continue
                keep = fn(td, depth, loo)
                assert len(keep) == depth, (name, depth, keep)
                maps[name] = keep
            print(f"\ndepth {depth} (drop {td - depth}):")
            result.setdefault("sweep", {})[depth] = run_pass(m, depth, maps, loader, td)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nwrote {args.json}")
        return

    n = td - args.depth
    cuts: dict[str, list[int]] = {}
    for name, drop in patterns(td, args.depth).items():
        keep = sorted(set(range(td)) - drop)
        # Shapes collide at large n (at depth 8, `middle` IS `quarter_2`); scoring both twice
        # would only add noise to the table.
        if keep not in cuts.values():
            cuts[name] = keep
    if loo:
        # Greedy from the leave-one-out ranking; ignores interactions, which is the point of
        # scoring it head-to-head with the structured shapes rather than trusting it.
        greedy = sorted(range(td), key=lambda i: -loo[f"drop_{i}"]["agreement"])[:n]
        cuts["greedy_loo"] = sorted(set(range(td)) - set(greedy))
    print(f"\ncut to depth {args.depth} (drop {n}):")
    result["cut"] = run_pass(m, args.depth, cuts, loader, td)

    best = max(result["cut"].items(), key=lambda kv: kv[1]["agreement"])
    print(f"\nbest by agreement: {best[0]}  keep={best[1]['keep']}")
    print(f"  hidden_layer_map: {best[1]['keep']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
