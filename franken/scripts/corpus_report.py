"""The corpus tables both tracks print: holdout overlap, length profile, and what the build landed.

Formatting only -- measurement lives in `franken.data.corpus`. Each track's script adds the verdict
its objective needs (the embed track gates on scoreability; `lm` has nothing to gate).
"""

from __future__ import annotations

import os
import time

from franken.data.corpus import SPLITS, describe, profile, realized_mix, source_texts
from franken.data.corpus.build import train_cache_path

SAMPLE = 300  # texts per source for the length profile
HOLDOUT_SAMPLE = 100  # texts per split per source, for the overlap check


def check(sources) -> bool:
    """Only upstream-split sources need streaming -- for a hash-split source disjointness is a
    theorem. Splits are compared on mean length too: clean is not the same as representative."""
    native = [s for s in sources if s.key is None]
    print(
        f"holdout, upstream-split sources only ({len(native)} of {len(sources)}; "
        f"hash-split sources are disjoint by construction)\n"
        f"{'source':<18} {'train':>7} {'val':>6} {'test':>6}  {'mean len':>22}  verdict",
        flush=True,
    )
    ok = True
    for src in native:
        try:
            drawn = {s: set(source_texts(src, s, HOLDOUT_SAMPLE)) for s in SPLITS}
        except Exception as e:
            print(f"{src.name:<18}  FAILED {type(e).__name__}: {e}"[:110], flush=True)
            ok = False
            continue
        bad = [
            f"{a}/{b}={n}"
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
            if (n := len(drawn[a] & drawn[b]))
        ]
        means = {s: sum(map(len, t)) / max(len(t), 1) for s, t in drawn.items()}
        hi = max(means.values())
        skewed = (hi - min(means.values())) / max(hi, 1e-9) > 0.25
        counts = "  ".join(f"{len(drawn[s]):>6,}" for s in SPLITS)
        lens = "  ".join(f"{means[s]:>6.0f}" for s in SPLITS)
        verdict = "OVERLAP " + " ".join(bad) if bad else ("SPLITS DISAGREE" if skewed else "ok")
        print(f"{src.name:<18} {counts}  {lens}  {verdict}", flush=True)
        ok = ok and not bad and not skewed
    return ok


def lengths(cfg, sources, tokenizer, extra_col: str = "") -> tuple[bool, list]:
    """Per-source length profile. Returns (every source loaded, the rows) so the caller can add
    its own verdict over the same measurement rather than re-streaming."""
    cap = cfg.train.max_seq_len
    print(f"\n{cfg.train.corpus}: {len(sources)} sources, max_seq_len {cap}")
    print(
        f"{'source':<18} {'domain':<14} {'weight':>7} {'mean':>7} {'median':>7} "
        f"{'trunc':>7} {'longest':>8} {extra_col}",
        flush=True,
    )
    rows = profile(sources, tokenizer, cap, n=SAMPLE)
    for r in rows:
        if r.error:
            print(f"{r.name:<18} {r.domain:<14}  FAILED {r.error}"[:110], flush=True)
            continue
        tail = f" {r.scoreable}" if extra_col else ""
        print(
            f"{r.name:<18} {r.domain:<14} {r.weight:>6.1%} {r.mean:>7.1f} {r.median:>7d} "
            f"{r.truncated:>6.1%} {r.longest:>8,}{tail}",
            flush=True,
        )
    if failed := [r.name for r in rows if r.error]:
        print(f"\nFAILED TO LOAD: {', '.join(failed)}")
    return not failed, rows


def build(cfg, task, tokenizer, sources) -> int:
    """Build and cache if needed, then report what the artifact holds. Prints on cache hits too,
    the only place the realized token count shows up. Returns unique train tokens."""
    cap, tokens, pack = cfg.train.max_seq_len, cfg.train.tokens_per_epoch, cfg.train.pack
    cached = os.path.isdir(train_cache_path(cfg.train.corpus, tokens, cap, tokenizer, pack))
    print(
        f"\n{'loading' if cached else 'BUILDING (hours)'} {cfg.train.corpus} "
        f"tokens={tokens:,.0f} max_seq_len={cap}\n"
    )
    start = time.time()
    data = task.datasets(tokenizer, cfg)
    print(f"\nready in {(time.time() - start) / 60:.1f} min\n")

    stats = {s: describe(data[s], cap) for s in ("train", "validation")}
    # Packed, the same predicate counts blocks that needed no padding rather than documents cut.
    label = "full" if pack else "truncated"
    for split, st in stats.items():
        line = (
            f"{split:11s} n={st.n:>10,}  tokens={st.tokens:>14,}  mean={st.mean:5.1f}  "
            f"median={st.median:3.0f}  {label}@{cap}={100 * st.truncated:5.1f}%"
        )
        if pack and st.n:
            # What best-fit costs. Above a few percent the bin heuristic is leaving room on the
            # table, and every pad token is compute the loss then throws away.
            line += f"  pad={100 * (1 - st.tokens / (st.n * cap)):4.1f}%"
        print(line)

    if counts := realized_mix(data["train"], len(sources)):
        # `weight` is a token share, so realized-vs-declared is the direct check on the declaration.
        print("\nrealized mix (train), by token share:")
        for src, n in zip(sources, counts, strict=True):
            print(
                f"  {src.name:<18} {n:>12,}  {n / max(sum(counts), 1):>6.1%}  "
                f"(declared {src.weight:>6.1%})"
            )

    unique = stats["train"].tokens
    print(
        f"\nrealized {unique:,} unique tokens over {stats['train'].n:,} texts "
        f"— asked {tokens:,.0f} ({unique / tokens - 1:+.1%})"
    )
    return unique
