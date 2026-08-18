"""The corpus workflow: verify the holdout, measure the mix, build the cache. Three stages in one
script because each gates the next. Measurement lives in `franken.data.embed_corpus`; this file is
argument parsing, tables and the verdict.

    uv run python -m franken.scripts.qwen3.corpus --config configs/qwen3/depth19_quad.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import datasets

from franken.config import Config
from franken.data.embed_corpus import (
    SPLITS,
    cache_path,
    describe,
    mix,
    profile,
    realized_mix,
    source_texts,
)
from franken.tasks import build_task

TOKEN_TARGET = 2e9  # token-passes per run; the only place this budget is written down
SAMPLE = 300  # texts per source for the length profile
HOLDOUT_SAMPLE = 100  # texts per split per source, for the overlap check


def check(sources) -> bool:
    """Only upstream-split sources need streaming -- for a hash-split source disjointness is a
    theorem. Splits are also compared on mean text length, since clean is not the same as
    representative."""
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
        if (hi - min(means.values())) / max(hi, 1e-9) > 0.25:
            bad.append("SPLITS DISAGREE")
        ok = ok and not bad
        lens = "  ".join(f"{s[:3]} {m:.0f}" for s, m in means.items())
        print(
            f"{src.name:<18} {len(drawn['train']):>7} {len(drawn['validation']):>6} "
            f"{len(drawn['test']):>6}  {lens:>22}  {', '.join(bad) if bad else 'ok'}",
            flush=True,
        )
    return ok


def measure(cfg, sources, tokenizer) -> tuple[bool, int]:
    """Per-source token length -> the `corpus_size` that spends TOKEN_TARGET, plus the
    scoreability gate."""
    cap, epochs = cfg.train.max_seq_len, cfg.train.distill.epochs
    print(f"\n{cfg.train.corpus}: {len(sources)} sources, max_seq_len {cap}")
    # mean/p50 are post-cap; cut%/max* are untruncated. The max, not a percentile: FHE has no
    # clamp.
    print(
        f"{'source':<18} {'domain':<14} {'w':>6} {'mean':>7} {'p50':>5} "
        f"{'cut%':>6} {'max*':>7} {'score':>6}",
        flush=True,
    )
    rows = profile(sources, tokenizer, cap, n=SAMPLE)
    for r in rows:
        if r.error:
            print(f"{r.name:<18} {r.domain:<14} {r.weight:>6.3f}  FAILED {r.error}"[:118])
            continue
        print(
            f"{r.name:<18} {r.domain:<14} {r.weight:>6.3f} {r.mean:>7.1f} {r.median:>5} "
            f"{100 * r.truncated:>6.1f} {r.longest:>7} {r.scoreable:>6}"
        )

    good = [r for r in rows if not r.error]
    covered = sum(r.weight for r in good)
    if not covered:
        print("\nno source produced a sample")
        return False, 0
    # Rescaled, so a failed source does not read as zero-length.
    full = sum(r.weight * r.mean for r in good) / covered
    size = int(round(TOKEN_TARGET / epochs / full / 1e5) * 1e5)
    print(
        f"\ntok/text {full:.1f} over {covered:.3f} of the mix -> corpus_size {size:,} "
        f"= {size * full / 1e9:.2f}B/epoch, x{epochs} epochs = {size * full * epochs / 1e9:.2f}B"
    )
    if failed := [r.name for r in rows if r.error]:
        print(f"NO SAMPLE: {', '.join(failed)} — excluded from the mean above")
    if blind := [r.name for r in good if r.scoreable == "none"]:
        print(f"NOT SCOREABLE: {', '.join(blind)} — needs an eval pair or a Qrels declaration")
    return not (failed or blind), size


def build(cfg, task, tokenizer, sources) -> None:
    """Build and cache if needed, then report what the artifact holds. The token count is a
    report, not a gate -- `tok/text` is sampled, so landing a few percent off is expected. It
    prints on cache hits too, since that is the only place a stale `corpus_size` shows up."""
    cap = cfg.train.max_seq_len
    cached = os.path.isdir(
        cache_path(cfg.train.corpus, "train", cfg.train.corpus_size, cap, tokenizer)
    )
    verb = "loading" if cached else "BUILDING (hours)"
    print(f"\n{verb} {cfg.train.corpus} size={cfg.train.corpus_size:,} max_seq_len={cap}\n")
    start = time.time()
    data = task.datasets(tokenizer, cfg)
    print(f"\nready in {(time.time() - start) / 60:.1f} min\n")

    stats = {s: describe(data[s], cap) for s in ("train", "validation")}
    for split, st in stats.items():
        print(
            f"{split:11s} n={st.n:>10,}  tokens={st.tokens:>14,}  mean={st.mean:5.1f}  "
            f"median={st.median:3.0f}  truncated@{cap}={100 * st.truncated:4.1f}%"
        )

    counts = realized_mix(data["train"], len(sources))
    if counts:
        print("\nrealized mix (train):")
        for src, n in zip(sources, counts, strict=True):
            print(
                f"  {src.name:<18} {n:>10,}  {n / max(sum(counts), 1):>6.1%}  "
                f"(declared {src.weight:>6.1%})"
            )

    epochs = cfg.train.distill.epochs
    passes = stats["train"].tokens * epochs
    off = passes / TOKEN_TARGET - 1
    print(f"\ntoken-passes at {epochs} epochs: {passes:,} (target {TOKEN_TARGET:,.0f}, {off:+.1%})")
    print(
        f"corpus_size {cfg.train.corpus_size:,} spends that; "
        f"{round(cfg.train.corpus_size / (1 + off) / 1e5) * 100_000:,} would spend the target"
    )


def main(argv: list[str] | None = None) -> bool:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="path to the experiment YAML")
    args = p.parse_args(argv)

    datasets.disable_progress_bars()
    cfg = Config.from_yaml(args.config)
    task = build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)
    sources = mix(cfg.train.corpus)

    ok = check(sources)
    scoreable, size = measure(cfg, sources, tokenizer)
    ok = ok and scoreable
    if ok and size and size != cfg.train.corpus_size:
        print(
            f"\nconfig corpus_size {cfg.train.corpus_size:,} != measured {size:,} "
            f"({cfg.train.corpus_size / size - 1:+.1%}) — the config's value is what gets built."
        )
    if ok:
        build(cfg, task, tokenizer, sources)

    print(f"\n{'CORPUS OK' if ok else 'CORPUS FAILED — do not train'}\n")
    return ok


if __name__ == "__main__":
    ok = main()
    # os._exit, not sys.exit: an HF retry thread aborts during interpreter *finalization* (SIGABRT,
    # the 134 that makes a successful build look failed), which would overwrite this exit code.
    # Skipping finalization dodges that phase, so flush first -- os._exit does not.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)
