"""Per-source token length, so `corpus_size` can be set from measurement instead of arithmetic.

`tok/text` is the factor between a text count and a token budget, and it is not predictable from
the source list: the mix mingles ~10-token queries with ~200-token abstracts, so adding short-pair
sources lowers it and the text count needed for a given token target rises. The documented
precedent is a 15% miss on an estimated mean, which here would only surface after a multi-hour
build -- hence a probe first.

It doubles as a smoke test: every source in the mix is streamed and its extractor run, so a dead
loader or a renamed column fails here rather than hours into `build_corpus.py`.

    uv run python scripts/qwen3/corpus_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datasets  # noqa: E402
from franken.config import Config  # noqa: E402
from franken.data.embed_corpus import MIXES  # noqa: E402
from franken.tasks import build_task  # noqa: E402

SAMPLE = 300  # texts per source; a mean over 300 is tight enough to size a corpus
TARGETS = (11_500_000, 15_500_000, 16_500_000)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/depth19_multi_domain.yaml")
    p.add_argument("--json", help="machine-readable output, so a driver script need not scrape")
    args = p.parse_args(argv)

    datasets.disable_progress_bars()
    cfg = Config.from_yaml(args.config)
    tokenizer = build_task(cfg.train.task).build_tokenizer(cfg)
    mix = MIXES[cfg.train.corpus]

    # flush=True throughout: stdout is block-buffered when redirected, and a probe that streams 26
    # sources for many minutes is useless if its table only appears at exit.
    print(f"\n{cfg.train.corpus}: {len(mix)} sources, max_seq_len {cfg.train.max_seq_len}\n")
    # `mean`/`p50` are post-cap (what the corpus stores, hence the token budget); `cut%`, `p99*`
    # and `max*` are UNtruncated, which is the only way to see whether max_seq_len is still right.
    # Measuring only truncated lengths hides the tail behind a p99 of exactly max_seq_len.
    print(
        f"{'source':<16} {'domain':<14} {'w':>6} {'mean':>7} {'p50':>5} "
        f"{'cut%':>6} {'p99*':>7} {'max*':>7}",
        flush=True,
    )

    rows, weighted, missing = [], 0.0, []
    for name, domain, source, weight in mix:
        try:
            texts = source("train", SAMPLE)
        except Exception as e:  # a dead loader must not hide the rest of the table
            print(
                f"{name:<16} {domain:<14} {weight:>6.3f}  FAILED {type(e).__name__}: {e}"[:118],
                flush=True,
            )
            missing.append(name)
            continue
        raw = sorted(len(x) for x in tokenizer(texts)["input_ids"])
        cap = cfg.train.max_seq_len
        capped = [min(x, cap) for x in raw]
        mean = sum(capped) / max(len(capped), 1)
        cut = 100 * sum(1 for x in raw if x > cap) / max(len(raw), 1)
        p99 = raw[min(len(raw) - 1, int(0.99 * len(raw)))]
        print(
            f"{name:<16} {domain:<14} {weight:>6.3f} {mean:>7.1f} "
            f"{capped[len(capped) // 2]:>5} {cut:>6.1f} {p99:>7} {raw[-1]:>7}",
            flush=True,
        )
        rows.append((name, domain, weight, mean, raw))
        weighted += weight * mean

    covered = sum(w for _n, _d, w, _m, _r in rows)
    print(f"\nweighted mean tok/text: {weighted:.1f}  (over {covered:.3f} of the mix)")
    if missing:
        print(f"⚠️  no sample from: {', '.join(missing)} — the mean above excludes them")
    if covered > 0:
        full = weighted / covered  # rescale so a failed source does not read as zero-length
        print(f"rescaled to the whole mix: {full:.1f} tok/text\n")
        for target in TARGETS:
            print(f"  corpus_size {target:>10,} -> {target * full / 1e9:.2f}B tokens/epoch")
        print(f"\n  for 2.0B tokens: corpus_size ~ {round(2e9 / full / 1e5) * 1e5:,.0f}")

    by_domain: dict[str, float] = {}
    for _n, domain, w, mean, _r in rows:
        by_domain[domain] = by_domain.get(domain, 0.0) + w * mean
    print("\ntokens by domain (share of corpus tokens):")
    for domain, tok in sorted(by_domain.items(), key=lambda x: -x[1]):
        print(f"  {domain:<14} {tok / max(weighted, 1e-9):>6.1%}")

    # What a wider cap would actually buy. The FHE polynomial domain is clamp-free, so it is set by
    # the MAX sequence length, not the mean -- which is why 32768 was rejected for +6.7% of tokens.
    print("\ncap sweep (weighted over the mix):")
    print(
        f"  {'cap':>6} {'tok/text':>9} {'vs 1024':>8} {'texts cut':>10} {'corpus_size for 2B':>19}"
    )
    for c in (256, 512, 1024, 1536, 2048, 4096, 8192):
        mean_c = sum(w * sum(min(x, c) for x in r) / len(r) for _n, _d, w, _m, r in rows) / covered
        cut_c = sum(w * sum(1 for x in r if x > c) / len(r) for _n, _d, w, _m, r in rows) / covered
        base = sum(w * sum(min(x, 1024) for x in r) / len(r) for _n, _d, w, _m, r in rows) / covered
        print(
            f"  {c:>6} {mean_c:>9.1f} {mean_c / base - 1:>+7.1%} {cut_c:>9.2%} "
            f"{round(2e9 / mean_c / 1e5) * 1e5:>18,.0f}"
        )

    print("\nsources with the most truncation at the current cap:")
    worst = sorted(
        ((sum(1 for x in r if x > cap) / len(r), n, w, r[-1]) for n, _d, w, _m, r in rows),
        reverse=True,
    )[:6]
    for frac, name, w, mx in worst:
        print(f"  {name:<16} w={w:<6.3f} cut {frac:>6.1%}  longest {mx:>6}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "corpus": cfg.train.corpus,
                    "max_seq_len": cap,
                    "tok_per_text": weighted / covered if covered else 0.0,
                    "covered_weight": covered,
                    "failed": missing,
                    "sources": {
                        n: {"domain": d, "weight": w, "mean": m, "max_untruncated": r[-1]}
                        for n, d, w, m, r in rows
                    },
                },
                f,
                indent=2,
            )

    # Non-zero exit so a driver script can gate on this: a dead loader or renamed column must stop
    # the pipeline here, not surface hours into build_corpus.py.
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
