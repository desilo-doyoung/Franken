"""Build and cache a corpus preset once, before torchrun launches.

`_build_split` runs per rank and each rank re-streams and re-tokenizes independently — ~5 min of a
71-min run at 1.58M texts, an hour or more per rank at 10M. Build it here first and the ranks all
cache-hit. Also the place to read the realized mix: a source that runs dry is flagged EXHAUSTED,
and its declared weight did not take effect.

Usage:
    uv run python scripts/qwen3/build_corpus.py --config configs/qwen3/depth19_multi_domain.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pyarrow.compute as pc
from franken.config import Config
from franken.tasks import build_task


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    task = build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)
    cap = cfg.train.max_seq_len

    print(f"corpus={cfg.train.corpus} size={cfg.train.corpus_size:,} max_seq_len={cap}\n")
    start = time.time()
    data = task.datasets(tokenizer, cfg)
    print(f"\nbuilt in {(time.time() - start) / 60:.1f} min\n")

    tokens = {}
    for split in ("train", "validation"):
        ds = data[split]
        # Length off the Arrow column: materializing 10M token lists as Python objects is GBs.
        lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(zero_copy_only=False)
        tokens[split] = int(lengths.sum())
        print(
            f"{split:11s} n={len(lengths):>10,}  tokens={tokens[split]:>14,}  "
            f"mean={lengths.mean():5.1f}  median={np.median(lengths):3.0f}  "
            f"truncated@{cap}={100 * (lengths >= cap).mean():4.1f}%"
        )

    epochs = cfg.train.distill.epochs
    print(f"\ntoken-passes at {epochs} epochs: {tokens['train'] * epochs:,}")


if __name__ == "__main__":
    main()
