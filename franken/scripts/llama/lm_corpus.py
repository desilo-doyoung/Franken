"""The LM corpus workflow: verify the holdout, measure the mix, build the cache.

No scoreability stage, unlike the embed track's: that gate exists because an unscoreable retrieval
slice is a permanent blind spot, and logit KD has no ranking to be blind to -- per-source perplexity
always exists.

    uv run python -m franken.scripts.llama.lm_corpus --config configs/llama/smoke.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import datasets

from franken.config import Config
from franken.data import corpus_sources
from franken.scripts import corpus_report
from franken.tasks import build_task


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
    sources = corpus_sources(cfg.train.corpus)

    ok = corpus_report.check(sources)
    loaded, _rows = corpus_report.lengths(cfg, sources, tokenizer)
    ok = ok and loaded
    if ok:
        unique = corpus_report.build(cfg, task, tokenizer, sources)
        print(
            f"token-passes at {cfg.train.distill.epochs} epochs: "
            f"{unique * cfg.train.distill.epochs:,}"
        )

    print(f"\n{'CORPUS OK' if ok else 'CORPUS FAILED — do not train'}\n")
    return ok


if __name__ == "__main__":
    ok = main()
    # See the embed track's note: an HF retry thread can SIGABRT during interpreter finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)
