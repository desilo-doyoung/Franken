"""The embed corpus workflow: verify the holdout, measure the mix, build the cache. Three stages in
one script because each gates the next. Tables live in `franken.scripts.corpus_report`; what is
here is the verdict the embed objective adds -- every source must be scoreable.

    uv run python -m franken.scripts.qwen3.corpus --config configs/qwen3/depth19_quad.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import datasets

from franken.config import Config
from franken.data.qwen3.registry import mix
from franken.scripts import corpus_report
from franken.tasks import build_task

TOKEN_TARGET = 2e9  # token-passes per run; the only place this budget is written down


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

    ok = corpus_report.check(sources)
    loaded, rows = corpus_report.lengths(cfg, sources, tokenizer, extra_col="scoreable")
    # An unscoreable slice is a permanent blind spot: `code_apps` -53.9% turned out to be measuring
    # corpus coverage rather than the depth cut.
    if blind := [r.name for r in rows if not r.error and r.scoreable == "none"]:
        print(f"NOT SCOREABLE: {', '.join(blind)} — needs an eval pair or a Qrels declaration")
    ok = ok and loaded and not blind

    if ok:
        unique = corpus_report.build(cfg, task, tokenizer, sources)
        passes = unique * cfg.train.distill.epochs
        print(
            f"token-passes at {cfg.train.distill.epochs} epochs: {passes:,} "
            f"(reference target {TOKEN_TARGET:,.0f}, {passes / TOKEN_TARGET - 1:+.1%})"
        )

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
