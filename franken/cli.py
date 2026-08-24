"""Command-line entrypoints: corpus | train-teacher | distill | eval. The single-config path;
`run_experiments` does the same per config across GPUs and prints one table.

Usage:
    python main.py corpus  --config configs/qwen3/depth19_quad.yaml
    python main.py distill --config configs/qwen3/depth19_quad.yaml
    python main.py eval    --config configs/qwen3/depth19_quad.yaml
"""

import argparse

from franken.config import Config


def _load_config(args: argparse.Namespace) -> Config:
    return Config.from_yaml(args.config)


def cmd_train_teacher(args: argparse.Namespace, extra: list[str]) -> None:
    from franken.tasks import build_task

    cfg = _load_config(args)
    path = build_task(cfg.train.task).train_teacher(cfg)
    if path:
        print(f"Teacher checkpoint saved to {path}")
    else:
        print(
            f"Task {cfg.train.task!r}: no teacher training needed "
            "(the pretrained checkpoint is the teacher)."
        )


def cmd_distill(args: argparse.Namespace, extra: list[str]) -> None:
    import os

    import torch

    from franken.distill.trainer import Distiller
    from franken.paths import RunPaths

    cfg = _load_config(args)  # validate config early
    d = Distiller(cfg)
    d.setup()
    d.train()

    from franken.distill.dist import barrier, shutdown

    # Rank 0 only: it alone holds the selected checkpoint, and the others would race on the path.
    if d.dist.is_main:
        paths = RunPaths(cfg)
        os.makedirs(paths.student, exist_ok=True)
        torch.save(d.student.state_dict(), paths.student_bin)
        print(f"Student saved to {paths.student}")

    # destroy_process_group is collective, so tearing down mid-save hangs the job.
    barrier(d.dist)
    shutdown(d.dist)


# Fully-qualified, so a scorer is reachable wherever it lives.
_EVALUATOR = {"bert": "franken.scripts.bert.evaluate", "qwen3": "franken.scripts.qwen3.eval"}

# Keyed on the TASK and consulted first: what a corpus must satisfy, and what scores a student,
# follow from the objective rather than the architecture.
_CORPUS = {"embed": "franken.scripts.qwen3.corpus", "lm": "franken.scripts.llama.lm_corpus"}
_TASK_EVALUATOR = {"lm": "franken.scripts.llama.lm_eval"}


def _delegate(module: str, argv: list[str]) -> None:
    import importlib

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise SystemExit(f"Cannot load {module}: {e}") from e
    return mod.main(argv)


def cmd_corpus(args: argparse.Namespace, extra: list[str]) -> None:
    cfg = _load_config(args)
    module = _CORPUS.get(cfg.train.task)
    if not module:
        raise SystemExit(f"Task {cfg.train.task!r} brings its own data; nothing to build.")
    if _delegate(module, ["--config", args.config] + extra) is False:
        raise SystemExit("corpus gates failed — do not train")


def cmd_eval(args: argparse.Namespace, extra: list[str]) -> None:
    import os

    cfg = _load_config(args)
    # Task first: fall back to the backend only where the scorer is genuinely model-specific.
    module = _TASK_EVALUATOR.get(cfg.train.task) or _EVALUATOR.get(cfg.model.backend)
    if not module:
        raise SystemExit(
            f"No evaluator for task {cfg.train.task!r} / backend {cfg.model.backend!r}."
        )
    argv = ["--config", args.config]
    if args.ckpt:  # a directory holding pytorch_model.bin, or the file itself
        ckpt = (
            os.path.join(args.ckpt, "pytorch_model.bin") if os.path.isdir(args.ckpt) else args.ckpt
        )
        argv += ["--student-ckpt", ckpt]
    _delegate(module, argv + extra)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="franken", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        # Required, not defaulted: the config IS the experiment.
        p.add_argument("--config", required=True, help="path to YAML config")

    p_corpus = sub.add_parser("corpus", help="gate + measure the corpus, caching it if needed")
    add_config(p_corpus)
    p_corpus.set_defaults(func=cmd_corpus)

    p_teacher = sub.add_parser(
        "train-teacher", help="prepare the task's teacher (fine-tune if needed)"
    )
    add_config(p_teacher)
    p_teacher.set_defaults(func=cmd_train_teacher)

    p_distill = sub.add_parser("distill", help="distill teacher -> custom student")
    add_config(p_distill)
    p_distill.set_defaults(func=cmd_distill)

    p_eval = sub.add_parser("eval", help="score teacher + student via the backend's evaluator")
    add_config(p_eval)
    p_eval.add_argument(
        "--ckpt", help="student checkpoint dir or .bin (default: <output_dir>/student)"
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    # Unrecognized flags pass through, so this file need not mirror every scorer flag.
    args, extra = parser.parse_known_args(argv)
    # Elsewhere an unknown flag is a typo.
    if extra and args.command not in ("corpus", "eval"):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.func(args, extra)


if __name__ == "__main__":
    main()
