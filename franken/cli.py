"""Command-line entrypoints: corpus | train-teacher | distill | eval.

The single-config path, start to finish. For a batch over several configs and GPUs use
`scripts/qwen3/run_experiments.py`, which does the same steps per config and prints one table.

Usage:
    python main.py corpus  --config configs/qwen3/depth19_multi_domain.yaml [--build]
    python main.py distill --config configs/qwen3/depth19_multi_domain.yaml
    python main.py eval    --config configs/qwen3/depth19_multi_domain.yaml
"""

import argparse

from franken.config import Config


def _load_config(args: argparse.Namespace) -> Config:
    return Config.from_yaml(args.config)


def cmd_train_teacher(args: argparse.Namespace) -> None:
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


def cmd_distill(args: argparse.Namespace) -> None:
    import os

    import torch

    from franken.distill.trainer import Distiller
    from franken.paths import RunPaths

    cfg = _load_config(args)  # validate config early
    d = Distiller(cfg)
    d.setup()
    d.train()

    from franken.distill.dist import barrier, shutdown

    # Rank 0 only: every rank would otherwise race on the same path, and only rank 0 holds the
    # selected checkpoint.
    if d.dist.is_main:
        paths = RunPaths(cfg)
        os.makedirs(paths.student, exist_ok=True)
        torch.save(d.student.state_dict(), paths.student_bin())
        print(f"Student saved to {paths.student}")

    # Every rank reaches teardown together: destroy_process_group is collective, so letting one
    # rank tear down while another is still saving hangs the job.
    barrier(d.dist)
    shutdown(d.dist)


# Per backend, because the two tracks score different things: MRPC is accuracy/F1 over both splits,
# qwen3 is teacher agreement + nDCG over three suites.
_EVALUATOR = {"bert": "evaluate.py", "qwen3": "eval.py"}


def _delegate(backend: str, script: str, argv: list[str]) -> None:
    """Run scripts/<backend>/<script> in-process, with its own directory importable (the qwen3
    scripts import a sibling `common`)."""
    import importlib.util
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "scripts", backend, script)
    if not os.path.exists(path):
        raise SystemExit(f"No {script} for backend {backend!r} at {path}.")
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(f"franken_{backend}_{script}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(argv)


def cmd_corpus(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    _delegate(
        cfg.model.backend,
        "corpus.py",
        ["--config", args.config] + (["--build"] if args.build else []),
    )


def cmd_eval(args: argparse.Namespace) -> None:
    import os

    cfg = _load_config(args)  # backend name selects which model's evaluator to run
    script = _EVALUATOR.get(cfg.model.backend)
    if not script:
        raise SystemExit(f"No evaluator registered for backend {cfg.model.backend!r}.")
    argv = ["--config", args.config]
    if args.ckpt:  # a directory holding pytorch_model.bin, or the file itself
        ckpt = (
            os.path.join(args.ckpt, "pytorch_model.bin") if os.path.isdir(args.ckpt) else args.ckpt
        )
        argv += ["--student-ckpt", ckpt]
    _delegate(cfg.model.backend, script, argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="franken", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        # Required, not defaulted: the config is the experiment, and these commands cost hours.
        p.add_argument("--config", required=True, help="path to YAML config")

    p_corpus = sub.add_parser("corpus", help="gate + measure the corpus; --build to cache it")
    add_config(p_corpus)
    p_corpus.add_argument("--build", action="store_true", help="also build the cache (hours)")
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
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
