"""Command-line entrypoints: train-teacher | distill | eval.

Usage:
    python main.py train-teacher --config configs/default.yaml
    python main.py distill       --config configs/default.yaml
    python main.py eval          --config configs/default.yaml --ckpt outputs/student
"""

import argparse

from franken.config import Config


def _load_config(args: argparse.Namespace) -> Config:
    return Config.from_yaml(args.config)


def cmd_train_teacher(args: argparse.Namespace) -> None:
    from franken import teacher

    cfg = _load_config(args)
    path = teacher.train_teacher(cfg)
    print(f"Teacher checkpoint saved to {path}")


def cmd_distill(args: argparse.Namespace) -> None:
    _load_config(args)  # validate config early
    raise SystemExit(
        "`distill` is not implemented yet — the distillation loop lives in "
        "franken/distill/ and is built interactively in the tutorial session."
    )


def cmd_eval(args: argparse.Namespace) -> None:
    _load_config(args)
    raise SystemExit(
        "`eval` is not implemented yet — student evaluation is added alongside "
        "the distillation loop in the tutorial session."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="franken", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="configs/default.yaml", help="path to YAML config")

    p_teacher = sub.add_parser("train-teacher", help="fine-tune the HF teacher on MRPC")
    add_config(p_teacher)
    p_teacher.set_defaults(func=cmd_train_teacher)

    p_distill = sub.add_parser("distill", help="distill teacher -> custom student")
    add_config(p_distill)
    p_distill.set_defaults(func=cmd_distill)

    p_eval = sub.add_parser("eval", help="evaluate a student checkpoint on MRPC")
    add_config(p_eval)
    p_eval.add_argument("--ckpt", help="path to a student checkpoint")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
