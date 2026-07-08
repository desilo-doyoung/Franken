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
    import os

    import torch

    from franken.distill.trainer import Distiller

    cfg = _load_config(args)  # validate config early
    d = Distiller(cfg)
    d.setup()
    d.train()

    # Save the student checkpoint
    path = os.path.join(cfg.train.output_dir, "student")
    os.makedirs(path, exist_ok=True)
    torch.save(d.student.state_dict(), os.path.join(path, "pytorch_model.bin"))
    print(f"Student saved to {path}")


def cmd_eval(args):
    import os

    import torch

    from franken.distill.trainer import Distiller

    cfg = _load_config(args)
    d = Distiller(cfg)
    d.setup()  # builds student (strided init) + teacher
    if args.ckpt:  # load trained student weights over the init
        sd = torch.load(os.path.join(args.ckpt, "pytorch_model.bin"), map_location=d.device)
        d.student.load_state_dict(sd)
    print(d.evaluate())


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
