"""Op-curriculum distillation: Stage A distils the easier op set from the strided init, Stage B
warm-starts from it and swaps in the harder op, so the model absorbs one approximation at a time.
The configs may differ in ANY op. Verified on MRPC test: quad+cgf 0.845 single-stage -> 0.873.

Usage:
    python -m franken.scripts.stage_distill \
        --config-a configs/bert/depth8_quad_dom32.yaml \
        --config-b configs/bert/depth8_quad_dom32_cgf.yaml \
        [--skip-stagea] [--stageb-lr 3e-5] [--stageb-epochs 8]
"""

from __future__ import annotations

import argparse
import os

import torch

from franken.config import Config
from franken.distill.trainer import Distiller
from franken.paths import RunPaths


def _save(student, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-a", default="configs/bert/depth8_quad_dom32.yaml")
    p.add_argument("--config-b", default="configs/bert/depth8_quad_dom32_cgf.yaml")
    p.add_argument(
        "--stagea-dir",
        default=None,
        help="default: <run>/stageA_quad (run namespace from config-A's run_name)",
    )
    p.add_argument(
        "--stageb-dir",
        default=None,
        help="default: <run>/stageB_quad_cgf (run namespace from config-B's run_name)",
    )
    p.add_argument(
        "--skip-stagea",
        action="store_true",
        help="reuse an existing Stage A checkpoint in --stagea-dir",
    )
    p.add_argument("--stageb-lr", type=float, default=3e-5, help="Stage B LR")
    p.add_argument("--stageb-epochs", type=int, default=8, help="Stage B epochs")
    args = p.parse_args()

    # Default to the run-namespaced base.
    cfg_a = Config.from_yaml(args.config_a)
    cfg_b = Config.from_yaml(args.config_b)
    stagea_dir = args.stagea_dir or RunPaths(cfg_a).subdir("stageA_quad")
    stageb_dir = args.stageb_dir or RunPaths(cfg_b).subdir("stageB_quad_cgf")
    stagea_ckpt = os.path.join(stagea_dir, "pytorch_model.bin")

    # ---- Stage A: easier op set from strided init -------------------------
    if args.skip_stagea:
        print(f"[stageA] skipped; reusing {stagea_ckpt}")
    else:
        print(f"[stageA] distilling {args.config_a} -> {stagea_dir}")
        da = Distiller(cfg_a)
        da.setup()
        da.train()
        _save(da.student, stagea_dir)
        print(f"[stageA] saved -> {stagea_ckpt}")

    # ---- Stage B: full op set, warm-started from Stage A ------------------
    if args.stageb_lr is not None:
        cfg_b.train.distill.lr = args.stageb_lr
    if args.stageb_epochs is not None:
        cfg_b.train.distill.epochs = args.stageb_epochs
    print(
        f"[stageB] distilling {args.config_b} "
        f"(lr={cfg_b.train.distill.lr}, ep={cfg_b.train.distill.epochs}) "
        f"warm-started from {stagea_ckpt}"
    )

    db = Distiller(cfg_b)
    db.setup()  # strided init (kept for any params the Stage B op adds)
    sd = torch.load(stagea_ckpt, map_location=db.device)
    # strict=False so the swapped op may carry parameters of its own; both cases are reported
    # below, so a silent name mismatch cannot hide.
    incompatible = db.student.load_state_dict(sd, strict=False)
    if incompatible.missing_keys:
        print(
            f"[stageB] newly-initialized (absent in Stage A): {len(incompatible.missing_keys)} "
            f"params, e.g. {incompatible.missing_keys[:5]}"
        )
    if incompatible.unexpected_keys:
        print(
            f"[stageB] dropped (absent in Stage B): {len(incompatible.unexpected_keys)} "
            f"params, e.g. {incompatible.unexpected_keys[:5]}"
        )
    db.student.to(db.device)
    db.train()
    _save(db.student, stageb_dir)
    print(f"[stageB] saved -> {os.path.join(stageb_dir, 'pytorch_model.bin')}")


if __name__ == "__main__":
    main()
