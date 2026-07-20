"""Op-curriculum (progressive op-replacement) distillation.

Stage A: distill the student with the *easier* op set (config-A) from the
         teacher-strided init -> a strong single-distortion student.
Stage B: warm-start from Stage A's weights, switch to the *harder* op set
         (config-B), and keep distilling.

The two configs may differ in *any* op — a softmax approximation, a polynomial
activation, or whatever op is added later; nothing here is softmax-specific.
(Example: A = quad + exact softmax, B = quad + cgf softmax.)

This differs from TinyBERT/MPCFormer two-stage KD (which stages *loss targets*
with all ops live throughout, ~ what beta=10 already does). Here we stage *which
ops are active*, so the model absorbs one approximation at a time. Weights
transfer by name (strict=False): the shared backbone warm-starts, and any params
a new op introduces are left at their init and reported.

Usage:
    python scripts/stage_distill.py \
        --config-a configs/quad_fhe.yaml \
        --config-b configs/quad_cgf_fhe.yaml \
        --stagea-dir outputs/stageA_quad \
        --stageb-dir outputs/stageB_quad_cgf \
        [--skip-stagea] [--stageb-lr 3e-5] [--stageb-epochs 8]

Stage B defaults to a gentle warm-start (lr 3e-5, 8 epochs), below the configs'
5e-5, so absorbing the new op doesn't wash out the Stage A init. Verified (MRPC
test): quad+cgf 0.845 (single-stage) -> 0.873 (staged).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from franken.config import Config
from franken.distill.trainer import Distiller


def _save(student, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-a", default="configs/quad_fhe.yaml")
    p.add_argument("--config-b", default="configs/quad_cgf_fhe.yaml")
    p.add_argument("--stagea-dir", default="outputs/stageA_quad")
    p.add_argument("--stageb-dir", default="outputs/stageB_quad_cgf")
    p.add_argument("--skip-stagea", action="store_true",
                   help="reuse an existing Stage A checkpoint in --stagea-dir")
    p.add_argument("--stageb-lr", type=float, default=3e-5,
                   help="Stage B LR (default 3e-5: a gentle warm-start below the "
                        "configs' 5e-5, so absorbing the new op doesn't wash out "
                        "the Stage A init; pass the config-B value to disable)")
    p.add_argument("--stageb-epochs", type=int, default=8,
                   help="Stage B epochs (default 8: the warm-started student needs "
                        "fewer epochs than a cold start to adapt to the new op)")
    args = p.parse_args()

    stagea_ckpt = os.path.join(args.stagea_dir, "pytorch_model.bin")

    # ---- Stage A: easier op set from strided init -------------------------
    if args.skip_stagea:
        print(f"[stageA] skipped; reusing {stagea_ckpt}")
    else:
        print(f"[stageA] distilling {args.config_a} -> {args.stagea_dir}")
        cfg_a = Config.from_yaml(args.config_a)
        da = Distiller(cfg_a)
        da.setup()
        da.train()
        _save(da.student, args.stagea_dir)
        print(f"[stageA] saved -> {stagea_ckpt}")

    # ---- Stage B: full op set, warm-started from Stage A ------------------
    cfg_b = Config.from_yaml(args.config_b)
    if args.stageb_lr is not None:
        cfg_b.train.distill.lr = args.stageb_lr
    if args.stageb_epochs is not None:
        cfg_b.train.distill.epochs = args.stageb_epochs
    print(f"[stageB] distilling {args.config_b} "
          f"(lr={cfg_b.train.distill.lr}, ep={cfg_b.train.distill.epochs}) "
          f"warm-started from {stagea_ckpt}")

    db = Distiller(cfg_b)
    db.setup()  # strided init (kept for any params the Stage B op adds)
    sd = torch.load(stagea_ckpt, map_location=db.device)
    # strict=False so the swapped op need not be parameter-free: shared backbone
    # params warm-start from Stage A; params only the Stage B op introduces keep
    # their init. Both cases are reported so a silent name mismatch can't hide.
    incompatible = db.student.load_state_dict(sd, strict=False)
    if incompatible.missing_keys:
        print(f"[stageB] newly-initialized (absent in Stage A): {len(incompatible.missing_keys)} "
              f"params, e.g. {incompatible.missing_keys[:5]}")
    if incompatible.unexpected_keys:
        print(f"[stageB] dropped (absent in Stage B): {len(incompatible.unexpected_keys)} "
              f"params, e.g. {incompatible.unexpected_keys[:5]}")
    db.student.to(db.device)
    db.train()
    _save(db.student, args.stageb_dir)
    print(f"[stageB] saved -> {os.path.join(args.stageb_dir, 'pytorch_model.bin')}")


if __name__ == "__main__":
    main()
