"""Export a distilled student into a THOR-loadable model directory.

THOR loads a stock HF ``BertForSequenceClassification`` from ``<dir>/model.safetensors``
and reads the franken extras (``softmax`` / ``activation`` + kwargs) from
``<dir>/config.json`` to pick its HE ops and to patch the plaintext reference.
``main.py distill`` / ``scripts/stage_distill.py`` only write ``pytorch_model.bin``, so
this turns a checkpoint into that directory layout (``thor/measure_ranges.py`` reads it too).

Usage:
    python scripts/bert/export_thor.py --config configs/bert/quad_cgf_ranged.yaml \
        --ckpt outputs/bert/stageB_quad_cgf_ranged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from franken.config import Config
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--ckpt", required=True, help="checkpoint dir holding pytorch_model.bin, or the .bin"
    )
    p.add_argument("--out", default=None, help="output dir (default: the checkpoint's dir)")
    args = p.parse_args()

    bin_path = (
        args.ckpt if args.ckpt.endswith(".bin") else os.path.join(args.ckpt, "pytorch_model.bin")
    )
    out_dir = args.out or os.path.dirname(os.path.abspath(bin_path))
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    state = torch.load(bin_path, map_location="cpu")

    # safetensors needs contiguous, non-shared tensors; this model has no tied weights.
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    config = asdict(cfg.model) | {
        "architectures": ["BertForClassification"],
        "model_type": "bert",
        "_source": "franken.models.bert.BertForClassification",
        "_ckpt": bin_path,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"exported -> {out_dir}/model.safetensors\n            {out_dir}/config.json")
    print(
        f"  softmax={cfg.model.softmax} {cfg.model.softmax_kwargs}  "
        f"activation={cfg.model.activation} {cfg.model.activation_kwargs}"
    )


if __name__ == "__main__":
    main()
