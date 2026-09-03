"""Export a distilled BERT student into the checkpoint dir THOR loads.

THOR's ``utils.load_model`` needs three things next to each other: a
``model.safetensors``, a ``config.json`` carrying the dims *and* the franken op
names (it patches the plaintext reference to match the HE op), and the tokenizer.
Distillation only writes ``pytorch_model.bin``, so this fills in the rest.

Deploy = point ``thor/distilled-model`` at the resulting directory.

Usage:
    python -m franken.scripts.bert.export_thor --config configs/bert/depth6_quad_dom32.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from franken.config import Config
from franken.paths import RunPaths

# config.json keys THOR's load_bert_config reads; franken extras ride along and are
# ignored there, but load_model dispatches the op patches off softmax/activation.
ARCH_KEYS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "type_vocab_size",
    "num_labels",
    "pad_token_id",
    "hidden_dropout_prob",
    "attention_dropout_prob",
    "layer_norm_eps",
    "softmax",
    "softmax_kwargs",
    "activation",
    "activation_kwargs",
    # Inert downstream, but without them two checkpoints differing only in whether the pooler
    # was range-penalized are indistinguishable -- and that is the difference that ships.
    "pooler",
    "pooler_kwargs",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--student-ckpt", default=None, help="default: <run>/student/pytorch_model.bin")
    p.add_argument("--out-dir", default=None, help="default: alongside the checkpoint")
    p.add_argument("--note", default="", help="free-text provenance recorded in config.json")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    paths = RunPaths(cfg)
    ckpt = args.student_ckpt or paths.student_bin
    out_dir = args.out_dir or os.path.dirname(ckpt)
    os.makedirs(out_dir, exist_ok=True)

    state = torch.load(ckpt, map_location="cpu")
    # safetensors rejects shared storage; the student has no tied weights.
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    model = asdict(cfg.model)
    blob = {k: model[k] for k in ARCH_KEYS}
    blob |= {
        "architectures": ["BertForClassification"],
        "model_type": "bert",
        "_source": "franken.model.bert.BertForClassification",
        "_config": args.config,
    }
    if args.note:
        blob["_note"] = args.note
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(blob, f, indent=2)

    AutoTokenizer.from_pretrained(cfg.train.teacher_model).save_pretrained(out_dir)
    print(f"Exported {ckpt} -> {out_dir} (safetensors + config.json + tokenizer)")
    rel = os.path.relpath(os.path.abspath(out_dir), os.path.abspath("thor"))
    print(f"Deploy: ln -sfn {rel} thor/distilled-model")


if __name__ == "__main__":
    main()
