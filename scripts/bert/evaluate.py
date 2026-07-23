"""Evaluate the teacher and/or distilled student on MRPC splits.

Reports accuracy + F1 for each (model, split) pair as a table. MRPC is the one
GLUE task whose `test` split ships with public labels, so validation *and* test
can both be scored locally.

Model construction and the forward pass go through the model backend
(``franken.models``) so this script is not tied to a specific model class; data
building stays MRPC-specific (this is an MRPC scorer).

Usage:
    python scripts/bert/evaluate.py --config configs/bert/default.yaml
    python scripts/bert/evaluate.py --models student --splits validation test
    python scripts/bert/evaluate.py \
        --student-ckpt outputs/bert/student/pytorch_model.bin --splits test
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/bert/evaluate.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from datasets import load_dataset
from franken.config import Config
from franken.data.mrpc import compute_metrics
from franken.models import build_backend
from franken.paths import RunPaths
from franken.tasks import build_task
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


@torch.no_grad()
def evaluate_split(model, backend, task, dl: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = backend.forward(model, task.model_inputs(batch))
        preds.append(out["output"].argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return compute_metrics(preds, labels)


def build_loaders(tokenizer, task, splits, max_seq_len, batch_size):
    ds = load_dataset("nyu-mll/glue", "mrpc")
    ds = ds.map(
        lambda b: tokenizer(
            b["sentence1"], b["sentence2"], truncation=True, max_length=max_seq_len
        ),
        batched=True,
    )
    collator = DataCollatorWithPadding(tokenizer)
    cols = task.torch_columns()
    loaders = {}
    for split in splits:
        # Guard: a split with only -1 labels is unlabeled (can't score locally).
        if set(ds[split].unique("label")) == {-1}:
            print(f"[skip] split '{split}' has no public labels (all -1).")
            continue
        d = ds[split].with_format("torch", columns=cols)
        loaders[split] = DataLoader(d, batch_size=batch_size, collate_fn=collator)
    return loaders


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="configs/bert/default.yaml", help="config (student arch + teacher_ckpt)"
    )
    p.add_argument(
        "--student-ckpt",
        default=None,
        help="student state_dict (default: <run>/student/pytorch_model.bin)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["teacher", "student"],
        default=["teacher", "student"],
        help="which models to evaluate",
    )
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backend = build_backend(cfg.model.backend)
    task = build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)
    loaders = build_loaders(tokenizer, task, args.splits, cfg.train.max_seq_len, args.batch_size)

    models = {}
    if "teacher" in args.models:
        models["teacher"] = backend.load_teacher(cfg).to(device)
    if "student" in args.models:
        sc = args.student_ckpt or RunPaths(cfg).student_bin()
        student = backend.build_student(cfg)
        student.load_state_dict(torch.load(sc, map_location=device))
        models["student"] = student.to(device)

    # Print table.
    print(f"\n{'model':10s}{'split':13s}{'accuracy':>10s}{'f1':>9s}")
    print("-" * 42)
    for name, model in models.items():
        for split, dl in loaders.items():
            m = evaluate_split(model, backend, task, dl, device)
            print(f"{name:10s}{split:13s}{m['accuracy']:>10.4f}{m['f1']:>9.4f}")


if __name__ == "__main__":
    main()
