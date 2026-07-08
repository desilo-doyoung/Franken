"""Evaluate the teacher and/or distilled student on MRPC splits.

Reports accuracy + F1 for each (model, split) pair as a table. MRPC is the one
GLUE task whose `test` split ships with public labels, so validation *and* test
can both be scored locally.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --student-ckpt outputs/student/pytorch_model.bin --splits test
    python scripts/evaluate.py --models student --splits validation test
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/evaluate.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from franken.config import Config
from franken.data.mrpc import compute_metrics
from franken.model.bert import BertForClassification


@torch.no_grad()
def evaluate_split(model, is_hf: bool, dl: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        logits = out.logits if is_hf else out["logits"]
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return compute_metrics(preds, labels)


def build_loaders(tokenizer, splits, max_seq_len, batch_size):
    ds = load_dataset("nyu-mll/glue", "mrpc")
    ds = ds.map(
        lambda b: tokenizer(b["sentence1"], b["sentence2"], truncation=True, max_length=max_seq_len),
        batched=True,
    )
    collator = DataCollatorWithPadding(tokenizer)
    cols = ["input_ids", "token_type_ids", "attention_mask", "label"]
    loaders = {}
    for split in splits:
        # Guard: a split with only -1 labels is unlabeled (can't score locally).
        if set(ds[split].unique("label")) == {-1}:
            print(f"[skip] split '{split}' has no public labels (all -1).")
            continue
        d = ds[split].with_format("torch", columns=cols)
        loaders[split] = DataLoader(d, batch_size=batch_size, collate_fn=collator)
    return loaders


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml", help="config (student arch + teacher_ckpt)")
    p.add_argument("--student-ckpt", default=None,
                   help="student state_dict (default: <output_dir>/student/pytorch_model.bin)")
    p.add_argument("--models", nargs="+", choices=["teacher", "student"],
                   default=["teacher", "student"], help="which models to evaluate")
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
    loaders = build_loaders(tokenizer, args.splits, cfg.train.max_seq_len, args.batch_size)

    models = {}
    if "teacher" in args.models:
        ckpt = cfg.train.teacher_ckpt or cfg.train.teacher_model
        models["teacher"] = (AutoModelForSequenceClassification.from_pretrained(ckpt).to(device), True)
    if "student" in args.models:
        sc = args.student_ckpt or os.path.join(cfg.train.output_dir, "student", "pytorch_model.bin")
        student = BertForClassification(cfg.model)
        student.load_state_dict(torch.load(sc, map_location=device))
        models["student"] = (student.to(device), False)

    # Print table.
    print(f"\n{'model':10s}{'split':13s}{'accuracy':>10s}{'f1':>9s}")
    print("-" * 42)
    for name, (model, is_hf) in models.items():
        for split, dl in loaders.items():
            m = evaluate_split(model, is_hf, dl, device)
            print(f"{name:10s}{split:13s}{m['accuracy']:>10.4f}{m['f1']:>9.4f}")


if __name__ == "__main__":
    main()
