"""Accuracy + F1 for the teacher and/or student on MRPC splits. MRPC is the one GLUE task whose
`test` split ships with public labels, so both splits score locally.

Usage:
    python -m franken.scripts.bert.evaluate --config configs/bert/depth8_exact.yaml
    python -m franken.scripts.bert.evaluate --models student --splits validation test
    python -m franken.scripts.bert.evaluate \
        --student-ckpt outputs/bert/student/pytorch_model.bin --splits test
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from franken.config import Config
from franken.data.mrpc import compute_metrics, load_mrpc
from franken.models import build_backend
from franken.paths import RunPaths
from franken.tasks import build_task


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
    data = load_mrpc(tokenizer, max_seq_len, splits=tuple(splits))
    cols = task.torch_columns()
    loaders = {}
    for split in splits:
        # Guard: a split with only -1 labels is unlabeled (can't score locally).
        if set(data[split].unique("label")) == {-1}:
            print(f"[skip] split '{split}' has no public labels (all -1).")
            continue
        d = data[split].with_format("torch", columns=cols)
        loaders[split] = DataLoader(d, batch_size=batch_size, collate_fn=data["collator"])
    return loaders


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/bert/depth8_exact.yaml",
        help="config (student arch + teacher_ckpt)",
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
        sc = args.student_ckpt or RunPaths(cfg).student_bin
        student = backend.build_student(cfg)
        student.load_state_dict(torch.load(sc, map_location=device))
        models["student"] = student.to(device)

    print(f"\n{'model':10s}{'split':13s}{'accuracy':>10s}{'f1':>9s}")
    print("-" * 42)
    for name, model in models.items():
        for split, dl in loaders.items():
            m = evaluate_split(model, backend, task, dl, device)
            print(f"{name:10s}{split:13s}{m['accuracy']:>10.4f}{m['f1']:>9.4f}")


if __name__ == "__main__":
    main()
