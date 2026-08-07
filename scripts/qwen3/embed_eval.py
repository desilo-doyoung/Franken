"""Score a distilled embedding student against the teacher.

`embed_dist` (the training metric) is per-vector agreement, which an embedding model is
never actually used through — retrieval consumes *relative* similarity. A student can hold
high per-vector cosine while its ranking degrades (uniform shrinkage of the near/far
spread), or the reverse (a global rotation preserves every ranking). So the headline here
is **recall@10**: how many of the teacher's top-10 neighbours the student also retrieves.
It needs no labels. STS-B adds a labelled anchor, reported as a delta — the claim is
preservation of the teacher, not absolute quality.

With no --student-ckpt the student is seeded from the teacher, i.e. the identity baseline:
recall@10 ~1.0 and delta ~0 are then a self-test of this script.

Usage:
    uv run python scripts/qwen3/embed_eval.py --config configs/qwen3/exact.yaml
    uv run python scripts/qwen3/embed_eval.py --student-ckpt outputs/qwen3/student/pytorch_model.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datasets
import torch
import torch.nn.functional as F
from franken.config import Config
from franken.models import build_backend
from franken.tasks import build_task
from franken.tasks.embed import RECALL_K as K
from franken.tasks.embed import recall_at_k
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


@torch.no_grad()
def _embed(backend, model, task, batches, device):
    out = []
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(backend.forward(model, task.model_inputs(batch))["output"].float().cpu())
    return torch.cat(out)


def _neighbour_agreement(student, teacher):
    """recall@K vs the teacher's neighbours, plus Spearman over all pairwise similarities.

    recall@K comes from `franken.tasks.embed.recall_at_k` — the same function the training loop
    selects checkpoints with, so this script cannot silently score a different quantity than
    the one that chose the checkpoint.
    """
    recall = recall_at_k(student, teacher, K)

    ss, st = student @ student.T, teacher @ teacher.T  # unit-norm rows -> cosine
    # Every off-diagonal pairwise similarity (self-similarity excluded), Spearman-correlated.
    off = ~torch.eye(ss.size(0), dtype=torch.bool)
    rho = spearmanr(ss[off].numpy(), st[off].numpy()).statistic
    return recall, rho


@torch.no_grad()
def _embed_texts(backend, model, tokenizer, cfg, texts, device, batch_size: int = 32):
    """Embed a plain list of strings (tokenize + pool), truncated at `cfg.train.max_seq_len`.

    Distinct from `_embed`, which consumes pre-collated batches from the corpus loader. Shared with
    `retrieval_eval.py` so the two scripts cannot drift on tokenization or truncation — the same
    reason `recall_at_k` lives in `franken/tasks/embed.py` rather than here.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=cfg.train.max_seq_len,
            return_tensors="pt",
        ).to(device)
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        out.append(backend.forward(model, inputs)["output"].float().cpu())
    return torch.cat(out)


def _stsb(backend, model, tokenizer, cfg, device):
    """Spearman of cosine(s1, s2) against the human similarity labels."""
    ds = datasets.load_dataset("nyu-mll/glue", "stsb", split="validation")
    s1 = _embed_texts(backend, model, tokenizer, cfg, ds["sentence1"], device)
    s2 = _embed_texts(backend, model, tokenizer, cfg, ds["sentence2"], device)
    sims = F.cosine_similarity(s1, s2, dim=-1)
    # label range is [0, 5], but Spearman is rank-based so it doesn't matter.
    return spearmanr(sims.numpy(), ds["label"]).statistic


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/exact.yaml")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    p.add_argument("--json", help="also dump the metrics here, for scripted runs (run_experiments)")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device)
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    if args.student_ckpt:
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    student = student.to(device).eval()
    print(f"\nstudent: {args.student_ckpt or 'IDENTITY (seeded from teacher)'}")
    print(
        f"depth={cfg.model.num_hidden_layers} "
        f"softmax={cfg.model.softmax} act={cfg.model.activation}"
    )

    data = task.datasets(tokenizer, cfg)
    ds = data["validation"].with_format("torch", columns=task.torch_columns())
    batches = list(DataLoader(ds, batch_size=16, collate_fn=data["collator"]))
    s_emb, t_emb = (_embed(backend, m, task, batches, device) for m in (student, teacher))

    dist = (1 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean().item()
    recall, rho = _neighbour_agreement(s_emb, t_emb)
    print(f"\npool: {s_emb.size(0)} held-out texts")
    print(f"  embed_dist    {dist:.6f}   (per-vector; the training metric)")
    print(f"  recall@{K}     {recall:.4f}     (teacher's top-{K} neighbours also found)")
    print(f"  sim-spearman  {rho:.6f}   (all pairwise similarities)")

    t_sts = _stsb(backend, teacher, tokenizer, cfg, device)
    s_sts = _stsb(backend, student, tokenizer, cfg, device)
    print(
        f"\nSTS-B spearman: teacher {t_sts:.4f}  student {s_sts:.4f}  delta {s_sts - t_sts:+.4f}\n"
    )

    # Same numbers, structurally: run_experiments.py collects these instead of re-parsing the
    # prose above, so a reworded print can't silently break the summary table.
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "config": args.config,
                    "student_ckpt": args.student_ckpt,
                    "depth": cfg.model.num_hidden_layers,
                    "softmax": cfg.model.softmax,
                    "activation": cfg.model.activation,
                    "k": K,
                    "pool": s_emb.size(0),
                    "recall": recall,
                    "embed_dist": dist,
                    "sim_rho": rho,
                    "stsb_teacher": t_sts,
                    "stsb_student": s_sts,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
