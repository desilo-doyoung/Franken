"""What layer dropping costs before any distillation, per depth and per layer map.

Seeds the strided student at every depth and scores it against the teacher. Teacher
embeddings are computed once and reused, so each cell costs one student forward pass over
the eval pool — the whole sweep runs in minutes instead of the ~9 hours a distillation run
per depth would take.

Two maps are compared, because the choice dominates the depth:
  * ``stride``      — ``resolve_layer_map``'s uniform stride, the repo default (from the BERT
                      work, where 12->8 worked fine). It disturbs every region of the network,
                      including the first blocks that establish the residual stream's scale.
  * ``drop-middle`` — keep both ends, delete one contiguous middle run. Consecutive mid-stack
                      blocks make small similar increments, so removing a run of them costs
                      less than perturbing everything.

This is a GUIDE, not a result. Distillation exists precisely to recover what the init loses,
so a depth that looks poor here may still train well. Use it to pick training targets.

Usage:
    uv run python scripts/qwen3/depth_sweep.py --config configs/qwen3/exact.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)  # sibling scripts, for the metric helpers

import torch
import torch.nn.functional as F
from embed_eval import _embed, _neighbour_agreement
from franken.config import Config
from franken.distill.layer_map import resolve_layer_map
from franken.models import build_backend
from franken.tasks import build_task
from torch.utils.data import DataLoader

MIN_DEPTH = 4


def _drop_middle(teacher_depth: int, depth: int) -> list[int]:
    head = depth // 2
    return list(range(head)) + list(range(teacher_depth - (depth - head), teacher_depth))


@torch.no_grad()
def _score(backend, task, cfg, teacher, batches, device, t_emb, depth, layer_map):
    cfg.model.num_hidden_layers = depth
    cfg.distill.hidden_layer_map = layer_map
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    student = student.to(device).eval()

    s_emb = _embed(backend, student, task, batches, device)
    dist = (1 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean().item()
    recall, _ = _neighbour_agreement(s_emb, t_emb)
    params = sum(p.numel() for p in student.parameters()) / 1e6

    del student, s_emb
    torch.cuda.empty_cache()
    return dist, recall, params


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/exact.yaml")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device)
    full = teacher.config.num_hidden_layers

    data = task.datasets(tokenizer, cfg)
    ds = data["validation"].with_format("torch", columns=task.torch_columns())
    batches = list(DataLoader(ds, batch_size=16, collate_fn=data["collator"]))
    t_emb = _embed(backend, teacher, task, batches, device)

    print(f"\ncorpus={cfg.train.corpus} pool={t_emb.size(0)} texts | teacher depth {full}")
    print(f"{'depth':>5} {'params(M)':>10} | {'stride dist':>11} {'R@10':>7} | {'mid dist':>9} {'R@10':>7}")
    for depth in range(full, MIN_DEPTH - 1, -1):
        s_dist, s_recall, params = _score(
            backend, task, cfg, teacher, batches, device, t_emb, depth, resolve_layer_map(full, depth)
        )
        m_dist, m_recall, _ = _score(
            backend, task, cfg, teacher, batches, device, t_emb, depth, _drop_middle(full, depth)
        )
        flag = "  <--" if m_recall >= 0.75 and depth < full else ""
        print(
            f"{depth:>5} {params:>10.1f} | {s_dist:>11.4f} {s_recall:>7.4f} "
            f"| {m_dist:>9.4f} {m_recall:>7.4f}{flag}"
        )
    print(f"\ndrop-middle map at depth {full // 2}: {_drop_middle(full, full // 2)}\n")


if __name__ == "__main__":
    main()
