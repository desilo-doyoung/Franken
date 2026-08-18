"""Shared helpers for the qwen3 scripts: flags, teacher+student loading, retrieval scoring."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import torch

from franken.config import Config
from franken.encode import embed_texts
from franken.metrics import ndcg_pool
from franken.models import build_backend
from franken.tasks import build_task

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parser(doc: str, json: bool = True) -> argparse.ArgumentParser:
    # REQUIRED, not defaulted: two scorers once ran bare and compared a 128-token model against
    # a 1024-token one.
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="path to the experiment YAML")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    if json:
        p.add_argument("--json", help="also dump the metrics here, for scripted runs")
    return p


def config(args) -> Config:
    """The config alone, for checks worth doing before a model load."""
    return Config.from_yaml(args.config)


@dataclass
class Models:
    cfg: Config
    backend: Any
    task: Any
    tokenizer: Any
    teacher: Any
    student: Any
    device: torch.device


def load(args) -> Models:
    """Teacher and student, eval-mode, student seeded from the teacher then optionally overwritten
    from `--student-ckpt`. Warns when a no-ckpt run is NOT an identity (i.e. below full depth)."""
    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device).eval()
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    if args.student_ckpt:
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    # requires_grad_(False), not just eval(): a scorer that forgets no_grad would OOM otherwise.
    student = student.to(device).eval().requires_grad_(False)

    depth, teacher_depth = cfg.model.num_hidden_layers, teacher.config.num_hidden_layers
    if not args.student_ckpt and depth != teacher_depth:
        print(
            f"\nNOT an identity: depth {depth} != teacher {teacher_depth} and no --student-ckpt. "
            f"The student is an untrained truncation and will read ~-100%. Use a depth-"
            f"{teacher_depth} config for the self-test."
        )
    print(f"\nstudent: {args.student_ckpt or 'IDENTITY (seeded from teacher)'}")
    print(
        f"depth={depth} softmax={cfg.model.softmax} act={cfg.model.activation} "
        f"max_seq_len={cfg.train.max_seq_len}"
    )
    return Models(cfg, backend, task, tokenizer, teacher, student, device)


# --------------------------------------------------------------- pool embedding cache

# Teacher embeddings for a given pool never change, so compute once and reuse.
_NDCG_CACHE = "outputs/ndcg_cache"
_NDCG_CACHE_VERSION = 2  # v2: Pool-based pools, rebuilt by the adapter rewrite


def teacher_cache(suite: str, task: str, cfg) -> str:
    slug = re.sub(r"[^\w.-]", "_", cfg.train.teacher_model)
    return os.path.join(
        _NDCG_CACHE,
        f"v{_NDCG_CACHE_VERSION}-{suite}-{task}-{slug}-{cfg.train.max_seq_len}.pt",
    )


def pool_digest(pool) -> str:
    h = hashlib.blake2b(digest_size=8)
    for t in (*pool.q_texts, *pool.d_texts):
        h.update(t.encode())
    return h.hexdigest()


def embed_pool(m: Models, model, pool, cache: str | None = None):
    # Keyed on pool CONTENT, not size: a rebuild keeps the count while swapping the documents,
    # which once read as +107.7% on the identity self-test.
    digest = pool_digest(pool)
    if cache and os.path.exists(cache):
        blob = torch.load(cache, weights_only=True)
        if blob.get("digest") == digest:
            return blob["d"], blob["q"]
    d = embed_texts(m.backend, model, m.tokenizer, m.cfg, pool.d_texts, m.device)
    q = embed_texts(m.backend, model, m.tokenizer, m.cfg, pool.q_texts, m.device)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"d": d, "q": q, "digest": digest}, cache)
    return d, q


@torch.no_grad()
def score(m: Models, model, pool, cache: str | None = None) -> float:
    return ndcg_pool(pool, *embed_pool(m, model, pool, cache))
