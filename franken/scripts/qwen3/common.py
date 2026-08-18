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
    # `--config` is REQUIRED, not defaulted: the config decides what gets measured and these
    # scripts cost minutes to hours, so a wrong default is expensive and silent. The defaults used
    # to disagree, and two scorers run bare compared a 128-token model against a 1024-token one.
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="path to the experiment YAML")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    if json:
        p.add_argument("--json", help="also dump the metrics here, for scripted runs")
    return p


def config(args) -> Config:
    """The config alone, for checks worth doing before paying for a model load."""
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
    """Teacher and student, both eval-mode on the config's device, student seeded from the teacher
    and then optionally overwritten from `--student-ckpt`.

    Warns when a no-ckpt run is NOT an identity: a seeded student only equals the teacher at full
    depth; truncated it scores near-random (-99.7% measured at depth 19) and says nothing.
    """
    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device).eval()
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    if args.student_ckpt:
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    # requires_grad_(False), not just eval(): these scripts only score. Without it a scorer that
    # forgets no_grad retains an autograd graph per batch and OOMs on the accumulation loop.
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
    # Validate the cache on pool CONTENT, not size: a change to how pools are built holds the count
    # at 500x5000 while swapping the documents, which once served an old pool's teacher embeddings
    # against new ids and read as +107.7% on the identity self-test.
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
