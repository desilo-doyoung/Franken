"""Shared setup for the qwen3 scripts.

Import this before any `franken` module — it puts the repo root on `sys.path`, which is why the
scripts no longer each carry that dance. isort keeps it first: `franken` is configured as
first-party, so it always sorts after this.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from franken.config import Config  # noqa: E402
from franken.models import build_backend  # noqa: E402
from franken.tasks import build_task  # noqa: E402

# One default for every script. They used to disagree, so two scorers run with no flags compared a
# 128-token model against a 1024-token one.
DEFAULT_CONFIG = "configs/qwen3/depth19_multi_domain.yaml"


def parser(doc: str, json: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
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
    student = student.to(device).eval()

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


@torch.no_grad()
def _embed_texts(backend, model, tokenizer, cfg, texts, device, batch_size: int = 32):
    """Embed a plain list of strings (tokenize + pool), truncated at `cfg.train.max_seq_len`.

    Here rather than in one scorer, so no two of them can drift on tokenization or truncation.

    Batches are length-sorted and the original order restored afterwards. Unsorted, each batch pads
    to its longest member: at `max_seq_len` 1024 that is ~1024 tokens per text against a median near
    130. Longest-first, so an over-large batch fails on step 1 rather than 90% of the way in.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out: list[torch.Tensor] = [None] * len(texts)
    for i in range(0, len(order), batch_size):
        chunk = order[i : i + batch_size]
        enc = tokenizer(
            [texts[j] for j in chunk],
            padding=True,
            truncation=True,
            max_length=cfg.train.max_seq_len,
            return_tensors="pt",
        ).to(device)
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        emb = backend.forward(model, inputs)["output"].float().cpu()
        for k, j in enumerate(chunk):
            out[j] = emb[k]
    return torch.stack(out)


# --------------------------------------------------------------- retrieval scoring

K = 10  # nDCG@K and recall@K everywhere; retrieval is consumed at this scale

# Teacher embeddings for a given pool never change, so compute once and reuse.
_NDCG_CACHE = "outputs/ndcg_cache"
_NDCG_CACHE_VERSION = 2  # v2: Pool-based pools, rebuilt by the adapter rewrite


def ndcg_at_k(ranked_ids, relevant: dict[str, float], k: int = K) -> float:
    # Exponential gain 2^rel-1, log2(rank+1) discount — what trec_eval/MTEB compute.
    dcg = sum(
        (2.0 ** relevant.get(did, 0.0) - 1.0) / math.log2(rank + 2)
        for rank, did in enumerate(ranked_ids[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


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


def _embed_pool(m: Models, model, pool, cache: str | None):
    # Validate the cache on pool CONTENT, not size: a change to how pools are built holds the count
    # at 500x5000 while swapping the documents, which once served an old pool's teacher embeddings
    # against new ids and read as +107.7% on the identity self-test.
    digest = pool_digest(pool)
    if cache and os.path.exists(cache):
        blob = torch.load(cache, weights_only=True)
        if blob.get("digest") == digest:
            return blob["d"], blob["q"]
    d = _embed_texts(m.backend, model, m.tokenizer, m.cfg, pool.d_texts, m.device)
    q = _embed_texts(m.backend, model, m.tokenizer, m.cfg, pool.q_texts, m.device)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"d": d, "q": q, "digest": digest}, cache)
    return d, q


@torch.no_grad()
def score(m: Models, model, pool, cache: str | None = None) -> float:
    """Mean nDCG@K over the pool's queries. The backend L2-norms its output, so this is cosine."""
    d_emb, q_emb = _embed_pool(m, model, pool, cache)
    total = 0.0
    for i in range(0, len(pool.q_ids), 256):  # the full queries x docs matrix is needlessly big
        sims = q_emb[i : i + 256] @ d_emb.T
        top = sims.topk(min(K, sims.size(-1)), dim=-1).indices
        for row, qid in zip(top, pool.q_ids[i : i + 256], strict=True):
            total += ndcg_at_k([pool.d_ids[j] for j in row.tolist()], pool.qrels[qid])
    return total / len(pool.q_ids)
