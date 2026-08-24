"""Retrieval scoring for the embed track: teacher pool embeddings, cached on pool content."""

from __future__ import annotations

import hashlib
import os
import re

import torch

from franken.encode import embed_texts
from franken.metrics import ndcg_pool
from franken.scripts.common import Models

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
