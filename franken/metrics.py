"""Retrieval metrics. One home, so training-time selection and end-of-run scoring cannot drift."""

from __future__ import annotations

import math

import torch

# nDCG@K and recall@K everywhere; retrieval is consumed at this scale. The selection pool is 500
# texts, so recall@10 has 5000 slots and a quantum of 0.0002.
K = 10


def recall_at_k(student: torch.Tensor, teacher: torch.Tensor, k: int = K) -> float:
    """Fraction of each text's top-k teacher neighbours the student also retrieves -- *relative*
    similarity, which is how an embedding model is actually used. THE selection metric; per-vector
    agreement (``embed_dist`` = 1-cos) is logging only, because uniform shrinkage keeps cosine high
    while destroying the ranking and a global rotation does the reverse.

    ⚠️ Comparable only at a FIXED pool size: difficulty is ``k/(n-1)``, so identical per-vector
    damage reads 1.000 at n=11, 0.110 at n=500, 0.039 at n=5000. Rows must be L2-normed.
    """
    ss, st = student @ student.T, teacher @ teacher.T
    # Mask self-similarity, else every row's nearest neighbour is itself and both models agree
    # on it for free.
    eye = torch.eye(ss.size(0), dtype=torch.bool, device=ss.device)
    ss.masked_fill_(eye, float("-inf"))
    st.masked_fill_(eye, float("-inf"))
    top_s, top_t = ss.topk(k, dim=-1).indices, st.topk(k, dim=-1).indices
    hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top_s, top_t, strict=True))
    return hits / (top_s.size(0) * k)


def gold_recall_at_k(ranked_ids, gold: dict[str, float], k: int = K) -> float:
    """Judged-gold recall -- a different question from ``recall_at_k``, which scores agreement with
    the teacher and takes no judgements.

    ``min(k, |gold|)``: a query with 40 judged documents could never exceed 0.25 otherwise, which
    reads as model damage rather than the ceiling it is.
    """
    return sum(d in gold for d in ranked_ids[:k]) / min(k, len(gold))


def ndcg_at_k(ranked_ids, relevant: dict[str, float], k: int = K) -> float:
    # Exponential gain 2^rel-1, log2(rank+1) discount — what trec_eval/MTEB compute.
    dcg = sum(
        (2.0 ** relevant.get(did, 0.0) - 1.0) / math.log2(rank + 2)
        for rank, did in enumerate(ranked_ids[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_pool(pool, d_emb, q_emb, k: int = K) -> float:
    """Mean nDCG@k over the pool's queries. The backend L2-norms its output, so this is cosine.

    Split from the embedding step so a caller can reuse the embeddings — per-task fidelity
    (recall@k, embed_dist) is then free rather than a second pass over the pool.
    """
    total = 0.0
    for i in range(0, len(pool.q_ids), 256):  # the full queries x docs matrix is needlessly big
        sims = q_emb[i : i + 256] @ d_emb.T
        top = sims.topk(min(k, sims.size(-1)), dim=-1).indices
        for row, qid in zip(top, pool.q_ids[i : i + 256], strict=True):
            total += ndcg_at_k([pool.d_ids[j] for j in row.tolist()], pool.qrels[qid], k)
    return total / len(pool.q_ids)
