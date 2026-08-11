"""The shape a dataset row is normalized to, shared by the training corpus and the eval pools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Task-specific by design (the model card recommends tailoring it, worth 1-5%). MS MARCO is web
# search, so this is its matching instruction.
INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}"
)

SPLITS = ("train", "validation", "test")

# Per split, NOT cumulative bounds: the previous `VAL_PCT, TEST_PCT = 2, 4` read as a 4% test split
# and was really 2%. Small because eval needs ~5k documents and most sources over-supply 2-50x.
SPLIT_PCT = {"validation": 2, "test": 2}


def split_of(key: str) -> str:
    """`hashlib`, never `hash()` — Python salts `str.__hash__` per process, so `hash()` would
    redraw the split on every run and every machine."""
    p = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % 100
    bound = 0
    for split, pct in SPLIT_PCT.items():
        bound += pct
        if p < bound:
            return split
    return "train"


@dataclass(frozen=True)
class Record:
    query: str = ""  # the asymmetric short side, and the only text a prefix is applied to
    positives: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()  # hard negatives: corpus text AND eval distractors
    docs: tuple[str, ...] = ()  # corpus text forming no pair


def corpus_texts(rec: Record, prefix_query: bool) -> list[str]:
    q = [INSTRUCT.format(rec.query) if prefix_query else rec.query] if rec.query else []
    return q + [*rec.positives, *rec.negatives, *rec.docs]


def eval_pair(rec: Record) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """(query, golds, distractors). Every positive is gold — `ndcg_at_k` builds IDCG from all of
    them — so the adapters, not this, are where the count is deliberately kept small."""
    return (rec.query, rec.positives, rec.negatives) if rec.query and rec.positives else None
