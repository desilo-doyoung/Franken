"""The shape a dataset row is normalized to, shared by the training corpus and the eval pools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Verified against the checkpoint's config, not model-card prose: no space after `Query:`.
_INSTRUCT_FMT = "Instruct: {task}\nQuery:{query}"

# A source names the task it actually retrieves; tailoring is worth 1-5% per the model card.
WEB_SEARCH = "Given a web search query, retrieve relevant passages that answer the query"


def instruct(task: str | None, query: str) -> str:
    """`None` means an unprefixed query -- correct for a symmetric task."""
    return _INSTRUCT_FMT.format(task=task, query=query) if task else query


SPLITS = ("train", "validation", "test")

# Per split, NOT cumulative bounds. Validation only draws 500 texts; test fills 500x5000 pools.
SPLIT_PCT = {"validation": 1, "test": 4}


def split_of(key: str) -> str:
    """`hashlib`, never `hash()`: Python salts `str.__hash__` per process."""
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


def corpus_texts(rec: Record, task: str | None) -> list[str]:
    q = [instruct(task, rec.query)] if rec.query else []
    return q + [*rec.positives, *rec.negatives, *rec.docs]


def eval_pair(rec: Record) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """(query, golds, distractors). Every positive is gold, so adapters keep the count small."""
    return (rec.query, rec.positives, rec.negatives) if rec.query and rec.positives else None
