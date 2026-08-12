"""The shape a dataset row is normalized to, shared by the training corpus and the eval pools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# The wire format, verified against this checkpoint's `config_sentence_transformers.json` rather
# than model-card prose: no space after `Query:`. Only the task description varies per source, so
# the format lives here once and nobody re-types the colon.
_INSTRUCT_FMT = "Instruct: {task}\nQuery:{query}"

# Task descriptions. Tailoring is worth 1-5% per the model card, and one string reused across
# unrelated tasks is the defect this replaces -- so a source names the task it actually retrieves.
WEB_SEARCH = "Given a web search query, retrieve relevant passages that answer the query"


def instruct(task: str | None, query: str) -> str:
    """Wrap a query in its task instruction. `None` means an unprefixed query: correct for a
    symmetric task, where there is no query/document asymmetry to instruct."""
    return _INSTRUCT_FMT.format(task=task, query=query) if task else query


SPLITS = ("train", "validation", "test")

# Per split, NOT cumulative bounds: the previous `VAL_PCT, TEST_PCT = 2, 4` read as a 4% test split
# and was really 2%. Validation only ever draws `build.VAL_POOL` (500) texts for checkpoint
# selection, so 1% is ~50x what it needs; test fills 500x5000 retrieval pools per source and at 2%
# three sources could not (codefeedback 2,900 / glaive_code 2,726 / stackexchange 4,365 docs).
SPLIT_PCT = {"validation": 1, "test": 4}


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


def corpus_texts(rec: Record, task: str | None) -> list[str]:
    q = [instruct(task, rec.query)] if rec.query else []
    return q + [*rec.positives, *rec.negatives, *rec.docs]


def eval_pair(rec: Record) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """(query, golds, distractors). Every positive is gold — `ndcg_at_k` builds IDCG from all of
    them — so the adapters, not this, are where the count is deliberately kept small."""
    return (rec.query, rec.positives, rec.negatives) if rec.query and rec.positives else None
