"""Rows out of a source: which upstream split, which hash split, and the text a row yields."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import datasets

from franken.data.corpus.source import Source
from franken.data.corpus.spec import Record, corpus_texts, split_of

# Shard-order shuffling does the global mixing, so the buffer stays small.
_SHUFFLE = 10_000


@cache
def _judged(spec) -> frozenset[str]:
    """Documents a `Qrels` source judges, held out of training wholesale: `evalset._from_qrels`
    force-adds every gold to its pool, so `split_of` alone would leave them in the draw."""
    _qid, pid, score = spec.cols
    rows = datasets.load_dataset(spec.repo, split=spec.split)
    return frozenset(str(r[pid]) for r in rows if float(r[score]) > 0)


def records(src: Source, split: str) -> Iterator[Record]:
    """Rows of one source belonging to `split`. The corpus and the eval both read this, so they
    cannot disagree about membership."""
    hf_split = src.hf_split if src.key else src.split_map.get(split, split)
    judged = _judged(src.qrels) if src.qrels and split == "train" else frozenset()
    rows = datasets.load_dataset(src.repo, src.config, split=hf_split, streaming=True)
    # Shuffled every split, not just train: several streams are grouped, so a prefix `take` would
    # be single-mode. Shard-order shuffling does the global mixing, so the buffer stays small.
    for row in rows.shuffle(seed=0, buffer_size=_SHUFFLE):
        if src.key and split_of(str(row[src.key])) != split:
            continue
        if judged and str(row[src.key]) in judged:
            continue
        rec = src.adapt(row)
        if rec is not None:
            yield rec


def source_texts(src: Source, split: str, n: int) -> list[str]:
    out: list[str] = []
    for rec in records(src, split):
        out += corpus_texts(rec, src.instruct)
        if len(out) >= n:
            break
    return out[:n]
