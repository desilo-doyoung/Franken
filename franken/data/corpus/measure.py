"""Measuring sources and built artifacts, for REPORTING.

Nothing here can change what gets built -- the build draws to a token quota and needs no estimate.
Before that was true, this module's sampling error moved the corpus composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow.compute as pc

from franken.data.corpus.read import records
from franken.data.corpus.source import Source
from franken.data.corpus.spec import corpus_texts, eval_pair


@dataclass(frozen=True)
class SourceProfile:
    """One source measured: text length, and whether it can be scored."""

    name: str
    domain: str
    weight: float
    mean: float  # post-cap mean tokens/text -- what the token budget is spent on
    median: int
    truncated: float  # fraction over the cap, on an UNtruncated basis
    longest: int  # max untruncated; the FHE polynomial domain is set by the max, not a percentile
    scoreable: str  # "pair" | "qrels" | "none"
    error: str = ""  # set instead of the measurements when the source failed to load


def profile(
    sources: list[Source], tokenizer: Any, max_seq_len: int, n: int = 300, split: str = "train"
) -> list[SourceProfile]:
    """Stream `n` texts per source and measure them. REPORTING only: the build draws to a token
    quota and needs no estimate, so nothing measured here can skew an artifact.

    A source that raises is reported, not fatal -- the gate must name every dead loader, not the
    first. The BUILD makes the opposite choice and lets it raise.
    """
    out = []
    for src in sources:
        head = dict(name=src.name, domain=src.domain, weight=src.weight)
        try:
            texts, pairs = [], 0
            for rec in records(src, split):
                texts += corpus_texts(rec, src.instruct)
                pairs += eval_pair(rec) is not None
                if len(texts) >= n:
                    break
            raw = sorted(len(x) for x in tokenizer(texts[:n])["input_ids"])
        except Exception as e:
            # Keyword-only: a positional construction here silently shifts every measurement left
            # when a field is added.
            out.append(
                SourceProfile(
                    **head,
                    mean=0.0,
                    median=0,
                    truncated=0.0,
                    longest=0,
                    scoreable="none",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            continue
        capped = [min(x, max_seq_len) for x in raw]
        out.append(
            SourceProfile(
                **head,
                mean=sum(capped) / max(len(capped), 1),
                median=capped[len(capped) // 2],
                truncated=sum(1 for x in raw if x > max_seq_len) / max(len(raw), 1),
                longest=raw[-1],
                scoreable="qrels" if src.qrels else ("pair" if pairs else "none"),
            )
        )
    return out


@dataclass(frozen=True)
class SplitStats:
    n: int
    tokens: int
    mean: float
    median: float
    truncated: float


def describe(ds, max_seq_len: int) -> SplitStats:
    # Off the Arrow column: materializing 9M token lists as Python objects costs GBs.
    lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(zero_copy_only=False)
    return SplitStats(
        n=len(lengths),
        tokens=int(lengths.sum()),
        mean=float(lengths.mean()),
        median=float(np.median(lengths)),
        truncated=float((lengths >= max_seq_len).mean()),
    )


def realized_mix(ds, n_sources: int) -> list[int]:
    """Rows per source on the BUILT artifact -- where a source that ran dry becomes visible."""
    if "source" not in ds.column_names:
        return []
    col = ds.data.column("source").to_numpy(zero_copy_only=False)
    return np.bincount(col, minlength=n_sources).tolist()
