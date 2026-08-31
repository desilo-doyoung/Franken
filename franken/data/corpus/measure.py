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


def real_tokens(ds) -> np.ndarray:
    """Unpadded tokens per row. A packed artifact has no padding and stores no `attention_mask`, so
    its row lengths ARE its real tokens; unpacked rows are ragged and all-real too. The mask branch
    survives for artifacts built before v13, whose bins were padded to the cap."""
    lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(zero_copy_only=False)
    if "attention_mask" not in ds.column_names or not len(lengths):
        return lengths
    flat = np.asarray(pc.list_flatten(ds.data.column("attention_mask")))
    return np.add.reduceat(flat, np.r_[0, np.cumsum(lengths)[:-1]])


def describe(ds, max_seq_len: int) -> SplitStats:
    # Off the Arrow column: materializing 9M token lists as Python objects costs GBs.
    real = real_tokens(ds)
    return SplitStats(
        n=len(real),
        tokens=int(real.sum()),
        mean=float(real.mean()),
        median=float(np.median(real)),
        # Unpacked this is "hit the cap"; packed it is "bin needed no padding". Same predicate,
        # because an unpacked row's mask is all ones.
        truncated=float((real >= max_seq_len).mean()),
    )


def realized_mix(ds, n_sources: int) -> list[int]:
    """Real tokens per source on the BUILT artifact -- where a source that ran dry becomes visible.
    Tokens, not rows: `Source.weight` is a token share, and a padded row is not a full one."""
    if "source" not in ds.column_names:
        return []
    col = ds.data.column("source").to_numpy(zero_copy_only=False)
    return np.bincount(col, weights=real_tokens(ds), minlength=n_sources).astype(int).tolist()


def split_doc_share(ds, bos_id: int | None, eos_id: int | None) -> float:
    """Share of documents the chop split -- directly comparable to the 16.6% that best-fit packing
    avoided, and the number that decides whether chopping is costing anything.

    A row opening on something other than BOS is one split event; EOS counts documents. Reporting
    the ROW share instead would read ~98% at any realistic block size, since a row holds several
    documents and almost never ends on a boundary -- alarming, and about nothing.
    """
    if bos_id is None or eos_id is None or not len(ds):
        return 0.0
    col = ds.data.column("input_ids")
    splits = int((pc.list_element(col, 0).to_numpy(zero_copy_only=False) != bos_id).sum())
    docs = int((np.asarray(pc.list_flatten(col)) == eos_id).sum())
    return splits / max(docs, 1)
