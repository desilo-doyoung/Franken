"""Stream a preset's sources, mix them by weight, tokenize and cache."""

from __future__ import annotations

import os
import random
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import datasets
import numpy as np
import pyarrow.compute as pc
import transformers

from franken.data.embed_corpus.registry import PRESETS, Source
from franken.data.embed_corpus.spec import Record, corpus_texts, eval_pair, split_of

# Bump when a source, a weight or an adapter changes — the key covers the request, not the recipe
# that answered it. Manual rather than a content digest, which would discard an hours-long build
# on a cosmetic edit; the cost is that forgetting to bump serves stale text silently.
# v5: one Record per row (adapter rewrite); marco emits positives before negatives.
# v6: SPLIT_PCT 2/2 -> 1/4 (split membership is baked in, not part of the key); per-source
#     `instruct` replaces `prefix_query`.
_CACHE_DIR = "outputs/corpus_cache"
_CACHE_VERSION = 6

# Shard-order shuffling does the global mixing, so the buffer stays small — a big one only adds
# download latency, and the assembled corpus is shuffled anyway.
_SHUFFLE = 10_000

# Rows in the held-out selection pool. Deliberately NOT a config knob: recall@10 is strongly
# pool-size dependent (measured, at fixed per-vector damage: 1.000 at n=11, 0.110 at 500, 0.039 at
# 5000), so a per-run value would silently void every comparison while nothing failed.
VAL_POOL = 500


def _stream(repo: str, config: str | None, hf_split: str):
    ds = datasets.load_dataset(repo, config, split=hf_split, streaming=True)
    # Shuffle every split, not just train: several streams are grouped (Wikipedia by article id), so
    # a prefix `take` would be single-mode.
    return ds.shuffle(seed=0, buffer_size=_SHUFFLE)


def records(src: Source, split: str) -> Iterator[Record]:
    """Rows of one source belonging to `split`, normalized. The corpus and the eval both read this,
    which is what stops them disagreeing about membership or about cleaning."""
    hf_split = src.hf_split if src.key else src.split_map.get(split, split)
    for row in _stream(src.repo, src.config, hf_split):
        if src.key and split_of(str(row[src.key])) != split:
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


def _mix(sources: list[Source], split: str, n: int) -> tuple[list[str], list[int]]:
    """Draw each source in proportion, then interleave. Unshuffled, every batch is single-mode.

    A source that runs dry contributes less and its declared weight silently did not take effect —
    `mixed` once asked for 20% queries and delivered 5.2%. Report it per source as it lands: a 10M
    build runs for hours and this is the only progress signal.
    """
    drawn: list[tuple[int, list[str]]] = []
    for i, src in enumerate(sources):
        want = max(1, round(n * src.weight))
        texts = source_texts(src, split, want)
        short = "  EXHAUSTED" if len(texts) < want else ""
        print(f"  {src.name:24s} want {want:>9,}  got {len(texts):>9,}{short}", flush=True)
        drawn.append((i, texts))

    total = sum(len(t) for _i, t in drawn)
    realized = "  ".join(f"{sources[i].name} {len(t) / max(total, 1):.1%}" for i, t in drawn)
    print(f"corpus mix [{split}]: {total:,} texts for a request of {n:,}\n  {realized}", flush=True)

    rows = [(text, i) for i, texts in drawn for text in texts]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    return [t for t, _i in rows], [i for _t, i in rows]


def cache_path(name: str, split: str, n: int, max_seq_len: int, tokenizer: Any) -> str:
    """Public: `run_experiments.py` checks the cache exists before launching a concurrent batch."""
    tok_id = re.sub(r"[^\w.-]", "_", str(getattr(tokenizer, "name_or_path", "tokenizer")))
    return os.path.join(_CACHE_DIR, f"v{_CACHE_VERSION}-{name}-{split}-{n}-{max_seq_len}-{tok_id}")


def _save_atomic(ds, path: str) -> None:
    # Under torchrun every rank builds concurrently, so publish by rename rather than let N ranks
    # interleave writes into one directory; losers discard.
    tmp = f"{path}.tmp{os.getpid()}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(tmp)
    try:
        os.rename(tmp, path)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)


@lru_cache(maxsize=4)
def _build_split(name: str, split: str, n: int, max_seq_len: int, tokenizer: Any):
    """One tokenized split, memoized in-process and on disk. Sources stream, so a rebuild re-pays
    network and parsing — minutes at 216k texts, hours at 10M, and every rank pays independently."""
    if name not in PRESETS:
        raise KeyError(f"Unknown corpus {name!r}; available: {sorted(PRESETS)}")

    cached = cache_path(name, split, n, max_seq_len, tokenizer)
    if os.path.isdir(cached):
        return datasets.load_from_disk(cached)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    texts, source_ids = _mix(PRESETS[name], split, n)
    # `source` keeps provenance on the artifact: it is what lets the realized mix be verified after
    # the fact and per-domain metrics be reported. uint8 so 10M rows cost 10 MB.
    ds = datasets.Dataset.from_dict({"text": texts, "source": source_ids})
    ds = ds.cast_column("source", datasets.Value("uint8"))
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    _save_atomic(ds, cached)
    return ds


def load_embed_corpus(
    tokenizer: Any,
    name: str,
    size: int,
    max_seq_len: int = 128,
    val_size: int = VAL_POOL,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Tokenized splits plus a dynamic-padding collator — the shape `franken.data.mrpc.load_mrpc`
    returns. `sources` names the mix in `source`-column order."""
    out: dict[str, Any] = {
        split: _build_split(
            name, split, size if split == "train" else val_size, max_seq_len, tokenizer
        )
        for split in splits
    }
    out["collator"] = transformers.DataCollatorWithPadding(tokenizer)
    out["sources"] = [s.name for s in PRESETS[name]]
    return out


@dataclass(frozen=True)
class SourceProfile:
    """One source measured: how long its texts are, and whether it can be scored."""

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
    """Stream `n` texts per source and measure them. `tok/text` is not predictable from the source
    list -- the mix mingles ~10-token queries with ~200-token abstracts -- and the documented
    precedent is a 15% miss on an estimate, which would only surface after a multi-hour build.

    A source that raises is reported, not fatal: one dead loader must not hide the rest.
    """
    out = []
    for src in sources:
        try:
            texts, pairs = [], 0
            for rec in records(src, split):
                texts += corpus_texts(rec, src.instruct)
                pairs += eval_pair(rec) is not None
                if len(texts) >= n:
                    break
            raw = sorted(len(x) for x in tokenizer(texts[:n])["input_ids"])
        except Exception as e:
            out.append(
                SourceProfile(
                    src.name,
                    src.domain,
                    src.weight,
                    0.0,
                    0,
                    0.0,
                    0,
                    "none",
                    f"{type(e).__name__}: {e}",
                )
            )
            continue
        capped = [min(x, max_seq_len) for x in raw]
        out.append(
            SourceProfile(
                name=src.name,
                domain=src.domain,
                weight=src.weight,
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
    # Off the Arrow column: materializing 9M token lists as Python objects is GBs.
    lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(zero_copy_only=False)
    return SplitStats(
        n=len(lengths),
        tokens=int(lengths.sum()),
        mean=float(lengths.mean()),
        median=float(np.median(lengths)),
        truncated=float((lengths >= max_seq_len).mean()),
    )


def realized_mix(ds, n_sources: int) -> list[int]:
    """Texts per source on the BUILT artifact, from the stored `source` column. A source that ran
    dry contributed less than its declared weight, and this is where that becomes visible."""
    if "source" not in ds.column_names:
        return []
    col = ds.data.column("source").to_numpy(zero_copy_only=False)
    return np.bincount(col, minlength=n_sources).tolist()
