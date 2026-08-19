"""Stream a preset's sources, mix them by weight, tokenize and cache."""

from __future__ import annotations

import os
import random
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Any

import datasets
import numpy as np
import pyarrow.compute as pc
import transformers

from franken.data.embed_corpus.registry import PRESETS, Source
from franken.data.embed_corpus.spec import Record, corpus_texts, eval_pair, split_of

_CACHE_DIR = "outputs/corpus_cache"
# Bump when a source, weight or adapter changes: the key covers the request, not the recipe that
# answered it. Manual, not a digest, which would discard an hours-long build on a typo.
_CACHE_VERSION = 9  # v9: keyed on a token budget, not a text count

# Shard-order shuffling does the global mixing, so the buffer stays small.
_SHUFFLE = 10_000

# Deliberately NOT a config knob: recall@10 is strongly pool-size dependent, so a per-run value
# would silently void every comparison.
VAL_POOL = 500

CALIB = 2000  # texts/source for the tokens->texts calibration; +-0.9% on the mix mean


def _stream(repo: str, config: str | None, hf_split: str):
    ds = datasets.load_dataset(repo, config, split=hf_split, streaming=True)
    # Every split, not just train: several streams are grouped, so a prefix `take` is single-mode.
    return ds.shuffle(seed=0, buffer_size=_SHUFFLE)


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
    for row in _stream(src.repo, src.config, hf_split):
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


def _mix(sources: list[Source], split: str, n: int) -> tuple[list[str], list[int]]:
    """Draw each source in proportion, then interleave. A source that runs dry silently misses its
    declared weight, so report per source as it lands -- also the only progress signal."""
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


def _tokens_label(tokens: float) -> str:
    return f"{int(tokens)}tok"


def cache_path(name: str, split: str, size_label: str, max_seq_len: int, tokenizer: Any) -> str:
    """`size_label` is a token budget for train (`…-2000000000tok-…`) and a text count for the
    fixed validation pool. Public: `run_experiments` checks the cache before launching a batch."""
    tok_id = re.sub(r"[^\w.-]", "_", str(getattr(tokenizer, "name_or_path", "tokenizer")))
    return os.path.join(
        _CACHE_DIR, f"v{_CACHE_VERSION}-{name}-{split}-{size_label}-{max_seq_len}-{tok_id}"
    )


def train_cache_path(name: str, tokens: float, max_seq_len: int, tokenizer: Any) -> str:
    return cache_path(name, "train", _tokens_label(tokens), max_seq_len, tokenizer)


def _calibrate(name: str, max_seq_len: int, tokenizer: Any, tokens: float) -> int:
    """How many texts sum to `tokens`. Measured here, not declared: tok/text depends on the mix AND
    the cap (~110 at 1024, ~97 at 256). The cache is keyed on the token budget, so sampling error
    moves the text count and can never rename a corpus."""
    rows = [r for r in profile(PRESETS[name], tokenizer, max_seq_len, n=CALIB) if not r.error]
    covered = sum(r.weight for r in rows)
    if not covered:
        raise RuntimeError(f"No source in {name!r} produced a sample; cannot size the corpus.")
    mean = sum(r.weight * r.mean for r in rows) / covered  # rescaled, so a dead source is not 0
    print(f"calibration: {mean:.1f} tok/text over {covered:.3f} of the mix", flush=True)
    return max(1, round(tokens / mean))


def _save_atomic(ds, path: str) -> None:
    # Every rank builds concurrently, so publish by rename; losers discard.
    tmp = f"{path}.tmp{os.getpid()}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(tmp)
    try:
        os.rename(tmp, path)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)


_MEMO: dict[str, Any] = {}


def _build_split(name, split, size_label, max_seq_len, tokenizer, resolve_n):
    """One tokenized split, memoized in-process and on disk: a rebuild re-pays network and parsing,
    hours at 10M texts, per rank. `resolve_n` is called only on a miss, so a cache hit never pays
    for calibration."""
    if name not in PRESETS:
        raise KeyError(f"Unknown corpus {name!r}; available: {sorted(PRESETS)}")

    cached = cache_path(name, split, size_label, max_seq_len, tokenizer)
    if cached in _MEMO:
        return _MEMO[cached]
    if os.path.isdir(cached):
        return _MEMO.setdefault(cached, datasets.load_from_disk(cached))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    texts, source_ids = _mix(PRESETS[name], split, resolve_n())
    # Provenance, so the realized mix can be verified after the fact. uint8: 10M rows cost 10 MB.
    ds = datasets.Dataset.from_dict({"text": texts, "source": source_ids})
    ds = ds.cast_column("source", datasets.Value("uint8"))
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    _save_atomic(ds, cached)
    return _MEMO.setdefault(cached, ds)


def load_embed_corpus(
    tokenizer: Any,
    name: str,
    tokens_per_epoch: float,
    max_seq_len: int = 128,
    val_size: int = VAL_POOL,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Tokenized splits plus a collator, the shape `load_mrpc` returns. `tokens_per_epoch` sizes
    train; validation is a fixed TEXT count, since recall@10 is pool-size dependent."""
    out: dict[str, Any] = {}
    for split in splits:
        if split == "train":
            out[split] = _build_split(
                name,
                split,
                _tokens_label(tokens_per_epoch),
                max_seq_len,
                tokenizer,
                lambda: _calibrate(name, max_seq_len, tokenizer, tokens_per_epoch),
            )
        else:
            out[split] = _build_split(
                name, split, str(val_size), max_seq_len, tokenizer, lambda: val_size
            )
    out["collator"] = transformers.DataCollatorWithPadding(tokenizer)
    out["sources"] = [s.name for s in PRESETS[name]]
    return out


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
    """Stream `n` texts per source and measure them: `tok/text` is not predictable from the source
    list, and a bad estimate only surfaces after a multi-hour build.

    A source that raises is reported, not fatal -- one dead loader must not hide the rest.
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
    """Texts per source on the BUILT artifact -- where a source that ran dry becomes visible."""
    if "source" not in ds.column_names:
        return []
    col = ds.data.column("source").to_numpy(zero_copy_only=False)
    return np.bincount(col, minlength=n_sources).tolist()
