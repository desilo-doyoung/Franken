"""Build a corpus artifact: draw each source to its token quota, tokenize, cache.

Reading rows is `read`; measuring them is `measure`, which nothing here calls -- drawing to a quota
needs no estimate, so a measurement cannot skew an artifact.
"""

from __future__ import annotations

import bisect
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import datasets
import transformers

from franken.data.corpus.read import records
from franken.data.corpus.source import Source
from franken.data.corpus.spec import corpus_texts

_CACHE_DIR = "outputs/corpus_cache"
# Bump when a source, weight or adapter changes: the key covers the request, not the recipe that
# answered it. Manual, not a digest, which would discard an hours-long build on a typo.
_CACHE_VERSION = 12  # v12: packed by best fit, so only over-long documents are split

# Deliberately NOT a config knob: recall@10 is strongly pool-size dependent, so a per-run value
# would silently void every comparison.
VAL_POOL = 500


def _tokens_label(tokens: float) -> str:
    return f"{int(tokens)}tok"


def cache_path(
    name: str,
    split: str,
    size_label: str,
    max_seq_len: int,
    tokenizer: Any,
    pack: bool = False,
) -> str:
    """`size_label` is a token budget for train (`…-2000000000tok-…`) and a document count for the
    fixed validation pool. Public: `run_experiments` checks the cache before launching a batch."""
    tok_id = re.sub(r"[^\w.-]", "_", str(getattr(tokenizer, "name_or_path", "tokenizer")))
    return os.path.join(
        _CACHE_DIR,
        f"v{_CACHE_VERSION}-{name}-{split}-{size_label}-{max_seq_len}-{tok_id}"
        + ("-packed" if pack else ""),
    )


def train_cache_path(
    name: str, tokens: float, max_seq_len: int, tokenizer: Any, pack: bool = False
) -> str:
    return cache_path(name, "train", _tokens_label(tokens), max_seq_len, tokenizer, pack)


@dataclass
class _Build:
    """One split's build request: what names the artifact, and what fills it.

    Exactly one of `tokens` / `docs` is set. `docs` over-draws `docs * max_seq_len` tokens -- a
    document cannot exceed the cap -- and the excess is dropped after the shuffle, so the pool is an
    exact size with the same token composition as train.
    """

    name: str
    sources: list[Source]
    split: str
    tokenizer: Any
    max_seq_len: int
    pack: bool = False
    tokens: float | None = None
    docs: int | None = None

    @property
    def budget(self) -> float:
        return self.tokens if self.tokens is not None else self.docs * self.max_seq_len

    @property
    def path(self) -> str:
        label = _tokens_label(self.tokens) if self.tokens is not None else str(self.docs)
        return cache_path(self.name, self.split, label, self.max_seq_len, self.tokenizer, self.pack)


def _batches(src: Source, split: str, size: int = 1000) -> Iterator[list[str]]:
    buf: list[str] = []
    for rec in records(src, split):
        buf += corpus_texts(rec, src.instruct)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# Documents held before packing. Sorting the buffer long-first is what makes best-fit near-optimal,
# and a buffer is packed and flushed whole, so no half-full bin survives to clog the next one.
_PACK_BUFFER_DOCS = 2048


def _padded_row(bin_tokens: list, block_size: int, source: int, pad: int) -> dict:
    """Padding needs no mask of its own: it is right-side and attention is causal, so no real token
    can reach it. A dedicated pad token keeps EOS meaning only "document ended", and the pads still
    form one isolated segment because the first of them follows the last document's EOS."""
    padding = block_size - len(bin_tokens)
    return {
        "input_ids": bin_tokens + [pad] * padding,
        "attention_mask": [1] * len(bin_tokens) + [0] * padding,
        "source": source,
    }


def _bin_pack(fragments: list[list], block_size: int, source: int, pad: int) -> Iterator[dict]:
    """Best-fit-decreasing: longest fragment first, into the tightest bin that still holds it.

    `rooms` keeps (remaining room, bin index) sorted so bisect finds that bin in log time; a linear
    scan over bins would be ~1e9 comparisons across a 2B-token draw.
    """
    bins: list[list] = []
    rooms: list[tuple[int, int]] = []
    for fragment in sorted(fragments, key=len, reverse=True):
        needed = len(fragment)
        tightest = bisect.bisect_left(rooms, (needed, -1))
        if tightest == len(rooms):
            bins.append(list(fragment))
            room_left, bin_index = block_size - needed, len(bins) - 1
        else:
            room_left, bin_index = rooms.pop(tightest)
            bins[bin_index].extend(fragment)
            room_left -= needed
        if room_left:  # a full bin can never be chosen again, and would only slow the search
            bisect.insort(rooms, (room_left, bin_index))
    for bin_tokens in bins:
        yield _padded_row(bin_tokens, block_size, source, pad)


def _rows(req: _Build) -> Iterator[dict]:
    """Every source drawn to its declared TOKEN share, tokenized once.

    Drawing to a quota rather than to a planned document count is what makes the realized shares
    exact: no tok/doc estimate stands between the declaration and the artifact.

    The quota counts tokens PLACED, not read, so the overshoot is one buffer. Counting reads let a
    single 36k-token article blow a 63k quota by 57% on `finewiki_hi`.
    """
    block_size = req.max_seq_len
    eos = getattr(req.tokenizer, "eos_token_id", None)
    # Falls back to EOS only if the tokenizer ships no pad token; then pads segment one-per-token
    # instead of as a block, which costs a little flex sparsity but is still correct.
    pad = getattr(req.tokenizer, "pad_token_id", None)
    pad = eos if pad is None else pad
    for i, src in enumerate(req.sources):
        quota = req.budget * src.weight
        got, pending = 0, []
        for batch in _batches(src, req.split):
            encoded = req.tokenizer(batch, truncation=not req.pack, max_length=block_size)
            for ids in encoded["input_ids"]:
                if not req.pack:
                    yield {"input_ids": ids, "attention_mask": [1] * len(ids), "source": i}
                    got += len(ids)
                else:
                    # Best fit, not concatenate-and-chop: a document is split only when it cannot
                    # fit a bin at all. Chopping cut 16.6% of documents that would have fit whole.
                    document = ids + [eos]
                    for start in range(0, len(document), block_size):
                        # Counted per fragment, not per document: one 36k-token article otherwise
                        # blows a 63k quota the moment it is read.
                        pending.append(document[start : start + block_size])
                        got += len(pending[-1])
                        if len(pending) >= _PACK_BUFFER_DOCS:
                            yield from _bin_pack(pending, block_size, i, pad)
                            pending = []
                        if got >= quota:
                            break
                if got >= quota:
                    break
            if got >= quota:
                break
        if pending:
            yield from _bin_pack(pending, block_size, i, pad)
        short = "  EXHAUSTED" if got < quota else ""
        print(f"  {src.name:24s} want {quota:>12,.0f}  got {got:>12,} tok{short}", flush=True)


# -------------------------------------------------------------------- materialize


_MEMO: dict[str, Any] = {}


def _build_split(req: _Build):
    """One tokenized split, memoized in-process and on disk: a rebuild re-pays network and parsing,
    hours at 10M texts, per rank. Nothing streams until past both cache checks."""
    cached = req.path
    if cached in _MEMO:
        return _MEMO[cached]
    if os.path.isdir(cached):
        return _MEMO.setdefault(cached, datasets.load_from_disk(cached))

    print(f"corpus [{req.split}]: drawing {req.budget:,.0f} tokens", flush=True)
    # 🚨 from_generator fingerprints gen_kwargs, NOT `_rows`, so its cache survives a logic change
    # and would silently republish a stale draw. Scratch dir, deleted after: ours is the only cache.
    scratch = f"{cached}.gen{os.getpid()}"
    try:
        # Writes Arrow incrementally, so memory does not scale with the corpus.
        ds = datasets.Dataset.from_generator(_rows, gen_kwargs={"req": req}, cache_dir=scratch)
        # Provenance, so the realized mix is verifiable after the fact. uint8: 10M rows cost 10 MB.
        ds = ds.cast_column("source", datasets.Value("uint8")).shuffle(seed=0)
        if req.docs is not None:
            ds = ds.select(range(min(req.docs, len(ds))))  # shuffled, so the trim is unbiased
        # Every rank builds concurrently, so publish by rename; losers discard.
        staged = f"{cached}.tmp{os.getpid()}"
        os.makedirs(_CACHE_DIR, exist_ok=True)
        ds.flatten_indices().save_to_disk(staged)
        try:
            os.rename(staged, cached)
        except OSError:
            shutil.rmtree(staged, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _MEMO.setdefault(cached, datasets.load_from_disk(cached))


def load_corpus(
    tokenizer: Any,
    name: str,
    sources: list[Source],
    tokens_per_epoch: float,
    max_seq_len: int = 128,
    val_size: int = VAL_POOL,
    splits: tuple[str, ...] = ("train", "validation"),
    pack: bool = False,
) -> dict[str, Any]:
    """Tokenized splits plus a collator, the shape `load_mrpc` returns. `name` is the cache-key
    identity of the request and `sources` the recipe answering it."""
    if pack and tokenizer.eos_token_id is None:
        raise ValueError("train.pack needs an eos_token_id to mark document boundaries.")

    out: dict[str, Any] = {}
    for split in splits:
        train = split == "train"
        out[split] = _build_split(
            _Build(
                name=name,
                sources=sources,
                split=split,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                pack=pack,
                tokens=tokens_per_epoch if train else None,
                docs=None if train else val_size,
            )
        )
    out["collator"] = transformers.DataCollatorWithPadding(tokenizer)
    return out
