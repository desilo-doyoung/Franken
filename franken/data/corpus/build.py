"""Build a corpus artifact: draw each source to its token quota, tokenize, cache.

Reading rows is `read`; measuring them is `measure`, which nothing here calls -- drawing to a quota
needs no estimate, so a measurement cannot skew an artifact.
"""

from __future__ import annotations

import json
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
from franken.paths import ROOT

# Absolute: a relative dir made the cache identity depend on the process CWD.
_CACHE_DIR = os.path.join(ROOT, "outputs", "corpus_cache")
# Bump when a source, weight or adapter changes: the key covers the request, not the recipe that
# answered it. Manual, not a digest, which would discard an hours-long build on a typo.
_CACHE_VERSION = 13  # v13: packed by concatenate-and-chop; no padding, no stored attention_mask

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


def cache_missing(cfg, tokenizer: Any, val_size: int = VAL_POOL) -> bool:
    """BOTH splits: `Distiller.train` builds train and validation, so a train-only probe would let
    a half-cached mix reach the ranks and trip `_refuse_under_ddp`."""
    t = cfg.train
    paths = (
        train_cache_path(t.corpus, t.tokens_per_epoch, t.max_seq_len, tokenizer, t.pack),
        cache_path(t.corpus, "validation", str(val_size), t.max_seq_len, tokenizer, t.pack),
    )
    return not all(os.path.isdir(p) for p in paths)


@dataclass
class _Build:
    """One split's build request: what names the artifact, and what fills it.

    Exactly one of `tokens` / `docs` is set. `docs` over-draws and the excess is dropped after the
    shuffle, so the pool is an exact size with the same token composition as train.
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
        if self.tokens is not None:
            return self.tokens
        # Packing drops each source's trailing partial block, so draw one spare block per source
        # or the fixed pool lands a few rows short of `docs`.
        spare = len(self.sources) if self.pack else 0
        return (self.docs + spare) * self.max_seq_len

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


def _rows(req: _Build) -> Iterator[dict]:
    """Every source drawn to its declared TOKEN share, tokenized once.

    Drawing to a quota rather than to a planned document count is what makes the realized shares
    exact: no tok/doc estimate stands between the declaration and the artifact.

    Packed, documents are concatenated and chopped into whole `max_seq_len` blocks. A document that
    runs past a boundary is cut and its tail continues on the next row as an independent sequence --
    which is exactly how `packing.doc_positions` already reads a fragment with no preceding EOS. The
    truncated context shifts the INPUT distribution; it does not corrupt the target, because the
    teacher is scored on the same input. That is what makes chopping cheap here and expensive in
    next-token pretraining, where the label itself would be wrong.
    """
    block_size = req.max_seq_len
    eos = getattr(req.tokenizer, "eos_token_id", None)
    for i, src in enumerate(req.sources):
        # Integer, so `room` reaches exactly 0: a fractional remainder would append nothing while
        # still reading, and the source loop would never terminate.
        quota = int(req.budget * src.weight)
        got, blocks, buf = 0, 0, []
        for batch in _batches(src, req.split):
            encoded = req.tokenizer(batch, truncation=not req.pack, max_length=block_size)
            for ids in encoded["input_ids"]:
                if not req.pack:
                    yield {"input_ids": ids, "attention_mask": [1] * len(ids), "source": i}
                    got += len(ids)
                else:
                    # Cut AT the remaining room, so the draw is exact rather than overshooting by
                    # a document: one 36k-token Hindi article blew a 63k quota by 57% when the
                    # quota counted tokens READ.
                    document = (ids + [eos])[: quota - got]
                    buf.extend(document)
                    got += len(document)
                    while len(buf) >= block_size:
                        # No attention_mask: every position is real, and the collator synthesizes
                        # the all-ones mask the loss reads.
                        yield {"input_ids": buf[:block_size], "source": i}
                        del buf[:block_size]
                        blocks += 1
                if got >= quota:
                    break
            if got >= quota:
                break
        # Judged on the DRAW, before the trailing block is dropped: emitted tokens are a whole
        # number of blocks and so are always a little under quota, which would flag every source.
        short = "  EXHAUSTED" if got < quota else ""
        if req.pack:
            # Report what REACHED the artifact, or the line disagrees with `realized_mix`.
            got = blocks * block_size
            if not blocks:
                # Louder than EXHAUSTED: the source is absent entirely, which biases the mix
                # toward whatever fills a block.
                short = "  PRODUCED NO BLOCKS"
        print(f"  {src.name:24s} want {quota:>12,.0f}  got {got:>12,} tok{short}", flush=True)


# -------------------------------------------------------------------- materialize


_MEMO: dict[str, Any] = {}

MANIFEST = "franken_manifest.json"


def _refuse_under_ddp(cached: str) -> None:
    """A cold cache under torchrun is a hard error, not an N-way race.

    Every rank calls this, so a miss meant N processes streaming and tokenizing the whole corpus
    and racing on the rename, with the losers deleting hours of work. Gating on rank 0 while the
    others wait is not the fix either: the build is hours and `init_process_group` times out at 60
    minutes. Same predicate as `distill.dist.init_distributed`, read from the env so `franken.data`
    need not import `franken.distill`.
    """
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    raise RuntimeError(
        f"corpus artifact missing under torchrun (WORLD_SIZE={os.environ['WORLD_SIZE']}):\n"
        f"  {cached}\n"
        "Build it once, serially, before launching the ranks:\n"
        "  python main.py corpus --config <your config>"
    )


def _manifest(req: _Build, ds) -> dict:
    """The recipe the cache KEY omits -- sources, weights, repos. `_CACHE_VERSION` stays manual (a
    digest would discard an hours-long build on a cosmetic edit), so this is what makes a stale
    artifact diagnosable instead of invisible."""
    from franken.data.corpus.measure import realized_mix

    return {
        "cache_version": _CACHE_VERSION,
        "mix": req.name,
        "split": req.split,
        "budget_tokens": req.budget,
        "max_seq_len": req.max_seq_len,
        "pack": req.pack,
        "tokenizer": str(getattr(req.tokenizer, "name_or_path", "tokenizer")),
        "rows": len(ds),
        "sources": [
            {"name": s.name, "weight": s.weight, "repo": s.repo, "config": s.config}
            for s in req.sources
        ],
        "realized_tokens_by_source": realized_mix(ds, len(req.sources)),
    }


def _build_split(req: _Build):
    """One tokenized split, memoized in-process and on disk: a rebuild re-pays network and parsing,
    hours at 10M texts, per rank. Nothing streams until past both cache checks."""
    cached = req.path
    if cached in _MEMO:
        return _MEMO[cached]
    if os.path.isdir(cached):
        return _MEMO.setdefault(cached, datasets.load_from_disk(cached))

    _refuse_under_ddp(cached)
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
        # Publish by rename: DDP is refused above, but two `main.py corpus` runs can still
        # overlap, and a half-written artifact must never be visible under the real name.
        staged = f"{cached}.tmp{os.getpid()}"
        os.makedirs(_CACHE_DIR, exist_ok=True)
        ds = ds.flatten_indices()
        ds.save_to_disk(staged)
        # Inside `staged`, so provenance publishes atomically with the rename rather than leaving a
        # window where the artifact exists without it. load_from_disk ignores extra files.
        with open(os.path.join(staged, MANIFEST), "w") as f:
            json.dump(_manifest(req, ds), f, indent=2)
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
