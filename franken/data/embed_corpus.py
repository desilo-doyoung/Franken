"""Corpora for the label-free embedding self-distillation task.

The teacher supplies the targets, so no labels are needed — the corpus only has to
resemble the text the student will embed. ``train.corpus`` names a *preset* (a recipe)
rather than a dataset id, so a mix stays one config value instead of a list of ids plus
weights.

Texts are kept as **natural units** (a paragraph, a query) rather than chunked into
fixed-length blocks: an embedding model is deployed on whole passages, so blocks that
start and end mid-sentence would be off-distribution. Each source owns its own cleaning —
what counts as junk is a property of that source, not a general rule.

Queries carry the instruction prefix the model card specifies and documents carry none.
The model has one forward pass for both — no query/document encoders — so this asymmetry
is purely about covering the input distribution: distillation only repairs the student
where data exists, and queries are short (~5-15 tokens), which is the regime where CGF
softmax behaves differently (it normalizes by the visible-token count).

Sources yield ``list[str]``; ``load_embed_corpus`` tokenizes and wraps them.
"""

import os
import random
import re
import shutil
from functools import lru_cache
from typing import Any

import datasets
import transformers

# Tokenized splits are cached here across runs. Bump _CACHE_VERSION when a source or a mix weight
# changes — the key covers the request, not the recipe that answered it.
_CACHE_DIR = "outputs/corpus_cache"
_CACHE_VERSION = 1

# Task-specific by design (the model card recommends tailoring it, worth 1-5%). MS MARCO is
# web search, so this is its matching instruction.
INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}"
)


def _wikitext(config_name: str):
    """Wikipedia paragraphs (documents, unprefixed). Drops wikitext's blank lines and
    " = = Heading = = " rows, which ship as records of their own."""
    min_chars = 32

    def build(split: str, n: int) -> list[str]:
        ds = datasets.load_dataset("Salesforce/wikitext", config_name, split=split, streaming=True)
        out = []
        for example in ds:
            text = example["text"].strip()
            if len(text) >= min_chars and not text.startswith("="):
                out.append(text)
                if len(out) >= n:
                    break
        return out

    return build


def _ms_marco(kind: str):
    """Real web-search queries, or the passages they retrieve. ``kind`` is
    ``"query"`` (instruction-prefixed, short) or ``"passage"`` (raw documents)."""

    def build(split: str, n: int) -> list[str]:
        ds = datasets.load_dataset("microsoft/ms_marco", "v1.1", split=split, streaming=True)
        out = []
        for example in ds:
            if kind == "query":
                out.append(INSTRUCT.format(example["query"].strip()))
            else:
                out += [p.strip() for p in example["passages"]["passage_text"] if p.strip()]
            if len(out) >= n:
                break
        return out[:n]

    return build


def _mixed(weighted_sources):
    """Draw each source in proportion, then interleave. The shuffle is seeded so a run is
    reproducible, and it matters: unshuffled, every batch would be single-mode."""

    def build(split: str, n: int) -> list[str]:
        out = []
        for source, weight in weighted_sources:
            out += source(split, max(1, round(n * weight)))
        random.Random(0).shuffle(out)
        return out[:n]

    return build


CORPORA = {
    # Pipeline proof only, never a result.
    "smoke": _wikitext("wikitext-2-raw-v1"),
    # Proportions are a judgment call, set from the length profile rather than derived:
    # enough query coverage that the mode is not unseen, without letting ~10-token texts
    # dominate a corpus whose max_seq_len is 128.
    "mixed": _mixed(
        [
            (_ms_marco("query"), 0.2),
            (_ms_marco("passage"), 0.4),
            (_wikitext("wikitext-103-raw-v1"), 0.4),
        ]
    ),
}


def _cache_path(name: str, split: str, n: int, max_seq_len: int, tokenizer: Any) -> str:
    tok_id = re.sub(r"[^\w.-]", "_", str(getattr(tokenizer, "name_or_path", "tokenizer")))
    return os.path.join(_CACHE_DIR, f"v{_CACHE_VERSION}-{name}-{split}-{n}-{max_seq_len}-{tok_id}")


def _save_atomic(ds, path: str) -> None:
    # Under torchrun every rank builds this concurrently, so publish by rename rather than let N
    # ranks interleave writes into one directory; losers (rename onto a non-empty dir) discard.
    tmp = f"{path}.tmp{os.getpid()}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(tmp)
    try:
        os.rename(tmp, path)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)


@lru_cache(maxsize=4)
def _build_split(name: str, split: str, n: int, max_seq_len: int, tokenizer: Any):
    """One tokenized split, memoized in-process and on disk.

    Sources are HF *streaming* datasets, so a rebuild re-pays network and parsing: 21s at
    ``corpus_size`` 24k, minutes at 216k, and every rank pays it independently. Reproducibility-
    neutral: building consumes no global RNG, and a cache hit returns identical content.
    """
    if name not in CORPORA:
        raise KeyError(f"Unknown corpus {name!r}; available: {sorted(CORPORA)}")

    cached = _cache_path(name, split, n, max_seq_len, tokenizer)
    if os.path.isdir(cached):
        return datasets.load_from_disk(cached)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    texts = CORPORA[name](split, n)
    ds = datasets.Dataset.from_dict({"text": texts}).map(tok, batched=True, remove_columns=["text"])
    _save_atomic(ds, cached)
    return ds


def load_embed_corpus(
    tokenizer: Any,
    name: str,
    size: int,
    max_seq_len: int = 128,
    val_size: int = 500,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Load and tokenize an embedding corpus preset.

    Returns the requested tokenized splits and a dynamic-padding collator — the same shape
    ``franken.data.mrpc.load_mrpc`` returns. Pass ``splits`` to build only what the caller
    needs: evaluation scores 500 validation rows and has no use for the training corpus.
    """
    out = {
        split: _build_split(
            name, split, size if split == "train" else val_size, max_seq_len, tokenizer
        )
        for split in splits
    }
    out["collator"] = transformers.DataCollatorWithPadding(tokenizer)
    return out
