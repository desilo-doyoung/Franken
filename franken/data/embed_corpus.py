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

Two presets. ``mixed`` is the original three-source recipe, unchanged so earlier results stay
reproducible. ``multi_domain`` adds science, code and non-Latin scripts; weights are set against
each source's *measured* row count:

    msmarco corpus   8,841,823   s2orc abstracts  39,567,485   code             1,880,853
    hotpotqa corpus  5,233,329   pubmed (MedRAG)        ~23M   all-nli (x3)     1,673,550
    nq corpus        2,681,468   arxiv abstracts     203,037   wikitext-103      ~816,520
    msmarco queries    509,962   hotpotqa queries     97,852   wiki zh/ja/ar/ru/es 1.2-1.9M ea
    T2Ranking / DuRetrieval / CmedqaRetrieval  ~100,000 each

A source that runs dry contributes less and the realized mix drifts from the declared weights
with nothing to show for it: ``mixed`` asks for 20% queries and delivers **5.2%**, which stood
unmeasured through every result in the tracker. ``_mixed`` prints requested vs delivered per
source — treat anything flagged EXHAUSTED as a weight that did not take effect.
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
_CACHE_VERSION = 2

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


# Rows held off the head of each stream so validation stays disjoint from train. Sized for a
# 10x-larger val pool than the current 500 (act_range needs one), but the smallest sources here
# are ~100k rows, so a bigger reserve would cost them real training data.
VAL_RESERVE = 5_000


def _partitioned(repo: str, config: str | None, hf_split: str, extract, shuffle: int = 10_000):
    """A source whose dataset ships no validation split: reserve the stream's first
    ``VAL_RESERVE`` rows for validation and let train skip past them."""

    def build(split: str, n: int) -> list[str]:
        ds = datasets.load_dataset(repo, config, split=hf_split, streaming=True)
        # Shuffle so a source many times larger than n is not drawn as one contiguous slice.
        # The buffer stays small because shard-order shuffling does the global mixing and
        # `_mixed` shuffles the assembled corpus anyway — a big buffer only adds download
        # latency before the first row.
        ds = (
            ds.take(VAL_RESERVE)
            if split != "train"
            else ds.skip(VAL_RESERVE).shuffle(seed=0, buffer_size=shuffle)
        )
        out = []
        for example in ds:
            out += extract(example)
            if len(out) >= n:
                break
        return out[:n]

    return build


def _titled(example) -> list[str]:
    """BeIR / Wikipedia / C-MTEB rows keep title and body in separate columns."""
    title, text = (example.get("title") or "").strip(), example["text"].strip()
    if len(text) < 32:
        return []
    return [f"{title}. {text}" if title else text]


def _query(example) -> list[str]:
    return [INSTRUCT.format(example["text"].strip())]


def _triplet(example) -> list[str]:
    # Three independent sentences per row. No length floor: NLI text is short by nature, and
    # short is the regime CGF normalizes differently.
    return [example[k].strip() for k in ("anchor", "positive", "negative") if example[k].strip()]


def _field(name: str):
    """Sources that keep the whole text in one column."""

    def extract(example) -> list[str]:
        text = example[name].strip()
        return [text] if len(text) >= 32 else []

    return extract


def _paragraphs(example) -> list[str]:
    """Wikipedia rows are whole articles — median 1,040 tokens (zh), 1,764 (ru). Taken whole, an
    article costs a full row to keep its first 128 tokens and the other ~93% is discarded, so the
    slice is nothing but lead paragraphs. Paragraphs are the natural unit at this length (median
    82-100) and yield many texts per article instead of one."""
    return [p.strip() for p in example["text"].split("\n") if len(p.strip()) >= 64]


def _mixed(weighted_sources):
    """Draw each source in proportion, then interleave. The shuffle is seeded so a run is
    reproducible, and it matters: unshuffled, every batch would be single-mode.

    Sources exhaust silently, so the realized mix drifts from the declared weights — `mixed` at
    2.1M asks for 20% queries and delivers 5.2%. Report it rather than let it hide again.
    """

    def build(split: str, n: int) -> list[str]:
        drawn = []
        for name, source, weight in weighted_sources:
            want = max(1, round(n * weight))
            texts = source(split, want)
            short = "  EXHAUSTED" if len(texts) < want else ""
            # Per source as it lands, not batched at the end: a 10M build runs for hours and
            # this is the only progress signal.
            print(f"  {name:24s} want {want:>9,}  got {len(texts):>9,}{short}", flush=True)
            drawn.append((name, texts))

        total = sum(len(texts) for _, texts in drawn)
        realized = "  ".join(f"{name} {len(texts) / max(total, 1):.1%}" for name, texts in drawn)
        print(
            f"corpus mix [{split}]: {total:,} texts for a request of {n:,}\n  {realized}",
            flush=True,
        )

        out = [text for _, texts in drawn for text in texts]
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
            ("msmarco_query", _ms_marco("query"), 0.2),
            ("msmarco_passage", _ms_marco("passage"), 0.4),
            ("wikitext103", _wikitext("wikitext-103-raw-v1"), 0.4),
        ]
    ),
    # Weights are set against each source's *measured* row count, not its intended share -- see
    # the exhaustion table in the module docstring. Every slice is a domain some eval task scores:
    # English retrieval prose (48%) and science (17%) under nfcorpus/scifact + fiqa, multilingual
    # (18%) under xpqa_cmn, code (8%) under code_apps. Code is here on capability grounds only --
    # `scripts/qwen3/domain_range.py` measured it as the *tamest* domain for activation range.
    "multi_domain": _mixed(
        [
            ("msmarco_passage", _partitioned("BeIR/msmarco", "corpus", "corpus", _titled), 0.24),
            ("nq_passage", _partitioned("BeIR/nq", "corpus", "corpus", _titled), 0.09),
            ("hotpotqa_passage", _partitioned("BeIR/hotpotqa", "corpus", "corpus", _titled), 0.07),
            ("wikitext103", _wikitext("wikitext-103-raw-v1"), 0.07),
            ("pubmed", _partitioned("MedRAG/pubmed", "default", "train", _field("contents")), 0.08),
            (
                "s2orc",
                _partitioned(
                    "sentence-transformers/s2orc",
                    "abstract-citation-pair",
                    "train",
                    _field("abstract"),
                ),
                0.07,
            ),
            (
                "code",
                _partitioned(
                    "code-search-net/code_search_net",
                    "all",
                    "train",
                    _field("whole_func_string"),
                ),
                0.08,
            ),
            (
                "arxiv",
                _partitioned("ccdv/arxiv-summarization", "document", "train", _field("abstract")),
                0.02,
            ),
            ("msmarco_query", _partitioned("BeIR/msmarco", "queries", "queries", _query), 0.04),
            ("hotpotqa_query", _partitioned("BeIR/hotpotqa", "queries", "queries", _query), 0.01),
            (
                "nli",
                _partitioned("sentence-transformers/all-nli", "triplet", "train", _triplet),
                0.05,
            ),
            ("t2ranking", _partitioned("C-MTEB/T2Retrieval", "default", "corpus", _titled), 0.01),
            ("duretrieval", _partitioned("C-MTEB/DuRetrieval", "default", "corpus", _titled), 0.01),
            ("cmedqa", _partitioned("C-MTEB/CmedqaRetrieval", "default", "corpus", _titled), 0.01),
        ]
        # Five scripts, not five languages: Arabic moved the teacher's activation profile most
        # (layers 21/23/24 over D=32), then Cyrillic and CJK.
        + [
            (
                f"wiki_{lang}",
                _partitioned("wikimedia/wikipedia", f"20231101.{lang}", "train", _paragraphs),
                0.030,
            )
            for lang in ("zh", "ja", "ar", "ru", "es")
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
    pad_to_multiple_of: int | None = None,
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
    out["collator"] = transformers.DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=pad_to_multiple_of
    )
    return out
