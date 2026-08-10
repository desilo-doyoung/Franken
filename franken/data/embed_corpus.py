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

**Every ``multi_domain`` source is scoreable** — one row carries both sides of a (query, document)
pair, or the dataset ships qrels — so ``scripts/qwen3/corpus_eval.py`` can build a retrieval task
from its held-out rows. A slice that cannot be evaluated is a permanent blind spot, and that is
the failure ``domain_drift.py`` diagnosed: the 8% CodeSearchNet slice fully protected CSN-style
code (drift 0.0104, matching nfcorpus's 0.0087) while APPS drifted 17x, so ``code_apps`` −53.9%
nDCG was measuring corpus coverage rather than the depth cut. Sources that could not form a pair
were replaced, not kept unscored: BeIR/msmarco (empty title column) → ``microsoft/ms_marco`` v2.1,
wikitext-103 (no title) → English Wikipedia, arXiv (26k-char articles) → folded into s2orc.

Two presets. ``mixed`` is the original three-source recipe, unchanged so earlier results stay
reproducible. ``multi_domain`` spans English/informal prose, science, code, nine Wikipedia
languages and Chinese retrieval; weights are set against each source's *measured* row count:

    ms_marco v2.1  808,731 x11   s2orc pairs      41,769,185   code (CSN)       1,880,853
    hotpotqa corpus  5,233,329   pubmed (MedRAG)        ~23M   all-nli (x3)     1,673,550
    nq corpus        2,681,468   gooaq (x2)          6,024,992 specter (x3)     2,052,294
    eli5 (x2)          650,950   stackexchange (x2)    501,038 quora (x3)         305,286
    codefeedback       156,526   wiki en/zh/ja/...   1.3-6.4M articles each
    T2Ranking / DuRetrieval / CmedqaRetrieval  ~100,000 each  (the ceiling on Chinese retrieval;
    mmarco and nli-zh-all are dead script-loaders, so multilingual growth comes from Wikipedia)

A source that runs dry contributes less and the realized mix drifts from the declared weights
with nothing to show for it: ``mixed`` asks for 20% queries and delivers **5.2%**, which stood
unmeasured through every result in the tracker. ``_mixed`` prints requested vs delivered per
source — treat anything flagged EXHAUSTED as a weight that did not take effect.
"""

import hashlib
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
_CACHE_VERSION = 3

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


# Percent of the hash space given to each held-out split; train takes the rest. Small because
# eval only needs a ~5k-document pool per task, and `corpus_size` is a *request* that most
# sources over-supply 2-50x, so the holdout costs training almost nothing.
VAL_PCT, TEST_PCT = 2, 4


def _split_of(key: str) -> str:
    """Which split a row belongs to, as a pure function of a stable key.

    Membership cannot depend on read order, so the three splits are disjoint however a build
    scans the stream, and identical text always lands in the same split (duplicates cannot
    straddle it). ⚠️ `hashlib`, never `hash()` — Python salts `str.__hash__` per process, so
    `hash()` would redraw the split on every run and every machine.
    """
    p = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % 100
    if p < VAL_PCT:
        return "validation"
    if p < TEST_PCT:
        return "test"
    return "train"


# Shard-order shuffling does the global mixing, so the buffer stays small — a big one only adds
# download latency before the first row, and `_mixed` shuffles the assembled corpus anyway.
_SHUFFLE = 10_000


def _stream(repo: str, config: str | None, hf_split: str):
    ds = datasets.load_dataset(repo, config, split=hf_split, streaming=True)
    # Shuffle every split, not just train. Several streams are grouped (CodeSearchNet by
    # language with python first, Wikipedia by article id), so a prefix `take` is
    # single-language — which is what the old reserve did, and why its code pool was 100% python.
    return ds.shuffle(seed=0, buffer_size=_SHUFFLE)


def _native(repo: str, config: str | None, extract, split_map: dict[str, str] | None = None):
    """A source that already ships train/validation/test upstream — use it rather than ours."""
    split_map = split_map or {}

    def build(split: str, n: int) -> list[str]:
        ds = _stream(repo, config, split_map.get(split, split))
        out: list[str] = []
        for example in ds:
            out += extract(example)
            if len(out) >= n:
                break
        return out[:n]

    build.meta = {"repo": repo, "config": config, "native": True, "split_map": split_map}
    return build


def _hashed(repo: str, config: str | None, hf_split: str, extract, key: str):
    """A source with one upstream split: derive all three by hashing the ``key`` column.

    A row's whole yield goes to one split, which keeps `_paragraphs` from scattering an article and
    `_triplet` from separating an anchor from its positive. ``key`` is explicit rather than derived
    from the extractor's output because `scripts/qwen3/corpus_eval.py` re-derives membership from
    `build.meta` — the two must hash the same string or the eval silently scores trained rows.
    """

    def build(split: str, n: int) -> list[str]:
        ds = _stream(repo, config, hf_split)
        out: list[str] = []
        for example in ds:
            if _split_of(str(example[key])) != split:
                continue
            out += extract(example)
            if len(out) >= n:
                break
        return out[:n]

    build.meta = {
        "repo": repo,
        "config": config,
        "native": False,
        "hf_split": hf_split,
        "key": key,
    }
    return build


def _titled(example) -> list[str]:
    """BeIR / Wikipedia / C-MTEB rows keep title and body in separate columns."""
    title, text = (example.get("title") or "").strip(), example["text"].strip()
    if len(text) < 32:
        return []
    return [f"{title}. {text}" if title else text]


def _query(example) -> list[str]:
    return [INSTRUCT.format(example["text"].strip())]


def _pair(a: str, b: str):
    """Two text columns per row. Both sides are kept, so a retrieval task built from the held-out
    rows has its query *and* its gold document in the training distribution."""

    def extract(example) -> list[str]:
        return [example[k].strip() for k in (a, b) if example[k] and example[k].strip()]

    return extract


def _marco(example) -> list[str]:
    """One MS MARCO row is a whole retrieval task: the query plus 10 passages, one flagged
    relevant. Replaces the old BeIR corpus+queries pair of slices — same content, but the query
    and its positive live in the same row, so no split can separate them."""
    query = example["query"].strip()
    passages = [p.strip() for p in example["passages"]["passage_text"] if p.strip()]
    return ([INSTRUCT.format(query)] if query else []) + passages


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
        for name, _domain, source, weight in weighted_sources:
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


_WIKI_MAIN = ("zh", "ja", "ar", "ru", "es")  # scripts that move the teacher's activation range
_WIKI_EXTRA = ("de", "fr", "ko", "vi")  # language coverage only

# Every source is *scoreable*: one row carries both sides of a (query, document) pair, or the
# dataset ships qrels. A slice that cannot be evaluated is a permanent blind spot, which is the
# failure `domain_drift.py` found -- 8% CodeSearchNet fully protected CSN-style code (drift 0.0104,
# = nfcorpus's 0.0087) while APPS drifted 17x, so `code_apps` -53.9% was measuring coverage.
MIXES: dict[str, list[tuple[str, str, Any, float]]] = {
    "multi_domain": [
        # English web / encyclopedia prose
        ("msmarco", "english_prose", _native("microsoft/ms_marco", "v2.1", _marco), 0.232),
        (
            "nq_passage",
            "english_prose",
            _hashed("BeIR/nq", "corpus", "corpus", _titled, "_id"),
            0.06,
        ),
        (
            "hotpotqa_passage",
            "english_prose",
            _hashed("BeIR/hotpotqa", "corpus", "corpus", _titled, "_id"),
            0.05,
        ),
        (
            "wiki_en",
            "english_prose",
            _hashed("wikimedia/wikipedia", "20231101.en", "train", _paragraphs, "id"),
            0.06,
        ),
        # Informal / long-form web prose -- the `fiqa` domain, -13.5% at depth 19
        (
            "gooaq",
            "informal",
            _hashed(
                "sentence-transformers/gooaq",
                None,
                "train",
                _pair("question", "answer"),
                "question",
            ),
            0.06,
        ),
        (
            "eli5",
            "informal",
            _hashed(
                "sentence-transformers/eli5", None, "train", _pair("question", "answer"), "question"
            ),
            0.02,
        ),
        (
            "stackexchange",
            "informal",
            _hashed(
                "sentence-transformers/stackexchange-duplicates",
                "post-post-pair",
                "train",
                _pair("post1", "post2"),
                "post1",
            ),
            0.02,
        ),
        # Science / medical
        (
            "pubmed",
            "science",
            _hashed("MedRAG/pubmed", "default", "train", _field("contents"), "PMID"),
            0.06,
        ),
        (
            "s2orc",
            "science",
            _hashed(
                "sentence-transformers/s2orc",
                "abstract-citation-pair",
                "train",
                _pair("abstract", "citation"),
                "abstract",
            ),
            0.08,
        ),
        (
            "specter",
            "science",
            _hashed("sentence-transformers/specter", "triplet", "train", _triplet, "anchor"),
            0.03,
        ),
        # Code -- two genres, because CSN alone does not cover APPS-style code
        (
            "code",
            "code",
            _native("code-search-net/code_search_net", "all", _field("whole_func_string")),
            0.08,
        ),
        (
            "codefeedback",
            "code",
            _hashed("CoIR-Retrieval/codefeedback-st", "corpus", "corpus", _titled, "_id"),
            0.01,
        ),
        # Chinese retrieval -- capped near 100k rows per dataset, so multilingual growth comes
        # from Wikipedia languages instead (mmarco and nli-zh-all are dead script-loaders).
        (
            "t2ranking",
            "chinese",
            _hashed("C-MTEB/T2Retrieval", "default", "corpus", _titled, "id"),
            0.006,
        ),
        (
            "duretrieval",
            "chinese",
            _hashed("C-MTEB/DuRetrieval", "default", "corpus", _titled, "id"),
            0.006,
        ),
        (
            "cmedqa",
            "chinese",
            _hashed("C-MTEB/CmedqaRetrieval", "default", "corpus", _titled, "id"),
            0.006,
        ),
        # Short text -- the regime CGF softmax normalizes differently
        (
            "nli",
            "short",
            _native("sentence-transformers/all-nli", "triplet", _triplet, {"validation": "dev"}),
            0.04,
        ),
        (
            "quora",
            "short",
            _hashed(
                "sentence-transformers/quora-duplicates", "triplet", "train", _triplet, "anchor"
            ),
            0.01,
        ),
    ]
    + [
        (
            f"wiki_{lang}",
            "multilingual",
            _hashed("wikimedia/wikipedia", f"20231101.{lang}", "train", _paragraphs, "id"),
            w,
        )
        for langs, w in ((_WIKI_MAIN, 0.022), (_WIKI_EXTRA, 0.015))
        for lang in langs
    ],
}

DOMAINS = {name: domain for mix in MIXES.values() for name, domain, _s, _w in mix}

CORPORA = {
    # Pipeline proof only, never a result.
    "smoke": _wikitext("wikitext-2-raw-v1"),
    # Proportions are a judgment call, set from the length profile rather than derived:
    # enough query coverage that the mode is not unseen, without letting ~10-token texts
    # dominate a corpus whose max_seq_len is 128.
    "mixed": _mixed(
        [
            ("msmarco_query", "query", _ms_marco("query"), 0.2),
            ("msmarco_passage", "english_prose", _ms_marco("passage"), 0.4),
            ("wikitext103", "english_prose", _wikitext("wikitext-103-raw-v1"), 0.4),
        ]
    ),
    "multi_domain": _mixed(MIXES["multi_domain"]),
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
