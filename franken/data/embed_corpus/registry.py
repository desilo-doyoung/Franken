"""The mix: one ``Source`` per dataset, and nothing else declares a dataset.

**Every source is scoreable** — it yields (query, positive) records or declares ``Qrels``. No escape
hatch; `scripts/qwen3/corpus.py` enforces it before a build is paid for. An unscoreable slice
is a permanent blind spot, which is how `code_apps` -53.9% turned out to be measuring coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from franken.data.embed_corpus import adapters
from franken.data.embed_corpus.spec import Record


@dataclass(frozen=True)
class Qrels:
    """Real judgements, for a source whose rows hold no pair. Layouts are not inferrable: BeIR ships
    query-id/corpus-id in `test`, C-MTEB ships qid/pid in `dev` with queries under config `default`.

    Weaker guarantee than a pair task — judgements pick the golds, so each lands in train with
    ~96% probability. The document side is seen; only distractors are held out.
    """

    repo: str
    split: str = "test"
    queries: tuple[str | None, str] = ("queries", "queries")  # (config, split)
    cols: tuple[str, str, str] = ("query-id", "corpus-id", "score")


@dataclass(frozen=True)
class Source:
    name: str
    domain: str
    repo: str
    config: str | None
    adapt: Callable[[dict], Record | None]
    weight: float
    # Column the split is hashed from; None means the dataset ships its own splits. Explicit rather
    # than derived, because `evalset` must hash the identical string or it scores trained rows.
    key: str | None = None
    hf_split: str = "train"  # the single upstream split to draw from; unused when key is None
    split_map: Mapping[str, str] = field(default_factory=dict)  # ours -> upstream, when key is None
    prefix_query: bool = False
    qrels: Qrels | None = None


_WIKI_LANGS = ("zh", "ja", "ar", "ru", "es")  # scripts that move the teacher activation range

# Relative weights, normalised below. Coverage is the motivation, not volume: training text was web
# + encyclopedia prose while nDCG is measured on biomedical and scientific benchmarks.
_MULTI_DOMAIN = [
    Source(
        "msmarco",
        "english_prose",
        "microsoft/ms_marco",
        "v2.1",
        adapters.marco,
        0.209,
        prefix_query=True,
    ),
    # `titled`, not a pair: nq packs 35.3 passages per article, so keyed on `_id` the title
    # straddles splits 60.7% of the time and leaks the query side. Scored on judgements instead.
    Source(
        "nq_passage",
        "english_prose",
        "BeIR/nq",
        "corpus",
        adapters.titled,
        0.06,
        key="_id",
        hf_split="corpus",
        qrels=Qrels("BeIR/nq-qrels"),
    ),
    Source(
        "hotpotqa_passage",
        "english_prose",
        "BeIR/hotpotqa",
        "corpus",
        adapters.titled,
        0.05,
        key="_id",
        hf_split="corpus",
        qrels=Qrels("BeIR/hotpotqa-qrels"),
    ),
    Source(
        "wiki_en",
        "english_prose",
        "wikimedia/wikipedia",
        "20231101.en",
        adapters.paragraphs,
        0.06,
        key="id",
    ),
    # Informal prose: the `fiqa` domain, -13.5% at depth 19.
    Source(
        "gooaq",
        "informal",
        "sentence-transformers/gooaq",
        None,
        adapters.pair("question", "answer"),
        0.1,
        key="question",
    ),
    Source(
        "eli5",
        "informal",
        "sentence-transformers/eli5",
        None,
        adapters.pair("question", "answer"),
        0.03,
        key="question",
    ),
    Source(
        "stackexchange",
        "informal",
        "sentence-transformers/stackexchange-duplicates",
        "post-post-pair",
        adapters.pair("post1", "post2"),
        0.03,
        key="post1",
    ),
    # Partial eval contamination, bounded by the sampling rate (~4% of pubmed, ~2% of s2orc):
    # pubmed overlaps nfcorpus's domain, s2orc is scifact's source. `fiqa` is the clean check.
    Source(
        "pubmed",
        "science",
        "MedRAG/pubmed",
        "default",
        adapters.pair("title", "content"),
        0.06,
        key="PMID",
    ),
    Source(
        "s2orc",
        "science",
        "sentence-transformers/s2orc",
        "abstract-citation-pair",
        adapters.pair("abstract", "citation"),
        0.08,
        key="abstract",
    ),
    Source(
        "specter",
        "science",
        "sentence-transformers/specter",
        "triplet",
        adapters.triplet,
        0.03,
        key="anchor",
    ),
    # CSN is library functions; the instruction sets are problem -> solution, i.e. APPS's task shape
    # without being APPS, so they close that genre gap while `code_apps` stays a clean canary.
    Source(
        "code",
        "code",
        "code-search-net/code_search_net",
        "python",
        adapters.pair("func_documentation_string", "whole_func_string"),
        0.04,
    ),
    Source(
        "codefeedback",
        "code",
        "m-a-p/CodeFeedback-Filtered-Instruction",
        None,
        adapters.pair("query", "answer"),
        0.015,
        key="query",
    ),
    Source(
        "glaive_code",
        "code",
        "glaiveai/glaive-code-assistant",
        None,
        adapters.pair("question", "answer"),
        0.013,
        key="question",
    ),
] + [
    # `datasets` 5.0 removed script loaders, killing miracl/mr-tydi/MLDR; wikipedia is
    # parquet-native with 323 language configs, so multilingual coverage comes from here.
    Source(
        f"wiki_{lang}",
        "multilingual",
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        adapters.paragraphs,
        0.026,
        key="id",
    )
    for lang in _WIKI_LANGS
]


def _normalized(sources: list[Source]) -> list[Source]:
    # Declared relative so dropping a source is a one-line delete, not a rebalance of 18 numbers.
    total = sum(s.weight for s in sources)
    return [replace(s, weight=s.weight / total) for s in sources]


# Scoreable mixes only — this is what the gates and the eval iterate.
MIXES: dict[str, list[Source]] = {"multi_domain": _normalized(_MULTI_DOMAIN)}

DOMAINS = {s.name: s.domain for mix in MIXES.values() for s in mix}

# Buildable presets. `smoke` is a pipeline proof, never a result, so it sits outside MIXES rather
# than weakening the scoreability rule for everything else.
PRESETS: dict[str, list[Source]] = {
    "smoke": [
        Source(
            "wikitext2",
            "english_prose",
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            adapters.wikitext,
            1.0,
        )
    ],
    **MIXES,
}


def mix(name: str) -> list[Source]:
    if name not in MIXES:
        raise KeyError(
            f"No source registry for corpus {name!r}; registered: {sorted(MIXES)}. "
            "('smoke' is a single-source pipeline proof and is not scoreable by design.)"
        )
    return MIXES[name]
