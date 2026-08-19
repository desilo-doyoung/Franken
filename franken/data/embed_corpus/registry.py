"""The training mix: one ``Source`` per dataset.

Every source is scoreable -- it yields (query, positive) records or declares ``Qrels``, enforced by
`corpus.py` before a build is paid for. An unscoreable slice is a permanent blind spot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from franken.data.embed_corpus import adapters
from franken.data.embed_corpus.spec import WEB_SEARCH, Record


@dataclass(frozen=True)
class Qrels:
    """Real judgements, for a source whose rows hold no pair. Layouts are not inferrable, hence the
    explicit repo/split/cols. Judged documents are held out of training by `build._judged`."""

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
    # Column the split is hashed from; None means the dataset ships its own splits. `evalset` must
    # hash the identical string or it scores trained rows.
    key: str | None = None
    hf_split: str = "train"  # the single upstream split to draw from; unused when key is None
    split_map: Mapping[str, str] = field(default_factory=dict)  # ours -> upstream, when key is None
    # Baked into the cached corpus text, so changing it needs a `build._CACHE_VERSION` bump.
    instruct: str | None = None
    qrels: Qrels | None = None
    # False where the gold is one arbitrary member of an equally valid set, so nDCG would score a
    # tie-break. Governs REPORTING only -- the source still yields pairs for recall@10.
    scores_ndcg: bool = True


_WIKI_LANGS = ("zh", "ja", "ar", "ru", "es")  # scripts that move the teacher activation range

# Relative weights, normalised below. Coverage is the motivation, not volume.
_MULTI_DOMAIN = [
    Source(
        "msmarco",
        "english_prose",
        "microsoft/ms_marco",
        "v2.1",
        adapters.marco,
        0.209,
        instruct=WEB_SEARCH,
    ),
    # `titled`, not a pair: keyed on `_id` the title straddles splits 60.7% of the time and leaks
    # the query side. Scored on judgements instead.
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
    # One sibling is promoted to gold and the rest become unjudged false negatives. Fidelity only.
    Source(
        "wiki_en",
        "english_prose",
        "wikimedia/wikipedia",
        "20231101.en",
        adapters.paragraphs,
        0.06,
        key="id",
        scores_ndcg=False,
    ),
    # Every instructed source carries WEB_SEARCH, measured rather than assumed: task-matching
    # strings came in at +0.0017 to -0.0067 against it. The model keys on the canonical string it
    # was trained with, not on semantic fit.
    Source(
        "gooaq",
        "informal",
        "sentence-transformers/gooaq",
        None,
        adapters.pair("question", "answer"),
        0.1,
        key="question",
        instruct=WEB_SEARCH,
    ),
    Source(
        "eli5",
        "informal",
        "sentence-transformers/eli5",
        None,
        adapters.pair("question", "answer"),
        0.03,
        key="question",
        instruct=WEB_SEARCH,  # a forum-question string measured -0.0067
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
    # Chosen over a PubMed corpus because nfcorpus IS PubMed; overlap here is 3 rows in 400,000.
    Source(
        "arxiv",
        "science",
        "gfissore/arxiv-abstracts-2021",
        None,
        adapters.pair("title", "abstract"),
        0.06,
        key="id",
        instruct=WEB_SEARCH,
    ),
    # Abstract -> abstract is SYMMETRIC, so there is no query side to instruct; the gate agrees
    # (-0.0088 for the web string, the only negative among the corpus sources).
    Source(
        "s2orc",
        "science",
        "sentence-transformers/s2orc",
        "abstract-citation-pair",
        adapters.pair("abstract", "citation"),
        0.08,
        key="abstract",
    ),
    # Title -> *a* related title: many are equally related, so the promoted one is arbitrary.
    Source(
        "specter",
        "science",
        "sentence-transformers/specter",
        "triplet",
        adapters.triplet,
        0.03,
        key="anchor",
        scores_ndcg=False,
    ),
    # APPS's task shape without being APPS, so `code_apps` stays a clean canary.
    Source(
        "code",
        "code",
        "code-search-net/code_search_net",
        "python",
        adapters.pair("func_documentation_string", "whole_func_string"),
        0.04,
        instruct=WEB_SEARCH,
    ),
    Source(
        "codefeedback",
        "code",
        "m-a-p/CodeFeedback-Filtered-Instruction",
        None,
        adapters.pair("query", "answer"),
        0.015,
        key="query",
        instruct=WEB_SEARCH,
    ),
    Source(
        "glaive_code",
        "code",
        "glaiveai/glaive-code-assistant",
        None,
        adapters.pair("question", "answer"),
        0.013,
        key="question",
        instruct=WEB_SEARCH,
    ),
] + [
    # `datasets` 5.0 removed script loaders, killing miracl/mr-tydi/MLDR.
    Source(
        f"wiki_{lang}",
        "multilingual",
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        adapters.paragraphs,
        0.026,
        key="id",
        scores_ndcg=False,  # same arbitrary-sibling gold as wiki_en
    )
    for lang in _WIKI_LANGS
]


def _normalized(sources: list[Source]) -> list[Source]:
    # Relative, so dropping a source is a one-line delete.
    total = sum(s.weight for s in sources)
    return [replace(s, weight=s.weight / total) for s in sources]


# Scoreable mixes only -- what the gates and the eval iterate.
MIXES: dict[str, list[Source]] = {"multi_domain": _normalized(_MULTI_DOMAIN)}


# `smoke` is a pipeline proof, never a result, so it sits outside MIXES.
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
    # Kept only so the pre-1024 configs naming it still run. Not scoreable, not in MIXES.
    "mixed": [
        Source(
            "msmarco_query",
            "query",
            "microsoft/ms_marco",
            "v1.1",
            adapters.marco_side("query"),
            0.2,
            instruct=WEB_SEARCH,
        ),
        Source(
            "msmarco_passage",
            "english_prose",
            "microsoft/ms_marco",
            "v1.1",
            adapters.marco_side("passage"),
            0.4,
        ),
        Source(
            "wikitext103",
            "english_prose",
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            adapters.wikitext,
            0.4,
        ),
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
