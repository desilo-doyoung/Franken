"""Qwen3-Embedding's training mix: one ``Source`` per dataset.

Every source is scoreable -- it yields (query, positive) records or declares ``Qrels``, enforced by
`franken.scripts.qwen3.corpus` before a build is paid for. An unscoreable slice is a permanent
blind spot.
"""

from __future__ import annotations

from franken.data.corpus import adapters
from franken.data.corpus.source import Qrels, Source, normalized
from franken.data.corpus.spec import WEB_SEARCH

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


# Scoreable mixes only -- what the gates and the eval iterate.
MIXES: dict[str, list[Source]] = {"multi_domain": normalized(_MULTI_DOMAIN)}


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
