"""Llama's training mix: plain documents for logit distillation.

Unscoreable by design -- `lm` supervises against the teacher's distribution, not a retrieval gold.
"""

from __future__ import annotations

from franken.data.corpus import adapters
from franken.data.corpus.source import Source, normalized

# Llama 3.2 officially supports en/de/fr/it/pt/hi/es/th -- NOT qwen3's `_WIKI_LANGS`, which was
# picked for a multilingual embedding model. `hi` and `th` are what keep a non-Latin script in the
# sample `act_range` reads the FHE polynomial domain from.
_LLAMA_LANGS = ("de", "fr", "es", "it", "pt", "hi", "th")

# TOKEN shares, normalised below: `_plan_draw` converts each to a document count with a live
# tok/doc measurement, so a source's length never silently moves its share of the budget.
_LLAMA_WEB = [
    Source(
        "fineweb_edu",
        "web_prose",
        "HuggingFaceFW/fineweb-edu",
        "sample-100BT",
        adapters.whole("text"),
        0.37,
        key="id",
        scores_ndcg=False,
    ),
    # FineWiki REPLACES `wikimedia/wikipedia` rather than joining it: same articles, cleaner text.
    Source(
        "finewiki_en",
        "english_prose",
        "HuggingFaceFW/finewiki",
        "en",
        adapters.whole("text"),
        0.1,
        key="id",
        scores_ndcg=False,
    ),
    # A pointwise loss makes the training distribution the region of fidelity, so a corpus with
    # no math leaves the student unconstrained there. Whether it moves the max: ask act_range.
    # open-web-math over finemath: +18% teacher entropy (2.178 vs 1.846) at the same LaTeX density.
    Source(
        "open_web_math",
        "math",
        "open-web-math/open-web-math",
        None,
        adapters.whole("text"),
        0.12,
        key="url",  # no id column; url is the stable key
        scores_ndcg=False,
    ),
    Source(
        "arxiv",
        "science",
        "gfissore/arxiv-abstracts-2021",
        None,
        adapters.whole("abstract"),
        0.06,
        key="id",
        scores_ndcg=False,
    ),
    # codeparrot over CodeSearchNet on volume, not information (entropy is a wash): 5.17M whole
    # files vs 457k functions, and CodeSearchNet cannot fill 12.9% of a 2B-token epoch.
    Source(
        "codeparrot",
        "code",
        "codeparrot/codeparrot-clean",
        None,
        adapters.whole("content"),
        0.13,
        key="hash",
        scores_ndcg=False,
    ),
] + [
    Source(
        f"finewiki_{lang}",
        "multilingual",
        "HuggingFaceFW/finewiki",
        lang,
        adapters.whole("text"),
        0.0314,
        key="id",
        scores_ndcg=False,
    )
    for lang in _LLAMA_LANGS
]

# No MIXES/PRESETS split: that exists for qwen3 because `smoke` and `mixed` sit outside the
# scoreable set, and nothing here is scoreable in the first place.
PRESETS: dict[str, list[Source]] = {"llama_web": normalized(_LLAMA_WEB)}
