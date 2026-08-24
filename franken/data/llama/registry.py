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

# Relative weights, normalised below.
_LLAMA_WEB = [
    Source(
        "fineweb_edu",
        "web_prose",
        "HuggingFaceFW/fineweb-edu",
        "sample-100BT",
        adapters.whole("text"),
        0.50,
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
        0.12,
        key="id",
        scores_ndcg=False,
    ),
    Source(
        "code",
        "code",
        "code-search-net/code_search_net",
        "python",
        adapters.whole("whole_func_string"),
        0.12,
        scores_ndcg=False,
    ),
] + [
    Source(
        f"finewiki_{lang}",
        "multilingual",
        "HuggingFaceFW/finewiki",
        lang,
        adapters.whole("text"),
        0.03,
        key="id",
        scores_ndcg=False,
    )
    for lang in _LLAMA_LANGS
]

# No MIXES/PRESETS split: that exists for qwen3 because `smoke` and `mixed` sit outside the
# scoreable set, and nothing here is scoreable in the first place.
PRESETS: dict[str, list[Source]] = {"llama_web": normalized(_LLAMA_WEB)}
