from types import SimpleNamespace

import pytest

from franken.data.embed_corpus.build import cache_path, train_cache_path
from franken.data.embed_corpus.registry import MIXES, PRESETS

# name -> (repo, config, key, raw weight, instructed, has qrels, scores_ndcg)
_MULTI_DOMAIN = {
    "msmarco": ("microsoft/ms_marco", "v2.1", None, 0.209, True, False, True),
    "nq_passage": ("BeIR/nq", "corpus", "_id", 0.06, False, True, True),
    "hotpotqa_passage": ("BeIR/hotpotqa", "corpus", "_id", 0.05, False, True, True),
    "wiki_en": ("wikimedia/wikipedia", "20231101.en", "id", 0.06, False, False, False),
    "gooaq": ("sentence-transformers/gooaq", None, "question", 0.1, True, False, True),
    "eli5": ("sentence-transformers/eli5", None, "question", 0.03, True, False, True),
    "stackexchange": (
        "sentence-transformers/stackexchange-duplicates",
        "post-post-pair",
        "post1",
        0.03,
        False,
        False,
        True,
    ),
    "arxiv": ("gfissore/arxiv-abstracts-2021", None, "id", 0.06, True, False, True),
    "s2orc": (
        "sentence-transformers/s2orc",
        "abstract-citation-pair",
        "abstract",
        0.08,
        False,
        False,
        True,
    ),
    "specter": ("sentence-transformers/specter", "triplet", "anchor", 0.03, False, False, False),
    "code": ("code-search-net/code_search_net", "python", None, 0.04, True, False, True),
    "codefeedback": (
        "m-a-p/CodeFeedback-Filtered-Instruction",
        None,
        "query",
        0.015,
        True,
        False,
        True,
    ),
    "glaive_code": ("glaiveai/glaive-code-assistant", None, "question", 0.013, True, False, True),
    **{
        f"wiki_{lang}": (
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            "id",
            0.026,
            False,
            False,
            False,
        )
        for lang in ("zh", "ja", "ar", "ru", "es")
    },
}

_RAW_TOTAL = sum(v[3] for v in _MULTI_DOMAIN.values())


def _tok(name: str):
    # cache_path reads nothing else off a tokenizer.
    return SimpleNamespace(name_or_path=name)


def _declared(src):
    return (
        src.repo,
        src.config,
        src.key,
        pytest.approx(src.weight * _RAW_TOTAL),
        src.instruct is not None,
        bool(src.qrels),
        src.scores_ndcg,
    )


@pytest.mark.parametrize(
    "name,split,label,cap,tok,expected",
    [
        (
            "multi_domain",
            "train",
            "1930000tok",
            256,
            "Qwen/Qwen3-Embedding-0.6B",
            "outputs/corpus_cache/v9-multi_domain-train-1930000tok-256-Qwen_Qwen3-Embedding-0.6B",
        ),
        (
            "multi_domain",
            "validation",
            "500",
            256,
            "Qwen/Qwen3-Embedding-0.6B",
            "outputs/corpus_cache/v9-multi_domain-validation-500-256-Qwen_Qwen3-Embedding-0.6B",
        ),
        (
            "mixed",
            "train",
            "140000tok",
            128,
            "unsloth/Llama-3.2-1B",
            "outputs/corpus_cache/v9-mixed-train-140000tok-128-unsloth_Llama-3.2-1B",
        ),
        (
            "mixed",
            "validation",
            "500",
            128,
            "unsloth/Llama-3.2-1B",
            "outputs/corpus_cache/v9-mixed-validation-500-128-unsloth_Llama-3.2-1B",
        ),
    ],
)
def test_cache_path_still_names_the_builds_already_on_disk(name, split, label, cap, tok, expected):
    # These four directories exist. A moved _CACHE_VERSION, _CACHE_DIR, size label or tokenizer
    # sanitizer renames the key and silently re-pays a multi-hour build.
    assert cache_path(name, split, label, cap, _tok(tok)) == expected


def test_train_cache_path_renders_the_budget_as_an_integer():
    # An f-string on the float would emit "1930000.0tok" and orphan the built cache.
    assert train_cache_path("multi_domain", 1.93e6, 256, _tok("t")).endswith("-1930000tok-256-t")


@pytest.mark.parametrize("name", sorted(_MULTI_DOMAIN), ids=lambda n: n)
def test_the_multi_domain_declaration_is_unchanged(name):
    # A transposed repo/config or a nudged weight rebuilds from the wrong dataset under an
    # unchanged cache key -- the key covers the request, not the recipe.
    src = next(s for s in MIXES["multi_domain"] if s.name == name)
    assert _declared(src) == _MULTI_DOMAIN[name]


def test_the_source_order_is_stable():
    # `source` is a uint8 INDEX into this list, baked into every built artifact and zipped against
    # realized_mix counts.
    assert [s.name for s in MIXES["multi_domain"]] == list(_MULTI_DOMAIN)


@pytest.mark.parametrize("name", sorted(MIXES))
def test_weights_normalize_to_one(name):
    assert sum(s.weight for s in MIXES[name]) == pytest.approx(1.0)


def test_every_mix_is_buildable():
    # `mix()` reads MIXES and `_build_split` reads PRESETS: a mix missing from PRESETS passes the
    # gate and then dies an hour into the build.
    assert set(MIXES) <= set(PRESETS)


def test_preset_names_are_unique_across_the_shared_cache_namespace():
    # cache_path has one flat directory, so two registries reusing a name serve each other's text.
    from franken.data.lm_corpus import MIXES as LM_MIXES

    assert not set(PRESETS) & set(LM_MIXES)
