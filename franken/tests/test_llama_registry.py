import pytest

from franken.data import corpus_sources
from franken.data.llama.registry import PRESETS
from franken.data.qwen3.registry import PRESETS as QWEN3_PRESETS

# name -> (repo, config, column, TOKEN share)
_LLAMA_WEB = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-100BT", "text", 0.37),
    "finewiki_en": ("HuggingFaceFW/finewiki", "en", "text", 0.10),
    "open_web_math": ("open-web-math/open-web-math", None, "text", 0.12),
    "arxiv": ("gfissore/arxiv-abstracts-2021", None, "abstract", 0.06),
    "codeparrot": ("codeparrot/codeparrot-clean", None, "content", 0.13),
    **{
        f"finewiki_{lang}": ("HuggingFaceFW/finewiki", lang, "text", 0.0314)
        for lang in ("de", "fr", "es", "it", "pt", "hi", "th")
    },
}

_RAW_TOTAL = sum(v[3] for v in _LLAMA_WEB.values())


@pytest.mark.parametrize("name", sorted(_LLAMA_WEB), ids=lambda n: n)
def test_the_llama_web_declaration_is_unchanged(name):
    # Same hazard as the qwen3 mix: the cache key covers the request, not the recipe, so a
    # transposed repo/config rebuilds from the wrong dataset under an unchanged name.
    src = next(s for s in PRESETS["llama_web"] if s.name == name)
    repo, config, column, raw = _LLAMA_WEB[name]
    assert (src.repo, src.config, src.adapt.shape) == (
        repo,
        config,
        f"no pair in the row -- {column} taken whole",
    )
    assert src.weight == pytest.approx(raw / _RAW_TOTAL)


def test_the_source_order_is_stable():
    # `source` is a uint8 INDEX into this list, baked into every built artifact.
    assert [s.name for s in PRESETS["llama_web"]] == list(_LLAMA_WEB)


def test_no_name_collides_with_the_qwen3_registry():
    # cache_path has ONE flat directory, so a reused name would serve the other model's text.
    assert not set(PRESETS) & set(QWEN3_PRESETS)


def test_every_source_hash_splits():
    # A `whole` source has no short side to average against, so an upstream split's length skew
    # shows through -- CodeSearchNet's tripped the holdout gate at 985/1354/1210.
    assert [s.name for s in PRESETS["llama_web"] if s.key is None] == []


def test_the_resolver_finds_both_registries():
    assert len(corpus_sources("llama_web")) == len(_LLAMA_WEB)
    assert len(corpus_sources("multi_domain")) == len(QWEN3_PRESETS["multi_domain"])
