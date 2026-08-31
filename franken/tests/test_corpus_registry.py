import os
from types import SimpleNamespace

import pytest

from franken.data.corpus.build import cache_path, train_cache_path
from franken.data.qwen3.registry import MIXES, PRESETS
from franken.paths import ROOT

# name -> (repo, config, key, TOKEN share, instructed, has qrels, scores_ndcg)
_MULTI_DOMAIN = {
    "msmarco": ("microsoft/ms_marco", "v2.1", None, 0.1544, True, False, True),
    "nq_passage": ("BeIR/nq", "corpus", "_id", 0.0785, False, True, True),
    "hotpotqa_passage": ("BeIR/hotpotqa", "corpus", "_id", 0.0517, False, True, True),
    "wiki_en": ("wikimedia/wikipedia", "20231101.en", "id", 0.0419, False, False, False),
    "gooaq": ("sentence-transformers/gooaq", None, "question", 0.0451, True, False, True),
    "eli5": ("sentence-transformers/eli5", None, "question", 0.02, True, False, True),
    "stackexchange": (
        "sentence-transformers/stackexchange-duplicates",
        "post-post-pair",
        "post1",
        0.0531,
        False,
        False,
        True,
    ),
    "arxiv": ("gfissore/arxiv-abstracts-2021", None, "id", 0.0649, True, False, True),
    "s2orc": (
        "sentence-transformers/s2orc",
        "abstract-citation-pair",
        "abstract",
        0.1965,
        False,
        False,
        True,
    ),
    "specter": ("sentence-transformers/specter", "triplet", "anchor", 0.0043, False, False, False),
    "code": ("code-search-net/code_search_net", "python", None, 0.0612, True, False, True),
    "codefeedback": (
        "m-a-p/CodeFeedback-Filtered-Instruction",
        None,
        "query",
        0.0438,
        True,
        False,
        True,
    ),
    "glaive_code": ("glaiveai/glaive-code-assistant", None, "question", 0.0332, True, False, True),
    **{
        f"wiki_{lang}": ("wikimedia/wikipedia", f"20231101.{lang}", "id", w, False, False, False)
        for lang, w in (
            ("zh", 0.0322),
            ("ja", 0.0298),
            ("ar", 0.0342),
            ("ru", 0.0292),
            ("es", 0.0260),
        )
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
            "v13-multi_domain-train-1930000tok-256-Qwen_Qwen3-Embedding-0.6B",
        ),
        (
            "multi_domain",
            "validation",
            "500",
            256,
            "Qwen/Qwen3-Embedding-0.6B",
            "v13-multi_domain-validation-500-256-Qwen_Qwen3-Embedding-0.6B",
        ),
        (
            "mixed",
            "train",
            "140000tok",
            128,
            "unsloth/Llama-3.2-1B",
            "v13-mixed-train-140000tok-128-unsloth_Llama-3.2-1B",
        ),
        (
            "mixed",
            "validation",
            "500",
            128,
            "unsloth/Llama-3.2-1B",
            "v13-mixed-validation-500-128-unsloth_Llama-3.2-1B",
        ),
    ],
)
def test_cache_path_still_names_the_builds_already_on_disk(name, split, label, cap, tok, expected):
    # v13 deliberately orphans the v12 dirs: chopping regroups documents into different blocks and
    # drops the stored mask. Pinned on the BASENAME so the NEXT change cannot move the key by
    # accident, while the directory itself is free to be anchored anywhere.
    got = cache_path(name, split, label, cap, _tok(tok))
    assert os.path.basename(got) == expected


def test_the_cache_lives_at_a_fixed_place_not_wherever_python_was_started():
    # A relative dir made two runs of the same config disagree about the cache from different CWDs.
    got = cache_path("mixed", "train", "140000tok", 128, _tok("t"))
    assert os.path.isabs(got)
    assert got.startswith(ROOT + os.sep)


def test_packing_is_a_different_artifact_not_a_reinterpreted_one():
    # An unpacked key must render exactly as before the flag existed, or turning packing on
    # elsewhere would silently reinterpret an existing build.
    bare = cache_path("multi_domain", "train", "1930000tok", 256, _tok("t"))
    assert cache_path("multi_domain", "train", "1930000tok", 256, _tok("t"), pack=False) == bare
    assert cache_path("multi_domain", "train", "1930000tok", 256, _tok("t"), pack=True) == (
        bare + "-packed"
    )


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


def test_declared_token_shares_are_realized(monkeypatch):
    # The point of drawing to a quota instead of a planned document count: no tok/doc estimate
    # stands between the declaration and the artifact, so uneven lengths cannot skew the mix.
    from collections import Counter

    from franken.data.corpus import build
    from franken.data.llama.registry import PRESETS as LLAMA

    sources = LLAMA["llama_web"]
    lens = {s.name: 5 + 40 * i for i, s in enumerate(sources)}  # 5..445 tokens, an 89x spread

    def endless(src, split, size=1000):
        # Never runs dry, so the quota is the only thing that stops the draw.
        while True:
            yield ["w " * lens[src.name]] * size

    monkeypatch.setattr(build, "_batches", endless)

    req = build._Build(
        name="t",
        sources=sources,
        split="train",
        tokenizer=_FakeTok(),
        max_seq_len=1024,
        tokens=2_000_000,
    )
    tokens = Counter()
    for row in build._rows(req):
        tokens[row["source"]] += len(row["input_ids"])

    total = sum(tokens.values())
    for i, src in enumerate(sources):
        assert tokens[i] / total == pytest.approx(src.weight, rel=0.01)


def test_a_source_that_fails_mid_draw_is_not_silently_dropped(monkeypatch):
    # `profile` tolerates a dead loader because the GATE must report every source. The build must
    # not: a silently short corpus trains a different experiment than the one declared.
    from franken.data.corpus import build
    from franken.data.llama.registry import PRESETS as LLAMA

    def boom(src, split, size=1000):
        raise RuntimeError("loader died")

    monkeypatch.setattr(build, "_batches", boom)
    req = build._Build(
        name="t",
        sources=LLAMA["llama_web"],
        split="train",
        tokenizer=_FakeTok(),
        max_seq_len=1024,
        tokens=1000,
    )
    with pytest.raises(RuntimeError, match="loader died"):
        list(build._rows(req))


class _FakeTok:
    """One token per whitespace-separated word, so a document's length is written in its text."""

    eos_token_id = 0
    name_or_path = "fake"

    def __call__(self, batch, truncation=False, max_length=None):
        cap = max_length if truncation and max_length else 10**9
        return {"input_ids": [[1] * min(len(t.split()), cap) for t in batch]}
