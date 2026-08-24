from collections import Counter

import pytest

from franken.data.embed_corpus.spec import (
    SPLIT_PCT,
    SPLITS,
    WEB_SEARCH,
    Record,
    corpus_texts,
    eval_pair,
    instruct,
    split_of,
)

KEYS = [f"k{i}" for i in range(40_000)]


def test_split_assignment_is_stable_across_processes():
    # Pinned, not just self-consistent: `hash()` would redraw the split every run and silently
    # move trained rows into the eval pool.
    assert split_of("k103") == "validation"
    assert split_of("k26") == "test"
    assert split_of("k0") == "train"


def test_split_is_a_pure_function_of_the_key():
    assert all(split_of(k) == split_of(k) for k in KEYS[:100])


def test_split_proportions_match_the_declaration():
    counts = Counter(split_of(k) for k in KEYS)
    for split, pct in SPLIT_PCT.items():
        assert abs(100 * counts[split] / len(KEYS) - pct) < 0.5
    assert set(counts) <= set(SPLITS)


def test_split_pct_are_per_split_not_cumulative():
    # The bug this replaced: VAL_PCT/TEST_PCT = 2/4 read as a 4% test split and delivered 2%.
    counts = Counter(split_of(k) for k in KEYS)
    assert counts["test"] > counts["validation"] * 2


def test_instruct_matches_the_checkpoint_wire_format():
    assert instruct("Given a query", "cats") == "Instruct: Given a query\nQuery:cats"


def test_the_web_search_task_string_is_pinned():
    # Baked into cached corpus text and into every pool's q_texts: a one-character edit
    # invalidates the corpus cache and all 18 pool JSONs, silently.
    assert WEB_SEARCH == (
        "Given a web search query, retrieve relevant passages that answer the query"
    )


def test_instruct_leaves_a_symmetric_query_bare():
    assert instruct(None, "cats") == "cats"


def test_corpus_texts_prefixes_only_the_query():
    rec = Record(query="q", positives=("p1", "p2"), negatives=("n",), docs=("d",))
    assert corpus_texts(rec, "T") == ["Instruct: T\nQuery:q", "p1", "p2", "n", "d"]


def test_corpus_texts_without_a_query_side():
    assert corpus_texts(Record(docs=("d1", "d2")), "T") == ["d1", "d2"]


def test_eval_pair_returns_query_golds_and_distractors():
    rec = Record(query="q", positives=("p1", "p2"), negatives=("n",))
    assert eval_pair(rec) == ("q", ("p1", "p2"), ("n",))


@pytest.mark.parametrize(
    "rec", [Record(positives=("p",)), Record(query="q"), Record(query="q", docs=("d",))]
)
def test_eval_pair_needs_both_sides(rec):
    assert eval_pair(rec) is None
