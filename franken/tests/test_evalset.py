from franken.data.embed_corpus.evalset import Pool, _Docs


def test_identical_text_gets_one_id():
    # A twin of a gold document is an exact tie in the ranking, so nDCG is decided by float noise.
    docs = _Docs(10)
    assert docs.add("same") == docs.add("same")
    assert docs.ids == ["d0"]


def test_cap_is_enforced():
    docs = _Docs(2)
    assert [docs.add(t) for t in ("a", "b", "c")] == ["d0", "d1", None]
    assert docs.full()


def test_force_admits_a_gold_past_the_cap():
    docs = _Docs(2)
    for t in ("a", "b"):
        docs.add(t)
    assert docs.add("gold", force=True) == "d2"
    assert docs.texts == ["a", "b", "gold"]


def test_force_still_deduplicates():
    docs = _Docs(2)
    docs.add("a")
    assert docs.add("a", force=True) == "d0"
    assert docs.ids == ["d0"]


def test_ids_index_their_own_text():
    docs = _Docs(10)
    for t in ("a", "b", "c"):
        docs.add(t)
    assert [docs.texts[docs.ids.index(f"d{i}")] for i in range(3)] == ["a", "b", "c"]


def test_pool_is_falsy_without_queries():
    assert not Pool(d_ids=["d0"], d_texts=["a"])
    assert Pool(q_ids=["q0"], q_texts=["q"])
