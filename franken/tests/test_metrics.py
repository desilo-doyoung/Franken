import math

import torch

from franken.data.embed_corpus import Pool
from franken.metrics import gold_recall_at_k, ndcg_at_k, ndcg_pool, recall_at_k


def circle(*degrees):
    angles = torch.tensor([math.radians(d) for d in degrees])
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def test_identical_embeddings_agree_completely():
    x = torch.nn.functional.normalize(torch.randn(50, 16), dim=-1)
    assert recall_at_k(x, x, 5) == 1.0


def test_self_similarity_is_masked():
    # Same four texts, neighbourhoods fully rearranged. Unmasked, every row's nearest neighbour
    # would be itself and the two models would agree for free.
    teacher = circle(0, 5, 180, 185)
    student = circle(0, 180, 5, 185)
    assert recall_at_k(student, teacher, 1) == 0.0


def test_agreement_is_partial_when_neighbourhoods_partly_move():
    # Three teacher pairs; the student pulls one text out of its pair and into another's.
    teacher = circle(0, 5, 120, 125, 240, 245)
    student = circle(0, 5, 8, 125, 240, 245)
    assert recall_at_k(student, teacher, 1) == 0.5


def test_whole_pool_neighbourhood_is_trivially_agreed():
    student = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    teacher = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    assert recall_at_k(student, teacher, 7) == 1.0


def test_inputs_are_not_mutated():
    # `masked_fill_` is in-place on the gram matrix; it must not reach the caller's embeddings.
    x = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    before = x.clone()
    recall_at_k(x, x, 3)
    assert torch.equal(x, before)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}) == 1.0


def test_ndcg_penalises_a_lower_rank():
    assert round(ndcg_at_k(["x", "a"], {"a": 1.0}), 4) == 0.6309


def test_ndcg_respects_graded_relevance():
    # Swapping a rel-2 gold below a rel-1 one must cost more than swapping two equal golds.
    assert round(ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}), 4) == 0.7967
    assert ndcg_at_k(["b", "a"], {"a": 1.0, "b": 1.0}) == 1.0


def test_ndcg_ignores_golds_below_k():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1.0}, k=2) == 0.0


def test_ndcg_without_judgements_is_zero():
    assert ndcg_at_k(["a"], {}) == 0.0


def test_gold_recall_ceiling_is_min_k_and_gold_count():
    # 40 judged docs and k=10: retrieving 10 of them is a perfect score, not 0.25.
    assert gold_recall_at_k([f"d{i}" for i in range(10)], {f"d{i}": 1.0 for i in range(40)}) == 1.0


def test_gold_recall_counts_only_judged_hits():
    assert gold_recall_at_k(["a", "x"], {"a": 1.0, "b": 1.0}) == 0.5
    assert gold_recall_at_k(["x", "y"], {"a": 1.0}) == 0.0


def one_query_pool(gold: str) -> Pool:
    return Pool(
        d_ids=["d0", "d1"],
        d_texts=["", ""],
        q_ids=["q0"],
        q_texts=[""],
        qrels={"q0": {gold: 1.0}},
    )


def test_ndcg_pool_scores_the_retrieved_ranking():
    d_emb, q_emb = torch.eye(2), torch.tensor([[1.0, 0.0]])
    assert ndcg_pool(one_query_pool("d0"), d_emb, q_emb) == 1.0
    assert round(ndcg_pool(one_query_pool("d1"), d_emb, q_emb), 4) == 0.6309
