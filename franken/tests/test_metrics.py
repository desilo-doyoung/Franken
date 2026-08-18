import math

import torch

from franken.tasks.embed import recall_at_k


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
