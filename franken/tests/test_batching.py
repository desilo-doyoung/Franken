import random

import pytest

from franken.distill.batching import plan_batches, shard

BUDGET, MAX_SEQS = 1000, 8


@pytest.fixture
def lengths():
    rng = random.Random(11)
    return [rng.randint(5, 200) for _ in range(500)]


def padded(batch, lengths):
    return len(batch) * max(lengths[i] for i in batch)


def test_every_index_appears_exactly_once(lengths):
    batches = plan_batches(lengths, BUDGET, MAX_SEQS, seed=0)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))


def test_respects_the_sequence_cap(lengths):
    for b in plan_batches(lengths, BUDGET, MAX_SEQS, seed=0):
        assert 0 < len(b) <= MAX_SEQS


def test_respects_the_padded_token_budget(lengths):
    # A lone sequence longer than the budget is admitted rather than dropped; anything with a
    # second member must fit.
    for b in plan_batches(lengths, BUDGET, MAX_SEQS, seed=0):
        if len(b) > 1:
            assert padded(b, lengths) <= BUDGET


def test_a_single_overlong_sequence_is_kept_not_dropped():
    batches = plan_batches([10, 10, 5000], token_budget=100, max_seqs=8, seed=0)
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_plan_ignores_global_rng_state(lengths):
    # Every rank builds the plan independently and they must agree, or the ranks step a different
    # number of times and deadlock in the allreduce.
    random.seed(1)
    a = plan_batches(lengths, BUDGET, MAX_SEQS, seed=0)
    random.seed(999)
    random.random()
    b = plan_batches(lengths, BUDGET, MAX_SEQS, seed=0)
    assert a == b


def test_different_seeds_give_different_plans(lengths):
    assert plan_batches(lengths, BUDGET, MAX_SEQS, 0) != plan_batches(lengths, BUDGET, MAX_SEQS, 1)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_shard_gives_every_rank_the_same_step_count(lengths, world_size):
    batches = plan_batches(lengths, BUDGET, MAX_SEQS, seed=0)
    shards = [shard(batches, r, world_size) for r in range(world_size)]
    assert len({len(s) for s in shards}) == 1
    assert len(shards[0]) == len(batches) // world_size


def test_shards_are_disjoint_and_drawn_from_the_plan(lengths):
    batches = plan_batches(lengths, BUDGET, MAX_SEQS, seed=0)
    seen = [i for r in range(3) for b in shard(batches, r, 3) for i in b]
    assert len(seen) == len(set(seen))
    assert set(seen) <= set(range(len(lengths)))
