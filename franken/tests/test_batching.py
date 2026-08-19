import random

import pytest

from franken.distill.batching import plan_batches, shard

BUDGET = 1000


@pytest.fixture
def lengths():
    rng = random.Random(11)
    return [rng.randint(5, 200) for _ in range(500)]


@pytest.fixture
def corpus_lengths():
    """Shaped like multi_domain at cap 1024: median ~74, mean ~112."""
    rng = random.Random(7)
    return [min(1024, max(4, int(rng.lognormvariate(4.3, 0.9)))) for _ in range(30_000)]


def padded(batch, lengths):
    return len(batch) * max(lengths[i] for i in batch)


def test_every_index_appears_exactly_once(lengths):
    batches = plan_batches(lengths, BUDGET, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_respects_the_padded_token_budget(lengths):
    # A lone sequence longer than the budget is admitted rather than dropped; anything with a
    # second member must fit.
    for b in plan_batches(lengths, BUDGET, seed=0):
        if len(b) > 1:
            assert padded(b, lengths) <= BUDGET


def test_a_single_overlong_sequence_is_kept_not_dropped():
    batches = plan_batches([10, 10, 5000], token_budget=100, seed=0)
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_plan_ignores_global_rng_state(lengths):
    # Every rank builds the plan independently and they must agree, or the ranks step a different
    # number of times and deadlock in the allreduce.
    random.seed(1)
    a = plan_batches(lengths, BUDGET, seed=0)
    random.seed(999)
    random.random()
    assert a == plan_batches(lengths, BUDGET, seed=0)


def test_different_seeds_give_different_plans(lengths):
    assert plan_batches(lengths, BUDGET, 0) != plan_batches(lengths, BUDGET, 1)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_shard_gives_every_rank_the_same_step_count(lengths, world_size):
    batches = plan_batches(lengths, BUDGET, seed=0)
    shards = [shard(batches, r, world_size) for r in range(world_size)]
    assert len({len(s) for s in shards}) == 1
    assert len(shards[0]) == len(batches) // world_size


def test_shards_are_disjoint_and_drawn_from_the_plan(lengths):
    batches = plan_batches(lengths, BUDGET, seed=0)
    seen = [i for r in range(3) for b in shard(batches, r, 3) for i in b]
    assert len(seen) == len(set(seen))
    assert set(seen) <= set(range(len(lengths)))


CORPUS_BUDGETS = [2048, 4096, 8192, 16384, 32768]


def test_the_budget_is_nearly_all_used(corpus_lengths):
    # Without a sequence cap the batch fills to the budget; a cap would only bind on short-text
    # batches and cost ~9% here for no memory saving.
    for budget in CORPUS_BUDGETS:
        plan = plan_batches(corpus_lengths, budget, seed=0)
        occupancy = sum(padded(b, corpus_lengths) for b in plan) / (len(plan) * budget)
        assert occupancy > 0.93, (budget, occupancy)


def test_doubling_the_budget_doubles_the_batch(corpus_lengths):
    # The budget must be a linear knob on the batch, or sqrt-batch LR and every budget-to-budget
    # comparison stop being interpretable.
    def seqs(budget):
        plan = plan_batches(corpus_lengths, budget, seed=0)
        return sum(len(b) for b in plan) / len(plan)

    for lo, hi in zip(CORPUS_BUDGETS, CORPUS_BUDGETS[1:], strict=False):
        assert 1.9 < seqs(hi) / seqs(lo) < 2.1, (lo, hi, seqs(lo), seqs(hi))
