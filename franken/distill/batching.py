"""Token-budgeted batching: hold padded tokens constant, let the sequence count float. Bounds
activation memory by construction and lifts padding efficiency from ~14% to ~98%."""

from __future__ import annotations

import random

import pyarrow.compute as pc

# Texts sorted together before cutting into batches. Local, not global: fully sorted batches would
# be single-mode, the failure the corpus shuffle exists to prevent.
SORT_WINDOW = 12_800  # texts, so the granularity does not move with the budget


def plan_batches(
    lengths,
    token_budget: int,
    seed: int,
    sort_window: int = SORT_WINDOW,
) -> list[list[int]]:
    # No sequence cap: it would bind only on short-text batches, costing ~9% of the budget
    # (98.5% -> 90.6% occupancy) for no memory saving -- activations are padded-tokens x hidden.
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)

    batches: list[list[int]] = []
    for start in range(0, len(order), sort_window):
        window = sorted(order[start : start + sort_window], key=lambda i: lengths[i])
        batch: list[int] = []
        w = 0
        for i in window:
            grown = max(w, int(lengths[i]))
            if batch and (len(batch) + 1) * grown > token_budget:
                batches.append(batch)
                batch, grown = [], int(lengths[i])
            batch.append(i)
            w = grown
        if batch:
            batches.append(batch)

    # Batches leave each window length-sorted; unshuffled, the LR warmup would see only the
    # shortest texts in the corpus.
    rng.shuffle(batches)
    return batches


def plan_fixed_batches(n_rows: int, rows_per_batch: int, seed: int) -> list[list[int]]:
    """Equal-width rows need no bucketing: a batch is a row count. Nothing here packs -- the corpus
    build already did that; this only chunks what it produced.

    Shuffled first because a packed block is single-source, so only the draw mixes sources within a
    step. The trailing partial is dropped: every step then carries identical tokens, and flex
    (compiled with `dynamic=False`) never sees a second shape.
    """
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    return [
        order[i : i + rows_per_batch] for i in range(0, n_rows - rows_per_batch + 1, rows_per_batch)
    ]


def row_plan(dataset, micro_tokens: int, seed: int, block_size: int | None = None):
    """The one planner the trainer and the eval loader share.

    `block_size` set means the artifact was bin-packed at BUILD time, so every row is already
    exactly that wide and grouping them is a row count. That width is checked, not assumed: a short
    row would silently overshoot the budget once the collator padded the batch to its longest.
    """
    lengths = pc.list_value_length(dataset.data.column("input_ids")).to_numpy(zero_copy_only=False)
    if block_size is None:
        return plan_batches(lengths, micro_tokens, seed)

    if len(lengths):
        lo, hi = int(lengths.min()), int(lengths.max())
        if lo != block_size or hi != block_size:
            raise ValueError(
                f"train.pack promises rows of exactly {block_size:,} tokens, but the artifact "
                f"holds {lo:,}..{hi:,}. Rebuild the corpus, or the batch budget is meaningless."
            )
    rows, spare = divmod(micro_tokens, block_size)
    if not rows:
        raise ValueError(
            f"micro-batch is {micro_tokens:,} tokens but a packed row is {block_size:,}. Raise "
            "FRANKEN_MAX_TOKENS_PER_RANK to at least max_seq_len, or lower max_seq_len."
        )
    if spare:
        raise ValueError(
            f"micro-batch {micro_tokens:,} is not a whole number of {block_size:,}-token rows; "
            f"{spare:,} tokens per step would go unused. Make tokens_per_step a multiple of "
            "max_seq_len x world_size."
        )
    plan = plan_fixed_batches(len(lengths), rows, seed)
    if not plan:
        # Otherwise `optimizer_steps` is 0 and the LR schedule divides by zero an import deep into
        # transformers, with nothing naming the corpus as the cause.
        raise ValueError(
            f"the artifact holds {len(lengths):,} rows of {block_size:,} tokens, too few for one "
            f"micro-batch of {rows:,}. Draw more tokens, or lower max_seq_len."
        )
    return plan


def shard(batches: list[list[int]], rank: int, world_size: int) -> list[list[int]]:
    """Drop the remainder: ranks that step a different number of times deadlock in the allreduce.
    Needs no collective -- `plan_batches` touches no global RNG, so every rank plans identically."""
    usable = len(batches) - len(batches) % world_size
    return batches[:usable][rank::world_size]
