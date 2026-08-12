"""Token-budgeted batching: fit the batch to the input, not the input to the batch.

Padding to the batch maximum means a fixed sequence count pays for what the longest member does not
use: on `multi_domain` @1024, 132 real tokens per text against ~950 padded (14% efficiency).
Holding *tokens* constant gives ~139 padded (95%), turning ~7x into ~1.1x, and bounds activation
memory by construction -- no micro-batch to size, no gradient accumulation.

Widths are the raw batch maximum, not bucketed. Rounding to 64 cost ~19% padding for nothing:
`unique_graphs` measured 8 (two per DDP bucket), so `automatic_dynamic_shapes` had already
generalized over shapes.
"""

from __future__ import annotations

import random


def plan_batches(
    lengths,
    token_budget: int,
    max_seqs: int,
    seed: int,
    mega: int = 50,
) -> list[list[int]]:
    """Index batches of at most ``token_budget`` padded tokens and ``max_seqs`` sequences.

    Sorting is local to a ``mega * max_seqs`` window, not global: fully sorted batches would be
    single-mode (all queries, then all abstracts), the failure the corpus shuffle exists to prevent.
    """
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)

    batches: list[list[int]] = []
    span = mega * max_seqs
    for start in range(0, len(order), span):
        window = sorted(order[start : start + span], key=lambda i: lengths[i])
        batch: list[int] = []
        w = 0
        for i in window:
            grown = max(w, int(lengths[i]))
            if batch and ((len(batch) + 1) * grown > token_budget or len(batch) >= max_seqs):
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


def shard(batches: list[list[int]], rank: int, world_size: int) -> list[list[int]]:
    """Ranks that step a different number of times deadlock in the gradient allreduce, so drop the
    remainder. Safe without a collective: `plan_batches` touches no global RNG, so every rank builds
    the identical plan from the identical cached lengths."""
    usable = len(batches) - len(batches) % world_size
    return batches[:usable][rank::world_size]
