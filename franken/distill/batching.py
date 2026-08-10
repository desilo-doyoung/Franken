"""Token-budgeted batching: fit the batch to the input, instead of the input to the batch.

``DataCollatorWithPadding`` pads to the batch maximum, so a fixed sequence count pays for whatever
the batch's longest member does not use. Measured on ``multi_domain`` @1024: 132 real tokens per
text, ~950 padded under shuffled batches of 128, i.e. 14% efficiency. Holding *tokens* per batch
constant instead gives 162 padded (81%), turning ~7x into ~1.4x.

It also bounds activation memory by construction -- a batch of 1024-token texts simply holds fewer
of them -- so there is no worst-case micro-batch to size and no gradient accumulation to add.
"""

from __future__ import annotations

import random


def plan_batches(
    lengths,
    token_budget: int,
    max_seqs: int,
    seed: int,
    bucket: int,
    mega: int = 50,
) -> list[list[int]]:
    """Index batches of at most ``token_budget`` padded tokens and ``max_seqs`` sequences.

    Sorting is local to a ``mega * max_seqs`` window, not global: fully sorted batches would be
    single-mode (all queries, then all abstracts), the failure the corpus shuffle exists to prevent.
    """
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)

    def width(i: int) -> int:
        return -(-int(lengths[i]) // bucket) * bucket

    batches: list[list[int]] = []
    span = mega * max_seqs
    for start in range(0, len(order), span):
        window = sorted(order[start : start + span], key=lambda i: lengths[i])
        batch: list[int] = []
        w = 0
        for i in window:
            grown = max(w, width(i))
            if batch and ((len(batch) + 1) * grown > token_budget or len(batch) >= max_seqs):
                batches.append(batch)
                batch, grown = [], width(i)
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
