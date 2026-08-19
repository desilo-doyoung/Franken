"""Distributed helpers. Global batch and steps-per-epoch are held constant across world sizes, so
the recorded LR schedule stays comparable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistEnv:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistEnv:
    """Join the process group under torchrun; otherwise create no group at all, which keeps a
    plain `python main.py distill` on exactly the single-process path."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return DistEnv()

    env = DistEnv(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )
    torch.cuda.set_device(env.local_rank)  # NCCL binds to whatever device is current
    # 60min, not NCCL's 600s: rank 0 evaluates alone while the others wait in barrier().
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=60),
        device_id=torch.device(f"cuda:{env.local_rank}"),
    )
    return env


def shutdown(env: DistEnv) -> None:
    if env.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(env: DistEnv) -> None:
    if env.enabled:
        dist.barrier()


def max_tokens_per_rank() -> int:
    """Padded tokens one GPU holds -- a machine property (card x depth), hence env not config:
    getting it wrong changes accumulation, i.e. speed, never the global batch. 16384 is safe at
    depth 28; an A6000 needs ~4096."""
    return int(os.environ.get("FRANKEN_MAX_TOKENS_PER_RANK", 16384))


def per_rank_batch(batch_size: int, env: DistEnv) -> int:
    """Split the GLOBAL batch across ranks, so steps-per-epoch matches a single-process run."""
    if not env.enabled:
        return batch_size
    if batch_size % env.world_size:
        raise ValueError(
            f"batch_size {batch_size} is not divisible by world_size {env.world_size}; "
            "the global batch must be preserved for results to stay comparable."
        )
    return batch_size // env.world_size
