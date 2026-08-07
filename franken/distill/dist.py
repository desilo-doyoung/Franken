"""Distributed helpers. Bare `python main.py distill` stays single-process and unchanged;
`torchrun --nproc_per_node=N main.py distill` goes data-parallel at any world size.

Global batch and steps-per-epoch are held constant (per-rank batch = batch_size // world_size),
so the recorded LR schedule and every comparison against PROGRESS.md stay valid.
"""

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
    """Join the process group when launched under torchrun. Without it (or at world size 1),
    return the single-process default and create no group at all — that is what keeps a plain
    `python main.py distill` on exactly the pre-DDP code path."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return DistEnv()

    env = DistEnv(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )
    # Before init_process_group: NCCL binds to whatever device is current.
    torch.cuda.set_device(env.local_rank)
    # 60min, not NCCL's 600s default: rank 0 evaluates alone while the others wait in barrier(),
    # and on a large corpus that overruns 600s and the watchdog aborts the whole job.
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


def per_rank_batch(batch_size: int, env: DistEnv) -> int:
    """Split the configured GLOBAL batch across ranks, so steps-per-epoch and the LR schedule
    match a single-process run exactly."""
    if not env.enabled:
        return batch_size
    if batch_size % env.world_size:
        raise ValueError(
            f"batch_size {batch_size} is not divisible by world_size {env.world_size}; "
            "the global batch must be preserved for results to stay comparable."
        )
    return batch_size // env.world_size
