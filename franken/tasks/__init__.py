"""Task registry (see ``franken.models`` for the parallel model-backend registry)."""

from __future__ import annotations

from franken.tasks.base import Task
from franken.tasks.embed import EmbedSelfDistillTask
from franken.tasks.mrpc import MrpcTask

TASKS: dict[str, type[Task]] = {
    "mrpc": MrpcTask,
    "embed": EmbedSelfDistillTask,
}


def build_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]()
