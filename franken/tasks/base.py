"""Task interface: everything about *what* is being learned, independent of the model family. So
swapping self-distillation for a fine-tuned downstream task needs no ``ModelBackend`` change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from franken.config import Config
from franken.models.base import ModelBackend


class Task(ABC):
    @abstractmethod
    def build_tokenizer(self, cfg: Config) -> Any: ...

    @abstractmethod
    def datasets(self, tokenizer: Any, cfg: Config) -> dict:
        """Return ``{"train": ds, "validation": ds, "collator": collator}``."""

    @abstractmethod
    def torch_columns(self) -> list[str]:
        """Dataset columns to expose as torch tensors (fed to the collator)."""

    @abstractmethod
    def model_inputs(self, batch: dict) -> dict:
        """Map a collated batch to the forward kwargs for ``ModelBackend.forward``."""

    @abstractmethod
    def compute_loss(self, student_out: dict, teacher_out: dict, batch: dict, cfg: Config) -> tuple:
        """Return ``(total_loss, components)`` where ``components`` is a dict of
        named scalar tensors for logging. ``*_out`` are backend forward outputs
        (``{"output", "hidden_states"}``); ``cfg`` supplies the distill weights."""

    @abstractmethod
    def select_metric(self) -> tuple[str, bool]:
        """``(metric_name, higher_is_better)`` used for best-checkpoint selection."""

    @abstractmethod
    def evaluate(
        self,
        backend: ModelBackend,
        model: nn.Module,
        tokenizer: Any,
        cfg: Config,
        split: str = "validation",
        teacher: nn.Module | None = None,
    ) -> dict:
        """Metrics dict containing ``select_metric``'s key. Tasks scored against labels ignore
        ``teacher``; label-free tasks whose metric *is* teacher agreement require it."""

    def train_teacher(self, cfg: Config) -> str | None:
        """Fine-tune and save a teacher if the task needs one; return its path.
        Default: no-op (the pretrained checkpoint is already the teacher)."""
        return None
