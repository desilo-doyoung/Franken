"""Task interface: what is being learned, independent of the model family."""

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
        """-> ``(total_loss, components)``; components are named scalars for logging."""

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
        """Metrics containing ``select_metric``'s key. Label-free tasks require ``teacher``."""

    def train_teacher(self, cfg: Config) -> str | None:
        """Fine-tune and save a teacher if the task needs one. Default: the checkpoint is it."""
        return None
