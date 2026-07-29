"""Model backend interface.

A ``ModelBackend`` is a strategy encapsulating everything model-family-specific
that ``Distiller`` (and the scripts) need: how to build/load the models, how to
seed the student from the teacher, how to run a forward pass into a normalized
output contract, and how to reach the submodules the range penalty hooks.

Nothing task-specific lives here (no data / labels / loss / metric — those are on
``franken.tasks.Task``). The interface is intentionally pinned to exactly the
methods the callers use today; it is expected to evolve when a second real
backend (Qwen3) is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from franken.config import Config


class ModelBackend(ABC):
    @abstractmethod
    def build_student(self, cfg: Config) -> nn.Module:
        """Construct the student with the configured FHE ops injected."""

    @abstractmethod
    def load_teacher(self, cfg: Config) -> nn.Module:
        """Load a frozen, eval-mode teacher with hidden states enabled."""

    @abstractmethod
    def seed_student(self, student: nn.Module, teacher: nn.Module, cfg: Config) -> None:
        """Initialize student weights from the teacher (in place)."""

    @abstractmethod
    def forward(self, model: nn.Module, inputs: dict) -> dict:
        """Run ``model`` on ``inputs`` (forward kwargs) and return a normalized
        ``{"output": Tensor, "hidden_states": Sequence[Tensor]}`` dict, where
        ``hidden_states[0]`` is the embedding output (HF convention). Works for
        both the teacher and the student."""

    @abstractmethod
    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        """Modules whose *output* is an FFN pre-activation — the tensors the
        range penalty pulls into the activation op's valid domain (hook targets)."""

    @abstractmethod
    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        """The per-layer activation op modules (some expose a ``.domain``)."""
