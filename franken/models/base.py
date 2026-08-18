"""Model backend interface: everything model-family-specific that ``Distiller`` and the scripts
need. Nothing task-specific (data/labels/loss/metric live on ``franken.tasks.Task``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from franken.config import Config
from franken.distill.layer_map import resolve_layer_map


class ModelBackend(ABC):
    # Substring identifying a per-layer weight key, with the block index directly after it
    # ("bert.encoder.layer.3.…", "layers.3.…"). Everything else is copied verbatim.
    layer_marker: str

    @abstractmethod
    def build_student(self, cfg: Config) -> nn.Module:
        """Construct the student with the configured FHE ops injected."""

    @abstractmethod
    def load_teacher(self, cfg: Config) -> nn.Module:
        """Load a frozen, eval-mode teacher with hidden states enabled."""

    def seed_student(self, student: nn.Module, teacher: nn.Module, cfg: Config) -> None:
        """Strided teacher -> student init, in place. Weights transfer by name, so a student block
        keeps the teacher block's key layout under depth reduction."""
        layer_map = resolve_layer_map(
            teacher.config.num_hidden_layers,
            cfg.model.num_hidden_layers,
            cfg.distill.hidden_layer_map,
        )
        marker = self.layer_marker
        new_state = {}
        for key, tensor in teacher.state_dict().items():
            if marker not in key:
                new_state[key] = tensor  # embeddings / norms / heads: verbatim
                continue
            t = int(key.split(marker)[1].split(".")[0])
            if t in layer_map:
                i = layer_map.index(t)  # student slot for teacher block t
                new_state[key.replace(f"{marker}{t}.", f"{marker}{i}.", 1)] = tensor
        student.load_state_dict(new_state, strict=False)

    @abstractmethod
    def forward(self, model: nn.Module, inputs: dict) -> dict:
        """Teacher or student -> ``{"output": Tensor, "hidden_states": Sequence[Tensor]}``, with
        ``hidden_states[0]`` the embedding output (HF convention)."""

    @abstractmethod
    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        """Modules whose *output* is an FFN pre-activation: the range penalty's hook targets."""

    @abstractmethod
    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        """The per-layer activation op modules (some expose a ``.domain``)."""

    @abstractmethod
    def softmax_ops(self, model: nn.Module) -> list[nn.Module]:
        """The per-layer softmax op modules, for hooking the scores that reach them."""
