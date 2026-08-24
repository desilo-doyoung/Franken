"""Model backend interface: everything model-family-specific. Nothing task-specific."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from franken.config import Config
from franken.distill.layer_map import resolve_layer_map


class ModelBackend(ABC):
    # Substring before a block index in a weight key ("layers.3."); everything else copies verbatim.
    layer_marker: str

    @abstractmethod
    def build_student(self, cfg: Config) -> nn.Module:
        """Construct the student with the configured FHE ops injected."""

    @abstractmethod
    def load_teacher(self, cfg: Config) -> nn.Module:
        """Load a frozen, eval-mode teacher with hidden states enabled."""

    def seed_student(self, student: nn.Module, teacher: nn.Module, cfg: Config) -> None:
        """Strided teacher -> student init, in place; weights transfer by name."""
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
        """-> ``{"output", "hidden_states"}``, with ``hidden_states[0]`` the embedding output.

        ``output`` is the backend's canonical representation -- logits for a classifier, the pooled
        vector for a decoder -- and is what `parity_gate` compares. A decoder may also supply
        ``lm_head_weight``: the output projection, for a task that needs vocabulary logits. It is a
        reference so the task can project in chunks; materializing 128k-vocab logits on every
        forward would cost GBs the embed path never reads.
        """

    @abstractmethod
    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        """Modules whose *output* is an FFN pre-activation: the range penalty's hook targets."""

    @abstractmethod
    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        """The per-layer activation op modules (some expose a ``.domain``)."""

    @abstractmethod
    def softmax_ops(self, model: nn.Module) -> list[nn.Module]:
        """The per-layer softmax op modules, for hooking the scores that reach them."""
