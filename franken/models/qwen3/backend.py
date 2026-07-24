"""Qwen3-Embedding-0.6B backend — STUB (to be implemented).

The Qwen3 student is built **from scratch**, mirroring ``franken.models.bert`` —
hand-written modules (RMSNorm, RoPE, GQA attention with QK-norm, SwiGLU MLP)
composed into a student whose softmax/activation are injected ops from
``franken.ops``, with teacher weights loaded by name + strided depth reduction.
This is the same design as the BERT student and keeps the ops genuinely swappable;
we do NOT inject ops into an HF ``Qwen3Model`` (see ``franken/models/qwen3/PROGRESS.md``
for the module checklist). The class is importable so the registry resolves; every
method raises until filled in.

Implementation notes (parallels ``franken.models.bert.backend.BertBackend``):

- build_student(cfg): construct the from-scratch student from a resolved Qwen3 config
    (dims read from ``AutoConfig(cfg.train.teacher_model)``, depth overridden by
    ``cfg.model.num_hidden_layers``), with the FHE ops injected at build time —
    ``build_softmax``/``build_activation`` on ``cfg.model.{softmax,activation}``.
    NOTE Qwen3's SwiGLU nonlinearity is SiLU, not GELU — the current ``ACTIVATION_OPS``
    are all GELU-family, so add SiLU-family ops (ExactSiLU + polynomial approximations
    exposing ``.domain``) before using a non-exact activation.
- load_teacher(cfg): HF ``AutoModel`` backbone, exact ops, ``output_hidden_states=True``,
    ``.eval()`` + ``requires_grad_(False)``.
- seed_student(student, teacher, cfg): name-matched load of the teacher state_dict with a
    strided ``resolve_layer_map`` for depth reduction (``embed_tokens``/final ``norm``
    verbatim), mirroring ``franken.models.bert.loader``.
- forward(model, inputs): return {"output": <L2-normed last-token pooled embedding>,
    "hidden_states": tuple}; hidden_states[0] must be the embedding output (HF convention).
    Pool the teacher the same way.
- ffn_preact_modules(model): the per-layer ``gate_proj`` modules (range-penalty hooks).
- activation_ops(model): the per-layer SwiGLU activation op modules (some expose ``.domain``).
"""

from __future__ import annotations

from torch import nn

from franken.config import Config
from franken.models.base import ModelBackend

_TODO = "Qwen3Backend.{} is not implemented yet — see franken/models/qwen3/backend.py docstring."


class Qwen3Backend(ModelBackend):
    def build_student(self, cfg: Config) -> nn.Module:
        raise NotImplementedError(_TODO.format("build_student"))

    def load_teacher(self, cfg: Config) -> nn.Module:
        raise NotImplementedError(_TODO.format("load_teacher"))

    def seed_student(self, student: nn.Module, teacher: nn.Module, cfg: Config) -> None:
        raise NotImplementedError(_TODO.format("seed_student"))

    def forward(self, model: nn.Module, inputs: dict) -> dict:
        raise NotImplementedError(_TODO.format("forward"))

    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        raise NotImplementedError(_TODO.format("ffn_preact_modules"))

    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        raise NotImplementedError(_TODO.format("activation_ops"))
