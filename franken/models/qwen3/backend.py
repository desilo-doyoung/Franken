"""Qwen3-Embedding-0.6B backend — STUB (to be implemented).

This is intentionally unimplemented: the model internals will be built by hand
(mirroring how the from-scratch BERT student was built), with guidance. The class
is importable so the registry resolves; every method raises until filled in.

Implementation notes (parallels ``franken.models.bert.backend.BertBackend``):

- build_student(cfg):
    Load the HF Qwen3-Embedding backbone from ``cfg.train.teacher_model`` with
    ``output_hidden_states=True``, then INJECT the FHE ops (reusing HF's RoPE / GQA /
    QK-norm / RMSNorm — only softmax + activation are FHE-relevant):
      * activation: for each ``layer`` in ``model.model.layers`` set ``layer.mlp.act_fn``
        to ``build_activation(cfg.model.activation, **cfg.model.activation_kwargs)``.
        NOTE Qwen3's SwiGLU nonlinearity is SiLU, not GELU — the current
        ``ACTIVATION_OPS`` are all GELU-family, so add SiLU-family ops (ExactSiLU +
        polynomial approximations with ``.domain``) before using non-exact activations.
      * softmax: register a custom attention interface once via
        ``transformers.AttentionInterface.register("franken", fn)`` where ``fn`` does
        ``repeat_kv`` (GQA) + scaled scores + ``module.franken_softmax(scores, mask, dim=-1)``;
        set ``model.config._attn_implementation = "franken"`` and attach
        ``layer.self_attn.franken_softmax = build_softmax(cfg.model.softmax, ...)`` per layer.
- load_teacher(cfg): same backbone, exact ops, ``.eval()`` + ``requires_grad_(False)``.
- seed_student(student, teacher, cfg): same width/depth => ``student.load_state_dict(
    teacher.state_dict(), strict=False)`` (strict=False so buffer-only injected ops don't error).
- forward(model, inputs): return {"output": <pooled last-token embedding>, "hidden_states": tuple};
    hidden_states[0] must be the embedding output (HF convention). Pooling may live in the task.
- ffn_preact_modules(model): ``[layer.mlp.gate_proj for layer in model.model.layers]``.
- activation_ops(model): ``[layer.mlp.act_fn for layer in model.model.layers]``.
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
