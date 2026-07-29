"""Qwen3-Embedding backend: the from-scratch Qwen3 student + HF Qwen3 teacher.

Builds the from-scratch ``Qwen3Model`` student (RMSNorm, RoPE, GQA attention with
QK-norm, SwiGLU MLP; softmax/activation injected from ``franken.ops``), seeds it
from the HF teacher via ``init_student_from_teacher`` (name-matched, strided for
depth reduction), and loads the frozen HF ``AutoModel`` backbone as teacher.

The distillation output contract is the *pooled sentence embedding*: Qwen3-Embedding
pools the **last non-pad token** of the final hidden state and L2-normalizes it.
Both models are pooled here by the same code path, so the task's loss only ever
compares like with like.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from franken.config import Config
from franken.distill.layer_map import resolve_layer_map
from franken.models.base import ModelBackend
from franken.models.qwen3.loader import init_student_from_teacher
from franken.models.qwen3.model import Qwen3Model


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor | None):
    """Qwen3-Embedding pooling: the hidden state at the last visible position.

    Indexed as "last position where the mask is 1" rather than ``mask.sum() - 1`` so
    it is correct under either padding side (Qwen3-Embedding's reference code pads
    left; HF's collator pads right).
    """
    if attention_mask is None:
        return last_hidden[:, -1]
    seq_len = attention_mask.shape[-1]
    idx = seq_len - 1 - attention_mask.flip(-1).argmax(-1)
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), idx]


class Qwen3Backend(ModelBackend):
    def build_student(self, cfg: Config) -> nn.Module:
        return Qwen3Model(cfg.model)

    def load_teacher(self, cfg: Config) -> nn.Module:
        # Frozen HF backbone (no lm_head), exact ops, per-layer hidden states enabled
        # so the distillation loss can read them. dtype is pinned to fp32: transformers
        # 5.x defaults to the checkpoint dtype (bf16 here), which would put the parity
        # gate and the distillation targets at bf16 precision.
        ckpt = cfg.train.teacher_ckpt or cfg.train.teacher_model
        model = AutoModel.from_pretrained(ckpt, dtype=torch.float32, output_hidden_states=True)
        model.eval()
        model.requires_grad_(False)
        return model

    def seed_student(self, student: nn.Module, teacher: nn.Module, cfg: Config) -> None:
        layer_map = resolve_layer_map(
            teacher.config.num_hidden_layers,
            cfg.model.num_hidden_layers,
            cfg.distill.hidden_layer_map,
        )
        init_student_from_teacher(student, teacher.state_dict(), layer_map)

    def forward(self, model: nn.Module, inputs: dict) -> dict:
        out = model(**inputs)
        # Custom student returns a dict; the HF teacher returns a ModelOutput.
        if isinstance(out, dict):
            last_hidden, hidden_states = out["last_hidden_state"], out["hidden_states"]
        else:
            last_hidden, hidden_states = out.last_hidden_state, out.hidden_states
        pooled = _last_token_pool(last_hidden, inputs.get("attention_mask"))
        return {
            "output": F.normalize(pooled, p=2, dim=-1),
            "hidden_states": hidden_states,
        }

    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        return [ly.mlp.gate_proj for ly in model.layers]

    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        return [ly.mlp.act_fn for ly in model.layers]
