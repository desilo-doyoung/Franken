"""Llama-3.2 backend: the from-scratch student against the frozen HF backbone. Both go through
the same pooling here, so the loss only ever compares like with like."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from franken.config import Config
from franken.models.base import ModelBackend
from franken.models.llama.model import LlamaModel


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor | None):
    """The hidden state at the last visible position, indexed so either padding side works."""
    if attention_mask is None:
        return last_hidden[:, -1]
    seq_len = attention_mask.shape[-1]
    idx = seq_len - 1 - attention_mask.flip(-1).argmax(-1)
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), idx]


class LlamaBackend(ModelBackend):
    layer_marker = "layers."

    def build_student(self, cfg: Config) -> nn.Module:
        return LlamaModel(cfg.model)

    def load_teacher(self, cfg: Config) -> nn.Module:
        # fp32 is pinned: transformers 5.x defaults to the checkpoint dtype (bf16), which would
        # put the parity gate and the distillation targets at bf16.
        ckpt = cfg.train.teacher_ckpt or cfg.train.teacher_model
        # attn_impl is a STUDENT setting; without this the teacher keeps a dense mask, which in
        # fp32 + GQA reaches only the MATH backend and materializes (B,H,S,S).
        extra = {"attn_implementation": "flex_attention"} if cfg.model.attn_impl == "flex" else {}
        model = AutoModel.from_pretrained(
            ckpt, dtype=torch.float32, output_hidden_states=True, **extra
        )
        model.eval()
        model.requires_grad_(False)
        # HF only isolates documents when past_key_values AND attention_mask are both None
        # (masking_utils). A cache here would silently cost the teacher its isolation.
        model.config.use_cache = False
        return model

    def forward(self, model: nn.Module, inputs: dict) -> dict:
        out = model(**inputs)
        # Student returns a dict; the HF teacher returns a ModelOutput.
        if isinstance(out, dict):
            last_hidden, hidden_states = out["last_hidden_state"], out["hidden_states"]
        else:
            last_hidden, hidden_states = out.last_hidden_state, out.hidden_states
        pooled = _last_token_pool(last_hidden, inputs.get("attention_mask"))
        return {
            "output": F.normalize(pooled, p=2, dim=-1),
            "hidden_states": hidden_states,
            # Tied embeddings, so this IS the LM head, and `hidden_states[-1]` is already
            # post-norm -- exactly what LlamaForCausalLM projects.
            "lm_head_weight": model.embed_tokens.weight,
        }

    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        return [ly.mlp.gate_proj for ly in model.layers]

    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        return [ly.mlp.act_fn for ly in model.layers]

    def softmax_ops(self, model: nn.Module) -> list[nn.Module]:
        return [ly.self_attn.softmax for ly in model.layers]
