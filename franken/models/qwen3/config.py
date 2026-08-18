from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from franken.config import ModelConfig

# Declared next to the field it constrains; attention.py imports it.
ATTN_IMPLS = ("manual", "sdpa_causal")


@dataclass
class Qwen3ModelConfig(ModelConfig):
    """Qwen3 0.6B dims."""

    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128  # different from hidden_size // num_attention_heads
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    vocab_size: int = 151669
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    # "manual" = materialized scores through the injected softmax op (required for approximate
    # softmaxes). "sdpa_causal" = fused SDPA, the only form reaching flash; exact softmax and
    # right padding only (see attention.py).
    attn_impl: str = "manual"

    def validate(self) -> Any:
        activation = super().validate()
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(
                f"Unknown model.attn_impl {self.attn_impl!r}; use {' | '.join(ATTN_IMPLS)}"
            )
        if self.attn_impl == "sdpa_causal" and self.softmax != "exact":
            raise ValueError(
                f"attn_impl 'sdpa_causal' fuses the softmax into the kernel, so it cannot run "
                f"softmax={self.softmax!r}. Approximate softmaxes need attn_impl 'manual'."
            )
        return activation
