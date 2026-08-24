from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from franken.config import ModelConfig

# Declared next to the field it constrains; attention.py imports it.
ATTN_IMPLS = ("manual", "sdpa_causal")


@dataclass
class LlamaModelConfig(ModelConfig):
    """Llama 3.2 1B dims."""

    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 64
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    # Llama 3's rope_scaling, flattened. The scaling rewrites inv_freq itself, so dropping it
    # shifts every angle away from the teacher's -- even at max_seq_len 128. See rope.py.
    rope_theta: float = 5e5
    rope_scaling_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192
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
