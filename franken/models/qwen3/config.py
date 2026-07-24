from __future__ import annotations

from dataclasses import dataclass

from franken.config import ModelConfig


@dataclass
class Qwen3ModelConfig(ModelConfig):
    """Qwen3 0.6B dims."""

    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128 # different from hidden_size // num_attention_heads
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    vocab_size: int = 151669
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    tie_word_embeddings: bool = True
