"""BERT student dims: the student is built from scratch, so nothing reads them off a
checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

from franken.config import ModelConfig


@dataclass
class BertModelConfig(ModelConfig):
    # Width matches the teacher, so hidden-state MSE needs no projection.

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    vocab_size: int = 30522
    type_vocab_size: int = 2
    num_labels: int = 2
    pad_token_id: int = 0
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
