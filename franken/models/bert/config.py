"""BERT student architecture config.

The BERT student is built from scratch (no pretrained checkpoint to read dims
from), so every architectural dimension is declared here. Extends the
model-agnostic ``franken.config.ModelConfig`` (backend / depth / ops) with the
BERT-specific widths, vocab, and dropout the from-scratch modules consume.
"""

from __future__ import annotations

from dataclasses import dataclass

from franken.config import ModelConfig


@dataclass
class BertModelConfig(ModelConfig):
    """BERT dims. Width matches the teacher (768) so hidden-state MSE needs no
    projection; only depth (``num_hidden_layers``) and ops change."""

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
