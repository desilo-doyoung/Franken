"""Franken: a configurable knowledge-distillation framework for HE-friendly BERT.

Distill ``google-bert/bert-base-uncased`` into a student whose internal ops
(softmax, GELU) and depth are swappable via configuration, so the student is
cheaper to evaluate under homomorphic encryption / MPC.
"""

from franken.config import Config, DistillConfig, ModelConfig, TrainConfig

__all__ = ["Config", "ModelConfig", "DistillConfig", "TrainConfig"]

__version__ = "0.1.0"
