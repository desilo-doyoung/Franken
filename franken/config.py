"""Configuration schema for Franken.

Experiments are declarative: a single YAML file selects the student depth, the
swappable ops (softmax / GELU) and their kwargs, the distillation loss weights,
and the training hyperparameters. Nothing about the three customizations
(layer reduction, softmax approximation, polynomial GELU) requires code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Student architecture. Width matches the teacher (768); only depth and
    ops change, so hidden-state MSE needs no projection."""

    num_hidden_layers: int = 6
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

    # Swappable ops: a registry name + optional construction kwargs.
    # Resolved via franken.ops.build_softmax / build_gelu.
    softmax: str = "exact"
    softmax_kwargs: dict[str, Any] = field(default_factory=dict)
    gelu: str = "exact"
    gelu_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillConfig:
    """Distillation loss weights and the teacher->student hidden-layer map.

    Loss = (1 - alpha) * CE
         + alpha * T^2 * KL(student/T, teacher/T)
         + beta * masked_MSE(student_hidden, teacher_hidden)
    """

    alpha: float = 0.5
    beta: float = 1.0
    temperature: float = 2.0
    # None -> auto uniform-stride map computed from teacher/student depths.
    hidden_layer_map: list[int] | None = None


@dataclass
class TrainConfig:
    teacher_model: str = "google-bert/bert-base-uncased"
    teacher_ckpt: str | None = None
    output_dir: str = "outputs"
    lr: float = 5e-5
    batch_size: int = 32
    epochs: int = 3
    max_seq_len: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cuda"


@dataclass
class Config:
    """Root config aggregating the three sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        return cls(
            model=_build(ModelConfig, raw.get("model", {})),
            distill=_build(DistillConfig, raw.get("distill", {})),
            train=_build(TrainConfig, raw.get("train", {})),
        )


def _build(dc_type: type, values: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(dc_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")
    return dc_type(**{k: v for k, v in values.items() if k in known})
