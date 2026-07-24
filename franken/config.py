"""Configuration schema for Franken.

Experiments are declarative: a single YAML file selects the student depth, the
swappable ops (softmax / activation) and their kwargs, the distillation loss
weights, and the training hyperparameters. Nothing about the three
customizations (layer reduction, softmax approximation, polynomial activation)
requires code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model-agnostic student knobs shared by every backend: which backend builds
    the model, the depth-reduction lever, and the swappable ops.

    Backend-specific architecture (widths, vocab, dropout, ...) lives in a
    per-backend subclass under that model's package — e.g.
    ``franken.models.bert.config.BertModelConfig`` and
    ``franken.models.qwen3.config.Qwen3ModelConfig``. Each from-scratch student
    hardcodes its dims as subclass defaults (matching the checkpoint it loads
    weights from); only depth + ops vary per experiment. The concrete subclass is
    chosen by ``backend`` when the config loads (see ``_model_config_cls``)."""

    # Which model backend builds/runs the model (franken.models registry).
    # "bert" = the from-scratch BERT student; "qwen3" = Qwen3-Embedding.
    backend: str = "bert"

    # Depth-reduction lever: number of student layers (< teacher for FHE). Strided
    # teacher->student init fills these from the teacher (see resolve_layer_map).
    num_hidden_layers: int = 6

    # Swappable ops: a registry name + optional construction kwargs.
    # Resolved via franken.ops.build_softmax / build_activation.
    softmax: str = "exact"
    softmax_kwargs: dict[str, Any] = field(default_factory=dict)
    activation: str = "exact"
    activation_kwargs: dict[str, Any] = field(default_factory=dict)


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
    # Squash-penalty weight: keeps FFN pre-activations inside a polynomial
    # activation's valid domain (e.g. cheb_gelu) so the bare poly is FHE-safe at
    # inference. 0 = off; ignored for ops without a bounded domain (e.g. exact).
    range_penalty: float = 0.0


@dataclass
class OptimConfig:
    """Optimization hyperparameters for a single training run.

    Teacher fine-tuning and student distillation each get their own block so
    they can be tuned independently (e.g. a low-lr / high-epoch teacher while
    distillation stays at its separately-tuned bs32/lr5e-5/3ep).
    """

    # Defaults from the original BERT/GLUE papers.
    lr: float = 5e-5
    epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass
class TrainConfig:
    teacher_model: str = "google-bert/bert-base-uncased"
    teacher_ckpt: str | None = None
    output_dir: str = "outputs"
    # Which task drives data/tokenizer/loss/metric/teacher-training (franken.tasks
    # registry). "mrpc" = GLUE MRPC classification; "embed" = embedding self-distill (stub).
    task: str = "mrpc"
    # Output namespace under output_dir: outputs/<run_name or model.backend>/...
    # None -> namespace by the model backend (e.g. outputs/bert/, outputs/qwen3/).
    # Set it to carve a specific experiment its own subtree.
    run_name: str | None = None
    max_seq_len: int = 128
    seed: int = 42
    device: str = "cuda"

    # Per-run optimization blocks (see OptimConfig).
    teacher: OptimConfig = field(default_factory=OptimConfig)
    distill: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class Config:
    """Root config aggregating the three sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        model_raw = raw.get("model", {})
        model_cls = _model_config_cls(model_raw.get("backend", ModelConfig.backend))
        return cls(
            model=_build(model_cls, model_raw),
            distill=_build(DistillConfig, raw.get("distill", {})),
            train=_build_train(raw.get("train", {})),
        )


def _model_config_cls(backend: str) -> type[ModelConfig]:
    """Resolve the per-backend ModelConfig subclass for a ``backend`` name.

    Lazy-imported to avoid a config<->models import cycle (the model packages import
    this module). Each from-scratch backend declares its architecture dims in a
    subclass; a backend with no subclass falls back to the agnostic base."""
    if backend == "bert":
        from franken.models.bert.config import BertModelConfig

        return BertModelConfig
    if backend == "qwen3":
        from franken.models.qwen3.config import Qwen3ModelConfig

        return Qwen3ModelConfig
    return ModelConfig


def _build(dc_type: type, values: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(dc_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")
    return dc_type(**{k: v for k, v in values.items() if k in known})


def _build_train(values: dict[str, Any]) -> TrainConfig:
    """Build TrainConfig, resolving the nested teacher/distill OptimConfig blocks.

    The plain ``_build`` can't descend into nested dataclasses (it would store the
    sub-dicts verbatim), so pop those two blocks, build each as an OptimConfig, then
    assemble TrainConfig from the remaining flat keys.
    """
    values = dict(values)
    teacher = _build(OptimConfig, values.pop("teacher", {}))
    distill = _build(OptimConfig, values.pop("distill", {}))
    return _build(TrainConfig, {**values, "teacher": teacher, "distill": distill})
