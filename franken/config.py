"""Configuration schema. One YAML picks depth, ops, loss weights and hyperparameters, so none of
the three customizations (layer reduction, softmax approximation, polynomial activation) needs a
code edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Backend-agnostic knobs. Architecture dims live in a per-backend subclass (BertModelConfig,
    Qwen3ModelConfig) picked by ``backend`` at load time; only depth + ops vary per experiment."""

    backend: str = "bert"  # franken.models registry

    # Depth-reduction lever. Strided teacher->student init fills these (see resolve_layer_map).
    num_hidden_layers: int = 6

    # Registry names + construction kwargs; see franken.ops.
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
    # "mse" (raw) | "relative" (per-layer, normalized by the teacher layer's own mean square).
    # Raw MSE lets large-activation layers own the gradient -- see distill.loss for the numbers.
    hidden_loss: str = "mse"
    # Squash-penalty weight keeping FFN pre-activations inside a polynomial activation's domain,
    # so the bare poly is FHE-safe at inference. 0 = off; ignored for ops with no domain.
    range_penalty: float = 0.0
    # Which STUDENT layers the penalty applies to; None = all. Constraining a layer costs accuracy:
    # on Qwen3, 27 of 28 sit inside +-24 while one outlier hits ~300, and penalizing all 28 to fix
    # that one cost 8.2 recall points. Measure with scripts/qwen3/act_range.py first.
    range_penalty_layers: list[int] | None = None


@dataclass
class OptimConfig:
    """One training run's hyperparameters. Teacher and distill get separate blocks so they tune
    independently."""

    # `null` = DERIVE by sqrt-batch scaling from trainer.BASE_LR/BASE_BATCH, using the batch the
    # run actually assembles (token budgeting only knows it at startup). Resolved value is logged.
    lr: float | None = 5e-5
    epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # Padded tokens per batch, sequence count floating to fit; set it and `batch_size` goes unused.
    # ⚠️ Per RANK, so tokens/step and steps/epoch scale with world size -- unlike `batch_size`,
    # which is global and divided by it.
    token_budget: int | None = None
    max_seqs: int = 256  # ceiling on sequences per batch, whatever the budget allows


@dataclass
class TrainConfig:
    teacher_model: str = "google-bert/bert-base-uncased"
    teacher_ckpt: str | None = None
    output_dir: str = "outputs"
    # Drives data/tokenizer/loss/metric/teacher-training (franken.tasks registry).
    task: str = "mrpc"
    # A named preset, not a dataset id, so a mix stays one config value (franken.data.embed_corpus).
    # Ignored by tasks bringing their own data (MRPC).
    corpus: str = "smoke"
    corpus_size: int = 2000
    # Output namespace: outputs/<run_name or model.backend>/... None = namespace by backend.
    run_name: str | None = None
    max_seq_len: int = 128
    seed: int = 42
    device: str = "cuda"
    # Distillation loop only; eval is forced back to fp32. "fp32" | "tf32" | "bf16" (tf32 plus
    # autocast over the student forward + loss). The teacher never enters autocast (trainer).
    precision: str = "fp32"

    # torch.compile student + teacher for training; eval stays eager (see trainer.evaluate).
    compile: bool = False

    teacher: OptimConfig = field(default_factory=OptimConfig)
    distill: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class Config:
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
    # Lazy imports: the model packages import this module, so a top-level import would cycle.
    if backend == "bert":
        from franken.models.bert.config import BertModelConfig

        return BertModelConfig
    if backend == "qwen3":
        from franken.models.qwen3.config import Qwen3ModelConfig

        return Qwen3ModelConfig
    return ModelConfig


def _build(dc_type: type, values: dict[str, Any]):
    known = {f.name for f in fields(dc_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")
    return dc_type(**{k: v for k, v in values.items() if k in known})


def _build_train(values: dict[str, Any]) -> TrainConfig:
    # `_build` can't descend into nested dataclasses, so build the two OptimConfig blocks first.
    values = dict(values)
    teacher = _build(OptimConfig, values.pop("teacher", {}))
    distill = _build(OptimConfig, values.pop("distill", {}))
    return _build(TrainConfig, {**values, "teacher": teacher, "distill": distill})
