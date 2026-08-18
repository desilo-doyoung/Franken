"""Configuration schema. One YAML picks depth, ops, loss weights and hyperparameters, so none of
the three customizations (layer reduction, softmax approximation, polynomial activation) needs a
code edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import yaml

# One source for the legal values; the code that consumes them imports these.
PRECISIONS = ("fp32", "tf32", "bf16")
HIDDEN_LOSSES = ("mse", "relative")
_TOP_LEVEL = ("model", "distill", "train")


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

    def validate(self) -> Any:
        # Returns the activation op: only an instance says what `domain` resolves to. Building
        # both here is also what rejects a bad op name or a kwarg the op does not take.
        from franken.ops import build_activation, build_softmax

        _build_op(build_softmax, "softmax", self.softmax, self.softmax_kwargs)
        return _build_op(build_activation, "activation", self.activation, self.activation_kwargs)


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
        # `_build` guards keys within a block; a misspelled BLOCK used to be dropped in silence.
        if unknown := set(raw) - set(_TOP_LEVEL):
            raise ValueError(
                f"Unknown top-level config keys: {sorted(unknown)}; expected {list(_TOP_LEVEL)}"
            )
        model_raw = raw.get("model", {})
        model_cls = _model_config_cls(model_raw.get("backend", ModelConfig.backend))
        cfg = cls(
            model=_build(model_cls, model_raw),
            distill=_build(DistillConfig, raw.get("distill", {})),
            train=_build_train(raw.get("train", {})),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Everything checkable without loading a model, so a typo fails in milliseconds."""
        # backend/task membership is left to build_backend/build_task, which raise on the first
        # line of every command: importing those registries here would pull in transformers.
        activation = self.model.validate()

        if self.train.precision not in PRECISIONS:
            raise ValueError(
                f"Unknown train.precision {self.train.precision!r}; use {' | '.join(PRECISIONS)}"
            )
        if self.distill.hidden_loss not in HIDDEN_LOSSES:
            raise ValueError(
                f"Unknown distill.hidden_loss {self.distill.hidden_loss!r}; "
                f"use {' | '.join(HIDDEN_LOSSES)}"
            )

        # Otherwise silent: the trainer skips the penalty when there is no domain, and the run
        # trains unpenalized while looking healthy.
        if self.distill.range_penalty > 0 and getattr(activation, "domain", None) is None:
            raise ValueError(
                f"distill.range_penalty is {self.distill.range_penalty} but activation "
                f"{self.model.activation!r} exposes no domain, so the penalty would do nothing. "
                f"Set activation_kwargs.domain, or set range_penalty to 0."
            )

        depth = self.model.num_hidden_layers
        layers = self.distill.range_penalty_layers
        if layers is not None and (bad := [i for i in layers if not 0 <= i < depth]):
            raise ValueError(
                f"distill.range_penalty_layers {bad} out of range for a {depth}-layer student "
                f"(valid 0..{depth - 1}; STUDENT indices, not teacher's)"
            )
        hidden_map = self.distill.hidden_layer_map
        if hidden_map is not None and len(hidden_map) != depth:
            raise ValueError(
                f"distill.hidden_layer_map has {len(hidden_map)} entries for a {depth}-layer "
                f"student; it names one teacher block per student block."
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


def _build_op(builder, kind: str, name: str, kwargs: dict[str, Any]):
    # An op that takes no `domain` raises TypeError rather than ignoring it; say so plainly.
    try:
        return builder(name, **kwargs)
    except TypeError as e:
        raise ValueError(f"model.{kind}_kwargs {kwargs} rejected by {kind} {name!r}: {e}") from e


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
