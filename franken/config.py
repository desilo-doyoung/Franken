"""Configuration schema: one YAML picks depth, ops, loss weights and hyperparameters."""

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
    """Backend-agnostic knobs; architecture dims live in the per-backend subclass."""

    backend: str = "bert"  # franken.models registry

    num_hidden_layers: int = 6  # strided teacher->student init fills these

    # Registry names + ctor kwargs; see franken.ops.
    softmax: str = "exact"
    softmax_kwargs: dict[str, Any] = field(default_factory=dict)
    activation: str = "exact"
    activation_kwargs: dict[str, Any] = field(default_factory=dict)
    # Only the pooled architectures have one; decoders leave it unused.
    pooler: str = "exact"
    pooler_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Any:
        # Returns the activation op: only an instance says what `domain` resolves to. Building both
        # is also what rejects a bad op name or an unsupported kwarg.
        from franken.ops import build_activation, build_pooler, build_softmax

        _build_op(build_softmax, "softmax", self.softmax, self.softmax_kwargs)
        _build_op(build_pooler, "pooler", self.pooler, self.pooler_kwargs)
        return _build_op(build_activation, "activation", self.activation, self.activation_kwargs)


@dataclass
class DistillConfig:
    """Loss = (1-alpha)*CE + alpha*T^2*KL(student/T, teacher/T) + beta*masked_MSE(hidden).

    That shape is `mrpc`'s. The label-free tasks drop the CE, so `alpha` stops trading against
    anything and becomes a plain weight: `embed` ignores alpha/temperature entirely, and `lm`
    weights its vocabulary KL with them. Set alpha explicitly on those -- the 0.5 default silently
    halves the only term that has a target.
    """

    alpha: float = 0.5
    beta: float = 1.0
    temperature: float = 2.0
    hidden_layer_map: list[int] | None = None  # None -> resolved by hidden_layer_mode
    # Rule used when hidden_layer_map is None. `stride` is the historical default.
    hidden_layer_mode: str = "stride"
    hidden_loss: str = "mse"  # see distill.loss
    # Keeps FFN pre-activations inside a polynomial activation's domain, so the bare poly is
    # FHE-safe at inference. 0 = off.
    range_penalty: float = 0.0
    # STUDENT layers the penalty applies to; None = all. Constraining a layer costs accuracy --
    # penalizing all 28 to fix one Qwen3 outlier cost 8.2 recall points. Measure with act_range.py.
    range_penalty_layers: list[int] | None = None
    # Same mechanism for the pooler pre-activation; its domain is model.pooler_kwargs.domain.
    pooler_penalty: float = 0.0


@dataclass
class OptimConfig:
    """One run's hyperparameters; teacher and distill tune independently."""

    lr: float | None = 5e-5  # null = derive by sqrt-batch scaling (see trainer.resolve_lr)
    epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # GLOBAL padded tokens per optimizer step, sequence count floating to fit; supersedes
    # `batch_size`. Machine-independent: the per-GPU slice and any gradient accumulation are derived
    # from world size and dist.max_tokens_per_rank(). Sets the step count --
    # steps = tokens_per_epoch * epochs / tokens_per_step.
    tokens_per_step: int | None = None


@dataclass
class TrainConfig:
    teacher_model: str = "google-bert/bert-base-uncased"
    teacher_ckpt: str | None = None
    output_dir: str = "outputs"
    task: str = "mrpc"  # franken.tasks registry
    # A named preset, not a dataset id, so a mix stays one config value. Ignored by MRPC.
    corpus: str = "smoke"
    # UNIQUE tokens per pass; total passes = this * distill.epochs, so `epochs` no longer resizes
    # the corpus. Nominal -- it names a corpus; the build reports the realized count.
    tokens_per_epoch: float | None = None
    run_name: str | None = None  # output namespace; None = the backend name
    max_seq_len: int = 128
    # Concatenate documents into whole max_seq_len blocks instead of truncating. Opt-in: it is
    # a different corpus under a different cache key, never a reinterpretation of an old build.
    pack: bool = False
    seed: int = 42
    device: str = "cuda"
    # Distillation loop only; eval is forced back to fp32 and the teacher never enters autocast.
    precision: str = "fp32"
    compile: bool = False  # training only; eval stays eager
    # Recompute each layer's activations in backward instead of storing them: 7x less memory per
    # token for ~16% wall clock, which is what lets tokens_per_step hold several blocks.
    grad_checkpoint: bool = False

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
        # backend/task membership is left to build_backend/build_task: importing those registries
        # here would pull in transformers.
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
        if getattr(self.model, "attn_impl", None) == "flex" and not self.train.pack:
            raise ValueError(
                "attn_impl 'flex' needs train.pack: its block mask is causal + same-document, "
                "and the padding an unpacked corpus needs is deliberately not implemented."
            )

        # Otherwise silent: the trainer skips the penalty when there is no domain, and the run
        # trains unpenalized while looking healthy.
        if self.distill.range_penalty > 0 and getattr(activation, "domain", None) is None:
            raise ValueError(
                f"distill.range_penalty is {self.distill.range_penalty} but activation "
                f"{self.model.activation!r} exposes no domain, so the penalty would do nothing. "
                f"Set activation_kwargs.domain, or set range_penalty to 0."
            )

        if self.distill.pooler_penalty > 0 and self.model.pooler_kwargs.get("domain") is None:
            raise ValueError(
                f"distill.pooler_penalty is {self.distill.pooler_penalty} but "
                f"model.pooler_kwargs exposes no domain, so the penalty would do nothing. "
                f"Set model.pooler_kwargs.domain, or set pooler_penalty to 0."
            )

        # MRPC brings its own data; a corpus task must say how much to draw.
        if self.train.task != "mrpc":
            if self.train.tokens_per_epoch is None:
                raise ValueError(
                    f"train.task {self.train.task!r} needs train.tokens_per_epoch: unique tokens "
                    "in one pass over the corpus."
                )
            # YAML 1.1 reads `1.0e9` as a STRING -- only a signed exponent is a float. Caught here
            # because the alternative is a TypeError an hour later, after the corpus gate.
            if isinstance(self.train.tokens_per_epoch, str):
                raise ValueError(
                    f"train.tokens_per_epoch {self.train.tokens_per_epoch!r} parsed as a string: "
                    "YAML needs `1_000_000_000` or `1.0e+9`, not `1.0e9`."
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
    # Lazy: the model packages import this module.
    if backend == "bert":
        from franken.models.bert.config import BertModelConfig

        return BertModelConfig
    if backend == "qwen3":
        from franken.models.qwen3.config import Qwen3ModelConfig

        return Qwen3ModelConfig
    if backend == "llama":
        from franken.models.llama.config import LlamaModelConfig

        return LlamaModelConfig
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
    # `_build` cannot descend into nested dataclasses.
    values = dict(values)
    teacher = _build(OptimConfig, values.pop("teacher", {}))
    distill = _build(OptimConfig, values.pop("distill", {}))
    return _build(TrainConfig, {**values, "teacher": teacher, "distill": distill})
