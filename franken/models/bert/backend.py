"""BERT backend: the from-scratch BERT student + HF BERT teacher.

Builds the from-scratch ``BertForClassification`` student, seeds it from the HF
teacher via ``init_student_from_teacher``, and loads the frozen HF
sequence-classification teacher for distillation. All BERT-specific model handling
lives here; task concerns (data, loss, teacher fine-tune) live in the task.
"""

from __future__ import annotations

from torch import nn
from transformers import AutoModelForSequenceClassification

from franken.config import Config
from franken.distill.layer_map import resolve_layer_map
from franken.models.base import ModelBackend
from franken.models.bert.bert import BertForClassification
from franken.models.bert.loader import init_student_from_teacher


class BertBackend(ModelBackend):
    def build_student(self, cfg: Config) -> nn.Module:
        return BertForClassification(cfg.model)

    def load_teacher(self, cfg: Config) -> nn.Module:
        # Frozen HF sequence-classification teacher, exact ops, per-layer hidden
        # states enabled so the distillation loss can read them.
        ckpt = cfg.train.teacher_ckpt or cfg.train.teacher_model
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, output_hidden_states=True)
        model.eval()
        model.requires_grad_(False)
        return model

    def seed_student(self, student: nn.Module, teacher: nn.Module, cfg: Config) -> None:
        layer_map = resolve_layer_map(
            teacher.config.num_hidden_layers,
            cfg.model.num_hidden_layers,
            cfg.distill.hidden_layer_map,
        )
        init_student_from_teacher(student, teacher.state_dict(), layer_map)

    def forward(self, model: nn.Module, inputs: dict) -> dict:
        out = model(**inputs)
        # Custom student returns a dict; the HF teacher returns a ModelOutput.
        if isinstance(out, dict):
            return {"output": out["logits"], "hidden_states": out["hidden_states"]}
        return {"output": out.logits, "hidden_states": out.hidden_states}

    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        return [ly.intermediate.dense for ly in model.bert.encoder.layer]

    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        return [ly.intermediate.intermediate_act_fn for ly in model.bert.encoder.layer]
