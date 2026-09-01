"""BERT backend: the from-scratch ``BertForClassification`` student, seeded from a frozen HF
sequence-classification teacher. Task concerns (data, loss, teacher fine-tune) live in the task.
"""

from __future__ import annotations

from torch import nn
from transformers import AutoModelForSequenceClassification

from franken.config import Config
from franken.models.base import ModelBackend
from franken.models.bert.bert import BertForClassification


class BertBackend(ModelBackend):
    layer_marker = ".encoder.layer."

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

    def forward(self, model: nn.Module, inputs: dict) -> dict:
        out = model(**inputs)
        # Custom student returns a dict; the HF teacher returns a ModelOutput.
        if isinstance(out, dict):
            return {"output": out["logits"], "hidden_states": out["hidden_states"]}
        return {"output": out.logits, "hidden_states": out.hidden_states}

    def ffn_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        return [ly.intermediate.dense for ly in model.bert.encoder.layer]

    def pooler_preact_modules(self, model: nn.Module) -> list[nn.Module]:
        return [model.bert.pooler.dense]

    def activation_ops(self, model: nn.Module) -> list[nn.Module]:
        return [ly.intermediate.intermediate_act_fn for ly in model.bert.encoder.layer]

    def softmax_ops(self, model: nn.Module) -> list[nn.Module]:
        return [ly.attention.self.softmax for ly in model.bert.encoder.layer]
