"""MRPC task: GLUE paraphrase classification (the original Franken task).

Owns everything task-specific: the MRPC data (``franken.data.mrpc``), the
distillation loss (``ClassificationDistillLoss``), the checkpoint metric, and the
classification teacher fine-tune (HF Trainer on a sequence-classification head).
The model backend stays task-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from franken.config import Config, DistillConfig
from franken.data.mrpc import compute_metrics, load_mrpc
from franken.distill.loss import layerwise_hidden_loss, masked_mse_loss
from franken.models.base import ModelBackend
from franken.paths import RunPaths
from franken.tasks.base import Task

_COLUMNS = ["input_ids", "token_type_ids", "attention_mask", "label"]


def _hf_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_metrics(np.argmax(logits, axis=-1), labels)


class ClassificationDistillLoss(nn.Module):
    """Classification KD loss: ``(1-alpha)*CE + alpha*T^2*KL + beta*masked_MSE(hidden)``.

    CE and the logit-KL are classification-specific (softmax over classes), so this
    lives with the classification task rather than the generic distill package. The
    hidden-state term reuses the shared ``layerwise_hidden_loss``.
    """

    def __init__(self, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, student_logits, teacher_logits, labels, student_hidden, teacher_hidden, attention_mask
    ):
        ce = F.cross_entropy(student_logits, labels)

        T = self.cfg.temperature
        kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        hidden = layerwise_hidden_loss(
            student_hidden,
            teacher_hidden,
            attention_mask,
            masked_mse_loss,
            self.cfg.hidden_layer_map,
            self.cfg.hidden_layer_mode,
        )

        total = (1 - self.cfg.alpha) * ce + self.cfg.alpha * kl + self.cfg.beta * hidden
        return total, ce, kl, hidden


class MrpcTask(Task):
    def __init__(self):
        self._loss_fn: ClassificationDistillLoss | None = None

    def build_tokenizer(self, cfg: Config) -> Any:
        return AutoTokenizer.from_pretrained(cfg.train.teacher_model)

    def datasets(self, tokenizer: Any, cfg: Config) -> dict:
        return load_mrpc(tokenizer, cfg.train.max_seq_len)

    def torch_columns(self) -> list[str]:
        return list(_COLUMNS)

    def model_inputs(self, batch: dict) -> dict:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }

    def compute_loss(self, student_out, teacher_out, batch, cfg: Config) -> tuple:
        if self._loss_fn is None:
            self._loss_fn = ClassificationDistillLoss(cfg.distill)
        total, ce, kl, hidden = self._loss_fn(
            student_out["output"],
            teacher_out["output"],
            batch["labels"],
            student_out["hidden_states"],
            teacher_out["hidden_states"],
            batch["attention_mask"],
        )
        # Components are logging-only; detach so the trainer can scalar-ize them
        # without dragging the autograd graph (total stays attached for backward).
        return total, {"ce": ce.detach(), "kl": kl.detach(), "hidden": hidden.detach()}

    def select_metric(self) -> tuple[str, bool]:
        return ("f1", True)

    @torch.no_grad()
    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation", teacher=None
    ) -> dict:
        # `teacher` is unused: MRPC scores against labels, not against the teacher.
        data = self.datasets(tokenizer, cfg)
        ds = data[split].with_format("torch", columns=self.torch_columns())
        loader = DataLoader(
            ds, batch_size=cfg.train.distill.batch_size, collate_fn=data["collator"]
        )
        device = next(model.parameters()).device

        model.eval()
        logits, labels = [], []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = backend.forward(model, self.model_inputs(batch))
            logits.append(out["output"].cpu())
            labels.append(batch["labels"].cpu())

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        return compute_metrics(logits.argmax(dim=-1).numpy(), labels.numpy())

    def train_teacher(self, cfg: Config) -> str | None:
        tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
        data = load_mrpc(tok, max_seq_len=cfg.train.max_seq_len)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.train.teacher_model, num_labels=cfg.model.num_labels
        )

        teacher_dir = RunPaths(cfg).teacher
        args = TrainingArguments(
            output_dir=teacher_dir,
            learning_rate=cfg.train.teacher.lr,
            per_device_train_batch_size=cfg.train.teacher.batch_size,
            per_device_eval_batch_size=cfg.train.teacher.batch_size,
            num_train_epochs=cfg.train.teacher.epochs,
            weight_decay=cfg.train.teacher.weight_decay,
            warmup_ratio=cfg.train.teacher.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=cfg.train.seed,
        )
        trainer = Trainer(
            model,
            args,
            train_dataset=data["train"],
            eval_dataset=data["validation"],
            data_collator=data["collator"],
            compute_metrics=_hf_metrics,
        )
        trainer.train()
        trainer.save_model(teacher_dir)
        tok.save_pretrained(teacher_dir)
        return teacher_dir
