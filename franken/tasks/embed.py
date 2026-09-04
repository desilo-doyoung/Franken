"""Embedding self-distillation: match the teacher's pooled embedding, no labels. The backend owns
pooling, so nothing here is Qwen3-specific."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from franken.config import Config, DistillConfig
from franken.distill.loss import PER_LAYER, layerwise_hidden_loss
from franken.encode import embed_batches
from franken.metrics import K, recall_at_k
from franken.models.base import ModelBackend
from franken.tasks.selfdistill import SelfDistillTask

_COLUMNS = ["input_ids", "attention_mask"]  # no token_type_ids for Qwen3

_RECALL_KEY = f"recall@{K}"


class EmbeddingDistillLoss(nn.Module):
    """``(1 - cos)`` on the pooled embedding + ``beta`` * masked hidden MSE (TinyBERT-style
    layerwise matching, which earns its place once the student is structurally different).

    No labels and no logits, so ``alpha``/``temperature`` go unused.
    """

    def __init__(self, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.hidden_loss not in PER_LAYER:
            raise ValueError(
                f"Unknown hidden_loss {cfg.hidden_loss!r}; use {' | '.join(PER_LAYER)}"
            )
        self.per_layer = PER_LAYER[cfg.hidden_loss]

    def forward(self, student_emb, teacher_emb, student_hidden, teacher_hidden, attention_mask):
        embed = (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean()

        hidden = layerwise_hidden_loss(
            student_hidden,
            teacher_hidden,
            attention_mask,
            self.per_layer,
            self.cfg.hidden_layer_map,
            self.cfg.hidden_layer_mode,
        )
        return embed + self.cfg.beta * hidden, embed, hidden


class EmbedSelfDistillTask(SelfDistillTask):
    def __init__(self):
        self._loss_fn: EmbeddingDistillLoss | None = None

    def compute_loss(self, student_out, teacher_out, batch, cfg: Config) -> tuple:
        if self._loss_fn is None:
            self._loss_fn = EmbeddingDistillLoss(cfg.distill)
        total, embed, hidden = self._loss_fn(
            student_out["output"],
            teacher_out["output"],
            student_out["hidden_states"],
            teacher_out["hidden_states"],
            batch["attention_mask"],
        )
        # Logging only; detached so scalar-izing them drags no autograd graph.
        return total, {"embed": embed.detach(), "hidden": hidden.detach()}

    def select_metric(self) -> tuple[str, bool]:
        # Deliberately NOT what the loss minimizes -- see `recall_at_k`.
        return (_RECALL_KEY, True)

    @torch.no_grad()
    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation", teacher=None
    ) -> dict:
        _ds, loader = self.eval_loader(tokenizer, cfg, split, teacher)
        device = next(model.parameters()).device

        # Whole-pool: recall@k is a property of the neighbourhood, not accumulable batch by batch.
        model.eval()
        student_emb, teacher_emb = embed_batches(backend, self, loader, device, model, teacher)

        mean_cos = F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean().item()
        return {
            _RECALL_KEY: recall_at_k(student_emb, teacher_emb),
            "embed_dist": 1.0 - mean_cos,
            "embed_cos": mean_cos,
        }

    # train_teacher inherits the base no-op (pretrained checkpoint is the teacher).
