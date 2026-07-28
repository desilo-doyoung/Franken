"""Embedding self-distillation: match a pretrained embedding teacher, no labels.

The teacher is the pretrained checkpoint (no fine-tune), and the target is its pooled
embedding, so the "task" is agreement with the teacher rather than any downstream
objective. Data is plain text from a corpus preset (``franken.data.embed_corpus``);
the backend owns pooling, so this module never sees hidden-state layout.

Pairs with the Qwen3 backend, but nothing here is Qwen3-specific.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from franken.config import Config, DistillConfig
from franken.data.embed_corpus import load_embed_corpus
from franken.distill.layer_map import resolve_layer_map
from franken.distill.loss import masked_mse_loss
from franken.models.base import ModelBackend
from franken.tasks.base import Task

_COLUMNS = ["input_ids", "attention_mask"]  # no token_type_ids for Qwen3

_TODO = "EmbedSelfDistillTask.{} is not implemented yet."


class EmbeddingDistillLoss(nn.Module):
    """``(1 - cos)`` on the pooled embedding + ``beta`` * masked hidden MSE.

    The embedding term is Reimers & Gurevych (2020) sentence-embedding distillation.
    They minimize MSE between student and teacher embeddings; on L2-normed vectors
    ``MSE == 2 - 2cos``, so cosine is the same objective with a bounded, readable scale.
    The hidden term is TinyBERT / Patient-KD layerwise matching, reusing the shared
    ``masked_mse_loss`` + ``resolve_layer_map`` — it earns its place when the student is
    structurally different from the teacher (reduced depth), and ``beta: 0`` turns it off.

    No CE / logit-KL here: there are no labels and no logits, which is why
    ``DistillConfig.alpha`` and ``temperature`` go unused by this task.
    """

    def __init__(self, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, student_emb, teacher_emb, student_hidden, teacher_hidden, attention_mask):
        embed = (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean()

        # hidden_states[0] is the embedding output, so drop it for the layer count.
        layer_map = resolve_layer_map(
            len(teacher_hidden) - 1, len(student_hidden) - 1, self.cfg.hidden_layer_map
        )
        hidden = 0.0
        for s_block, t_block in enumerate(layer_map):
            hidden = hidden + masked_mse_loss(
                student_hidden[s_block + 1], teacher_hidden[t_block + 1], attention_mask
            )
        hidden = hidden / len(layer_map)

        return embed + self.cfg.beta * hidden, embed, hidden


class EmbedSelfDistillTask(Task):
    def __init__(self):
        self._loss_fn: EmbeddingDistillLoss | None = None

    def build_tokenizer(self, cfg: Config) -> Any:
        tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
        # Right padding — the tokenizer's default, pinned for consistency, NOT correctness.
        # Padding side does not change the embeddings: `model.py` (and HF) build position_ids
        # as a plain arange, so left padding shifts every real token's RoPE position by the
        # pad count — but RoPE is *relative*, attention depends only on position differences,
        # and a constant shift cancels. Verified: same text alone vs batched with a much
        # longer neighbour gives cosine 1.0 under either side, for teacher and student.
        # (The model card recommends left; it makes no difference here.)
        tok.padding_side = "right"
        return tok

    def datasets(self, tokenizer: Any, cfg: Config) -> dict:
        return load_embed_corpus(
            tokenizer, cfg.train.corpus, cfg.train.corpus_size, cfg.train.max_seq_len
        )

    def torch_columns(self) -> list[str]:
        return list(_COLUMNS)

    def model_inputs(self, batch: dict) -> dict:
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

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
        # Components are logging-only; detach so the trainer can scalar-ize them without
        # dragging the autograd graph (total stays attached for backward).
        return total, {"embed": embed.detach(), "hidden": hidden.detach()}

    def select_metric(self) -> tuple[str, bool]:
        raise NotImplementedError(_TODO.format("select_metric"))

    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation"
    ) -> dict:
        raise NotImplementedError(_TODO.format("evaluate"))

    # train_teacher inherits the base no-op (pretrained checkpoint is the teacher).
