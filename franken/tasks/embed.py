"""Embedding self-distillation: match the teacher's pooled embedding, no labels. The backend owns
pooling, so nothing here is Qwen3-specific."""

from __future__ import annotations

from typing import Any

import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from franken.config import Config, DistillConfig
from franken.data.embed_corpus import load_embed_corpus
from franken.distill.batching import plan_batches
from franken.distill.dist import max_tokens_per_rank
from franken.distill.layer_map import resolve_layer_map
from franken.distill.loss import masked_mse_loss, masked_relative_mse_loss
from franken.encode import embed_batches
from franken.metrics import K, recall_at_k
from franken.models.base import ModelBackend
from franken.tasks.base import Task

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
        # Resolved once, not per step.
        if cfg.hidden_loss == "relative":
            self.per_layer = masked_relative_mse_loss
        elif cfg.hidden_loss == "mse":
            self.per_layer = masked_mse_loss
        else:
            raise ValueError(f"Unknown hidden_loss {cfg.hidden_loss!r}; use mse | relative")

    def forward(self, student_emb, teacher_emb, student_hidden, teacher_hidden, attention_mask):
        embed = (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean()

        # hidden_states[0] is the embedding output, so drop it for the layer count.
        layer_map = resolve_layer_map(
            len(teacher_hidden) - 1, len(student_hidden) - 1, self.cfg.hidden_layer_map
        )
        hidden = 0.0
        for s_block, t_block in enumerate(layer_map):
            hidden = hidden + self.per_layer(
                student_hidden[s_block + 1], teacher_hidden[t_block + 1], attention_mask
            )
        hidden = hidden / len(layer_map)

        return embed + self.cfg.beta * hidden, embed, hidden


class EmbedSelfDistillTask(Task):
    def __init__(self):
        self._loss_fn: EmbeddingDistillLoss | None = None

    def build_tokenizer(self, cfg: Config) -> Any:
        tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
        # RoPE is relative, so padding side normally cancels -- EXCEPT under attn_impl
        # "sdpa_causal", which drops the pad mask and needs pads AFTER real tokens.
        tok.padding_side = "right"
        return tok

    def datasets(self, tokenizer: Any, cfg: Config, splits=("train", "validation")) -> dict:
        return load_embed_corpus(
            tokenizer,
            cfg.train.corpus,
            cfg.train.tokens_per_epoch,
            cfg.train.max_seq_len,
            splits=splits,
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
        # Logging only; detached so scalar-izing them drags no autograd graph.
        return total, {"embed": embed.detach(), "hidden": hidden.detach()}

    def select_metric(self) -> tuple[str, bool]:
        # Deliberately NOT what the loss minimizes -- see `recall_at_k`.
        return (_RECALL_KEY, True)

    @torch.no_grad()
    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation", teacher=None
    ) -> dict:
        if teacher is None:
            raise ValueError(
                "EmbedSelfDistillTask.evaluate needs `teacher`: the metric is agreement with "
                "the teacher, so there is nothing to score against without it."
            )
        # Only the scored split: rebuilding the training corpus per epoch costs minutes.
        data = self.datasets(tokenizer, cfg, splits=(split,))
        ds = data[split].with_format("torch", columns=self.torch_columns())
        # Batched like training so one knob bounds memory everywhere; recall@10 is bit-identical
        # across eval batch sizes anyway.
        opt = cfg.train.distill
        if opt.tokens_per_step:
            lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(
                zero_copy_only=False
            )
            # Eval is single-process, so the whole step lands on one device; cap it at what the
            # machine holds rather than assuming the training world size.
            budget = min(opt.tokens_per_step, max_tokens_per_rank())
            plan = plan_batches(lengths, budget, cfg.train.seed)
            loader = DataLoader(ds, batch_sampler=plan, collate_fn=data["collator"])
        else:
            loader = DataLoader(ds, batch_size=opt.batch_size, collate_fn=data["collator"])
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
