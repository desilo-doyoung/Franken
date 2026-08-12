"""Embedding self-distillation: match a pretrained teacher's pooled embedding, no labels, so the
"task" is agreement with the teacher rather than any downstream objective. The backend owns
pooling, so nothing here sees hidden-state layout or is Qwen3-specific.
"""

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
from franken.distill.layer_map import resolve_layer_map
from franken.distill.loss import masked_mse_loss, masked_relative_mse_loss
from franken.models.base import ModelBackend
from franken.tasks.base import Task

_COLUMNS = ["input_ids", "attention_mask"]  # no token_type_ids for Qwen3

# Neighbourhood size for the selection metric. 10 because that is the scale retrieval is
# consumed at; the pool is 500 texts, so recall@10 has 5000 slots and a quantum of 0.0002.
RECALL_K = 10
_RECALL_KEY = f"recall@{RECALL_K}"


def recall_at_k(student: torch.Tensor, teacher: torch.Tensor, k: int = RECALL_K) -> float:
    """Fraction of each text's top-k teacher neighbours the student also retrieves -- *relative*
    similarity, which is how an embedding model is actually used. This is THE selection metric;
    per-vector agreement (``embed_dist`` = 1-cos) is logging only, because uniform shrinkage keeps
    cosine high while destroying the ranking and a global rotation does the reverse. Lives here,
    not in the eval script, so training-time selection and end-of-run scoring cannot drift.

    ⚠️ Comparable only at a FIXED pool size: difficulty is ``k/(n-1)``, so identical per-vector
    damage reads 1.000 at n=11, 0.110 at n=500, 0.039 at n=5000. Rows must be L2-normed.
    """
    ss, st = student @ student.T, teacher @ teacher.T
    # Mask self-similarity, else every row's nearest neighbour is itself and both models
    # agree on it for free.
    eye = torch.eye(ss.size(0), dtype=torch.bool, device=ss.device)
    ss.masked_fill_(eye, float("-inf"))
    st.masked_fill_(eye, float("-inf"))
    top_s, top_t = ss.topk(k, dim=-1).indices, st.topk(k, dim=-1).indices
    hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top_s, top_t, strict=True))
    return hits / (top_s.size(0) * k)


class EmbeddingDistillLoss(nn.Module):
    """``(1 - cos)`` on the pooled embedding + ``beta`` * masked hidden MSE.

    Embedding term is Reimers & Gurevych (2020): they minimize MSE, and on L2-normed vectors
    ``MSE == 2 - 2cos``, so cosine is the same objective on a bounded, readable scale. Hidden term
    is TinyBERT/Patient-KD layerwise matching, which earns its place once the student is
    structurally different (reduced depth); ``beta: 0`` turns it off.

    No CE / logit-KL: no labels and no logits, which is why ``alpha``/``temperature`` go unused.
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
        if self.cfg.hidden_loss == "relative":
            per_layer = masked_relative_mse_loss
        elif self.cfg.hidden_loss == "mse":
            per_layer = masked_mse_loss
        else:
            raise ValueError(f"Unknown hidden_loss {self.cfg.hidden_loss!r}; use mse | relative")

        hidden = 0.0
        for s_block, t_block in enumerate(layer_map):
            hidden = hidden + per_layer(
                student_hidden[s_block + 1], teacher_hidden[t_block + 1], attention_mask
            )
        hidden = hidden / len(layer_map)

        return embed + self.cfg.beta * hidden, embed, hidden


class EmbedSelfDistillTask(Task):
    def __init__(self):
        self._loss_fn: EmbeddingDistillLoss | None = None

    def build_tokenizer(self, cfg: Config) -> Any:
        tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
        # Pinned for consistency, not correctness: RoPE is *relative*, so left padding's constant
        # position shift cancels (verified cosine 1.0 alone vs batched, both models). EXCEPT under
        # attn_impl "sdpa_causal", which drops the pad mask and needs pads AFTER real tokens.
        tok.padding_side = "right"
        return tok

    def datasets(self, tokenizer: Any, cfg: Config, splits=("train", "validation")) -> dict:
        return load_embed_corpus(
            tokenizer,
            cfg.train.corpus,
            cfg.train.corpus_size,
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
        # Components are logging-only; detach so the trainer can scalar-ize them without
        # dragging the autograd graph (total stays attached for backward).
        return total, {"embed": embed.detach(), "hidden": hidden.detach()}

    def select_metric(self) -> tuple[str, bool]:
        # Deliberately NOT what the loss minimizes: per-vector closeness is not how the model is
        # used -- see `recall_at_k`.
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
        # Only the scored split: rebuilding the training corpus per epoch is minutes at 216k.
        data = self.datasets(tokenizer, cfg, splits=(split,))
        ds = data[split].with_format("torch", columns=self.torch_columns())
        # Batched like training, so one knob bounds memory everywhere (a fixed sequence count at
        # max_seq_len 1024 puts the whole 500-row pool in one ~20 GB batch). Safe to differ from
        # training: recall@10 is bit-identical at eval batch 8/16/32/64, only embed_dist moves.
        opt = cfg.train.distill
        if opt.token_budget:
            lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(
                zero_copy_only=False
            )
            plan = plan_batches(lengths, opt.token_budget, opt.max_seqs, cfg.train.seed)
            loader = DataLoader(ds, batch_sampler=plan, collate_fn=data["collator"])
        else:
            loader = DataLoader(ds, batch_size=opt.batch_size, collate_fn=data["collator"])
        device = next(model.parameters()).device

        # Whole-pool embeddings, not a streaming mean: recall@k is a property of the pool's
        # neighbourhood structure, so it cannot be accumulated batch by batch. 500x1024 fp32.
        model.eval()
        student_emb, teacher_emb = [], []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = self.model_inputs(batch)
            student_emb.append(backend.forward(model, inputs)["output"].float().cpu())
            teacher_emb.append(backend.forward(teacher, inputs)["output"].float().cpu())
        student_emb, teacher_emb = torch.cat(student_emb), torch.cat(teacher_emb)

        mean_cos = F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean().item()
        return {
            _RECALL_KEY: recall_at_k(student_emb, teacher_emb),
            "embed_dist": 1.0 - mean_cos,
            "embed_cos": mean_cos,
        }

    # train_teacher inherits the base no-op (pretrained checkpoint is the teacher).
