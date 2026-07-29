"""Embedding self-distillation: match a pretrained embedding teacher, no labels.

The teacher is the pretrained checkpoint (no fine-tune), and the target is its pooled
embedding, so the "task" is agreement with the teacher rather than any downstream
objective. Data is plain text from a corpus preset (``franken.data.embed_corpus``);
the backend owns pooling, so this module never sees hidden-state layout.

Pairs with the Qwen3 backend, but nothing here is Qwen3-specific.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from franken.config import Config, DistillConfig
from franken.data.embed_corpus import load_embed_corpus
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
    """Fraction of each text's top-k teacher neighbours that the student also retrieves.

    This is what an embedding model is actually used through — *relative* similarity — and it
    is why per-vector agreement (``embed_dist`` = 1-cos) is not enough on its own: uniform
    shrinkage of the near/far spread keeps cosine high while destroying the ranking, and a
    global rotation does the reverse. Measured to disagree with ``embed_dist`` repeatedly, so
    this is the selection metric and ``embed_dist`` is logging only.

    Rows must be L2-normed (the backend's pooled output is), so ``x @ x.T`` is cosine.
    Defined here rather than in the eval script so training-time selection and end-of-run
    scoring cannot drift apart.
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
        # Deliberately NOT the quantity the loss minimizes. There are no labels, so "best"
        # means "closest to the teacher", but per-vector closeness (what the loss and
        # `embed_dist` measure) is not what the model is used through — see `recall_at_k`.
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
        data = self.datasets(tokenizer, cfg)
        ds = data[split].with_format("torch", columns=self.torch_columns())
        # The run's batch size, from config — eval needs no separate knob. Dynamic padding does
        # perturb the embeddings across batch compositions, but only at ~5e-7: measured on the
        # depth-19 checkpoint, recall@10 is *bit-identical* at eval batch 8/16/32/64 (0.863000)
        # and only `embed_dist` moves, in its 8th decimal. Selection is therefore unaffected.
        loader = DataLoader(
            ds, batch_size=cfg.train.distill.batch_size, collate_fn=data["collator"]
        )
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
