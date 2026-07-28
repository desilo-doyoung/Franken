"""Embedding self-distillation: match a pretrained embedding teacher, no labels.

The teacher is the pretrained checkpoint (no fine-tune), and the target is its pooled
embedding, so the "task" is agreement with the teacher rather than any downstream
objective. Data is plain text from a corpus preset (``franken.data.embed_corpus``);
the backend owns pooling, so this module never sees hidden-state layout.

Pairs with the Qwen3 backend, but nothing here is Qwen3-specific.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from franken.config import Config
from franken.data.embed_corpus import load_embed_corpus
from franken.models.base import ModelBackend
from franken.tasks.base import Task

_COLUMNS = ["input_ids", "attention_mask"]  # no token_type_ids for Qwen3

_TODO = "EmbedSelfDistillTask.{} is not implemented yet."


class EmbedSelfDistillTask(Task):
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
        raise NotImplementedError(_TODO.format("compute_loss"))

    def select_metric(self) -> tuple[str, bool]:
        raise NotImplementedError(_TODO.format("select_metric"))

    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation"
    ) -> dict:
        raise NotImplementedError(_TODO.format("evaluate"))

    # train_teacher inherits the base no-op (pretrained checkpoint is the teacher).
