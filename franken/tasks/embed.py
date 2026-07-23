"""Embedding self-distillation task — STUB (to be implemented).

Objective: distill a pretrained Qwen3-Embedding teacher into an FHE-op student of the
same architecture, matching the teacher's OUTPUT EMBEDDINGS (no labels, no fine-tune).
Switching to a fine-tuned downstream task later is a Task swap only (the backend is unchanged).

Implementation notes (parallels ``franken.tasks.mrpc.MrpcTask``):

- build_tokenizer(cfg): ``AutoTokenizer.from_pretrained(cfg.train.teacher_model)``.
- datasets(tokenizer, cfg): a representative text corpus (wikitext-2 for a smoke test;
    wikitext-103 or a broad-corpus slice for real runs). Consider a config field for the
    corpus id. Tokenize plain text (single field, no sentence pairs).
- torch_columns(): ``["input_ids", "attention_mask"]``  (NO token_type_ids for Qwen3).
- model_inputs(batch): ``{"input_ids": ..., "attention_mask": ...}``.
- compute_loss(student_out, teacher_out, batch, cfg): embedding-match loss
    (cosine distance or MSE between student_out["output"] and teacher_out["output"]) +
    hidden-state MSE reusing ``franken.distill.loss.masked_mse_loss`` +
    ``franken.distill.layer_map.resolve_layer_map``. Return (total, {"embed": ..., "hidden": ...}).
- select_metric(): ``("embed_dist", False)``  (lower distance-to-teacher is better).
- evaluate(...): compute mean embedding distance (and/or cosine sim) to the teacher on the
    validation split. NOTE: needs the teacher too — thread it in when implementing (the
    Distiller has both models), or recompute teacher embeddings here.
- train_teacher(cfg): return None (the pretrained checkpoint is the teacher).
"""

from __future__ import annotations

from typing import Any

from franken.config import Config
from franken.models.base import ModelBackend
from franken.tasks.base import Task

_TODO = "EmbedSelfDistillTask.{} is not implemented yet — see franken/tasks/embed.py docstring."


class EmbedSelfDistillTask(Task):
    def build_tokenizer(self, cfg: Config) -> Any:
        raise NotImplementedError(_TODO.format("build_tokenizer"))

    def datasets(self, tokenizer: Any, cfg: Config) -> dict:
        raise NotImplementedError(_TODO.format("datasets"))

    def torch_columns(self) -> list[str]:
        raise NotImplementedError(_TODO.format("torch_columns"))

    def model_inputs(self, batch: dict) -> dict:
        raise NotImplementedError(_TODO.format("model_inputs"))

    def compute_loss(self, student_out, teacher_out, batch, cfg: Config) -> tuple:
        raise NotImplementedError(_TODO.format("compute_loss"))

    def select_metric(self) -> tuple[str, bool]:
        raise NotImplementedError(_TODO.format("select_metric"))

    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation"
    ) -> dict:
        raise NotImplementedError(_TODO.format("evaluate"))

    # train_teacher inherits the base no-op (pretrained checkpoint is the teacher).
