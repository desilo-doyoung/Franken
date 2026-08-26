"""What the label-free, corpus-backed tasks share: tokenizer, data, batching, the teacher guard.

`embed` and `lm` differ only in what they compare -- a pooled vector against the teacher's, or the
next-token distribution against it. Everything up to that point is the same.
"""

from __future__ import annotations

from typing import Any

import pyarrow.compute as pc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from franken.config import Config
from franken.data import corpus_sources
from franken.data.corpus import load_corpus
from franken.distill.batching import plan_batches
from franken.distill.dist import max_tokens_per_rank
from franken.distill.packing import doc_positions
from franken.tasks.base import Task

_COLUMNS = ["input_ids", "attention_mask"]


class SelfDistillTask(Task):
    # Set by `datasets`. Paths that never build a corpus (the embed scorer, act_range) keep the
    # defaults and so keep pre-packing behaviour.
    _pack = False
    _eos_id: int | None = None

    def build_tokenizer(self, cfg: Config) -> Any:
        tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
        # RoPE is relative, so padding side normally cancels -- EXCEPT under attn_impl
        # "sdpa_causal", which drops the pad mask and needs pads AFTER real tokens.
        tok.padding_side = "right"
        return tok

    def datasets(self, tokenizer: Any, cfg: Config, splits=("train", "validation")) -> dict:
        self._pack = cfg.train.pack
        self._eos_id = tokenizer.eos_token_id
        return load_corpus(
            tokenizer,
            cfg.train.corpus,
            corpus_sources(cfg.train.corpus),
            cfg.train.tokens_per_epoch,
            cfg.train.max_seq_len,
            splits=splits,
            pack=cfg.train.pack,
        )

    def torch_columns(self) -> list[str]:
        return list(_COLUMNS)

    def model_inputs(self, batch: dict) -> dict:
        if not self._pack:
            # A pad token equal to eos would read right-padding as document starts.
            return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        # The mask is all ones under packing, and sending it would cost the HF teacher its
        # document isolation (masking_utils). The loss still reads it off `batch`.
        return {
            "input_ids": batch["input_ids"],
            "position_ids": doc_positions(batch["input_ids"], self._eos_id),
        }

    def eval_loader(self, tokenizer: Any, cfg: Config, split: str, teacher) -> tuple:
        """The scored split only -- rebuilding the training corpus per epoch costs minutes.
        Batched like training, so one knob bounds memory everywhere."""
        if teacher is None:
            raise ValueError(
                f"{type(self).__name__}.evaluate needs `teacher`: the metric is agreement with "
                "the teacher, so there is nothing to score against without it."
            )
        data = self.datasets(tokenizer, cfg, splits=(split,))
        ds = data[split].with_format("torch", columns=self.torch_columns())
        opt = cfg.train.distill
        if not opt.tokens_per_step:
            return ds, DataLoader(ds, batch_size=opt.batch_size, collate_fn=data["collator"])
        lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(zero_copy_only=False)
        # Eval is single-process, so the whole step lands on one device; cap it at what the
        # machine holds rather than assuming the training world size.
        plan = plan_batches(
            lengths, min(opt.tokens_per_step, max_tokens_per_rank()), cfg.train.seed
        )
        return ds, DataLoader(ds, batch_sampler=plan, collate_fn=data["collator"])
