"""What the label-free, corpus-backed tasks share: tokenizer, data, batching, the teacher guard.

`embed` and `lm` differ only in what they compare -- a pooled vector against the teacher's, or the
next-token distribution against it. Everything up to that point is the same.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from franken.config import Config
from franken.data import corpus_sources
from franken.data.corpus import load_corpus
from franken.distill.batching import row_plan
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
        # Packed rows are all real, so the artifact stores no mask and `with_format` would raise on
        # a column that is not there; the collator synthesizes the all-ones mask the loss reads.
        return ["input_ids"] if self._pack else list(_COLUMNS)

    def model_inputs(self, batch: dict) -> dict:
        if not self._pack:
            # A pad token equal to eos would read right-padding as document starts.
            return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        # Sending a mask would cost the HF teacher its document isolation (masking_utils). The
        # loss still reads one off `batch` -- the collator's, all ones, since packed rows have no
        # padding at all.
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
        # Eval is single-process, so the whole step lands on one device; cap it at what the
        # machine holds rather than assuming the training world size. Same planner as training, or
        # a packed artifact would be batched two different ways in one run.
        plan = row_plan(
            ds,
            min(opt.tokens_per_step, max_tokens_per_rank()),
            cfg.train.seed,
            cfg.train.max_seq_len if cfg.train.pack else None,
        )
        return ds, DataLoader(ds, batch_sampler=plan, collate_fn=data["collator"])
