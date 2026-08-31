"""The artifact loaded the way training loads it: build -> with_format -> collator -> model_inputs.

A green unit suite is not a usable artifact. `_rows` once dropped `attention_mask` while every test
passed, the gate passed, and the realized-mix table was exact -- and `torch_columns()` could not
load the result. These tests go through the real path for that reason.
"""

import datasets
import pytest
import torch
import transformers

from franken.config import Config
from franken.data.corpus import adapters, build
from franken.data.corpus.source import Source
from franken.tasks.lm import LMDistillTask

CAP = 8


class _Tok:
    """One token per whitespace word, plus the special ids `_rows` and the collator need."""

    eos_token_id = 99
    pad_token_id = 98
    bos_token_id = 97
    name_or_path = "fake"
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"

    def __call__(self, batch, truncation=False, max_length=None):
        cap = max_length if truncation and max_length else 10**9
        return {"input_ids": [[97] + [1] * min(len(t.split()), cap) for t in batch]}


def _artifact(pack: bool):
    src = Source("s0", "d", "r", None, adapters.whole("text"), 1.0)
    req = build._Build(
        name="t",
        sources=[src],
        split="train",
        tokenizer=_Tok(),
        max_seq_len=CAP,
        pack=pack,
        tokens=200,
    )

    def batches(_src, _split, size=1000):
        for _ in range(40):
            yield ["w " * 5]

    return req, batches


@pytest.fixture
def rows(monkeypatch):
    def make(pack):
        req, batches = _artifact(pack)
        monkeypatch.setattr(build, "_batches", batches)
        return datasets.Dataset.from_list(list(build._rows(req)))

    return make


def _cfg(pack: bool) -> Config:
    return Config.from_dict(
        {"train": {"task": "lm", "max_seq_len": CAP, "pack": pack, "tokens_per_epoch": 200}}
    )


@pytest.mark.parametrize("pack", [True, False])
def test_the_artifact_loads_through_torch_columns(rows, pack):
    """`with_format` raises on a column the artifact does not hold, so the two must agree."""
    task = LMDistillTask()
    task._pack, task._eos_id = pack, _Tok.eos_token_id
    ds = rows(pack)

    assert set(task.torch_columns()) <= set(ds.column_names)
    ds.with_format("torch", columns=task.torch_columns())[0]  # the call that used to raise


def test_a_packed_batch_reaches_the_loss_with_an_all_ones_mask(rows):
    """The collator synthesizes the mask the artifact no longer stores. That is an UPSTREAM
    behaviour the packed path now depends on, so pin it against the real tokenizer and the real
    `DataCollatorWithPadding`, not a stand-in that could agree by construction."""
    real = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    task = LMDistillTask()
    task._pack, task._eos_id = True, _Tok.eos_token_id
    ds = rows(True).with_format("torch", columns=task.torch_columns())

    collator = transformers.DataCollatorWithPadding(real)
    batch = collator([ds[i] for i in range(3)])

    assert batch["input_ids"].shape == (3, CAP)
    assert torch.equal(batch["attention_mask"], torch.ones(3, CAP, dtype=torch.long))
    # and the forward still withholds it, or the HF teacher loses document isolation
    assert "attention_mask" not in task.model_inputs(batch)
