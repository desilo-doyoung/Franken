"""GLUE MRPC: paraphrase detection over sentence pairs (~3.7k train), scored on accuracy + F1.

MRPC is the one GLUE task whose `test` split ships public labels, so ask for it by name and it
scores locally like any other.
"""

from typing import Any

import datasets
import transformers
from sklearn.metrics import accuracy_score, f1_score


def load_mrpc(
    tokenizer: Any, max_seq_len: int = 128, splits: tuple[str, ...] = ("train", "validation")
) -> dict[str, Any]:
    def tok(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"], truncation=True, max_length=max_seq_len
        )

    ds = datasets.load_dataset("nyu-mll/glue", "mrpc")
    ds = ds.map(tok, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    out: dict[str, Any] = {s: ds[s] for s in splits}
    out["collator"] = transformers.DataCollatorWithPadding(tokenizer)
    return out


def compute_metrics(predictions: Any, labels: Any) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }
