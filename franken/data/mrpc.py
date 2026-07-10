"""GLUE MRPC loading, tokenization, and metrics.

MRPC is paraphrase detection over sentence pairs (~3.7k train). Metrics are
accuracy + F1, per the GLUE convention.
"""

from typing import Any

import datasets
import transformers
from sklearn.metrics import accuracy_score, f1_score


def load_mrpc(tokenizer: Any, max_seq_len: int = 128) -> dict[str, Any]:
    """Load and tokenize MRPC splits.

    Returns a dict with ``train`` / ``validation`` tokenized datasets and a
    dynamic-padding collator, ready for a DataLoader.
    """

    def tok(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"], truncation=True, max_length=max_seq_len
        )

    ds = datasets.load_dataset("nyu-mll/glue", "mrpc")
    ds = ds.map(tok, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    collator = transformers.DataCollatorWithPadding(tokenizer)

    return {
        "train": ds["train"],
        "validation": ds["validation"],
        "collator": collator,
    }


def compute_metrics(predictions: Any, labels: Any) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }
