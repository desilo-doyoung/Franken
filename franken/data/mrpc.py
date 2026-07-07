"""GLUE MRPC loading, tokenization, and metrics (skeleton).

MRPC is paraphrase detection over sentence pairs (~3.7k train). Metrics are
accuracy + F1, per the GLUE convention.
"""

from __future__ import annotations

from typing import Any

TASK = "mrpc"
SENTENCE_KEYS = ("sentence1", "sentence2")
NUM_LABELS = 2


def load_mrpc(tokenizer: Any, max_seq_len: int = 128) -> dict[str, Any]:
    """Load and tokenize MRPC splits.

    Returns a dict with ``train`` / ``validation`` tokenized datasets and a
    dynamic-padding collator, ready for a DataLoader.
    """
    # TODO:
    #   ds = datasets.load_dataset("glue", "mrpc")
    #   tokenize (sentence1, sentence2) with truncation to max_seq_len
    #   collator = transformers.DataCollatorWithPadding(tokenizer)
    #   return {"train": ..., "validation": ..., "collator": collator}
    raise NotImplementedError("load_mrpc is a stub.")


def compute_metrics(predictions: Any, labels: Any) -> dict[str, float]:
    """Return {'accuracy': ..., 'f1': ...} for MRPC."""
    # TODO: sklearn.metrics.accuracy_score / f1_score.
    raise NotImplementedError("compute_metrics is a stub.")
