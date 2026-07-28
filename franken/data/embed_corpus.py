"""Corpora for the label-free embedding self-distillation task.

The teacher supplies the targets, so no labels are needed — the corpus only has to
resemble the text the student will embed. ``train.corpus`` names a *preset* (a recipe)
rather than a dataset id, so when the corpus becomes a mix it stays one config value
instead of a list of ids plus weights.

Texts are kept as **natural units** (a paragraph, later a query) rather than chunked into
fixed-length blocks: an embedding model is deployed on whole passages, so blocks that start
and end mid-sentence would be off-distribution. Each preset owns its own cleaning — what
counts as junk is a property of the source, not a general rule.
"""

from typing import Any

import datasets
import transformers


def _wikitext(config_name: str):
    """Wikipedia paragraphs. Drops wikitext's blank lines and " = = Heading = = " rows,
    which ship as records of their own and are junk to embed."""
    min_chars = 32

    def is_paragraph(example: dict) -> bool:
        text = example["text"].strip()
        return len(text) >= min_chars and not text.startswith("=")

    def build(split: str, n: int):
        ds = datasets.load_dataset("Salesforce/wikitext", config_name, split=split)
        ds = ds.filter(is_paragraph)
        return ds.select(range(min(n, len(ds))))

    return build


# name -> (split, n) -> dataset with a "text" column
CORPORA = {
    "smoke": _wikitext("wikitext-2-raw-v1"),
}


def load_embed_corpus(
    tokenizer: Any, name: str, size: int, max_seq_len: int = 128, val_size: int = 500
) -> dict[str, Any]:
    """Load and tokenize an embedding corpus preset.

    Returns ``train`` / ``validation`` tokenized datasets and a dynamic-padding collator —
    the same shape ``franken.data.mrpc.load_mrpc`` returns.
    """
    if name not in CORPORA:
        raise KeyError(f"Unknown corpus {name!r}; available: {sorted(CORPORA)}")
    build = CORPORA[name]

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    splits = {"train": build("train", size), "validation": build("validation", val_size)}
    splits = {
        k: v.map(tok, batched=True, remove_columns=v.column_names) for k, v in splits.items()
    }

    return {
        "train": splits["train"],
        "validation": splits["validation"],
        "collator": transformers.DataCollatorWithPadding(tokenizer),
    }
