"""Corpora for the label-free embedding self-distillation task.

The teacher supplies the targets, so no labels are needed — the corpus only has to resemble the text
the student will embed. `train.corpus` names a *preset*, not a dataset id, so a mix stays one config
value. Every dataset is declared exactly once, in `registry`, and both the training corpus and the
eval pools derive from that declaration.
"""

from franken.data.embed_corpus.build import (
    cache_path,
    describe,
    load_embed_corpus,
    profile,
    realized_mix,
    source_texts,
    train_cache_path,
)
from franken.data.embed_corpus.evalset import Pool, pool
from franken.data.embed_corpus.registry import Qrels, Source, mix
from franken.data.embed_corpus.spec import SPLITS, WEB_SEARCH, Record, instruct

__all__ = [
    "SPLITS",
    "WEB_SEARCH",
    "Pool",
    "Qrels",
    "Record",
    "Source",
    "cache_path",
    "describe",
    "instruct",
    "load_embed_corpus",
    "mix",
    "pool",
    "profile",
    "realized_mix",
    "source_texts",
    "train_cache_path",
]
