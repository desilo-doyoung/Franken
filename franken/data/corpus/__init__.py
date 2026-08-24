"""Shared corpus machinery: stream a mix, tokenize it, cache it, score retrieval pools over it.

The mixes are declared per model, in `franken/data/<model>/registry.py`.
"""

from franken.data.corpus.build import (
    describe,
    load_corpus,
    profile,
    realized_mix,
    source_texts,
    train_cache_path,
)
from franken.data.corpus.evalset import Pool, pool
from franken.data.corpus.spec import SPLITS, WEB_SEARCH, instruct

__all__ = [
    "SPLITS",
    "WEB_SEARCH",
    "Pool",
    "describe",
    "instruct",
    "load_corpus",
    "pool",
    "profile",
    "realized_mix",
    "source_texts",
    "train_cache_path",
]
