"""Shared corpus machinery: read a source, measure it, build an artifact from it.

The mixes are declared per model, in `franken/data/<model>/registry.py`.
"""

from franken.data.corpus.build import cache_missing, load_corpus, train_cache_path
from franken.data.corpus.evalset import Pool, pool
from franken.data.corpus.measure import describe, profile, realized_mix
from franken.data.corpus.read import source_texts
from franken.data.corpus.spec import SPLITS, WEB_SEARCH, instruct

__all__ = [
    "SPLITS",
    "WEB_SEARCH",
    "Pool",
    "cache_missing",
    "describe",
    "instruct",
    "load_corpus",
    "pool",
    "profile",
    "realized_mix",
    "source_texts",
    "train_cache_path",
]
