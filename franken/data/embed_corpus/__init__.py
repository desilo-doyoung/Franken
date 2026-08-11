"""Corpora for the label-free embedding self-distillation task.

The teacher supplies the targets, so no labels are needed — the corpus only has to resemble the text
the student will embed. `train.corpus` names a *preset*, not a dataset id, so a mix stays one config
value. Every dataset is declared exactly once, in `registry`, and both the training corpus and the
eval pools derive from that declaration.
"""

from franken.data.embed_corpus.build import (
    SourceProfile,
    SplitStats,
    cache_path,
    describe,
    load_embed_corpus,
    profile,
    realized_mix,
    records,
    source_texts,
)
from franken.data.embed_corpus.evalset import DOCS, QUERIES, Pool, pool
from franken.data.embed_corpus.registry import DOMAINS, MIXES, PRESETS, Qrels, Source, mix
from franken.data.embed_corpus.spec import (
    INSTRUCT,
    SPLIT_PCT,
    SPLITS,
    Record,
    corpus_texts,
    eval_pair,
    split_of,
)

__all__ = [
    "DOCS",
    "DOMAINS",
    "INSTRUCT",
    "MIXES",
    "PRESETS",
    "QUERIES",
    "Pool",
    "Qrels",
    "Record",
    "SPLITS",
    "SPLIT_PCT",
    "Source",
    "SourceProfile",
    "SplitStats",
    "cache_path",
    "describe",
    "corpus_texts",
    "eval_pair",
    "load_embed_corpus",
    "mix",
    "pool",
    "profile",
    "realized_mix",
    "records",
    "source_texts",
    "split_of",
]
