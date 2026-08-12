"""Retrieval pools over a source's held-out rows.

The split guarantee is inherited rather than restated: `records` assigns a row to exactly one split
as a pure function of its key, so a task cannot silently score trained rows, and a changed key
column moves the eval with it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import datasets

from franken.data.embed_corpus.build import records
from franken.data.embed_corpus.registry import Source
from franken.data.embed_corpus.spec import eval_pair, instruct, split_of

QUERIES, DOCS = 500, 5_000

# Bump alongside an adapter change: pools are text, so a cleaning edit invalidates them.
# v2: SPLIT_PCT 2/2 -> 1/4 moves pool membership; per-source `instruct`.
_CACHE_DIR = "outputs/corpus_pool_cache"
_CACHE_VERSION = 2


@dataclass
class Pool:
    """The (documents, queries, judgements) triple every scorer consumes."""

    d_ids: list[str] = field(default_factory=list)
    d_texts: list[str] = field(default_factory=list)
    q_ids: list[str] = field(default_factory=list)
    q_texts: list[str] = field(default_factory=list)
    qrels: dict[str, dict[str, float]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.q_ids)


class _Docs:
    """Deduplicates by text. An identical twin of a gold document is an exact tie in the ranking, so
    it caps nDCG below 1 and the winner is decided by ~1e-7 of float noise — measured at 108
    duplicates in gooaq's 5,000, enough to move the identity self-test off zero."""

    def __init__(self, limit: int):
        self.limit = limit
        self.ids: list[str] = []
        self.texts: list[str] = []
        self._seen: dict[str, str] = {}

    def add(self, text: str, force: bool = False) -> str | None:
        if text in self._seen:
            return self._seen[text]
        if not force and len(self.ids) >= self.limit:
            return None
        did = f"d{len(self.ids)}"
        self._seen[text] = did
        self.ids.append(did)
        self.texts.append(text)
        return did

    def full(self) -> bool:
        return len(self.ids) >= self.limit


def _from_pairs(src: Source, split: str, n_queries: int, n_docs: int) -> Pool:
    docs = _Docs(n_docs)
    pool = Pool()
    for rec in records(src, split):
        pair = eval_pair(rec)
        if pair is None:
            for text in rec.docs:
                docs.add(text)
        else:
            query, golds, negatives = pair
            if len(pool.q_ids) < n_queries:
                qid = f"q{len(pool.q_ids)}"
                pool.q_ids.append(qid)
                # The source's own instruction, so the eval matches training in format as well as
                # content -- a query the student never saw prefixed must not arrive prefixed here.
                pool.q_texts.append(instruct(src.instruct, query))
                # `force`: a gold must be in the pool even once the doc cap is reached, else the
                # query is unanswerable and scores 0 for a reason that is not the model.
                pool.qrels[qid] = {docs.add(g, force=True): 1.0 for g in golds}
            else:
                for g in golds:
                    docs.add(g)
            for text in (*negatives, *rec.docs):
                docs.add(text)
        if len(pool.q_ids) >= n_queries and docs.full():
            break
    pool.d_ids, pool.d_texts = docs.ids, docs.texts
    return pool


def _from_qrels(src: Source, split: str, n_queries: int, n_docs: int) -> Pool:
    spec = src.qrels
    qid_col, pid_col, score_col = spec.cols

    rel: dict[str, dict[str, float]] = {}
    for row in datasets.load_dataset(spec.repo, split=spec.split):
        if float(row[score_col]) > 0:
            rel.setdefault(str(row[qid_col]), {})[str(row[pid_col])] = float(row[score_col])

    # BeIR and C-MTEB both name the query id column the same as the corpus id column.
    q_cfg, q_split = spec.queries
    q_ids, q_texts = [], []
    for row in datasets.load_dataset(src.repo, q_cfg, split=q_split):
        qid = str(row[src.key])
        if qid in rel and len(q_ids) < n_queries:
            text = row["text"].strip()
            q_ids.append(qid)
            q_texts.append(instruct(src.instruct, text))
    wanted = {pid for qid in q_ids for pid in rel[qid]}

    # Stream the corpus rather than download it whole (BeIR/nq is 2.68M documents), and run the
    # source's own adapter so an eval document is byte-identical to the corpus text.
    docs = _Docs(n_docs)
    found: dict[str, str] = {}
    for row in datasets.load_dataset(src.repo, src.config, split=src.hf_split, streaming=True):
        rec = src.adapt(row)
        if rec is None or not rec.docs:
            continue
        did = str(row[src.key])
        if did in wanted:
            found[did] = docs.add(rec.docs[0], force=True)
        elif not docs.full() and split_of(did) == split:
            docs.add(rec.docs[0])  # distractors from the held-out split only
        if docs.full() and len(found) == len(wanted):
            break

    pool = Pool(d_ids=docs.ids, d_texts=docs.texts)
    for qid, text in zip(q_ids, q_texts, strict=True):
        judged = {found[pid]: score for pid, score in rel[qid].items() if pid in found}
        if judged:  # a query whose gold never appeared is unanswerable, not hard
            pool.q_ids.append(qid)
            pool.q_texts.append(text)
            pool.qrels[qid] = judged
    return pool


def _cache_path(corpus: str, name: str, split: str) -> str:
    return os.path.join(
        _CACHE_DIR, f"v{_CACHE_VERSION}-{corpus}-{name}-{split}-{QUERIES}x{DOCS}.json"
    )


def pool(
    src: Source,
    split: str,
    corpus: str,
    n_queries: int = QUERIES,
    n_docs: int = DOCS,
    cache: bool = True,
) -> Pool:
    """Every source is scoreable, so this always returns a pool — empty only if a source ran out of
    held-out rows, which is a finding, not a configuration."""
    path = _cache_path(corpus, src.name, split)
    if cache and n_queries == QUERIES and n_docs == DOCS and os.path.exists(path):
        with open(path) as f:
            return Pool(**json.load(f))

    built = (
        _from_qrels(src, split, n_queries, n_docs)
        if src.qrels
        else _from_pairs(src, split, n_queries, n_docs)
    )
    if cache and n_queries == QUERIES and n_docs == DOCS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(built.__dict__, f)
    return built
