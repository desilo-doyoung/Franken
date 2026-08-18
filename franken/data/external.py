"""External retrieval benchmarks. Not `embed_corpus.Source`: that is a training slice carrying a
weight, a split hash and an adapter, while these are pure eval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import datasets

from franken.data.embed_corpus import WEB_SEARCH, Pool, instruct


@dataclass(frozen=True)
class Benchmark:
    """A judged retrieval task and the query instruction it is scored with."""

    load: Callable[[str], Pool]
    instruct: str

    def pool(self) -> Pool:
        return self.load(self.instruct)


def _assemble(corpus, queries, qrels_rows, id_field: str, task: str) -> Pool:
    qrels: dict[str, dict[str, float]] = {}
    for r in qrels_rows:
        rel = float(r["score"])  # XPQA stores it as a string
        if rel > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = rel

    # The queries file bundles train/dev/test; keep only judged ones.
    q_ids, q_texts = [], []
    for r in queries:
        if (qid := str(r[id_field])) in qrels:
            q_ids.append(qid)
            q_texts.append(instruct(task, r["text"].strip()))

    # Documents take no instruction prefix. The space join is BEIR's own, not our choice.
    d_ids = [str(x) for x in corpus[id_field]]
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(d_ids)
    d_texts = [f"{t} {x}".strip() for t, x in zip(titles, corpus["text"], strict=True)]
    return Pool(d_ids=d_ids, d_texts=d_texts, q_ids=q_ids, q_texts=q_texts, qrels=qrels)


def _beir(repo: str, task: str) -> Pool:
    """BEIR/MTEB layout: corpus(_id,title,text), queries(_id,text), qrels in "default"/"test"."""
    return _assemble(
        datasets.load_dataset(repo, "corpus", split="corpus"),
        datasets.load_dataset(repo, "queries", split="queries"),
        datasets.load_dataset(repo, "default", split="test"),
        "_id",
        task,
    )


def _xpqa(pair: str, task: str) -> Pool:
    """XPQA layout: one config per language pair, everything in split "test", `id` not `_id`."""
    repo = "mteb/XPQARetrieval"
    return _assemble(
        datasets.load_dataset(repo, f"{pair}-corpus", split="test"),
        datasets.load_dataset(repo, f"{pair}-queries", split="test"),
        datasets.load_dataset(repo, f"{pair}-qrels", split="test"),
        "id",
        task,
    )


# Small on purpose, and all clean w.r.t. the training corpus -- which rules out MS MARCO / NQ /
# HotpotQA, CoIR's CodeSearchNet and MIRACL / Mr.TyDi. Every query is instructed, with WEB_SEARCH
# unless a sweep measured a task string beating it by more than the ~0.005 floor.
EXTERNAL: dict[str, Benchmark] = {
    # biomedical, GRADED rel (0-2); +0.0110 over web
    "nfcorpus": Benchmark(
        partial(_beir, "mteb/nfcorpus"),
        "Given a medical question, retrieve documents that best answer it",
    ),
    # claim verification; a claim-specific string measured -0.0024
    "scifact": Benchmark(partial(_beir, "mteb/scifact"), WEB_SEARCH),
    # informal web prose; a finance string measured +0.0048, on the noise floor
    "fiqa": Benchmark(partial(_beir, "mteb/fiqa"), WEB_SEARCH),
    # zh, the best-covered language; a product string measured -0.0269, the worst candidate
    "xpqa_cmn": Benchmark(partial(_xpqa, "cmn-cmn"), WEB_SEARCH),
    # The code slice. The web string COSTS 0.0734 here -- calling an APPS problem a web query
    # misdirects; this recovers 0.0694 of it while keeping the query instructed.
    "code_apps": Benchmark(
        partial(_beir, "CoIR-Retrieval/apps"),
        "Given a programming problem, retrieve the code that solves it",
    ),
}
