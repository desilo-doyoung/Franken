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

    # The queries file bundles train/dev/test; keep only judged (test) ones.
    q_ids, q_texts = [], []
    for r in queries:
        if (qid := str(r[id_field])) in qrels:
            q_ids.append(qid)
            q_texts.append(instruct(task, r["text"].strip()))

    # document = title + text, and documents take no instruction prefix. The space join is BEIR's
    # own: `extract_corpus_sentences` is (title + sep + text).strip() with sep=" " — not our choice.
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


# Small on purpose: MS MARCO-scale tasks cost hours per checkpoint, and documents are ~90% of the
# runtime while the statistics live in the query count. All are clean w.r.t. the training corpus,
# which rules out the obvious picks -- the MS MARCO / NQ / HotpotQA benchmarks (the corpus takes
# 27 / 10 / 7% of those very corpora), CoIR's CodeSearchNet and `cosqa` (CodeSearchNet-derived, as
# is the corpus), and MIRACL / Mr.TyDi (Wikipedia-derived).
#
#
# EVERY query is instructed -- that asymmetry (instruction on the query,
# `"document": ""`) is the model's contract, and stripping it measures symmetric similarity instead
# of retrieval. WEB_SEARCH is the default; a task-specific string is kept ONLY where a teacher-only
# sweep measured it beating web by more than the ~0.005 floor -- two of five here, none of the 18
# corpus sources. Deltas in the qwen3 tracker; the sweep script is
# `git log -- franken/scripts/qwen3`.
EXTERNAL: dict[str, Benchmark] = {
    # 3.6k docs, biomedical, GRADED rel (0-2). +0.0110 over web.
    "nfcorpus": Benchmark(
        partial(_beir, "mteb/nfcorpus"),
        "Given a medical question, retrieve documents that best answer it",
    ),
    # 5.2k docs, claim verification, binary. A claim-specific string measured -0.0024: keep web.
    "scifact": Benchmark(partial(_beir, "mteb/scifact"), WEB_SEARCH),
    # 58k docs / 1.7k q, informal web prose. A finance-specific string measured +0.0048, on the
    # noise floor: not enough to justify a second string.
    "fiqa": Benchmark(partial(_beir, "mteb/fiqa"), WEB_SEARCH),
    # 1.7k docs / 824 q, zh = best-covered language. A product-specific string measured -0.0269,
    # the worst candidate in the gate: keep web.
    "xpqa_cmn": Benchmark(partial(_xpqa, "cmn-cmn"), WEB_SEARCH),
    # Scores the code slice. The web string COSTS 0.0734 here (0.6623 vs 0.7357 bare) -- calling an
    # APPS problem statement a web search query actively misdirects. This recovers 0.0694 of that
    # while keeping the query instructed; bare buys the last 0.0040 by dropping the asymmetry, which
    # is not a trade worth making when the point is to measure retrieval.
    "code_apps": Benchmark(
        partial(_beir, "CoIR-Retrieval/apps"),
        "Given a programming problem, retrieve the code that solves it",
    ),
}
