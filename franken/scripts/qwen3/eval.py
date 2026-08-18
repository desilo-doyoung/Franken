"""Score a checkpoint. Three suites in one script, because the reading is a comparison across them.

    fidelity   recall@10 + STS-B    did the student track THIS teacher on the selection pool?
    corpus     nDCG@10 + recall@10  quality AND fidelity per training slice, held-out rows
    external   nDCG@10 benchmarks   quality lost in the wild -- the number to quote

`external - corpus` is the coverage gap, printed at the bottom, and that subtraction is the point:
a small in-distribution deficit with a large external one means the fix is data; both large means
it is capacity and more data will not help.

Validation selects, test reports: the corpus suite defaults to `--split test` because validation is
what `Distiller.train` scores recall@10 on to pick the checkpoint.

    uv run python -m franken.scripts.qwen3.eval \
        --student-ckpt outputs/<run>/student/pytorch_model.bin
    uv run python -m franken.scripts.qwen3.eval --suite corpus --split validation  # selection saw
    uv run python -m franken.scripts.qwen3.eval --config configs/qwen3/depth28_exact.yaml

With no --student-ckpt the student is seeded from the teacher, so at FULL depth every delta reads
~0 -- the self-test. Below full depth it is an untrained truncation and reads ~-100%.
"""

from __future__ import annotations

import json
from functools import partial

import datasets
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from franken.data.embed_corpus import WEB_SEARCH, Pool, instruct, mix, pool
from franken.encode import embed_batches, embed_texts
from franken.metrics import K, ndcg_pool, recall_at_k
from franken.scripts.qwen3 import common, report
from franken.scripts.qwen3.common import Models, embed_pool, load, score, teacher_cache

SUITES = ("fidelity", "corpus", "external")


# --------------------------------------------------------------- external benchmarks


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
# (loader, instruction). EVERY query is instructed -- that asymmetry (instruction on the query,
# `"document": ""`) is the model's contract, and stripping it measures symmetric similarity instead
# of retrieval. WEB_SEARCH is the default; a task-specific string is kept ONLY where a teacher-only
# sweep measured it beating web by more than the ~0.005 floor -- two of five here, none of the 18
# corpus sources. Deltas in the qwen3 tracker; the sweep script is
# `git log -- franken/scripts/qwen3`.
EXTERNAL = {
    # 3.6k docs, biomedical, GRADED rel (0-2). +0.0110 over web.
    "nfcorpus": (
        partial(_beir, "mteb/nfcorpus"),
        "Given a medical question, retrieve documents that best answer it",
    ),
    # 5.2k docs, claim verification, binary. A claim-specific string measured -0.0024: keep web.
    "scifact": (partial(_beir, "mteb/scifact"), WEB_SEARCH),
    # 58k docs / 1.7k q, informal web prose. A finance-specific string measured +0.0048, on the
    # noise floor: not enough to justify a second string.
    "fiqa": (partial(_beir, "mteb/fiqa"), WEB_SEARCH),
    # 1.7k docs / 824 q, zh = best-covered language. A product-specific string measured -0.0269,
    # the worst candidate in the gate: keep web.
    "xpqa_cmn": (partial(_xpqa, "cmn-cmn"), WEB_SEARCH),
    # Scores the code slice. The web string COSTS 0.0734 here (0.6623 vs 0.7357 bare) -- calling an
    # APPS problem statement a web search query actively misdirects. This recovers 0.0694 of that
    # while keeping the query instructed; bare buys the last 0.0040 by dropping the asymmetry, which
    # is not a trade worth making when the point is to measure retrieval.
    "code_apps": (
        partial(_beir, "CoIR-Retrieval/apps"),
        "Given a programming problem, retrieve the code that solves it",
    ),
}


# --------------------------------------------------------------- suites


def _pairs(m: Models, pools: dict[str, Pool], suite: str, shapes=None, emit=report.silent) -> dict:
    """Score teacher and student over named pools, one row each, teacher cached. `shapes` says what
    a task retrieves -- without it a reader cannot tell that `gooaq` is question->answer while
    `specter` is anchor->cited-title against a hard negative, difficulties that are not comparable.
    """
    out = {}
    for name, p in pools.items():
        shape = (shapes or {}).get(name, "")
        if not p:
            emit(report.empty_row(name))
            continue
        t = score(m, m.teacher, p, cache=teacher_cache(suite, name, m.cfg))
        s = score(m, m.student, p)
        # Emitted as it lands, not at the end: a pool takes minutes and the run is otherwise silent.
        emit(report.task_row(name, "", len(p.q_ids), len(p.d_ids), report.quality(t, s), shape))
        out[name] = {"teacher": t, "student": s, "queries": len(p.q_ids), "docs": len(p.d_ids)}
        if shape:
            out[name]["retrieves"] = shape
    return out


def _corpus_rows(m, pools, kinds: dict, shapes: dict, ndcg: dict, emit=report.silent) -> dict:
    """Quality AND fidelity per task, off one pass of embeddings (free: `score` already embedded
    every pool twice and threw the vectors away). nDCG@K is quality lost; recall@K and embed_dist
    are how far the student's geometry moved ON THAT SLICE. Different questions, read as a ratio,
    so both belong per task rather than pooled.

    `ndcg` is `Source.scores_ndcg`: blank rather than caveated where the gold is arbitrary, because
    a printed number gets read as one."""
    out = {}
    for name, p in pools.items():
        kind, shape = kinds.get(name, ""), shapes.get(name, "")
        if not p:
            emit(report.empty_row(name, kind))
            continue
        td, tq = embed_pool(m, m.teacher, p, teacher_cache("corpus", name, m.cfg))
        sd, sq = embed_pool(m, m.student, p)
        # The WHOLE doc pool: read across MODELS within a task, so pool size cancels, and
        # truncating would only make recall@K easier (k/(n-1) rises) and less sensitive to damage.
        rec = recall_at_k(sd, td, K)
        dist = 1.0 - F.cosine_similarity(sd, td, dim=-1).mean().item()
        row = {
            "queries": len(p.q_ids),
            "docs": len(p.d_ids),
            "tag": kind,
            "retrieves": shape,
            f"recall@{K}": rec,
            "embed_dist": dist,
            "scores_ndcg": ndcg.get(name, True),
        }
        if row["scores_ndcg"]:
            t, s = ndcg_pool(p, td, tq), ndcg_pool(p, sd, sq)
            row |= {"teacher": t, "student": s}
            cells = report.quality(t, s)
        else:
            cells = report.BLANK_QUALITY
        emit(
            report.task_row(
                name, kind, len(p.q_ids), len(p.d_ids), f"{cells} {rec:>9.4f} {dist:>8.4f}", shape
            )
        )
        out[name] = row
    return out


def _macro(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    return sum(r["teacher"] for r in rows) / n, sum(r["student"] for r in rows) / n


def _macro_of(label: str, rows: list[dict], emit) -> dict:
    t, s = _macro(rows)
    emit(report.macro_row(label, t, s, len(rows)))
    return {"teacher": t, "student": s, "n": len(rows)}


@torch.no_grad()
def fidelity(m: Models, emit=report.silent) -> dict:
    """Teacher agreement on the corpus's own held-out pool, plus STS-B as a labelled anchor.

    recall@10 is ALREADY teacher-relative -- feeding it the teacher gives exactly 1.0, so there is
    no teacher column to print. It comes from the same `recall_at_k` that selected the checkpoint,
    so this is the number that chose it; comparable only at a fixed pool size.
    """
    data = m.task.datasets(m.tokenizer, m.cfg, splits=("validation",))
    ds = data["validation"].with_format("torch", columns=m.task.torch_columns())
    loader = DataLoader(ds, batch_size=16, collate_fn=data["collator"])
    s_emb, t_emb = embed_batches(m.backend, m.task, loader, m.device, m.student, m.teacher)

    stsb = {}
    ds_sts = datasets.load_dataset("nyu-mll/glue", "stsb", split="validation")
    for who, model in (("teacher", m.teacher), ("student", m.student)):
        a, b = (
            embed_texts(m.backend, model, m.tokenizer, m.cfg, ds_sts[c], m.device)
            for c in ("sentence1", "sentence2")
        )
        # Label range is [0, 5]; Spearman is rank-based so the scale does not matter.
        stsb[who] = spearmanr(F.cosine_similarity(a, b, dim=-1).numpy(), ds_sts["label"]).statistic

    out = {
        "metric": f"recall@{K} (teacher-neighbour agreement), embed_dist, STS-B spearman",
        "pool": s_emb.size(0),
        f"recall@{K}": recall_at_k(s_emb, t_emb, K),
        "embed_dist": 1.0 - F.cosine_similarity(s_emb, t_emb, dim=-1).mean().item(),
        "stsb_teacher": stsb["teacher"],
        "stsb_student": stsb["student"],
    }
    emit(report.fidelity_block(out))
    return out


def corpus(m: Models, split: str, names: list[str], emit=report.silent) -> dict:
    """nDCG@10 on held-out rows of the training slices, so coverage is out of the equation.

    Two macros, deliberately not one: a qrels task's golds are ~96% likely to be train rows, so
    only its distractors are held out. Averaging that into a clean headline is the CORE mistake.
    """
    sources = {s.name: s for s in mix(m.cfg.train.corpus)}
    emit(report.corpus_header(m.cfg.train.corpus, split))
    pools = {n: pool(sources[n], split, m.cfg.train.corpus) for n in names}
    kinds = {n: ("qrels" if sources[n].qrels else "pair") for n in names}
    # For a qrels source the row holds no pair, so the adapter's shape describes the corpus text
    # rather than the eval task — which is real judged queries run against it.
    shapes = {
        n: (
            "judged query -> gold passage"
            if sources[n].qrels
            else getattr(sources[n].adapt, "shape", "")
        )
        for n in names
    }
    ndcg = {n: sources[n].scores_ndcg for n in names}
    rows = _corpus_rows(m, pools, kinds, shapes, ndcg, emit)
    for n, r in rows.items():
        r["domain"] = sources[n].domain

    out: dict = {"metric": f"ndcg@{K}", "sources": rows}
    if not rows:
        return out
    emit("")
    # Every macro is over scored tasks only -- a suppressed task has no nDCG to average.
    scored = [r for r in rows.values() if r["scores_ndcg"]]
    if blind := [n for n, r in rows.items() if not r["scores_ndcg"]]:
        emit(report.unscored_note(blind, sum(sources[n].weight for n in blind)))
    for kind in ("pair", "qrels"):
        if group := [r for r in scored if r["tag"] == kind]:
            out[f"macro_{kind}"] = _macro_of(f"MACRO-{kind}", group, emit)
    # Pair tasks only. Mixing in qrels tasks would fold ~96%-train-row documents into a domain
    # average, which is the reason the macros are split in the first place.
    pair_rows = [r for r in scored if r["tag"] == "pair"]
    if pair_rows:
        emit(f"\nby domain (pair tasks only, nDCG@{K}):")
        for domain in sorted({r["domain"] for r in pair_rows}):
            group = [r for r in pair_rows if r["domain"] == domain]
            emit(report.domain_row(domain, len(group), *_macro(group)))
    return out


def external(m: Models, names: list[str], emit=report.silent) -> dict:
    """nDCG@10 against ground-truth judgements. NOT comparable to the published MTEB table: task
    subset, the config's own max_seq_len (the FHE condition) vs MTEB's 512, one generic instruction.

    ⚠️ The macro is EVERY scored task. Two tasks ("CORE") read the depth-19 cut at +0.4% where five
    put it at -16.0%, inverting the ratio column too -- it reversed the conclusion, not the value.
    """
    emit(report.header("external benchmarks", f"nDCG@{K}"))
    rows = _pairs(
        m,
        {n: EXTERNAL[n][0](EXTERNAL[n][1]) for n in names},
        "external",
        shapes=dict.fromkeys(names, "judged query -> gold document"),
        emit=emit,
    )
    out: dict = {"metric": f"ndcg@{K}", "tasks": rows}
    if rows:
        emit("")
        out["macro"] = _macro_of("MACRO", list(rows.values()), emit)
    return out


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument(
        "--suite", default=",".join(SUITES), help=f"comma-separated: {', '.join(SUITES)}"
    )
    # test, not validation: validation selects the checkpoint (`Distiller.train` -> recall@10), so
    # reporting there would score the split that picked the model. Selection never reads test.
    p.add_argument("--split", default="test", choices=("validation", "test"))
    p.add_argument("--sources", default="", help="corpus suite: subset of the mix; default all")
    p.add_argument(
        "--tasks", default=",".join(EXTERNAL), help="external suite: subset; default all"
    )
    args = p.parse_args(argv)

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    if unknown := [s for s in suites if s not in SUITES]:
        raise SystemExit(f"Unknown suite(s) {unknown}; available: {', '.join(SUITES)}")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if unknown := [t for t in tasks if t not in EXTERNAL]:
        raise SystemExit(f"Unknown task(s) {unknown}; available: {', '.join(EXTERNAL)}")

    # Resolved before the model load, so a typo fails in milliseconds.
    cfg_sources = [s.name for s in mix(common.config(args).train.corpus)]
    names = [n.strip() for n in args.sources.split(",") if n.strip()] or cfg_sources
    if unknown := [n for n in names if n not in cfg_sources]:
        raise SystemExit(f"Not in the mix: {unknown}; have {sorted(cfg_sources)}")

    datasets.disable_progress_bars()

    def emit(line: str) -> None:
        # Flushed: a suite takes minutes per task and stdout is block-buffered into a log file.
        print(line, flush=True)

    m = load(args)
    result: dict = {"config": args.config, "student_ckpt": args.student_ckpt, "k": K}
    if "fidelity" in suites:
        result["fidelity"] = fidelity(m, emit)
    if "corpus" in suites:
        result["corpus"] = corpus(m, args.split, names, emit)
    if "external" in suites:
        result["external"] = external(m, tasks, emit)

    # The subtraction the two nDCG suites exist for. Pair-derived only: the qrels macro's documents
    # are mostly train rows, so it would flatter the in-distribution side.
    in_dist = result.get("corpus", {}).get("macro_pair")
    off_dist = result.get("external", {}).get("macro")
    if in_dist and off_dist:
        a = report.relative_delta(in_dist["teacher"], in_dist["student"])
        b = report.relative_delta(off_dist["teacher"], off_dist["student"])
        result["coverage_gap"] = b - a
        emit(report.coverage_gap_block(a, b))
    emit("")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
