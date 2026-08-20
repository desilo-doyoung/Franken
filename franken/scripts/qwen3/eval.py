"""Score a checkpoint. Three suites in one script, because the reading is a comparison across them.

    fidelity   recall@10 + STS-B    did the student track THIS teacher, with no gold needed?
    corpus     nDCG@10 + recall@10  quality AND fidelity per training slice, held-out rows
    external   nDCG@10 benchmarks   quality lost in the wild -- the number to quote

`external - corpus` is the coverage gap: a small in-distribution deficit with a large external one
means the fix is data; both large means it is capacity. The corpus suite defaults to `--split test`
because training scored validation every epoch.

    uv run python -m franken.scripts.qwen3.eval \
        --student-ckpt outputs/<run>/student/pytorch_model.bin
    uv run python -m franken.scripts.qwen3.eval --suite corpus --split validation
    uv run python -m franken.scripts.qwen3.eval --config configs/qwen3/depth28_exact.yaml

With no --student-ckpt the student is seeded from the teacher, so at FULL depth every delta reads
~0 -- the self-test. Below full depth it is an untrained truncation and reads ~-100%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import datasets
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from franken.data.embed_corpus import Pool, mix, pool
from franken.data.external import EXTERNAL
from franken.encode import embed_batches, embed_texts
from franken.metrics import K, ndcg_pool, recall_at_k
from franken.scripts.qwen3 import common, report
from franken.scripts.qwen3.common import Models, embed_pool, load, score, teacher_cache

SUITES = ("fidelity", "corpus", "external")


@dataclass
class Task:
    """One scoring target. Everything a row needs travels together -- the alternative is four
    dicts keyed by the same names, which is what this replaced."""

    name: str
    pool: Pool = field(default_factory=Pool)
    kind: str = ""  # "pair" | "qrels" | "" -- qrels golds are mostly train rows
    shape: str = ""  # what it retrieves; the tasks differ wildly in difficulty
    scores_ndcg: bool = True
    domain: str = ""


# --------------------------------------------------------------- suites


def _pairs(m: Models, tasks: list[Task], suite: str, emit=report.silent) -> dict:
    """One row per pool, teacher cached."""
    out = {}
    for t in tasks:
        if not t.pool:
            emit(report.empty_row(t.name))
            continue
        n_q, n_d = len(t.pool.q_ids), len(t.pool.d_ids)
        teach = score(m, m.teacher, t.pool, cache=teacher_cache(suite, t.name, m.cfg))
        stud = score(m, m.student, t.pool)
        # As it lands: a pool takes minutes and the run is otherwise silent.
        emit(report.task_row(t.name, "", n_q, n_d, report.quality(teach, stud), t.shape))
        out[t.name] = {"teacher": teach, "student": stud, "queries": n_q, "docs": n_d}
        if t.shape:
            out[t.name]["retrieves"] = t.shape
    return out


def _corpus_rows(m: Models, tasks: list[Task], emit=report.silent) -> dict:
    """Quality AND fidelity per task off one pass of embeddings. nDCG is quality lost; recall and
    embed_dist are how far the geometry moved -- different questions, read as a ratio.

    nDCG is blanked where `Source.scores_ndcg` is false, since a printed number gets read as one."""
    out = {}
    for t in tasks:
        if not t.pool:
            emit(report.empty_row(t.name, t.kind))
            continue
        n_q, n_d = len(t.pool.q_ids), len(t.pool.d_ids)
        td, tq = embed_pool(m, m.teacher, t.pool, teacher_cache("corpus", t.name, m.cfg))
        sd, sq = embed_pool(m, m.student, t.pool)
        # The WHOLE doc pool: read across MODELS within a task, so pool size cancels, and
        # truncating would only make recall@K easier (k/(n-1) rises) and less sensitive to damage.
        rec = recall_at_k(sd, td, K)
        dist = 1.0 - F.cosine_similarity(sd, td, dim=-1).mean().item()
        row = {
            "queries": n_q,
            "docs": n_d,
            "tag": t.kind,
            "retrieves": t.shape,
            "domain": t.domain,
            f"recall@{K}": rec,
            "embed_dist": dist,
            "scores_ndcg": t.scores_ndcg,
        }
        if t.scores_ndcg:
            teach, stud = ndcg_pool(t.pool, td, tq), ndcg_pool(t.pool, sd, sq)
            row |= {"teacher": teach, "student": stud}
            cells = report.quality(teach, stud)
        else:
            cells = report.BLANK_QUALITY
        emit(
            report.task_row(t.name, t.kind, n_q, n_d, f"{cells} {rec:>9.4f} {dist:>8.4f}", t.shape)
        )
        out[t.name] = row
    return out


def _macro(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    return sum(r["teacher"] for r in rows) / n, sum(r["student"] for r in rows) / n


def _macro_of(label: str, rows: list[dict], emit) -> dict:
    t, s = _macro(rows)
    emit(report.macro_row(label, t, s, len(rows)))
    return {"teacher": t, "student": s, "n": len(rows)}


# --------------------------------------------------------------- entry points


@torch.no_grad()
def fidelity(m: Models, emit=report.silent) -> dict:
    """Teacher agreement on the held-out pool, plus STS-B as a labelled anchor. recall@10 is
    already teacher-relative (the teacher scores 1.0) and needs no judgements at all -- the only
    quality signal here that no gold set gets a vote in."""
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
        # Spearman is rank-based, so the [0, 5] label scale does not matter.
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


def _corpus_tasks(sources: dict, corpus_name: str, split: str, names: list[str]) -> list[Task]:
    tasks = []
    for n in names:
        src = sources[n]
        tasks.append(
            Task(
                name=n,
                pool=pool(src, split, corpus_name),
                kind="qrels" if src.qrels else "pair",
                # A qrels row holds no pair, so the adapter's shape would describe the corpus.
                shape=(
                    "judged query -> gold passage" if src.qrels else getattr(src.adapt, "shape", "")
                ),
                scores_ndcg=src.scores_ndcg,
                domain=src.domain,
            )
        )
    return tasks


def _corpus_summary(rows: dict, sources: dict, emit) -> dict:
    """Two macros, not one: a qrels task's golds are mostly train rows, so only its distractors
    are held out. Every macro is over scored tasks only."""
    out: dict = {}
    scored = [r for r in rows.values() if r["scores_ndcg"]]
    if blind := [n for n, r in rows.items() if not r["scores_ndcg"]]:
        emit(report.unscored_note(blind, sum(sources[n].weight for n in blind)))
    for kind in ("pair", "qrels"):
        if group := [r for r in scored if r["tag"] == kind]:
            out[f"macro_{kind}"] = _macro_of(f"MACRO-{kind}", group, emit)
    # Pair tasks only -- the same reason the macros are split.
    if pair_rows := [r for r in scored if r["tag"] == "pair"]:
        emit(f"\nby domain (pair tasks only, nDCG@{K}):")
        for domain in sorted({r["domain"] for r in pair_rows}):
            group = [r for r in pair_rows if r["domain"] == domain]
            emit(report.domain_row(domain, len(group), *_macro(group)))
    return out


def corpus(m: Models, split: str, names: list[str], emit=report.silent) -> dict:
    """nDCG on held-out rows of the training slices, so coverage is out of the equation."""
    sources = {s.name: s for s in mix(m.cfg.train.corpus)}
    emit(report.corpus_header(m.cfg.train.corpus, split))
    tasks = _corpus_tasks(sources, m.cfg.train.corpus, split, names)
    rows = _corpus_rows(m, tasks, emit)
    out: dict = {"metric": f"ndcg@{K}", "sources": rows}
    if rows:
        emit("")
        out |= _corpus_summary(rows, sources, emit)
    return out


def external(m: Models, names: list[str], emit=report.silent) -> dict:
    """nDCG against ground-truth judgements. NOT comparable to the published MTEB table: task
    subset, the config's own max_seq_len, one generic instruction.

    The macro is EVERY scored task -- a two-task subset once reversed the conclusion outright.
    """
    emit(report.header("external benchmarks", f"nDCG@{K}"))
    tasks = [Task(n, EXTERNAL[n].pool(), shape="judged query -> gold document") for n in names]
    rows = _pairs(m, tasks, "external", emit)
    out: dict = {"metric": f"ndcg@{K}", "tasks": rows}
    if rows:
        emit("")
        out["macro"] = _macro_of("MACRO", list(rows.values()), emit)
    return out


def _subset(raw: str, available, label: str) -> list[str]:
    """Comma-separated subset of `available`, empty meaning all. Checked before the model load, so
    a typo fails in milliseconds rather than after minutes of embedding."""
    picked = [x.strip() for x in raw.split(",") if x.strip()] or list(available)
    if unknown := [x for x in picked if x not in available]:
        raise SystemExit(f"Unknown {label} {unknown}; available: {', '.join(sorted(available))}")
    return picked


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument(
        "--suite", default=",".join(SUITES), help=f"comma-separated: {', '.join(SUITES)}"
    )
    # test, not validation: training scored validation every epoch.
    p.add_argument("--split", default="test", choices=("validation", "test"))
    p.add_argument("--sources", default="", help="corpus suite: subset of the mix; default all")
    p.add_argument(
        "--tasks", default=",".join(EXTERNAL), help="external suite: subset; default all"
    )
    args = p.parse_args(argv)

    suites = _subset(args.suite, SUITES, "suite(s)")
    tasks = _subset(args.tasks, EXTERNAL, "task(s)")
    names = _subset(args.sources, [s.name for s in mix(common.config(args).train.corpus)], "source")

    datasets.disable_progress_bars()

    def emit(line: str) -> None:
        print(line, flush=True)  # flushed: stdout is block-buffered into a log file

    m = load(args)
    result: dict = {"config": args.config, "student_ckpt": args.student_ckpt, "k": K}
    if "fidelity" in suites:
        result["fidelity"] = fidelity(m, emit)
    if "corpus" in suites:
        result["corpus"] = corpus(m, args.split, names, emit)
    if "external" in suites:
        result["external"] = external(m, tasks, emit)

    # The subtraction the two nDCG suites exist for. Pair-derived: qrels docs are mostly train
    # rows and would flatter the in-distribution side.
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
