"""Score a checkpoint. Three suites in one script, because the reading is a comparison across them.

    fidelity   recall@10 + STS-B    did the student track THIS teacher on the selection pool?
    corpus     nDCG@10 + recall@10  quality AND fidelity per training slice, held-out rows
    external   nDCG@10 benchmarks   quality lost in the wild -- the number to quote

`external - corpus` is the coverage gap: a small in-distribution deficit with a large external one
means the fix is data; both large means it is capacity. The corpus suite defaults to `--split test`
because validation is what picked the checkpoint.

    uv run python -m franken.scripts.qwen3.eval \
        --student-ckpt outputs/<run>/student/pytorch_model.bin
    uv run python -m franken.scripts.qwen3.eval --suite corpus --split validation  # selection saw
    uv run python -m franken.scripts.qwen3.eval --config configs/qwen3/depth28_exact.yaml

With no --student-ckpt the student is seeded from the teacher, so at FULL depth every delta reads
~0 -- the self-test. Below full depth it is an untrained truncation and reads ~-100%.
"""

from __future__ import annotations

import json

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


# --------------------------------------------------------------- suites


def _pairs(m: Models, pools: dict[str, Pool], suite: str, shapes=None, emit=report.silent) -> dict:
    """One row per pool, teacher cached. `shapes` says what each task retrieves -- the tasks have
    very different difficulty and the column is what makes that visible."""
    out = {}
    for name, p in pools.items():
        shape = (shapes or {}).get(name, "")
        if not p:
            emit(report.empty_row(name))
            continue
        t = score(m, m.teacher, p, cache=teacher_cache(suite, name, m.cfg))
        s = score(m, m.student, p)
        # As it lands: a pool takes minutes and the run is otherwise silent.
        emit(report.task_row(name, "", len(p.q_ids), len(p.d_ids), report.quality(t, s), shape))
        out[name] = {"teacher": t, "student": s, "queries": len(p.q_ids), "docs": len(p.d_ids)}
        if shape:
            out[name]["retrieves"] = shape
    return out


def _corpus_rows(m, pools, kinds: dict, shapes: dict, ndcg: dict, emit=report.silent) -> dict:
    """Quality AND fidelity per task off one pass of embeddings. nDCG is quality lost; recall and
    embed_dist are how far the geometry moved -- different questions, read as a ratio.

    `ndcg` is `Source.scores_ndcg`: blanked where the gold is arbitrary, since a printed number
    gets read as one."""
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
    """Teacher agreement on the held-out pool, plus STS-B as a labelled anchor. recall@10 is
    already teacher-relative (the teacher scores 1.0), and is the number that chose the
    checkpoint."""
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


def corpus(m: Models, split: str, names: list[str], emit=report.silent) -> dict:
    """nDCG on held-out rows of the training slices, so coverage is out of the equation. Two
    macros, not one: a qrels task's golds are mostly train rows, so only its distractors are held
    out."""
    sources = {s.name: s for s in mix(m.cfg.train.corpus)}
    emit(report.corpus_header(m.cfg.train.corpus, split))
    pools = {n: pool(sources[n], split, m.cfg.train.corpus) for n in names}
    kinds = {n: ("qrels" if sources[n].qrels else "pair") for n in names}
    # A qrels row holds no pair, so the adapter's shape would describe the corpus, not the task.
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
    # Pair tasks only -- the same reason the macros are split.
    pair_rows = [r for r in scored if r["tag"] == "pair"]
    if pair_rows:
        emit(f"\nby domain (pair tasks only, nDCG@{K}):")
        for domain in sorted({r["domain"] for r in pair_rows}):
            group = [r for r in pair_rows if r["domain"] == domain]
            emit(report.domain_row(domain, len(group), *_macro(group)))
    return out


def external(m: Models, names: list[str], emit=report.silent) -> dict:
    """nDCG against ground-truth judgements. NOT comparable to the published MTEB table: task
    subset, the config's own max_seq_len, one generic instruction.

    The macro is EVERY scored task -- a two-task subset once reversed the conclusion outright.
    """
    emit(report.header("external benchmarks", f"nDCG@{K}"))
    rows = _pairs(
        m,
        {n: EXTERNAL[n].pool() for n in names},
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
    # test, not validation: validation is what selected the checkpoint.
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

    # Before the model load, so a typo fails in milliseconds.
    cfg_sources = [s.name for s in mix(common.config(args).train.corpus)]
    names = [n.strip() for n in args.sources.split(",") if n.strip()] or cfg_sources
    if unknown := [n for n in names if n not in cfg_sources]:
        raise SystemExit(f"Not in the mix: {unknown}; have {sorted(cfg_sources)}")

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
