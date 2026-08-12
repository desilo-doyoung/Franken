"""Score a checkpoint. Three suites in one script, because the reading is a comparison across them.

    fidelity   recall@10 + STS-B    did the student track THIS teacher on the selection pool?
    corpus     nDCG@10 + recall@10  quality AND fidelity per training slice, held-out rows
    external   nDCG@10 benchmarks   quality lost in the wild -- the number to quote

`external - corpus` is the coverage gap, printed at the bottom, and that subtraction is the point:
a small in-distribution deficit with a large external one means the fix is data; both large means
it is capacity and more data will not help.

    uv run python scripts/qwen3/eval.py --student-ckpt outputs/<run>/student/pytorch_model.bin
    uv run python scripts/qwen3/eval.py --suite corpus --split test        # touched once
    uv run python scripts/qwen3/eval.py --config configs/qwen3/depth28_multi_domain.yaml

With no --student-ckpt the student is seeded from the teacher, so at FULL depth every delta reads
~0 -- the self-test. Below full depth it is an untrained truncation and reads ~-100%.
"""

from __future__ import annotations

import json
from functools import partial

import common
import datasets
import torch
import torch.nn.functional as F
from common import K, Models, _embed_texts, embed_pool, load, ndcg_pool, score, teacher_cache
from franken.data.embed_corpus import INSTRUCT, Pool, mix, pool
from franken.tasks.embed import recall_at_k
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

SUITES = ("fidelity", "corpus", "external")


# --------------------------------------------------------------- external benchmarks


def _assemble(corpus, queries, qrels_rows, id_field: str) -> Pool:
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
            q_texts.append(INSTRUCT.format(r["text"].strip()))

    # document = title + text, and documents take no instruction prefix.
    d_ids = [str(x) for x in corpus[id_field]]
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(d_ids)
    d_texts = [f"{t} {x}".strip() for t, x in zip(titles, corpus["text"], strict=True)]
    return Pool(d_ids=d_ids, d_texts=d_texts, q_ids=q_ids, q_texts=q_texts, qrels=qrels)


def _beir(repo: str) -> Pool:
    """BEIR/MTEB layout: corpus(_id,title,text), queries(_id,text), qrels in "default"/"test"."""
    return _assemble(
        datasets.load_dataset(repo, "corpus", split="corpus"),
        datasets.load_dataset(repo, "queries", split="queries"),
        datasets.load_dataset(repo, "default", split="test"),
        "_id",
    )


def _xpqa(pair: str) -> Pool:
    """XPQA layout: one config per language pair, everything in split "test", `id` not `_id`."""
    repo = "mteb/XPQARetrieval"
    return _assemble(
        datasets.load_dataset(repo, f"{pair}-corpus", split="test"),
        datasets.load_dataset(repo, f"{pair}-queries", split="test"),
        datasets.load_dataset(repo, f"{pair}-qrels", split="test"),
        "id",
    )


# Small on purpose: MS MARCO-scale tasks cost hours per checkpoint, and documents are ~90% of the
# runtime while the statistics live in the query count. All are clean w.r.t. the training corpus,
# which rules out the obvious picks -- the MS MARCO / NQ / HotpotQA benchmarks (the corpus takes
# 27 / 10 / 7% of those very corpora), CoIR's CodeSearchNet and `cosqa` (CodeSearchNet-derived, as
# is the corpus), and MIRACL / Mr.TyDi (Wikipedia-derived).
EXTERNAL = {
    "nfcorpus": partial(_beir, "mteb/nfcorpus"),  # 3.6k docs, biomedical, GRADED rel (0-2)
    "scifact": partial(_beir, "mteb/scifact"),  # 5.2k docs, claim verification, binary
    "fiqa": partial(_beir, "mteb/fiqa"),  # 58k docs / 1.7k q, informal web prose
    "xpqa_cmn": partial(_xpqa, "cmn-cmn"),  # 1.7k docs / 824 q, zh = best-covered language
    # Scores the code slice. It read 0.0797 at max_seq_len 128 purely because 92.8% of its QUERIES
    # overflowed; at 1024 they fit. Confirm the teacher clears that floor before trusting it.
    "code_apps": partial(_beir, "CoIR-Retrieval/apps"),
}


# --------------------------------------------------------------- suites


def _pairs(m: Models, pools: dict[str, Pool], suite: str, shapes: dict | None = None) -> dict:
    """Score teacher and student over named pools, one row each, teacher cached. `shapes` says what
    a task retrieves -- without it a reader cannot tell that `gooaq` is question->answer while
    `specter` is anchor->cited-title against a hard negative, difficulties that are not comparable.
    """
    out = {}
    for name, p in pools.items():
        shape = (shapes or {}).get(name, "")
        if not p:
            print(f"{name:>18} {'':>6}   no queries", flush=True)
            continue
        t = score(m, m.teacher, p, cache=teacher_cache(suite, name, m.cfg))
        s = score(m, m.student, p)
        print(
            f"{name:>18} {'':>6} {len(p.q_ids):>5} {len(p.d_ids):>6} "
            f"{t:>9.4f} {s:>9.4f} {s - t:>+9.4f} {100 * (s - t) / t if t else 0:>7.1f}%"
            f"   {shape}",
            flush=True,
        )
        out[name] = {"teacher": t, "student": s, "queries": len(p.q_ids), "docs": len(p.d_ids)}
        if shape:
            out[name]["retrieves"] = shape
    return out


def _corpus_rows(m: Models, pools: dict[str, Pool], kinds: dict, shapes: dict) -> dict:
    """Quality AND fidelity per task, off one pass of embeddings (free: `score` already embedded
    every pool twice and threw the vectors away). nDCG@K is quality lost; recall@K and embed_dist
    are how far the student's geometry moved ON THAT SLICE. Different questions, read as a ratio,
    so both belong per task rather than pooled."""
    out = {}
    for name, p in pools.items():
        kind, shape = kinds.get(name, ""), shapes.get(name, "")
        if not p:
            print(f"{name:>18} {kind:>6}   no queries", flush=True)
            continue
        td, tq = embed_pool(m, m.teacher, p, teacher_cache("corpus", name, m.cfg))
        sd, sq = embed_pool(m, m.student, p)
        t, s = ndcg_pool(p, td, tq), ndcg_pool(p, sd, sq)
        # The WHOLE doc pool: read across MODELS within a task, so pool size cancels, and
        # truncating would only make recall@K easier (k/(n-1) rises) and less sensitive to damage.
        rec = recall_at_k(sd, td, K)
        dist = 1.0 - F.cosine_similarity(sd, td, dim=-1).mean().item()
        print(
            f"{name:>18} {kind:>6} {len(p.q_ids):>5} {len(p.d_ids):>6} "
            f"{t:>9.4f} {s:>9.4f} {s - t:>+9.4f} {100 * (s - t) / t if t else 0:>7.1f}% "
            f"{rec:>9.4f} {dist:>8.4f}   {shape}",
            flush=True,
        )
        out[name] = {
            "teacher": t,
            "student": s,
            "queries": len(p.q_ids),
            "docs": len(p.d_ids),
            "tag": kind,
            "retrieves": shape,
            f"recall@{K}": rec,
            "embed_dist": dist,
        }
    return out


def _header(what: str, metric: str) -> None:
    # The metric always goes in the header: `recall@10` here means teacher-neighbour agreement,
    # MTEB's means something else, so an unlabelled column is a number waiting to be misread.
    print(
        f"\n== {what} -- {metric} ==\n{'task':>18} {'kind':>6} {'q':>5} {'docs':>6} "
        f"{'teacher':>9} {'student':>9} {'delta':>9} {'rel':>8}   retrieves",
        flush=True,
    )


def _macro(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    return sum(r["teacher"] for r in rows) / n, sum(r["student"] for r in rows) / n


def _print_macro(label: str, rows: list[dict]) -> dict:
    t, s = _macro(rows)
    print(
        f"{f'{label}({len(rows)})':>18} {'':>14} {'':>5} {'':>6} "
        f"{t:>9.4f} {s:>9.4f} {s - t:>+9.4f} {100 * (s - t) / t:>7.1f}%"
    )
    return {"teacher": t, "student": s, "n": len(rows)}


def fidelity(m: Models) -> dict:
    """Teacher agreement on the corpus's own held-out pool, plus STS-B as a labelled anchor.

    recall@10 is ALREADY teacher-relative -- feeding it the teacher gives exactly 1.0, so there is
    no teacher column to print. It comes from the same `recall_at_k` that selected the checkpoint,
    so this is the number that chose it; comparable only at a fixed pool size.
    """
    data = m.task.datasets(m.tokenizer, m.cfg, splits=("validation",))
    ds = data["validation"].with_format("torch", columns=m.task.torch_columns())
    embs = {"student": [], "teacher": []}
    for batch in DataLoader(ds, batch_size=16, collate_fn=data["collator"]):
        batch = {k: v.to(m.device) for k, v in batch.items()}
        inputs = m.task.model_inputs(batch)
        for who, model in (("student", m.student), ("teacher", m.teacher)):
            embs[who].append(m.backend.forward(model, inputs)["output"].float().cpu())
    s_emb, t_emb = torch.cat(embs["student"]), torch.cat(embs["teacher"])

    stsb = {}
    ds_sts = datasets.load_dataset("nyu-mll/glue", "stsb", split="validation")
    for who, model in (("teacher", m.teacher), ("student", m.student)):
        a, b = (
            _embed_texts(m.backend, model, m.tokenizer, m.cfg, ds_sts[c], m.device)
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
    print(
        f"\n== agreement: {out['pool']} held-out corpus texts -- recall@{K} vs THIS teacher "
        f"(not MTEB's recall) =="
    )
    print(
        f"  recall@{K}     {out[f'recall@{K}']:.4f}   of the teacher's top-{K} neighbours found;"
        f" teacher = 1.0 by construction"
    )
    print(f"  embed_dist    {out['embed_dist']:.6f}   (per-vector; logging only, it misranks)")
    print(
        f"  STS-B         teacher {stsb['teacher']:.4f}  student {stsb['student']:.4f}  "
        f"delta {stsb['student'] - stsb['teacher']:+.4f}"
    )
    return out


def corpus(m: Models, split: str, names: list[str]) -> dict:
    """nDCG@10 on held-out rows of the training slices, so coverage is out of the equation.

    Two macros, deliberately not one: a qrels task's golds are ~96% likely to be train rows, so
    only its distractors are held out. Averaging that into a clean headline is the CORE mistake.
    """
    sources = {s.name: s for s in mix(m.cfg.train.corpus)}
    print(
        f"\n== corpus: held-out rows of {m.cfg.train.corpus}, split={split} ==\n"
        f"   quality = nDCG@{K} (teacher/student/delta/rel);  fidelity = recall@{K} + embed_dist\n"
        f"   both over the task's whole doc pool. Read across MODELS: `docs` differs per task and\n"
        f"   both metrics are pool-size dependent, so task-to-task is not comparable\n"
        f"{'task':>18} {'kind':>6} {'q':>5} {'docs':>6} {'teacher':>9} {'student':>9} "
        f"{'delta':>9} {'rel':>8} {f'recall@{K}':>9} {'dist':>8}   retrieves",
        flush=True,
    )
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
    rows = _corpus_rows(m, pools, kinds, shapes)
    for n, r in rows.items():
        r["domain"] = sources[n].domain

    out: dict = {"metric": f"ndcg@{K}", "sources": rows}
    if not rows:
        return out
    print()
    for kind in ("pair", "qrels"):
        if group := [r for r in rows.values() if r["tag"] == kind]:
            out[f"macro_{kind}"] = _print_macro(f"MACRO-{kind}", group)
    # Pair tasks only. Mixing in qrels tasks would fold ~96%-train-row documents into a domain
    # average, which is the reason the macros are split in the first place.
    pair_rows = [r for r in rows.values() if r["tag"] == "pair"]
    if pair_rows:
        print(f"\nby domain (pair tasks only, nDCG@{K}):")
        for domain in sorted({r["domain"] for r in pair_rows}):
            group = [r for r in pair_rows if r["domain"] == domain]
            dt, ds = _macro(group)
            print(f"  {domain:<14} n={len(group)}  {dt:.4f} -> {ds:.4f}  {ds - dt:+.4f}")
    return out


def external(m: Models, names: list[str]) -> dict:
    """nDCG@10 against ground-truth judgements. NOT comparable to the published MTEB table: task
    subset, the config's own max_seq_len (the FHE condition) vs MTEB's 512, one generic instruction.

    ⚠️ The macro is EVERY scored task. Two tasks ("CORE") read the depth-19 cut at +0.4% where five
    put it at -16.0%, inverting the ratio column too -- it reversed the conclusion, not the value.
    """
    _header("external benchmarks", f"nDCG@{K}")
    rows = _pairs(
        m,
        {n: EXTERNAL[n]() for n in names},
        "external",
        shapes=dict.fromkeys(names, "judged query -> gold document"),
    )
    out: dict = {"metric": f"ndcg@{K}", "tasks": rows}
    if rows:
        print()
        out["macro"] = _print_macro("MACRO", list(rows.values()))
    return out


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument(
        "--suite", default=",".join(SUITES), help=f"comma-separated: {', '.join(SUITES)}"
    )
    p.add_argument("--split", default="validation", choices=("validation", "test"))
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
    m = load(args)
    result: dict = {"config": args.config, "student_ckpt": args.student_ckpt, "k": K}
    if "fidelity" in suites:
        result["fidelity"] = fidelity(m)
    if "corpus" in suites:
        result["corpus"] = corpus(m, args.split, names)
    if "external" in suites:
        result["external"] = external(m, tasks)

    # The subtraction the two nDCG suites exist for. Pair-derived only: the qrels macro's documents
    # are mostly train rows, so it would flatter the in-distribution side.
    in_dist = result.get("corpus", {}).get("macro_pair")
    off_dist = result.get("external", {}).get("macro")
    if in_dist and off_dist:
        a = 100 * (in_dist["student"] - in_dist["teacher"]) / in_dist["teacher"]
        b = 100 * (off_dist["student"] - off_dist["teacher"]) / off_dist["teacher"]
        result["coverage_gap"] = b - a
        print(
            f"\nin-distribution {a:+.1f}%   external {b:+.1f}%   coverage gap {b - a:+.1f}%\n"
            f"  a large gap means the fix is corpus coverage; both large means capacity."
        )
    print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
