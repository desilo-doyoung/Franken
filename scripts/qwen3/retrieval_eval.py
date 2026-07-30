"""nDCG@10 on labelled retrieval tasks — the only ABSOLUTE, top-k-sensitive metric in this repo.

Everything else here measures one of two other things. `recall@10` (see `embed_eval.py`) is
agreement with *this teacher's* neighbourhoods: it can say the student retrieves **differently**,
never that it retrieves **worse**. STS-B is absolute but coarse — Spearman over 1500 independent
sentence pairs, i.e. global ordering — and so is blind to exactly the top-of-list damage `recall@10`
detects. That leaves the cell this script fills:

                     teacher-fidelity      absolute quality
    local / top-k    recall@10             *** this script ***
    coarse / global  sim-rho (rejected)    STS-B

Scores teacher and student against ground-truth relevance judgements, so a drop here is real
degradation rather than divergence. With no --student-ckpt the student is seeded from the teacher,
i.e. the identity baseline: every delta must be ~0, which is this script's self-test.

⚠️ NOT comparable to the published Qwen3-Embedding MTEB table. Three-task subset (not the full
suite), `max_seq_len` from the config (128 — the FHE deployment condition) rather than MTEB's 512,
and one generic instruction rather than per-task ones. Valid for teacher-vs-student, invalid as a
leaderboard figure. Report it strictly as **nDCG@10**: MTEB retrieval also defines a "recall@10",
and ours means teacher-neighbour agreement.

Usage:
    uv run python scripts/qwen3/retrieval_eval.py --config configs/qwen3/depth14.yaml \
        --student-ckpt outputs/qwen3_depth14/student/pytorch_model.bin
    uv run python scripts/qwen3/retrieval_eval.py --tasks scifact   # subset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import datasets  # noqa: E402  (repo root must reach sys.path first)
import torch  # noqa: E402
from embed_eval import _embed_texts  # noqa: E402
from franken.config import Config  # noqa: E402
from franken.data.embed_corpus import INSTRUCT  # noqa: E402
from franken.models import build_backend  # noqa: E402
from franken.tasks import build_task  # noqa: E402

K = 10

# Standard BEIR/MTEB copies, all sharing one schema: corpus(_id,title,text),
# queries(_id,text), qrels via config "default" split "test" (query-id,corpus-id,score).
# Kept small on purpose — the full suite's retrieval tasks (MSMARCO 8.8M docs, ClimateFEVER 5.4M)
# would take hours per checkpoint, and we have a dozen checkpoints.
TASKS = {
    "nfcorpus": "mteb/nfcorpus",  # 3.6k docs, biomedical, GRADED relevance (0-2)
    "scifact": "mteb/scifact",  # 5.2k docs, scientific claim verification, binary
    # ⚠️ ArguAna's queries are whole arguments, far longer than 128 tokens, so its absolute score is
    # depressed by truncation. Kept anyway: teacher and student are truncated identically, so the
    # DELTA stays valid, and its 1406 queries add statistical weight. Its corpus also contains each
    # query's own document (a known quirk), which depresses everyone equally.
    "arguana": "mteb/arguana",  # 8.7k docs, argument retrieval, binary
}


def _load(task: str):
    """corpus texts + ids, query texts + ids (test only), and qrels as {qid: {did: score}}."""
    name = TASKS[task]
    corpus = datasets.load_dataset(name, "corpus", split="corpus")
    queries = datasets.load_dataset(name, "queries", split="queries")
    qrels_rows = datasets.load_dataset(name, "default", split="test")

    qrels: dict[str, dict[str, float]] = {}
    for r in qrels_rows:
        if r["score"] > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = float(r["score"])

    # The queries file bundles train/dev/test; keep only judged (test) queries.
    q_ids, q_texts = [], []
    for r in queries:
        qid = str(r["_id"])
        if qid in qrels:
            q_ids.append(qid)
            q_texts.append(INSTRUCT.format(r["text"].strip()))  # queries get the instruction prefix

    # BEIR convention: the document is title + text. Documents get NO prefix (verified against the
    # checkpoint's config_sentence_transformers.json, where "document" is "").
    d_ids = [str(r) for r in corpus["_id"]]
    d_texts = [f"{t} {x}".strip() for t, x in zip(corpus["title"], corpus["text"], strict=True)]
    return d_ids, d_texts, q_ids, q_texts, qrels


def ndcg_at_k(ranked_ids, relevant: dict[str, float], k: int = K) -> float:
    """nDCG@k with the exponential gain `2^rel - 1` and `log2(rank+1)` discount — what trec_eval /
    pytrec_eval (and therefore MTEB) compute, so teacher scores land near published values. For the
    binary tasks the gain reduces to `rel`; it only matters for graded NFCorpus."""
    dcg = sum(
        (2.0 ** relevant.get(did, 0.0) - 1.0) / math.log2(rank + 2)
        for rank, did in enumerate(ranked_ids[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def score(backend, model, tokenizer, cfg, device, d_ids, d_texts, q_ids, q_texts, qrels):
    """Mean nDCG@10 over queries. The backend L2-norms its output, so `q @ d.T` is cosine."""
    d_emb = _embed_texts(backend, model, tokenizer, cfg, d_texts, device)
    q_emb = _embed_texts(backend, model, tokenizer, cfg, q_texts, device)
    total = 0.0
    # Chunk the query side: the full (queries x docs) matrix is fine here but needlessly large.
    for i in range(0, len(q_ids), 256):
        sims = q_emb[i : i + 256] @ d_emb.T
        top = sims.topk(K, dim=-1).indices
        for row, qid in zip(top, q_ids[i : i + 256], strict=True):
            total += ndcg_at_k([d_ids[j] for j in row.tolist()], qrels[qid])
    return total / len(q_ids)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/qwen3/exact.yaml")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    p.add_argument("--tasks", default=",".join(TASKS), help=f"comma-separated: {', '.join(TASKS)}")
    p.add_argument("--json", help="also dump the metrics here, for scripted runs")
    args = p.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s) {unknown}; available: {sorted(TASKS)}")

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device).eval()
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    if args.student_ckpt:
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    student = student.to(device).eval()

    print(f"\nstudent: {args.student_ckpt or 'IDENTITY (seeded from teacher)'}")
    print(
        f"depth={cfg.model.num_hidden_layers} softmax={cfg.model.softmax} "
        f"act={cfg.model.activation} max_seq_len={cfg.train.max_seq_len}"
    )
    print(f"query prefix: {INSTRUCT.format('...')!r}\n")
    print(
        f"{'task':>10} {'queries':>8} {'docs':>7} "
        f"{'teacher':>9} {'student':>9} {'delta':>9} {'rel':>8}"
    )

    result: dict = {"config": args.config, "student_ckpt": args.student_ckpt, "k": K, "tasks": {}}
    for name in tasks:
        d_ids, d_texts, q_ids, q_texts, qrels = _load(name)
        t = score(backend, teacher, tokenizer, cfg, device, d_ids, d_texts, q_ids, q_texts, qrels)
        s = score(backend, student, tokenizer, cfg, device, d_ids, d_texts, q_ids, q_texts, qrels)
        rel = 100 * (s - t) / t if t > 0 else 0.0
        print(
            f"{name:>10} {len(q_ids):>8} {len(d_ids):>7} {t:>9.4f} {s:>9.4f} "
            f"{s - t:>+9.4f} {rel:>7.1f}%"
        )
        result["tasks"][name] = {
            "teacher": t, "student": s, "queries": len(q_ids), "docs": len(d_ids),
        }

    ts = [v["teacher"] for v in result["tasks"].values()]
    ss = [v["student"] for v in result["tasks"].values()]
    t_avg, s_avg = sum(ts) / len(ts), sum(ss) / len(ss)
    result |= {"teacher_avg": t_avg, "student_avg": s_avg}
    print(
        f"{'MACRO AVG':>10} {'':>8} {'':>7} {t_avg:>9.4f} {s_avg:>9.4f} "
        f"{s_avg - t_avg:>+9.4f} {100 * (s_avg - t_avg) / t_avg:>7.1f}%\n"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
