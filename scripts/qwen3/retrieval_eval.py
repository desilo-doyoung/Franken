"""nDCG@10 against ground-truth judgements — the only ABSOLUTE, top-k-sensitive metric here.

`recall@10` measures agreement with *this teacher's* neighbourhoods (different, not worse), and
STS-B is absolute but too coarse for top-of-list damage. This fills the remaining cell.

⚠️ NOT comparable to the published MTEB table: task subset, `max_seq_len` 128 (the FHE deployment
condition) vs MTEB's 512, one generic instruction. Valid teacher-vs-student only. With no
--student-ckpt the student IS the teacher, so every delta must be ~0 — the self-test.

    uv run python scripts/qwen3/retrieval_eval.py --config configs/qwen3/depth19_max.yaml \
        --student-ckpt outputs/qwen3_depth19_max/student/pytorch_model.bin
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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

# BEIR/MTEB copies sharing one schema: corpus(_id,title,text), queries(_id,text), qrels via config
# "default" split "test". Small on purpose — MSMARCO-scale tasks cost hours per checkpoint, and
# documents are ~90% of the runtime while the statistics live in the query count.
TASKS = {
    "nfcorpus": "mteb/nfcorpus",  # 3.6k docs, biomedical, GRADED relevance (0-2)
    "scifact": "mteb/scifact",  # 5.2k docs, claim verification, binary
}

# Teacher embeddings are identical across every run, so compute once and reuse.
_CACHE_DIR = "outputs/ndcg_cache"
_CACHE_VERSION = 1


def _load(task: str):
    name = TASKS[task]
    corpus = datasets.load_dataset(name, "corpus", split="corpus")
    queries = datasets.load_dataset(name, "queries", split="queries")
    qrels_rows = datasets.load_dataset(name, "default", split="test")

    qrels: dict[str, dict[str, float]] = {}
    for r in qrels_rows:
        if r["score"] > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = float(r["score"])

    # The queries file bundles train/dev/test; keep only judged (test) ones.
    q_ids, q_texts = [], []
    for r in queries:
        qid = str(r["_id"])
        if qid in qrels:
            q_ids.append(qid)
            q_texts.append(INSTRUCT.format(r["text"].strip()))

    # BEIR: document = title + text, and documents take no instruction prefix.
    d_ids = [str(r) for r in corpus["_id"]]
    d_texts = [f"{t} {x}".strip() for t, x in zip(corpus["title"], corpus["text"], strict=True)]
    return d_ids, d_texts, q_ids, q_texts, qrels


def ndcg_at_k(ranked_ids, relevant: dict[str, float], k: int = K) -> float:
    # Exponential gain 2^rel-1, log2(rank+1) discount — what trec_eval/MTEB compute.
    dcg = sum(
        (2.0 ** relevant.get(did, 0.0) - 1.0) / math.log2(rank + 2)
        for rank, did in enumerate(ranked_ids[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _teacher_cache(task: str, cfg) -> str:
    slug = re.sub(r"[^\w.-]", "_", cfg.train.teacher_model)
    return os.path.join(_CACHE_DIR, f"v{_CACHE_VERSION}-{task}-{slug}-{cfg.train.max_seq_len}.pt")


def _embed_pair(backend, model, tokenizer, cfg, device, d_texts, q_texts, cache):
    if cache and os.path.exists(cache):
        blob = torch.load(cache, weights_only=True)
        # Corruption check only; model and max_seq_len are already in the path.
        if blob["counts"] == [len(d_texts), len(q_texts)]:
            return blob["d"], blob["q"]
    d = _embed_texts(backend, model, tokenizer, cfg, d_texts, device)
    q = _embed_texts(backend, model, tokenizer, cfg, q_texts, device)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"d": d, "q": q, "counts": [len(d_texts), len(q_texts)]}, cache)
    return d, q


@torch.no_grad()
def score(
    backend, model, tokenizer, cfg, device, d_ids, d_texts, q_ids, q_texts, qrels, cache=None
):
    """Mean nDCG@10 over queries. The backend L2-norms its output, so `q @ d.T` is cosine."""
    d_emb, q_emb = _embed_pair(backend, model, tokenizer, cfg, device, d_texts, q_texts, cache)
    total = 0.0
    for i in range(0, len(q_ids), 256):  # chunked: the full queries x docs matrix is needlessly big
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
        t = score(
            backend,
            teacher,
            tokenizer,
            cfg,
            device,
            d_ids,
            d_texts,
            q_ids,
            q_texts,
            qrels,
            cache=_teacher_cache(name, cfg),
        )
        s = score(backend, student, tokenizer, cfg, device, d_ids, d_texts, q_ids, q_texts, qrels)
        rel = 100 * (s - t) / t if t > 0 else 0.0
        print(
            f"{name:>10} {len(q_ids):>8} {len(d_ids):>7} {t:>9.4f} {s:>9.4f} "
            f"{s - t:>+9.4f} {rel:>7.1f}%"
        )
        result["tasks"][name] = {
            "teacher": t,
            "student": s,
            "queries": len(q_ids),
            "docs": len(d_ids),
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
