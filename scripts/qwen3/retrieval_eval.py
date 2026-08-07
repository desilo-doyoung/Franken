"""nDCG@10 against ground-truth judgements — the only ABSOLUTE, top-k-sensitive metric here.

`recall@10` measures agreement with *this teacher's* neighbourhoods (different, not worse), and
STS-B is absolute but too coarse for top-of-list damage. This fills the remaining cell.

⚠️ NOT comparable to the published MTEB table: task subset, `max_seq_len` 128 (the FHE deployment
condition) vs MTEB's 512, one generic instruction. Valid teacher-vs-student only. With no
--student-ckpt the student IS the teacher, so every delta must be ~0 — the self-test.

The macro average covers `CORE` only and `--tasks` defaults to it, so the teacher 0.5083 reference
stays fixed however many extra tasks are scored. `xpqa_ara` / `xpqa_cmn` cover the multilingual
side of the `multi_domain` corpus and are opt-in.

    uv run python scripts/qwen3/retrieval_eval.py --config configs/qwen3/depth19.yaml \
        --student-ckpt outputs/qwen3_depth19/student/pytorch_model.bin

    # multilingual, alongside the core two
    uv run python scripts/qwen3/retrieval_eval.py --tasks nfcorpus,scifact,xpqa_ara,xpqa_cmn ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from functools import partial

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

# Teacher embeddings are identical across every run, so compute once and reuse.
_CACHE_DIR = "outputs/ndcg_cache"
_CACHE_VERSION = 1


def _assemble(corpus, queries, qrels_rows, id_field: str):
    qrels: dict[str, dict[str, float]] = {}
    for r in qrels_rows:
        score = float(r["score"])  # XPQA stores it as a string
        if score > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = score

    # The queries file bundles train/dev/test; keep only judged (test) ones.
    q_ids, q_texts = [], []
    for r in queries:
        qid = str(r[id_field])
        if qid in qrels:
            q_ids.append(qid)
            q_texts.append(INSTRUCT.format(r["text"].strip()))

    # document = title + text, and documents take no instruction prefix.
    d_ids = [str(x) for x in corpus[id_field]]
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(d_ids)
    d_texts = [f"{t} {x}".strip() for t, x in zip(titles, corpus["text"], strict=True)]
    return d_ids, d_texts, q_ids, q_texts, qrels


def _load_beir(repo: str):
    """BEIR/MTEB layout: corpus(_id,title,text), queries(_id,text), qrels in "default"/"test"."""
    return _assemble(
        datasets.load_dataset(repo, "corpus", split="corpus"),
        datasets.load_dataset(repo, "queries", split="queries"),
        datasets.load_dataset(repo, "default", split="test"),
        "_id",
    )


def _load_xpqa(pair: str):
    """XPQA layout: one config per language pair, everything in split "test", `id` not `_id`."""
    repo = "mteb/XPQARetrieval"
    return _assemble(
        datasets.load_dataset(repo, f"{pair}-corpus", split="test"),
        datasets.load_dataset(repo, f"{pair}-queries", split="test"),
        datasets.load_dataset(repo, f"{pair}-qrels", split="test"),
        "id",
    )


# Small on purpose — MSMARCO-scale tasks cost hours per checkpoint, and documents are ~90% of the
# runtime while the statistics live in the query count.
TASKS = {
    "nfcorpus": partial(_load_beir, "mteb/nfcorpus"),  # 3.6k docs, biomedical, GRADED rel (0-2)
    "scifact": partial(_load_beir, "mteb/scifact"),  # 5.2k docs, claim verification, binary
    # `multi_domain` coverage CORE cannot see. All three are clean w.r.t. the training corpus,
    # which is what rules out the obvious picks: MS MARCO / NQ / HotpotQA benchmarks (the corpus
    # takes 27% / 10% / 7% of those very corpora), CoIR's own CodeSearchNet and `cosqa` (both
    # CodeSearchNet-derived, as is the corpus), and MIRACL / Mr.TyDi (Wikipedia-derived, and the
    # corpus trains on wikimedia/wikipedia). CoIR and BEIR share a layout, so only XPQA needs
    # its own loader.
    "fiqa": partial(_load_beir, "mteb/fiqa"),  # 58k docs / 1.7k q, informal web prose
    "xpqa_cmn": partial(_load_xpqa, "cmn-cmn"),  # 1.7k docs / 824 q, zh = best-covered language
    # Scores the restored 8% code slice. It read 0.0797 at max_seq_len 128 purely because 92.8% of
    # its QUERIES overflowed; at 1024 they fit. ⚠️ Confirm the teacher clears that floor before
    # trusting it — a task with the teacher near zero cannot detect student damage. `cosqa` and
    # CoIR's CodeSearchNet are the wrong fallbacks (CodeSearchNet-derived, hence contaminated by
    # the restored slice); `CoIR-Retrieval/codefeedback-st` is the clean one.
    "code_apps": partial(_load_beir, "CoIR-Retrieval/apps"),
}

# The macro covers CORE ONLY, whatever else is scored alongside it. `run_experiments.py` reads
# `teacher_avg`/`student_avg` into the headline table and asserts the teacher value is constant
# across runs, so folding a task in would silently re-base every published number (teacher
# 0.5083 = nfcorpus 0.3506 + scifact 0.6660). Non-core tasks print below it, unaveraged.
CORE = ("nfcorpus", "scifact")


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
    # Defaults to CORE, not every task: `run_experiments.py` calls this without --tasks, so a
    # wider default would silently add cost and non-core rows to every experiment.
    p.add_argument("--tasks", default=",".join(CORE), help=f"comma-separated: {', '.join(TASKS)}")
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
    core = [n for n in tasks if n in CORE]
    extra = [n for n in tasks if n not in CORE]
    for name in core + extra:
        if extra and name == extra[0]:
            print(f"{'-' * 24} below: not in the macro {'-' * 15}")
        d_ids, d_texts, q_ids, q_texts, qrels = TASKS[name]()
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

    if core:
        ts = [result["tasks"][n]["teacher"] for n in core]
        ss = [result["tasks"][n]["student"] for n in core]
        t_avg, s_avg = sum(ts) / len(ts), sum(ss) / len(ss)
        result |= {"teacher_avg": t_avg, "student_avg": s_avg, "macro_tasks": core}
        # Flag a partial macro loudly: it is not the 0.5083-comparable number.
        label = "MACRO AVG" if tuple(core) == CORE else "MACRO(PART)"
        print(
            f"{label:>10} {'':>8} {'':>7} {t_avg:>9.4f} {s_avg:>9.4f} "
            f"{s_avg - t_avg:>+9.4f} {100 * (s_avg - t_avg) / t_avg:>7.1f}%\n"
        )
    else:
        print("\nno CORE task scored — no macro average, and no comparable number\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
