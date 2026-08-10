"""nDCG@10 and recall@10 on held-out rows of the training corpus itself.

`retrieval_eval.py` scores external benchmarks, which are off-distribution to an unknown degree, so
its deficits mix the cost of the depth cut and the FHE ops with the cost of corpus coverage.
`domain_drift.py` showed how badly those separate: held-out CodeSearchNet python drifts 0.0104
(= nfcorpus's 0.0087) while APPS drifts 0.179, so `code_apps` −53.9% nDCG was largely a coverage
measurement. This script removes coverage from the equation — every task is built from rows of a
corpus source that training never read.

The split guarantee is inherited, not restated: `embed_corpus._hashed` assigns a row to exactly one
split as a pure function of its key column, and `_rows` below re-derives membership from the same
`build.meta`. A task therefore cannot silently score trained rows, and if a source's key ever
changes the eval follows it.

Query sides carry a prefix only where the corpus prefixes them (MS MARCO queries, via `_marco`), so
the eval matches training in *format* as well as content — the point of the exercise.

Read it against `retrieval_eval.py` rather than alone: small deficits here with large ones there
means coverage, not architecture. Both near-equal means the architecture really does cost that.

    uv run python scripts/qwen3/corpus_eval.py --config configs/qwen3/depth19_multi_domain.yaml \
        --student-ckpt outputs/qwen3_depth19_multi_domain/student/pytorch_model.bin
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
from typing import NamedTuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import torch  # noqa: E402
from franken.config import Config  # noqa: E402
from franken.data.embed_corpus import (  # noqa: E402  # noqa: E402
    INSTRUCT,
    MIXES,
    _split_of,
    _stream,
)
from franken.models import build_backend  # noqa: E402
from franken.tasks import build_task  # noqa: E402
from retrieval_eval import K, score  # noqa: E402

QUERIES, DOCS = 500, 5_000
_CACHE_DIR = "outputs/corpus_eval_cache"
_CACHE_VERSION = 1


def _marco_fields(row) -> tuple[str, str, list[str]]:
    """MS MARCO keeps the whole task inside one row: a query, one flagged passage, nine hard
    negatives. The positive is not a plain column, hence a `fields` hook, not column names."""
    texts = [p.strip() for p in row["passages"]["passage_text"]]
    flags = row["passages"]["is_selected"]
    sel = [t for t, f in zip(texts, flags, strict=True) if f and t]
    neg = [t for t, f in zip(texts, flags, strict=True) if not f and t]
    return row["query"].strip(), (sel[0] if sel else ""), neg


class Task(NamedTuple):
    """One retrieval task over a corpus source's held-out rows.

    `extra` names columns that become hard distractors (a triplet's `negative`, MS MARCO's nine
    unselected passages) — the pool then contains near-misses rather than only random documents.
    `fields` replaces the column lookup for sources whose pair is not two plain columns.
    """

    slice: str  # must exist in the mix; asserted, so a task cannot drift off the corpus
    query: str = ""
    doc: str = ""
    extra: tuple[str, ...] = ()
    prefix: bool = False  # only where the corpus prefixes this source's query side
    keep: Callable | None = None
    fields: Callable | None = None  # row -> (query, doc, extras)


TASKS: dict[str, Task] = {
    "msmarco": Task("msmarco", prefix=True, fields=_marco_fields),
    "gooaq": Task("gooaq", "question", "answer"),
    "eli5": Task("eli5", "question", "answer"),
    "stackexchange": Task("stackexchange", "post1", "post2"),
    "s2orc": Task("s2orc", "citation", "abstract"),
    "specter": Task("specter", "anchor", "positive", ("negative",)),
    "nli": Task("nli", "anchor", "positive", ("negative",)),
    "quora": Task("quora", "anchor", "positive", ("negative",)),
    "csn_python": Task(
        "code",
        "func_documentation_string",
        "whole_func_string",
        keep=lambda r: r["language"] == "python",
    ),
    "codefeedback": Task("codefeedback", "title", "text"),
    "pubmed": Task("pubmed", "title", "content"),
    "nq": Task("nq_passage", "title", "text"),
    "hotpotqa": Task("hotpotqa_passage", "title", "text"),
    **{
        f"wiki_{lang}": Task(f"wiki_{lang}", "title", "text")
        for lang in ("en", "zh", "ja", "ar", "ru", "es", "de", "fr", "ko", "vi")
    },
}

SOURCES = {name: source for name, _d, source, _w in MIXES["multi_domain"]}
DOMAIN = {name: domain for name, domain, _s, _w in MIXES["multi_domain"]}


def _rows(slice_name: str, split: str):
    """Held-out rows of a corpus source, using that source's own split rule."""
    meta = SOURCES[slice_name].meta
    if meta["native"]:
        yield from _stream(meta["repo"], meta["config"], meta["split_map"].get(split, split))
        return
    for row in _stream(meta["repo"], meta["config"], meta["hf_split"]):
        if _split_of(str(row[meta["key"]])) == split:
            yield row


def build_task_data(name: str, split: str):
    """(d_ids, d_texts, q_ids, q_texts, qrels) — the shape `retrieval_eval.score` consumes."""
    t = TASKS[name]
    d_ids, d_texts, q_ids, q_texts, qrels = [], [], [], [], {}
    for row in _rows(t.slice, split):
        if t.keep and not t.keep(row):
            continue
        if t.fields:
            query, doc, extras = t.fields(row)
        else:
            query, doc = (row[t.query] or "").strip(), (row[t.doc] or "").strip()
            extras = [s for c in t.extra if (s := (row[c] or "").strip())]
        if not doc or (len(q_ids) < QUERIES and not query):
            continue

        if len(q_ids) < QUERIES:
            qid, did = f"q{len(q_ids)}", f"d{len(d_ids)}"
            q_ids.append(qid)
            q_texts.append(INSTRUCT.format(query) if t.prefix else query)
            qrels[qid] = {did: 1.0}
            d_ids.append(did)
            d_texts.append(doc)
        elif len(d_ids) < DOCS:
            d_ids.append(f"d{len(d_ids)}")
            d_texts.append(doc)
        for x in extras:  # hard distractors, never gold
            if len(d_ids) >= DOCS:
                break
            d_ids.append(f"d{len(d_ids)}")
            d_texts.append(x)
        if len(q_ids) >= QUERIES and len(d_ids) >= DOCS:
            break
    return d_ids, d_texts, q_ids, q_texts, qrels


def _cache(name: str, split: str, cfg) -> str:
    slug = re.sub(r"[^\w.-]", "_", cfg.train.teacher_model)
    return os.path.join(
        _CACHE_DIR,
        f"v{_CACHE_VERSION}-{name}-{split}-{QUERIES}x{DOCS}-{slug}-{cfg.train.max_seq_len}.pt",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/qwen3/depth19_multi_domain.yaml")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    # validation while iterating; test is touched once, which is the reason for a third split.
    p.add_argument("--split", default="validation", choices=("validation", "test"))
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--json")
    args = p.parse_args(argv)

    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if unknown := [t for t in names if t not in TASKS]:
        raise SystemExit(f"Unknown task(s) {unknown}; available: {sorted(TASKS)}")
    if missing := [TASKS[t].slice for t in names if TASKS[t].slice not in SOURCES]:
        raise SystemExit(f"Task(s) reference slices absent from the mix: {missing}")

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
        f"act={cfg.model.activation} max_seq_len={cfg.train.max_seq_len} split={args.split}"
    )
    print(
        f"\n{'task':>14} {'domain':>14} {'q':>5} {'docs':>6} "
        f"{'teacher':>9} {'student':>9} {'delta':>9} {'rel':>8}",
        flush=True,
    )

    result: dict = {
        "config": args.config,
        "student_ckpt": args.student_ckpt,
        "split": args.split,
        "k": K,
        "tasks": {},
    }
    for name in names:
        d_ids, d_texts, q_ids, q_texts, qrels = build_task_data(name, args.split)
        if not q_ids:
            print(f"{name:>14} {DOMAIN[TASKS[name].slice]:>14}   no held-out pairs", flush=True)
            continue
        common = (backend, tokenizer, cfg, device, d_ids, d_texts, q_ids, q_texts, qrels)
        t = score(common[0], teacher, *common[1:], cache=_cache(name, args.split, cfg))
        s = score(common[0], student, *common[1:])
        rel = 100 * (s - t) / t if t > 0 else 0.0
        print(
            f"{name:>14} {DOMAIN[TASKS[name].slice]:>14} {len(q_ids):>5} {len(d_ids):>6} "
            f"{t:>9.4f} {s:>9.4f} {s - t:>+9.4f} {rel:>7.1f}%",
            flush=True,
        )
        result["tasks"][name] = {
            "teacher": t,
            "student": s,
            "queries": len(q_ids),
            "docs": len(d_ids),
            "domain": DOMAIN[TASKS[name].slice],
        }

    scored = result["tasks"]
    if scored:
        ts = [v["teacher"] for v in scored.values()]
        ss = [v["student"] for v in scored.values()]
        t_avg, s_avg = sum(ts) / len(ts), sum(ss) / len(ss)
        result |= {"teacher_avg": t_avg, "student_avg": s_avg, "macro_tasks": list(scored)}
        print(
            f"\n{f'MACRO({len(ts)})':>14} {'':>14} {'':>5} {'':>6} {t_avg:>9.4f} {s_avg:>9.4f} "
            f"{s_avg - t_avg:>+9.4f} {100 * (s_avg - t_avg) / t_avg:>7.1f}%"
        )
        print("\nby domain:")
        for domain in sorted({v["domain"] for v in scored.values()}):
            rows = [v for v in scored.values() if v["domain"] == domain]
            dt = sum(r["teacher"] for r in rows) / len(rows)
            ds = sum(r["student"] for r in rows) / len(rows)
            print(
                f"  {domain:<14} n={len(rows)}  teacher {dt:.4f}  student {ds:.4f}  "
                f"{100 * (ds - dt) / dt:+.1f}%"
            )
        print()

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
