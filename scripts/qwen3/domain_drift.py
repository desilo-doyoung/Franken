"""Where the depth cut damaged the student: per-domain teacher-student drift, no training.

The 5-task nDCG puts the depth-19 cut at -16.0% overall but -53.9% on `code_apps` alone, and two
mechanisms predict that equally well:

1. **Off-distribution.** The corpus trains on CodeSearchNet library functions (docstring + body,
   6 languages); APPS is competitive-programming solutions (bare Python, stdin/stdout). Label-free
   self-distillation only constrains the student where the corpus has mass, so an unexercised
   region is free to drift once 32% of the layers are gone.
2. **Sensitivity.** APPS discriminates among ~8.8k homogeneous Python files at identifier level,
   where nfcorpus only needs coarse topical geometry. The same embedding damage then costs far
   more nDCG.

`csn_python` is in-distribution code held out from training; `apps_docs` is the same *language* in
a different *genre*. Drift far worse on APPS => (1), and the fix is corpus genre. Drift comparable
while nDCG collapses => (2), and the fix is capacity; neither more data nor a different layer map
would help.

Read the two metrics along different axes. `embed_dist` is pool-independent and comparable ACROSS
slices; recall@10 is not -- a pool of 500 APPS solutions is intrinsically harder than 500 nfcorpus
abstracts -- so compare recall ACROSS MODELS within a slice. Run the depth-28 control through it
as well: it is healthy on every task, so its per-slice drift is the floor to subtract.

Self-test: at FULL depth with no --student-ckpt the seeded student *is* the teacher, so every slice
must read dist 0.0 / recall 1.0. That needs a depth-28 config -- under a depth-19 one the splice
alone moves the student, so its no-ckpt run is the init baseline, not an identity.

Usage:
    uv run python scripts/qwen3/domain_drift.py \
      --config configs/qwen3/depth19_multi_domain_exact.yaml \
      --student-ckpt outputs/qwen3_depth19_multi_domain_exact/student/pytorch_model.bin

    uv run python scripts/qwen3/domain_drift.py \
      --config configs/qwen3/depth28_multi_domain.yaml          # identity self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datasets  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from embed_eval import _embed_texts  # noqa: E402
from franken.config import Config  # noqa: E402
from franken.models import build_backend  # noqa: E402
from franken.tasks import build_task  # noqa: E402
from franken.tasks.embed import RECALL_K as K  # noqa: E402
from franken.tasks.embed import recall_at_k  # noqa: E402
from retrieval_eval import _load_beir  # noqa: E402

# Every recall@10 in the trackers is measured at a 500-text pool and recall@10 is pool-size
# dependent, so this holds the numbers comparable to them.
POOL = 500


def _csn_python():
    """Held-out CodeSearchNet Python: same language as APPS, so genre is the only moving part.

    Read from CodeSearchNet's own `validation` split rather than the corpus loader, so this slice
    does not shift when the mix does. Filtered to python explicitly — the stream is grouped by
    language, and relying on that grouping is exactly the bias the corpus loader now shuffles away.
    """
    ds = datasets.load_dataset(
        "code-search-net/code_search_net", "all", split="validation", streaming=True
    )
    out = []
    for r in ds:
        if r["language"] == "python":
            out.append(r["whole_func_string"])
            if len(out) >= POOL:
                break
    return out


def _slices() -> dict[str, list[str]]:
    """Documents take no instruction prefix; query sides are wrapped by `_assemble`."""
    apps_d, apps_q = _load_beir("CoIR-Retrieval/apps")[1::2]
    nf_d, nf_q = _load_beir("mteb/nfcorpus")[1::2]
    fiqa_d = _load_beir("mteb/fiqa")[1]
    return {
        "csn_python": _csn_python(),  # in-distribution code, held out
        "apps_docs": apps_d,  # same language, different genre
        "apps_queries": apps_q,  # long NL problem statements
        "fiqa_docs": fiqa_d,  # the intermediate case (-12.4% nDCG)
        "nfcorpus_docs": nf_d,  # control: the domain that survives
        "nfcorpus_queries": nf_q,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/depth19_multi_domain_exact.yaml")
    p.add_argument("--student-ckpt", default=None, help="default: identity (seeded from teacher)")
    p.add_argument("--json", help="also dump the metrics here, for scripted runs")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device)
    student = backend.build_student(cfg)
    backend.seed_student(student, teacher, cfg)
    if args.student_ckpt:
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    student = student.to(device).eval()

    print(f"\nstudent: {args.student_ckpt or 'IDENTITY (seeded from teacher)'}")
    print(
        f"depth={cfg.model.num_hidden_layers} act={cfg.model.activation} "
        f"seq={cfg.train.max_seq_len}"
    )

    rows = {}
    for name, texts in _slices().items():
        texts = texts[:POOL]
        s_emb, t_emb = (
            _embed_texts(backend, m, tokenizer, cfg, texts, device) for m in (student, teacher)
        )
        rows[name] = {
            "n": len(texts),
            "embed_dist": (1 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean().item(),
            "recall": recall_at_k(s_emb, t_emb, K),
        }
        r = rows[name]
        print(
            f"  {name:<18} n={r['n']:<4} dist {r['embed_dist']:.5f}   recall@{K} {r['recall']:.4f}"
        )

    print(
        f"\n  dist is comparable across slices; recall@{K} only across models within a slice "
        "(pool homogeneity differs).\n"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "config": args.config,
                    "student_ckpt": args.student_ckpt,
                    "depth": cfg.model.num_hidden_layers,
                    "pool": POOL,
                    "k": K,
                    "slices": rows,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
