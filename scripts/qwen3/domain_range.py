"""Does the FFN pre-activation profile move when the corpus does?

The `{2, 18}` targeted penalty set was derived from activation outliers measured on English prose
only. `multi_domain` adds code, CJK, Arabic and Cyrillic — if those shift *which* layers leave
D=32, not just how far, the targeted penalty leaks on the layers it does not cover and the [6]
design is wrong for this corpus. This measures the teacher directly, per domain, before any
training happens.

Recorded English-only reference (teacher, `mixed`): layer 2 = 62.3, 27 = 322.7, 26 = 102.5,
19/24/25 = 32-36, everything else under 32.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/qwen3/domain_range.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import transformers
from franken.config import Config
from franken.data.embed_corpus import _field, _partitioned, _query, _titled
from franken.models import build_backend
from franken.tasks import build_task

N = 1500  # texts per domain; the max is exact, so this bounds tail coverage, not accuracy
DOMAIN = 32.0

DOMAINS = [
    ("english_web", _partitioned("BeIR/msmarco", "corpus", "corpus", _titled)),
    ("queries", _partitioned("BeIR/msmarco", "queries", "queries", _query)),
    (
        "code",
        _partitioned(
            "code-search-net/code_search_net", "all", "train", _field("whole_func_string")
        ),
    ),
    ("pubmed", _partitioned("MedRAG/pubmed", "default", "train", _field("contents"))),
    ("arxiv", _partitioned("ccdv/arxiv-summarization", "document", "train", _field("abstract"))),
    ("wiki_zh", _partitioned("wikimedia/wikipedia", "20231101.zh", "train", _titled)),
    ("wiki_ar", _partitioned("wikimedia/wikipedia", "20231101.ar", "train", _titled)),
    ("wiki_ru", _partitioned("wikimedia/wikipedia", "20231101.ru", "train", _titled)),
]


@torch.no_grad()
def measure(model, backend, tokenizer, texts, cap, device) -> dict[int, float]:
    """Exact max |gate_proj output| per layer over real tokens. Never subsampled — with no clamp
    at inference the single largest value decides safety."""
    peak, mask_holder = {}, {}
    hooks = [
        module.register_forward_hook(
            lambda m, inp, out, i=i: peak.__setitem__(
                i, max(peak.get(i, 0.0), out[mask_holder["m"]].abs().max().item())
            )
        )
        for i, module in enumerate(backend.ffn_preact_modules(model))
    ]
    collator = transformers.DataCollatorWithPadding(tokenizer)
    for start in range(0, len(texts), 16):
        chunk = texts[start : start + 16]
        enc = tokenizer(chunk, truncation=True, max_length=cap)
        batch = collator([{k: enc[k][j] for k in enc} for j in range(len(chunk))])
        batch = {k: v.to(device) for k, v in batch.items()}
        mask_holder["m"] = batch["attention_mask"].bool()
        backend.forward(model, {k: batch[k] for k in ("input_ids", "attention_mask")})
    for h in hooks:
        h.remove()
    return peak


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/depth19_multi_domain.yaml")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)
    model = backend.load_teacher(cfg).to(device).eval()
    n_layers = len(backend.ffn_preact_modules(model))
    print(f"teacher {cfg.train.teacher_model}, {n_layers} layers, {N} texts/domain\n", flush=True)

    peaks = {}
    for name, source in DOMAINS:
        texts = source("train", N)
        peaks[name] = measure(model, backend, tokenizer, texts, cfg.train.max_seq_len, device)
        over = sorted(i for i, v in peaks[name].items() if v > DOMAIN)
        print(
            f"{name:12s} n={len(texts):>5}  global max {max(peaks[name].values()):8.1f}"
            f"  layers over {DOMAIN:.0f}: {over}",
            flush=True,
        )

    print(f"\nper-layer max (blank = under {DOMAIN:.0f} everywhere)")
    header = "layer " + "".join(f"{name[:9]:>10}" for name, _ in DOMAINS)
    print(header)
    for i in range(n_layers):
        row = [peaks[name][i] for name, _ in DOMAINS]
        if max(row) <= DOMAIN:
            continue
        print(f"{i:>5} " + "".join(f"{v:>10.1f}" for v in row))

    baseline = {i for i, v in peaks["english_web"].items() if v > DOMAIN}
    extra = sorted({i for d in peaks.values() for i, v in d.items() if v > DOMAIN} - baseline)
    print(f"\nlayers over {DOMAIN:.0f} on english_web: {sorted(baseline)}")
    print(f"layers the new domains ADD: {extra or 'none'}")


if __name__ == "__main__":
    main()
