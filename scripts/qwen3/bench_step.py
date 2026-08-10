"""Time one distillation step, so speed claims are measured rather than assumed.

    uv run python scripts/qwen3/bench_step.py --config configs/qwen3/depth28_exact.yaml
    uv run python scripts/qwen3/bench_step.py --config ... --precision bf16 --compile

`tok/s` counts REAL (non-pad) tokens, matching the recipe table in PROGRESS.md. Does not
attach the range-penalty hooks, so penalized configs read slightly fast.
"""

import argparse
import os
import sys
import time

import pyarrow.compute as pc
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from franken.config import Config  # noqa: E402
from franken.distill.batching import plan_batches  # noqa: E402
from franken.distill.trainer import (  # noqa: E402
    Distiller,
    _apply_precision,
    _autocast,
    _maybe_compile,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=20, help="timed steps")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps (compile, autotune)")
    p.add_argument("--precision", choices=("fp32", "tf32", "bf16"), help="override config")
    p.add_argument("--compile", action="store_true", help="override config: torch.compile on")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.precision:
        cfg.train.precision = args.precision
    if args.compile:
        cfg.train.compile = True

    # Before the corpus build and the weight load: `batch_size` is the GLOBAL batch, split across
    # ranks at train time, and this script is single-process, so it would put all of it on one card.
    # token_budget is already per-rank and needs no such check.
    opt = cfg.train.distill
    if not opt.token_budget and opt.batch_size > 128:
        raise SystemExit(
            f"batch_size {opt.batch_size} is the GLOBAL batch ({opt.batch_size // 4} per rank on 4 "
            f"cards); this script is single-process and would OOM on one card. Bench a "
            f"token-budgeted config, or one whose batch_size is already per-rank."
        )

    d = Distiller(cfg)
    d.setup()
    _apply_precision(cfg.train.precision)

    data = d.task.datasets(d.tokenizer, cfg)
    train_data = data["train"].with_format("torch", columns=d.task.torch_columns())
    if opt.token_budget:
        lengths = pc.list_value_length(train_data.data.column("input_ids")).to_numpy(
            zero_copy_only=False
        )
        plan = plan_batches(lengths, opt.token_budget, opt.max_seqs, cfg.train.seed, opt.bucket)
        loader = DataLoader(train_data, batch_sampler=plan, collate_fn=data["collator"])
    else:
        loader = DataLoader(
            train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            collate_fn=data["collator"],
        )

    optimizer = AdamW(d.student.parameters(), lr=cfg.train.distill.lr)

    student = _maybe_compile(d.student, cfg)
    teacher = _maybe_compile(d.teacher, cfg)
    d.student.train()

    def step(batch):
        batch = {k: v.to(d.device) for k, v in batch.items()}
        inputs = d.task.model_inputs(batch)
        with torch.no_grad():
            teacher_out = d.backend.forward(teacher, inputs)
        with _autocast(cfg.train.precision):
            student_out = d.backend.forward(student, inputs)
            total, _ = d.task.compute_loss(student_out, teacher_out, batch, cfg)
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(d.student.parameters(), 1.0)
        optimizer.step()
        return int(batch["attention_mask"].sum()), int(batch["attention_mask"].numel())

    # Stop at `need`. `extend(loader)` would collate EVERY batch in the corpus into host RAM to use
    # 25 of them -- ~30 GB at 11.5M texts, which is an OOM long before any GPU work starts.
    need = args.warmup + args.steps
    batches = []
    while len(batches) < need:
        for batch in loader:
            batches.append(batch)
            if len(batches) >= need:
                break

    for batch in batches[: args.warmup]:
        step(batch)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    real_tokens = padded_tokens = 0
    t0 = time.perf_counter()
    for batch in batches[args.warmup :]:
        real, padded = step(batch)
        real_tokens += real
        padded_tokens += padded
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed / args.steps * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    batching = (
        f"token_budget={opt.token_budget:,} max_seqs={opt.max_seqs} bucket={opt.bucket}"
        if opt.token_budget
        else f"bs={opt.batch_size}"
    )
    print(
        f"{os.path.basename(args.config)}  depth={cfg.model.num_hidden_layers} "
        f"seq={cfg.train.max_seq_len} {batching} precision={cfg.train.precision} "
        f"compile={cfg.train.compile}"
    )
    if cfg.train.compile:
        print(f"  dynamo unique_graphs: {torch._dynamo.utils.counters['stats']['unique_graphs']}")
    print(
        f"  {ms:.0f} ms/step   {real_tokens / elapsed:.0f} tok/s (real)   "
        f"{padded_tokens / elapsed:.0f} tok/s (padded)   {peak:.1f} GB peak"
    )


if __name__ == "__main__":
    main()
