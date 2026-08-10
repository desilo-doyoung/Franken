"""Run a batch of distillation experiments across the available GPUs and print one table.

The remote workflow this exists for: pull, run this with the configs you want, copy the
final block back. Each config is distilled and then scored by `embed_eval.py` (recall@10,
STS-B) and `retrieval_eval.py` (nDCG@10 — the absolute number to quote), one config per GPU
at a time, and everything a decision needs ends up in a single markdown table that pastes
straight into `franken/models/qwen3/PROGRESS.md`.

Why a runner rather than a shell loop: results are collected from `--json`, not
scraped from prose, so a reworded print can't corrupt the table; a crashed run degrades to
one FAILED row with its log tail instead of losing the whole batch; and the GPU assignment
is a work queue, so N configs over 2 devices finish in ceil(N/2) slots even when runs differ
in length.

Devices are passed as CUDA indices and exported per-subprocess as CUDA_VISIBLE_DEVICES, so
`train.device: cuda` in the config always resolves to the intended card. Pass only cards you
actually own — a co-tenant's idle GPU is not free capacity.

`--ddp` switches the other way: configs run one at a time, each spread over ALL the devices via
torchrun. The queue maximizes throughput (N configs in ceil(N/devices) slots, zero communication);
DDP maximizes the compute a SINGLE run can absorb, which is what a "push one config as far as it
goes" experiment needs. `train.distill.batch_size` stays the global batch and is split across ranks,
so raising it to keep the per-rank batch fixed also means raising LR by sqrt(batch) in the config.

⚠️ Under `train.distill.token_budget` that last sentence does NOT apply: the budget is PER RANK, so
tokens/step scale with the device count, and `batch_size` sizes only the evaluation loader.
Steps/epoch is data-dependent — the trainer logs the real count at startup, and the LR must be
reconciled against that, not against `corpus_size / batch_size`.

Usage:
    uv run python scripts/qwen3/run_experiments.py --devices 0,1 \
        configs/qwen3/depth19.yaml configs/qwen3/quad_silu_nopen.yaml
    uv run python scripts/qwen3/run_experiments.py --devices 2,3 --eval-only configs/qwen3/*.yaml
    uv run python scripts/qwen3/run_experiments.py --devices 0,1,2,3 --ddp \
        configs/qwen3/depth28_exact.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from franken.config import Config  # noqa: E402  (repo root must reach sys.path first)
from franken.paths import RunPaths  # noqa: E402

_PRINT_LOCK = threading.Lock()


def _say(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def _run(cmd: list[str], device: str, log_path: str) -> int:
    """Run one subprocess pinned to `device`, streaming its output to `log_path`."""
    env = os.environ | {"CUDA_VISIBLE_DEVICES": device}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def _tail(path: str, n: int = 15) -> str:
    with open(path) as f:
        return "".join(f.readlines()[-n:])


def _trace(path: str) -> list[str]:
    """The per-epoch metric lines from a distill log — the convergence story in ~4 lines."""
    with open(path) as f:
        return [ln.rstrip() for ln in f if ln.startswith(("init:", "epoch "))]


def _distill_cmd(config: str, nproc: int) -> list[str]:
    """`main.py distill` directly, or under torchrun for DDP.

    Launched as `-m torch.distributed.run` rather than the `torchrun` console script so the
    ranks inherit exactly this interpreter — on a remote box the bare `torchrun` on PATH is
    often a different venv's."""
    argv = ["main.py", "distill", "--config", config]
    if nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", *argv]
    return [sys.executable, *argv]


def one_experiment(
    config: str,
    device: str,
    out_dir: str,
    eval_only: bool,
    ddp: bool = False,
    tasks: str | None = None,
) -> dict:
    stem = os.path.splitext(os.path.basename(config))[0]
    tag = f"[gpu{device}] {stem}"
    ckpt = RunPaths(Config.from_yaml(config)).student_bin()
    nproc = len(device.split(",")) if ddp else 1
    result = {"stem": stem, "config": config, "device": device, "minutes": None, "trace": []}
    if nproc > 1:
        result["world_size"] = nproc

    if not eval_only:
        log = os.path.join(out_dir, f"{stem}.distill.log")
        _say(f"{tag}: distill ({nproc} rank(s)) -> {log}")
        start = time.monotonic()
        code = _run(_distill_cmd(config, nproc), device, log)
        result["minutes"] = (time.monotonic() - start) / 60
        result["trace"] = _trace(log)
        if code != 0:
            _say(f"{tag}: DISTILL FAILED (exit {code})\n{_tail(log)}")
            return result | {"error": f"distill exit {code}", "log": log}
        _say(f"{tag}: distilled in {result['minutes']:.0f} min")

    if not os.path.exists(ckpt):
        _say(f"{tag}: no checkpoint at {ckpt}")
        return result | {"error": "missing checkpoint"}

    log = os.path.join(out_dir, f"{stem}.eval.log")
    metrics_path = os.path.join(out_dir, f"{stem}.json")
    _say(f"{tag}: eval -> {log}")
    code = _run(
        [
            sys.executable,
            os.path.join("scripts", "qwen3", "embed_eval.py"),
            "--config",
            config,
            "--student-ckpt",
            ckpt,
            "--json",
            metrics_path,
        ],
        device.split(",")[0],  # scoring is single-process; don't hand it the whole DDP card set
        log,
    )
    if code != 0 or not os.path.exists(metrics_path):
        _say(f"{tag}: EVAL FAILED (exit {code})\n{_tail(log)}")
        return result | {"error": f"eval exit {code}", "log": log}

    with open(metrics_path) as f:
        result |= json.load(f)
    _say(f"{tag}: recall@{result['k']} {result['recall']:.4f}  STS-B {result['stsb_student']:.4f}")

    log = os.path.join(out_dir, f"{stem}.ndcg.log")
    ndcg_path = os.path.join(out_dir, f"{stem}.ndcg.json")
    _say(f"{tag}: ndcg -> {log}")
    code = _run(
        [
            sys.executable,
            os.path.join("scripts", "qwen3", "retrieval_eval.py"),
            "--config",
            config,
            "--student-ckpt",
            ckpt,
            "--json",
            ndcg_path,
        ]
        + (["--tasks", tasks] if tasks else []),
        device.split(",")[0],
        log,
    )
    if code != 0 or not os.path.exists(ndcg_path):
        _say(f"{tag}: NDCG FAILED (exit {code})\n{_tail(log)}")
        return result | {"error": f"ndcg exit {code}", "log": log}

    with open(ndcg_path) as f:
        nd = json.load(f)
    # By name, not `|=`: both payloads carry `k` and `config`.
    result |= {
        "ndcg": nd["student_avg"],
        "ndcg_teacher": nd["teacher_avg"],
        "ndcg_tasks": nd["tasks"],
        "ndcg_n": len(nd.get("macro_tasks", [])),
    }
    _say(
        f"{tag}: nDCG@10 {result['ndcg']:.4f} (teacher {result['ndcg_teacher']:.4f}, "
        f"{result['ndcg_n']} tasks)"
    )
    return result


def report(results: list[dict], out_dir: str) -> None:
    """Print the summary and save it. Terminal scrollback is not storage — a remote batch has to
    leave a durable artifact, or a closed ssh session loses the only copy of an hour of GPU time."""
    ok = [r for r in results if not r.get("error")]
    out: list[str] = []

    def emit(line: str = "") -> None:
        out.append(line)
        print(line)

    emit("\n" + "=" * 78)
    emit("RESULTS — paste into franken/models/qwen3/PROGRESS.md\n")
    emit(
        "| run | depth | ops | recall@10 | vs teacher | nDCG@10 | vs teacher | ratio¹ "
        "| embed_dist | STS-B | Δ teacher | relative | min |"
    )
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        mins = f"{r['minutes']:.0f}" if r.get("minutes") else "—"
        delta = r["stsb_student"] - r["stsb_teacher"]
        # Relative to the teacher's own STS-B, which is the reference the claim is about
        # ("preserves the teacher"), not an absolute-quality score.
        rel = 100 * delta / r["stsb_teacher"]
        # recall@10 is ALREADY teacher-relative (1.0 is the ceiling), so its deficit is 1 - recall.
        recall_def = 1.0 - r["recall"]
        ndcg_def = (r["ndcg_teacher"] - r["ndcg"]) / r["ndcg_teacher"]
        ratio = f"{ndcg_def / recall_def:.2f}" if recall_def > 1e-9 else "—"
        emit(
            f"| {r['stem']} | {r['depth']} | {r['softmax']}/{r['activation']} "
            f"| {r['recall']:.4f} | {-100 * recall_def:+.1f}% "
            f"| {r['ndcg']:.4f} | {-100 * ndcg_def:+.1f}% | {ratio} "
            f"| {r['embed_dist']:.5f} | {r['stsb_student']:.4f} "
            f"| {delta:+.4f} | {rel:+.1f}% | {mins} |"
        )

    # Failures stay out of the table so the table is always pasteable, but they are listed
    # loudly: a batch that quietly dropped a run reads as "that config wasn't tried".
    for r in results:
        if r.get("error"):
            log = f"  ({r['log']})" if r.get("log") else ""
            emit(f"\nFAILED {r['stem']}: {r['error']}{log}")

    emit(
        "\n¹ nDCG deficit ÷ recall deficit. <1 means recall@10 overstates the damage; >1 means it "
        "understates it. Measured 1.6 for the depth-19 cut once every task is in the macro — it "
        "read 0.08 on the old 2-task one."
    )

    # Always print the per-task breakdown: the macro is an average over domains that move in
    # opposite directions (code -61%, Chinese +1%), and only this table shows that.
    names = sorted({t for r in ok for t in r.get("ndcg_tasks", {})})
    if names:
        emit(f"\nper-task nDCG@10, student (teacher) — the {len(names)} tasks in the macro\n")
        emit("| run | " + " | ".join(names) + " |")
        emit("|---" * (len(names) + 1) + "|")
        for r in ok:
            cells = []
            for name in names:
                task = r.get("ndcg_tasks", {}).get(name)
                cells.append(f"{task['student']:.4f} ({task['teacher']:.4f})" if task else "—")
            emit(f"| {r['stem']} | " + " | ".join(cells) + " |")
    ndcg_teachers = {round(r["ndcg_teacher"], 4) for r in ok}
    if ndcg_teachers:
        emit(f"teacher nDCG@10: {', '.join(f'{t:.4f}' for t in sorted(ndcg_teachers))}")

    teachers = {round(r["stsb_teacher"], 4) for r in ok}
    if teachers:
        # One value unless configs differ in max_seq_len — the teacher is deterministic, so a
        # second value means the runs aren't comparable on STS-B and the deltas hide it.
        emit(f"\nteacher STS-B: {', '.join(f'{t:.4f}' for t in sorted(teachers))}")
        if len(teachers) > 1:
            emit("  WARNING: teacher differs across runs (max_seq_len?) — deltas not comparable")

    for r in results:
        if r.get("trace"):
            emit(f"\n{r['stem']} training trace:")
            for line in r["trace"]:
                emit(f"  {line}")
    emit("=" * 78)

    # results.md is the printed block verbatim (copy/paste or scp it); results.json is every field
    # for later re-analysis, including the fields the table has no room for (sim_rho, pool, ckpt).
    md, js = os.path.join(out_dir, "results.md"), os.path.join(out_dir, "results.json")
    with open(md, "w") as f:
        f.write("\n".join(out) + "\n")
    with open(js, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {md}\n       {js}")


# Below this a rebuild is seconds, so a missing cache is not worth blocking on.
_PREBUILD_THRESHOLD = 100_000


def _require_corpus_cache(cfg, config_path: str) -> None:
    """Refuse to start a big embed batch with no corpus cache.

    Runs go one per device CONCURRENTLY, and `_build_split` is per process — with no cache every
    run streams and tokenizes the whole corpus independently (hours each at 11.5M texts). A fresh
    remote checkout never has one: `outputs/corpus_cache` is gitignored.
    """
    if cfg.train.task != "embed" or cfg.train.corpus_size < _PREBUILD_THRESHOLD:
        return
    from franken.data.embed_corpus import _cache_path  # noqa: PLC0415  (heavy import, rare path)
    from franken.tasks import build_task  # noqa: PLC0415

    tokenizer = build_task(cfg.train.task).build_tokenizer(cfg)
    cached = os.path.join(
        _ROOT,
        _cache_path(
            cfg.train.corpus, "train", cfg.train.corpus_size, cfg.train.max_seq_len, tokenizer
        ),
    )
    if not os.path.isdir(cached):
        raise SystemExit(
            f"{config_path}: no corpus cache for {cfg.train.corpus} "
            f"({cfg.train.corpus_size:,} texts) at {cached}\n"
            f"Build it once before the batch, or every run rebuilds it in parallel:\n"
            f"    uv run python scripts/qwen3/build_corpus.py --config {config_path}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("configs", nargs="+", help="config YAMLs to run, in reporting order")
    p.add_argument("--devices", default="0,1", help="CUDA indices to use, comma-separated")
    p.add_argument("--eval-only", action="store_true", help="score checkpoints, no training")
    p.add_argument(
        "--ddp",
        action="store_true",
        help="run configs one at a time, each across ALL --devices (torchrun), "
        "instead of one config per device",
    )
    p.add_argument("--out", default="outputs/experiments", help="logs + per-run metrics JSON")
    p.add_argument(
        "--tasks",
        help="nDCG tasks, comma-separated (default: every task in retrieval_eval). All scored "
        "tasks enter the macro, so a narrower list is a different, non-comparable number.",
    )
    args = p.parse_args(argv)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = os.path.join(_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    pending: queue.Queue = queue.Queue()
    for i, config in enumerate(args.configs):
        cfg = Config.from_yaml(config)  # fail on a bad config now, not 30 min into the batch
        _require_corpus_cache(cfg, config)
        pending.put((i, config))
    results: dict[int, dict] = {}

    def worker(device: str) -> None:
        while True:
            try:
                i, config = pending.get_nowait()
            except queue.Empty:
                return
            try:
                results[i] = one_experiment(
                    config, device, out_dir, args.eval_only, args.ddp, args.tasks
                )
            except Exception as exc:  # keep the other runs alive; report it as a row
                results[i] = {"stem": os.path.basename(config), "error": repr(exc)}

    mode = f"DDP across {len(devices)}" if args.ddp else f"queued over {len(devices)}"
    print(f"{len(args.configs)} experiment(s), {mode} device(s): {', '.join(devices)}")
    print(f"logs: {out_dir}")
    started = time.monotonic()
    # DDP already owns every card, so the queue collapses to one worker holding the whole set.
    threads = [
        threading.Thread(target=worker, args=(d,))
        for d in ([",".join(devices)] if args.ddp else devices)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report([results[i] for i in sorted(results)], out_dir)
    print(f"total wall time: {(time.monotonic() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
