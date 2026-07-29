"""Run a batch of distillation experiments across the available GPUs and print one table.

The remote workflow this exists for: pull, run this with the configs you want, copy the
final block back. Each config is distilled and then scored by `embed_eval.py`, one config
per GPU at a time, and everything a decision needs ends up in a single markdown table that
pastes straight into `franken/models/qwen3/PROGRESS.md`.

Why a runner rather than a shell loop: results are collected from `embed_eval --json`, not
scraped from prose, so a reworded print can't corrupt the table; a crashed run degrades to
one FAILED row with its log tail instead of losing the whole batch; and the GPU assignment
is a work queue, so N configs over 2 devices finish in ceil(N/2) slots even when runs differ
in length.

Devices are passed as CUDA indices and exported per-subprocess as CUDA_VISIBLE_DEVICES, so
`train.device: cuda` in the config always resolves to the intended card. Pass only cards you
actually own — a co-tenant's idle GPU is not free capacity.

Usage:
    uv run python scripts/qwen3/run_experiments.py --devices 0,1 \
        configs/qwen3/depth19.yaml configs/qwen3/depth14.yaml
    uv run python scripts/qwen3/run_experiments.py --devices 2,3 --eval-only configs/qwen3/*.yaml
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


def one_experiment(config: str, device: str, out_dir: str, eval_only: bool) -> dict:
    stem = os.path.splitext(os.path.basename(config))[0]
    tag = f"[gpu{device}] {stem}"
    ckpt = RunPaths(Config.from_yaml(config)).student_bin()
    result = {"stem": stem, "config": config, "device": device, "minutes": None, "trace": []}

    if not eval_only:
        log = os.path.join(out_dir, f"{stem}.distill.log")
        _say(f"{tag}: distill -> {log}")
        start = time.monotonic()
        code = _run([sys.executable, "main.py", "distill", "--config", config], device, log)
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
        device,
        log,
    )
    if code != 0 or not os.path.exists(metrics_path):
        _say(f"{tag}: EVAL FAILED (exit {code})\n{_tail(log)}")
        return result | {"error": f"eval exit {code}", "log": log}

    with open(metrics_path) as f:
        result |= json.load(f)
    _say(f"{tag}: recall@{result['k']} {result['recall']:.4f}  STS-B {result['stsb_student']:.4f}")
    return result


def report(results: list[dict]) -> None:
    ok = [r for r in results if not r.get("error")]
    print("\n" + "=" * 78)
    print("RESULTS — paste into franken/models/qwen3/PROGRESS.md\n")
    print("| run | depth | ops | recall@10 | embed_dist | STS-B | Δ teacher | relative | min |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        mins = f"{r['minutes']:.0f}" if r.get("minutes") else "—"
        delta = r["stsb_student"] - r["stsb_teacher"]
        # Relative to the teacher's own STS-B, which is the reference the claim is about
        # ("preserves the teacher"), not an absolute-quality score.
        rel = 100 * delta / r["stsb_teacher"]
        print(
            f"| {r['stem']} | {r['depth']} | {r['softmax']}/{r['activation']} "
            f"| {r['recall']:.4f} | {r['embed_dist']:.5f} | {r['stsb_student']:.4f} "
            f"| {delta:+.4f} | {rel:+.1f}% | {mins} |"
        )

    # Failures stay out of the table so the table is always pasteable, but they are listed
    # loudly: a batch that quietly dropped a run reads as "that config wasn't tried".
    for r in results:
        if r.get("error"):
            log = f"  ({r['log']})" if r.get("log") else ""
            print(f"\nFAILED {r['stem']}: {r['error']}{log}")

    teachers = {round(r["stsb_teacher"], 4) for r in ok}
    if teachers:
        # One value unless configs differ in max_seq_len — the teacher is deterministic, so a
        # second value means the runs aren't comparable on STS-B and the deltas hide it.
        print(f"\nteacher STS-B: {', '.join(f'{t:.4f}' for t in sorted(teachers))}")
        if len(teachers) > 1:
            print("  WARNING: teacher differs across runs (max_seq_len?) — deltas not comparable")

    for r in results:
        if r.get("trace"):
            print(f"\n{r['stem']} training trace:")
            for line in r["trace"]:
                print(f"  {line}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("configs", nargs="+", help="config YAMLs to run, in reporting order")
    p.add_argument("--devices", default="0,1", help="CUDA indices to use, comma-separated")
    p.add_argument("--eval-only", action="store_true", help="score checkpoints, no training")
    p.add_argument("--out", default="outputs/experiments", help="logs + per-run metrics JSON")
    args = p.parse_args(argv)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = os.path.join(_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    pending: queue.Queue = queue.Queue()
    for i, config in enumerate(args.configs):
        Config.from_yaml(config)  # fail on a bad config now, not 30 min into the batch
        pending.put((i, config))
    results: dict[int, dict] = {}

    def worker(device: str) -> None:
        while True:
            try:
                i, config = pending.get_nowait()
            except queue.Empty:
                return
            try:
                results[i] = one_experiment(config, device, out_dir, args.eval_only)
            except Exception as exc:  # keep the other runs alive; report it as a row
                results[i] = {"stem": os.path.basename(config), "error": repr(exc)}

    print(f"{len(args.configs)} experiment(s) over {len(devices)} device(s): {', '.join(devices)}")
    print(f"logs: {out_dir}")
    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report([results[i] for i in sorted(results)])
    print(f"total wall time: {(time.monotonic() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
