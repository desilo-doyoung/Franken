"""Run a batch of distillation experiments across the available GPUs and print one table.

Per config: corpus gates (once per corpus), distill, then `eval.py`. Everything a decision needs
lands in one markdown table for PROGRESS.md. Numbers come from each scorer's `--json`, never
scraped from prose, so a reworded print cannot corrupt the table; a crashed run degrades to one
FAILED row with its log tail.

Configs are a work queue over the given cards, so N configs cost ceil(N/devices) slots. Pass only
cards you actually own -- a co-tenant's idle GPU is not free capacity. `--ddp` switches the other
way: one config at a time over ALL devices. The queue maximizes throughput; DDP maximizes what a
SINGLE run can absorb.

⚠️ `token_budget` is PER RANK, so tokens/step scales with the device count and steps/epoch is
data-dependent -- the trainer logs the real count at startup and derives lr from it.

Usage:
    uv run python scripts/qwen3/run_experiments.py --devices 2,3 configs/qwen3/depth19*.yaml
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

import common
from franken.config import Config
from franken.paths import RunPaths

_ROOT = common.ROOT

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
    # `-m torch.distributed.run` rather than the `torchrun` console script, so the ranks inherit
    # exactly this interpreter -- on a remote box the bare `torchrun` on PATH is often another venv.
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
    cfg = Config.from_yaml(config)
    ckpt = RunPaths(cfg).student_bin()
    nproc = len(device.split(",")) if ddp else 1
    # depth/ops come from the config, not the scorer: they label the run, and `report` needs them
    # for every row including a FAILED one.
    result = {
        "stem": stem,
        "config": config,
        "device": device,
        "minutes": None,
        "trace": [],
        "depth": cfg.model.num_hidden_layers,
        "softmax": cfg.model.softmax,
        "activation": cfg.model.activation,
    }
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
            os.path.join("scripts", "qwen3", "eval.py"),
            "--config",
            config,
            "--student-ckpt",
            ckpt,
            "--json",
            metrics_path,
        ]
        + (["--tasks", tasks] if tasks else []),
        device.split(",")[0],  # scoring is single-process; don't hand it the whole DDP card set
        log,
    )
    if code != 0 or not os.path.exists(metrics_path):
        _say(f"{tag}: EVAL FAILED (exit {code})\n{_tail(log)}")
        return result | {"error": f"eval exit {code}", "log": log}

    with open(metrics_path) as f:
        ev = json.load(f)
    fid, ext = ev.get("fidelity", {}), ev.get("external", {}).get("macro", {})
    result |= {
        "k": ev["k"],
        "recall": fid.get(f"recall@{ev['k']}"),
        "embed_dist": fid.get("embed_dist"),
        "stsb_teacher": fid.get("stsb_teacher"),
        "stsb_student": fid.get("stsb_student"),
        "pool": fid.get("pool"),
        "ndcg": ext.get("student"),
        "ndcg_teacher": ext.get("teacher"),
        "ndcg_n": ext.get("n"),
        "ndcg_tasks": ev.get("external", {}).get("tasks", {}),
        "macro_pair": ev.get("corpus", {}).get("macro_pair"),
        "macro_qrels": ev.get("corpus", {}).get("macro_qrels"),
        "coverage_gap": ev.get("coverage_gap"),
    }
    _say(
        f"{tag}: recall@{result['k']} {result['recall']:.4f}  "
        f"nDCG@10 {result['ndcg']:.4f} (teacher {result['ndcg_teacher']:.4f})  "
        f"coverage gap {result['coverage_gap']:+.1f}%"
        if result.get("ndcg") and result.get("coverage_gap") is not None
        else f"{tag}: recall@{result['k']} {result['recall']:.4f}"
    )
    return result


def report(results: list[dict], out_dir: str) -> None:
    # Saved as well as printed: scrollback is not storage, and a closed ssh session would lose the
    # only copy of an hour of GPU time.
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


def _cache_missing(cfg) -> bool:
    # Runs go one per device CONCURRENTLY and `_build_split` is per process, so with no cache every
    # run tokenizes the whole corpus independently -- hours each at 9M texts.
    from franken.data.embed_corpus import cache_path  # noqa: PLC0415  (heavy import, rare path)
    from franken.tasks import build_task  # noqa: PLC0415

    tokenizer = build_task(cfg.train.task).build_tokenizer(cfg)
    cached = os.path.join(
        _ROOT,
        cache_path(
            cfg.train.corpus, "train", cfg.train.corpus_size, cfg.train.max_seq_len, tokenizer
        ),
    )
    return not os.path.isdir(cached)


def _corpus(cfg, config_path: str, out_dir: str, build: bool) -> None:
    # Gates (holdout, every source loading and scoreable, corpus_size still matching) are pure
    # checks, cheap next to a distill. corpus.py decides whether to build; `build` only logs it.
    if cfg.train.task != "embed" or cfg.train.corpus_size < _PREBUILD_THRESHOLD:
        return
    log = os.path.join(out_dir, "corpus.log")
    _say(f"corpus: gates{' + BUILD (hours)' if build else ''} -> {log}")
    code = _run(
        [sys.executable, os.path.join("scripts", "qwen3", "corpus.py"), "--config", config_path],
        "",  # no GPU needed
        log,
    )
    # 134/-6 is SIGABRT from an HF retry thread during interpreter shutdown, historically AFTER the
    # results print. corpus.py exits via os._exit to dodge it, but trust the verdict over the code.
    if code != 0 and not (code in (134, -6) and "CORPUS OK" in _tail(log, 4)):
        raise SystemExit(f"corpus FAILED (exit {code}) — not training\n{_tail(log)}")


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
        help="external nDCG tasks, comma-separated (default: all of them). All scored "
        "tasks enter the macro, so a narrower list is a different, non-comparable number.",
    )
    args = p.parse_args(argv)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = os.path.join(_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    pending: queue.Queue = queue.Queue()
    done: set[str] = set()  # the corpus step is per CORPUS, not per config
    for i, config in enumerate(args.configs):
        cfg = Config.from_yaml(config)  # fail on a bad config now, not 30 min into the batch
        if cfg.train.corpus not in done:
            missing = _cache_missing(cfg)
            if args.eval_only and missing:
                raise SystemExit(
                    f"{config}: no corpus cache for {cfg.train.corpus} "
                    f"({cfg.train.corpus_size:,} texts) and --eval-only will not build one.\n"
                    f"    uv run python scripts/qwen3/corpus.py --config {config}"
                )
            if not args.eval_only:
                _corpus(cfg, config, out_dir, build=missing)
            done.add(cfg.train.corpus)
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
