"""Run a batch of distillation experiments across the given GPUs and print one table.

Per config: corpus gates (once per corpus), distill, then eval. Numbers come from each scorer's
`--json`, never scraped from prose. A crashed run degrades to one FAILED row with its log tail.

Configs are a work queue over the cards; `--ddp` instead runs one config at a time across all of
them. Pass only cards you own. `tokens_per_step` is global, so results do not move with it.

Usage:
    uv run python -m franken.scripts.qwen3.run_experiments --devices 2,3 configs/qwen3/depth19*.yaml
    uv run python -m franken.scripts.qwen3.run_experiments --devices 2,3 --eval-only \
        configs/qwen3/*.yaml
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

from franken.config import Config
from franken.paths import RunPaths
from franken.scripts import common

_ROOT = common.ROOT

_PRINT_LOCK = threading.Lock()


def deficits(r: dict) -> dict:
    """The derived columns, out of the f-strings so they can be checked without a GPU run. Empty
    when a suite did not run, so callers test the dict instead of each ingredient."""
    if any(r.get(k) is None for k in ("recall", "ndcg", "ndcg_teacher", "stsb_teacher")):
        return {}
    # recall@10 is already teacher-relative, so its deficit is 1 - recall.
    recall_def = 1.0 - r["recall"]
    ndcg_def = (r["ndcg_teacher"] - r["ndcg"]) / r["ndcg_teacher"]
    delta = r["stsb_student"] - r["stsb_teacher"]
    return {
        "recall_def": recall_def,
        "ndcg_def": ndcg_def,
        # Undefined when the student matched the teacher.
        "ratio": ndcg_def / recall_def if recall_def > 1e-9 else None,
        "stsb_delta": delta,
        # Relative to the teacher's own STS-B, the reference the claim is about.
        "stsb_rel": 100 * delta / r["stsb_teacher"],
    }


def _say(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def _run(cmd: list[str], device: str, log_path: str) -> int:
    """One subprocess pinned to `device`, output streamed to `log_path`."""
    env = os.environ | {"CUDA_VISIBLE_DEVICES": device}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def _tail(path: str, n: int = 15) -> str:
    with open(path) as f:
        return "".join(f.readlines()[-n:])


def _trace(path: str) -> list[str]:
    """The per-epoch metric lines from a distill log."""
    with open(path) as f:
        return [ln.rstrip() for ln in f if ln.startswith(("init:", "epoch "))]


def _distill_cmd(config: str, nproc: int) -> list[str]:
    # `-m torch.distributed.run`, not the `torchrun` script: the ranks must inherit THIS venv.
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
    ckpt = RunPaths(cfg).student_bin
    nproc = len(device.split(",")) if ddp else 1
    # From the config, not the scorer: `report` needs them for FAILED rows too.
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
            "-m",
            "franken.scripts.qwen3.eval",
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
    # Folded in, not left to render(): otherwise the derived columns reach neither results.json
    # nor the table once a column is demoted.
    result |= deficits(result)
    _say(
        f"{tag}: recall@{result['k']} {result['recall']:.4f}  "
        f"nDCG@10 {result['ndcg']:.4f} (teacher {result['ndcg_teacher']:.4f})  "
        f"coverage gap {result['coverage_gap']:+.1f}%"
        if result.get("ndcg") and result.get("coverage_gap") is not None
        else f"{tag}: recall@{result['k']} {result['recall']:.4f}"
    )
    return result


def _rel(macro: dict | None) -> str:
    if not macro or not macro.get("teacher"):
        return "—"
    return f"{100 * (macro['student'] - macro['teacher']) / macro['teacher']:+.1f}%"


def render(results: list[dict]) -> list[str]:
    """The results block, as lines. Pure: no printing, no files, no GPU."""
    ok = [r for r in results if not r.get("error")]
    out: list[str] = ["\n" + "=" * 78, "RESULTS — paste into franken/models/qwen3/PROGRESS.md\n"]
    out.append("| run | depth | ops | recall@10¹ | embed_dist¹ | external² | min |")
    out.append("|---|---|---|---|---|---|---|")
    for r in ok:
        d = deficits(r)
        cells = [
            r["stem"],
            str(r["depth"]),
            f"{r['softmax']}/{r['activation']}",
            f"{r['recall']:.4f}" if r.get("recall") is not None else "—",
            f"{r['embed_dist']:.6f}" if r.get("embed_dist") is not None else "—",
            f"{-100 * d['ndcg_def']:+.1f}%" if d else "—",
            f"{r['minutes']:.0f}" if r.get("minutes") else "—",
        ]
        out.append("| " + " | ".join(cells) + " |")

    # Out of the table so it stays pasteable, but listed loudly.
    for r in results:
        if r.get("error"):
            log = f"  ({r['log']})" if r.get("log") else ""
            out.append(f"\nFAILED {r['stem']}: {r['error']}{log}")

    out.append(
        "\n¹ both gold-free, both vs THIS teacher on the held-out validation pool. They fail "
        "differently and that is the\n  point: recall@10 catches a moved RANKING, embed_dist "
        "(1 − mean cosine) catches moved GEOMETRY that a\n  ranking metric can absorb. "
        "embed_dist is also pool-size independent, where recall@10's difficulty is k/(n−1).\n"
        "² nDCG@10 vs teacher, macro over the external tasks. In-distribution retention is not a "
        "column because it does\n  not move — −0.2%/−0.2%/+0.1% across a 9-layer cut plus op "
        "replacement — so the coverage gap equals this\n  column minus a constant. It stays in "
        "results.json. The MAGNITUDE here is diluted (scifact and xpqa_cmn are\n  flat across the "
        "same cut); read the per-task table, and calibrate against depth28_exact at +0.3%."
    )

    # The macro averages tasks that move in opposite directions; only this shows that.
    names = sorted({t for r in ok for t in r.get("ndcg_tasks", {})})
    if names:
        out.append(
            f"\nper-task external nDCG@{10}, relative to teacher — the {len(names)} tasks "
            f"in the macro\n"
        )
        out.append("| run | " + " | ".join(names) + " |")
        out.append("|---" * (len(names) + 1) + "|")
        for r in ok:
            cells = []
            for name in names:
                task = r.get("ndcg_tasks", {}).get(name)
                cells.append(_rel(task) if task else "—")
            out.append(f"| {r['stem']} | " + " | ".join(cells) + " |")
    ndcg_teachers = {round(r["ndcg_teacher"], 4) for r in ok if r.get("ndcg_teacher")}
    if ndcg_teachers:
        out.append(f"\nteacher nDCG@10: {', '.join(f'{t:.4f}' for t in sorted(ndcg_teachers))}")

    teachers = {round(r["stsb_teacher"], 4) for r in ok if r.get("stsb_teacher")}
    if teachers:
        # The teacher is deterministic, so a second value means the runs are not comparable.
        out.append(f"\nteacher STS-B: {', '.join(f'{t:.4f}' for t in sorted(teachers))}")
        if len(teachers) > 1:
            out.append(
                "  WARNING: teacher differs across runs (max_seq_len?) — deltas not comparable"
            )

    for r in results:
        if r.get("trace"):
            out.append(f"\n{r['stem']} training trace:")
            out.extend(f"  {line}" for line in r["trace"])
    out.append("=" * 78)
    return out


def report(results: list[dict], out_dir: str) -> None:
    # Saved as well as printed: scrollback is not storage.
    out = render(results)
    print("\n".join(out))

    # results.md is the printed block verbatim; results.json keeps the fields the table drops.
    md, js = os.path.join(out_dir, "results.md"), os.path.join(out_dir, "results.json")
    with open(md, "w") as f:
        f.write("\n".join(out) + "\n")
    with open(js, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {md}\n       {js}")


# Below this a rebuild is seconds, so a missing cache is not worth blocking on.
_PREBUILD_TOKENS = 1e7


def _cache_missing(cfg) -> bool:
    # Runs go one per device concurrently, so with no cache each tokenizes the whole corpus.
    from franken.data.corpus import (
        train_cache_path,  # noqa: PLC0415  (heavy import, rare path)
    )
    from franken.tasks import build_task  # noqa: PLC0415

    tokenizer = build_task(cfg.train.task).build_tokenizer(cfg)
    cached = os.path.join(
        _ROOT,
        train_cache_path(
            cfg.train.corpus, cfg.train.tokens_per_epoch, cfg.train.max_seq_len, tokenizer
        ),
    )
    return not os.path.isdir(cached)


def _corpus(cfg, config_path: str, out_dir: str, build: bool) -> None:
    # Pure checks, cheap next to a distill. corpus.py decides whether to build; `build` logs it.
    if cfg.train.task != "embed" or cfg.train.tokens_per_epoch < _PREBUILD_TOKENS:
        return
    log = os.path.join(out_dir, "corpus.log")
    _say(f"corpus: gates{' + BUILD (hours)' if build else ''} -> {log}")
    code = _run(
        [sys.executable, "-m", "franken.scripts.qwen3.corpus", "--config", config_path],
        "",  # no GPU needed
        log,
    )
    # 134/-6 is SIGABRT from an HF retry thread at shutdown; trust the verdict over the code.
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
        cfg = Config.from_yaml(config)  # fail now, not 30 min into the batch
        if cfg.train.corpus not in done:
            missing = _cache_missing(cfg)
            if args.eval_only and missing:
                raise SystemExit(
                    f"{config}: no corpus cache for {cfg.train.corpus} "
                    f"({cfg.train.tokens_per_epoch:,.0f} tokens) and --eval-only will not\n"
                    f"    build one.\n"
                    f"    uv run python -m franken.scripts.qwen3.corpus --config {config}"
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
            except Exception as exc:  # keep the other runs alive
                results[i] = {"stem": os.path.basename(config), "error": repr(exc)}

    mode = f"DDP across {len(devices)}" if args.ddp else f"queued over {len(devices)}"
    print(f"{len(args.configs)} experiment(s), {mode} device(s): {', '.join(devices)}")
    print(f"logs: {out_dir}")
    started = time.monotonic()
    # DDP owns every card, so the queue collapses to one worker.
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
