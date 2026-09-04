"""Run the lm ladder across the given GPUs and print one table.

Per config: build the corpus (once per cache key), distill, then score. Everything goes through
`main.py`, so the TASK registry picks the corpus builder and the scorer -- the embed track's runner
hardcodes `franken.scripts.qwen3.{corpus,eval}` and is therefore unusable here.

Configs are a work queue over the cards; `--ddp` instead runs one config at a time across all of
them. `tokens_per_step` is global, so results do not move with the device count.

Usage:
    uv run python -m franken.scripts.llama.run_experiments --devices 0,1,2,3 --ddp \
        configs/llama/depth16_exact.yaml configs/llama/depth12_exact.yaml
    uv run python -m franken.scripts.llama.run_experiments --devices 0,1 --eval-only \
        configs/llama/*.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

from franken.config import Config
from franken.paths import RunPaths
from franken.scripts import common

_ROOT = common.ROOT
_PRINT_LOCK = threading.Lock()

# Columns the lm objective actually produces. `agreement` selects, `ppl` reports -- the same split
# of roles as recall@10 vs nDCG on the embed track.
_COLUMNS = ("agreement", "kl", "ppl", "teacher_ppl")


def _say(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def _run(cmd: list[str], devices: str, log_path: str) -> int:
    env = os.environ | ({"CUDA_VISIBLE_DEVICES": devices} if devices else {})
    with open(log_path, "w") as log:
        return subprocess.run(
            cmd, cwd=_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def _tail(path: str, n: int = 15) -> str:
    with open(path) as f:
        return "".join(f.readlines()[-n:])


def _trace(path: str) -> list[str]:
    """The lines worth surfacing from a distill log: the per-epoch metrics and the memory peak."""
    with open(path) as f:
        return [ln.rstrip() for ln in f if ln.startswith(("init:", "epoch ", "peak GPU"))]


def _cmd(action: str, config: str, nproc: int = 1, *extra: str) -> list[str]:
    # `-m torch.distributed.run`, not the `torchrun` script: the ranks must inherit THIS venv.
    argv = ["main.py", action, "--config", config, *extra]
    if nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", *argv]
    return [sys.executable, *argv]


def _cache_missing(cfg) -> bool:
    # Without this every concurrent run re-streams and re-tokenizes the whole corpus.
    from franken.data.corpus import cache_missing  # noqa: PLC0415  (heavy import, rare path)
    from franken.tasks import build_task  # noqa: PLC0415

    return cache_missing(cfg, build_task(cfg.train.task).build_tokenizer(cfg))


def build_corpus(config: str, out_dir: str) -> None:
    """Serial and before anything launches, because the cache is the shared prerequisite."""
    name = os.path.splitext(os.path.basename(config))[0]
    log = os.path.join(out_dir, f"{name}.corpus.log")
    _say(f"corpus: BUILDING (hours) -> {log}")
    code = _run(_cmd("corpus", config), "", log)  # CPU + network; no GPU
    if code != 0 and "CORPUS OK" not in _tail(log, 4):
        raise SystemExit(f"corpus FAILED (exit {code}) -- not training\n{_tail(log)}")


def _drop_checkpoint(ckpt: str) -> None:
    """Called only after the eval has read it. The metrics are the deliverable; the weights are
    3-5 GiB each and a full depth x beta x lr grid is ~185 GiB. A FAILED eval keeps its checkpoint,
    so a scoring bug never costs the training run."""
    path = os.path.join(_ROOT, ckpt)
    if not os.path.isdir(path):
        return
    freed = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
    shutil.rmtree(path)
    _say(f"  removed {ckpt} ({freed / 2**30:.1f} GiB)")


def one_experiment(
    config: str, devices: str, out_dir: str, eval_only: bool, ddp: bool, rm_ckpt: bool = False
) -> dict:
    name = os.path.splitext(os.path.basename(config))[0]
    cfg = Config.from_yaml(config)
    nproc = len(devices.split(",")) if ddp else 1
    row: dict = {"name": name, "layers": cfg.model.num_hidden_layers, "seq": cfg.train.max_seq_len}

    if not eval_only:
        log = os.path.join(out_dir, f"{name}.distill.log")
        _say(f"[{devices}] distill {name} -> {log}")
        if code := _run(_cmd("distill", config, nproc), devices, log):
            return row | {"error": f"distill exit {code}", "tail": _tail(log)}
        row["trace"] = _trace(log)

    metrics_path = os.path.join(out_dir, f"{name}.metrics.json")
    log = os.path.join(out_dir, f"{name}.eval.log")
    ckpt = RunPaths(cfg).student
    _say(f"[{devices}] eval {name} -> {log}")
    # --split test: validation selected the checkpoint, so reporting on it would score its own pool.
    argv = _cmd("eval", config, 1, "--json", metrics_path, "--split", "test")
    if os.path.isdir(os.path.join(_ROOT, ckpt)):
        argv += ["--ckpt", ckpt]
    if code := _run(argv, devices, log):
        return row | {"error": f"eval exit {code}", "tail": _tail(log)}

    with open(metrics_path) as f:
        row |= json.load(f)
    # Never under --eval-only: there we did not produce the checkpoint, the user is re-scoring it.
    if rm_ckpt and not eval_only:
        _drop_checkpoint(ckpt)
    return row


def render(results: list[dict]) -> list[str]:
    head = "| config | layers | seq | " + " | ".join(_COLUMNS) + " | ppl delta |"
    out = [head, "|" + "---|" * (len(_COLUMNS) + 4)]
    for r in results:
        if "error" in r:
            out.append(
                f"| {r['name']} | {r['layers']} | {r['seq']} | " + "FAILED | " * (len(_COLUMNS) + 1)
            )
            continue
        cells = " | ".join(f"{r[c]:.4f}" if r.get(c) is not None else "—" for c in _COLUMNS)
        delta = f"{r['ppl'] / r['teacher_ppl'] - 1:+.2%}" if r.get("teacher_ppl") else "—"
        out.append(f"| {r['name']} | {r['layers']} | {r['seq']} | {cells} | {delta} |")
    return out


def render_by_source(results: list[dict]) -> list[str]:
    """One column per config, so a depth cut or an ablation is read down the source axis -- which
    slice it cost is the question an aggregate row cannot answer."""
    scored = [r for r in results if r.get("by_source")]
    if not scored:
        return []
    names = list(dict.fromkeys(n for r in scored for n in r["by_source"]))
    out = [
        "",
        "| agreement by source | " + " | ".join(r["name"] for r in scored) + " |",
        "|" + "---|" * (len(scored) + 1),
    ]
    for n in names:
        cells = " | ".join(
            f"{r['by_source'][n]['agreement']:.4f}" if n in r["by_source"] else "—" for r in scored
        )
        out.append(f"| {n} | {cells} |")
    return out


def report(results: list[dict], out_dir: str) -> None:
    lines = render(results) + render_by_source(results)
    path = os.path.join(out_dir, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    _say("\n" + "\n".join(lines) + f"\n\nwrote {path}")
    for r in results:
        for ln in r.get("trace", []):
            _say(f"  {r['name']}: {ln}")
        if "error" in r:
            _say(f"  {r['name']}: {r['error']}\n{r.get('tail', '')}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("configs", nargs="+", help="config YAMLs to run, in reporting order")
    p.add_argument("--devices", default="0,1", help="CUDA indices to use, comma-separated")
    p.add_argument("--ddp", action="store_true", help="one config across all devices, in turn")
    p.add_argument("--eval-only", action="store_true", help="score checkpoints, no training")
    p.add_argument("--out", default="outputs/experiments", help="logs + per-run metrics JSON")
    p.add_argument(
        "--rm-checkpoint",
        action="store_true",
        help="delete each student checkpoint once its eval succeeds (~4 GiB per cell)",
    )
    args = p.parse_args(argv)

    out_dir = os.path.join(_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    if not args.eval_only:
        for config in dict.fromkeys(args.configs):  # one build per distinct cache key
            cfg = Config.from_yaml(config)
            if _cache_missing(cfg):
                build_corpus(config, out_dir)

    results: list[dict] = []
    if args.ddp:
        for config in args.configs:
            results.append(
                one_experiment(
                    config, args.devices, out_dir, args.eval_only, True, args.rm_checkpoint
                )
            )
    else:
        free: queue.Queue = queue.Queue()
        for d in args.devices.split(","):
            free.put(d)
        slots: dict[str, dict] = {}

        def work(config: str) -> None:
            device = free.get()
            try:
                slots[config] = one_experiment(
                    config, device, out_dir, args.eval_only, False, args.rm_checkpoint
                )
            finally:
                free.put(device)

        threads = [threading.Thread(target=work, args=(c,)) for c in args.configs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [slots[c] for c in args.configs if c in slots]

    report(results, out_dir)


if __name__ == "__main__":
    main()
