"""Seed sweep: best teacher seed by lowest validation CE, then the best student seed on top of it
by validation F1 *and* two FHE risk numbers -- see `select_winner`.

Every seed's checkpoint is kept: nothing here forces deterministic cuBLAS/cuDNN reductions, so
re-running a seed gives different risk numbers -- what was measured has to be what ships.

Seeds are split across --gpus, one single-GPU worker subprocess per chunk -- HF Trainer's
DataParallel would change the effective batch size and break bit-reproducibility.

Usage:
    # Orchestrate across GPUs 2 and 3 (default):
    uv run python -m franken.scripts.bert.seed_sweep --config configs/bert/depth8_exact.yaml \
        --seeds 42-51 --gpus 2,3 --sweep-dir outputs/bert/seed_sweep \
        --student-out outputs/bert/student

    # Re-select from an existing sweep at a different threshold (no GPU):
    uv run python -m franken.scripts.bert.seed_sweep select \
        --sweep-dir outputs/bert/seed_sweep --student-out outputs/bert/student --flip-margin 0.2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from franken.config import Config
from franken.data.mrpc import compute_metrics, load_mrpc
from franken.distill.trainer import Distiller
from franken.models import build_backend
from franken.tasks import build_task
from thor.measure_ranges import POOLER_TANH_WALL

MODULE = "franken.scripts.bert.seed_sweep"  # workers are relaunched as `python -m MODULE`

# The FHE runtime runs at 64 tokens and truncation moves CLS, so the risk numbers are only
# comparable to an encrypted run when measured here -- not at cfg.train.max_seq_len.
FHE_MAX_SEQ_LEN = 64
# A flip needs the noise to eat the whole margin, so the count below a threshold IS the exposed
# set. Measured on 408 encrypted examples: noise moved a margin by at most 0.265.
MARGIN_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)


# --------------------------------------------------------------------------- utils
def parse_seeds(spec: str) -> list[int]:
    """'42-51' -> [42..51]; '42,44,46' -> [42,44,46]; also a mix of both."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def split_chunks(items: list, n: int) -> list[list]:
    """Split into n near-equal contiguous chunks (drops empty trailing chunks)."""
    k, r = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        out.append(items[i : i + size])
        i += size
    return [c for c in out if c]


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------- teacher work
@torch.no_grad()
def score_teacher(
    ckpt_dir: str, cfg: Config, device: torch.device, backend, task
) -> dict[str, float]:
    """Validation CE (mean) + F1/acc for a saved teacher checkpoint."""
    tokenizer = task.build_tokenizer(cfg)
    data = task.datasets(tokenizer, cfg)
    val = data["validation"].with_format("torch", columns=task.torch_columns())
    dl = DataLoader(val, batch_size=64, collate_fn=data["collator"])

    cfg.train.teacher_ckpt = ckpt_dir  # backend.load_teacher reads this
    model = backend.load_teacher(cfg).to(device)
    model.eval()

    ce_sum, n = 0.0, 0
    logits_all, labels_all = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = backend.forward(model, task.model_inputs(batch))
        logits, labels = out["output"], batch["labels"]
        ce_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        n += labels.numel()
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())

    m = compute_metrics(torch.cat(logits_all).argmax(-1).numpy(), torch.cat(labels_all).numpy())
    del model
    _free()
    return {"val_ce": ce_sum / n, "val_acc": m["accuracy"], "val_f1": m["f1"]}


def cmd_teacher_worker(args: argparse.Namespace) -> None:
    """Train + score one teacher per seed in this worker's chunk; write results JSON."""
    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend = build_backend(cfg.model.backend)
    task = build_task(cfg.train.task)
    seeds = parse_seeds(args.seeds)
    results = []
    for seed in seeds:
        run_dir = os.path.join(args.sweep_dir, "teacher", f"seed{seed}")
        cfg.train.seed = seed
        cfg.train.output_dir = run_dir  # train_teacher saves to <output_dir>/teacher
        print(f"\n=== [teacher] seed {seed} -> {run_dir}/teacher ===", flush=True)
        ckpt = task.train_teacher(cfg)
        # Trim the Trainer's per-epoch checkpoint-* dirs (optimizer states etc.):
        # only the saved best model under <ckpt> is needed downstream.
        for name in os.listdir(ckpt):
            if name.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(ckpt, name), ignore_errors=True)
        _free()
        scores = score_teacher(ckpt, cfg, device, backend, task)
        print(f"[teacher] seed {seed}: {scores}", flush=True)
        results.append({"seed": seed, "ckpt": ckpt, **scores})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------- student work
@torch.no_grad()
def score_student_split(
    student, tokenizer, split: str, max_seq_len: int, device, backend, task, risk: bool = False
) -> dict:
    """Accuracy/F1, plus the two FHE risk quantities when `risk`. A flip is a wrong label and a
    breach is a divergence -- different failure classes, so they are never summed."""
    data = load_mrpc(tokenizer, max_seq_len, splits=(split,))
    ds = data[split].with_format("torch", columns=task.torch_columns())
    dl = DataLoader(ds, batch_size=64, collate_fn=data["collator"])
    pooler_preact = []
    hook = (
        backend.pooler_preact_modules(student)[0].register_forward_hook(
            lambda module, inputs, out: pooler_preact.append(out.detach().float().abs().cpu())
        )
        if risk
        else None
    )
    student.eval()
    logits, labels = [], []
    try:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = backend.forward(student, task.model_inputs(batch))
            logits.append(out["output"].detach().float().cpu())
            labels.append(batch["labels"].cpu())
    finally:
        if hook is not None:
            hook.remove()
    logits, labels = torch.cat(logits), torch.cat(labels)
    scores = compute_metrics(logits.argmax(-1).numpy(), labels.numpy())
    if not risk:
        return scores
    margins = (logits[:, 1] - logits[:, 0]).abs()
    pooler = torch.cat(pooler_preact).flatten()
    return {
        **scores,
        "examples": int(labels.numel()),
        "flip_exposure": {f"{t:g}": int((margins < t).sum()) for t in MARGIN_THRESHOLDS},
        # Not scale-free: larger logits look safer without being safer. `select` flags a spread.
        "margin_median": margins.median().item(),
        # The whole tail, not a top-N: counts run to ~60 per split and a truncated list would
        # silently saturate a later re-threshold.
        "margins_exposed": sorted(m for m in margins.tolist() if m < max(MARGIN_THRESHOLDS)),
        "pooler_breaches": int((pooler > POOLER_TANH_WALL).sum()),
        "pooler_max": pooler.max().item(),
        "pooler_median": pooler.median().item(),
    }


def cmd_student_worker(args: argparse.Namespace) -> None:
    """Distil one student per seed in this worker's chunk from the fixed best teacher.

    Scores and saves the best-VAL-F1 epoch. Accuracy/F1 stay at the config's max_seq_len to stay
    comparable with earlier sweeps; the risk numbers are measured at FHE_MAX_SEQ_LEN.
    """
    cfg = Config.from_yaml(args.config)
    cfg.train.teacher_ckpt = args.teacher_ckpt
    backend = build_backend(cfg.model.backend)
    task = build_task(cfg.train.task)
    seeds = parse_seeds(args.seeds)
    students = os.path.join(args.sweep_dir, "students")
    os.makedirs(students, exist_ok=True)
    results = []
    for seed in seeds:
        cfg.train.seed = seed
        print(f"\n=== [student] seed {seed} (teacher={args.teacher_ckpt}) ===", flush=True)
        d = Distiller(cfg)
        d.setup()
        d.train()  # restores this run's best-val-F1 checkpoint in-place
        v = d.evaluate()  # val metrics of the restored best checkpoint
        t = score_student_split(
            d.student, d.tokenizer, "test", cfg.train.max_seq_len, d.device, backend, task
        )
        risk = {
            split: score_student_split(
                d.student, d.tokenizer, split, FHE_MAX_SEQ_LEN, d.device, backend, task, risk=True
            )
            for split in ("validation", "test")
        }
        checkpoint = os.path.join(students, f"seed{seed}.pt")
        torch.save({k: vv.detach().cpu() for k, vv in d.student.state_dict().items()}, checkpoint)
        row = {
            "seed": seed,
            "val_acc": v["accuracy"],
            "val_f1": v["f1"],
            "test_acc": t["accuracy"],
            "test_f1": t["f1"],
            "checkpoint": checkpoint,
            "risk": risk,
            # Both splits: these are label-free functions of the network, so combining them is
            # resolution (2133 examples), not leakage.
            "pooler_breaches": sum(r["pooler_breaches"] for r in risk.values()),
            "pooler_max": max(r["pooler_max"] for r in risk.values()),
            "flip_exposure": {
                f"{t_:g}": sum(r["flip_exposure"][f"{t_:g}"] for r in risk.values())
                for t_ in MARGIN_THRESHOLDS
            },
        }
        print(
            f"[student] seed {seed}: val_f1={v['f1']:.4f} test_f1={t['f1']:.4f} "
            f"flip<0.3={row['flip_exposure']['0.3']} breaches={row['pooler_breaches']} "
            f"pooler_max={row['pooler_max']:.2f}",
            flush=True,
        )
        results.append(row)
        del d
        _free()

    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)


# ------------------------------------------------------------------------- export
def export_student(state: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "pytorch_model.bin")
    st_path = os.path.join(out_dir, "model.safetensors")
    torch.save(state, bin_path)
    # safetensors requires contiguous, non-shared tensors; the model has no tied
    # weights (classification head only), so a plain contiguous copy is safe.
    save_file({k: v.contiguous() for k, v in state.items()}, st_path, metadata={"format": "pt"})
    print(f"\nExported best student -> {bin_path}\n                        {st_path}", flush=True)


# ---------------------------------------------------------------------- selection
def select_winner(
    rows: list[dict], val_f1_band: float, flip_margin: float
) -> tuple[dict, str, list[dict]]:
    """Best val F1 within a band, then fewest flip-exposed examples; breaches veto rather than
    score, since that count moves run to run and ranking on it would be ranking on noise."""
    key = f"{flip_margin:g}"
    floor = max(r["val_f1"] for r in rows) - val_f1_band
    pool = [r for r in rows if r["val_f1"] >= floor]
    clean = [r for r in pool if r["pooler_breaches"] == 0]
    smallest = min(MARGIN_THRESHOLDS)
    winner = min(
        clean or pool,
        key=lambda r: (r["flip_exposure"][key], r["flip_exposure"][f"{smallest:g}"], -r["val_f1"]),
    )
    rule = (
        f"val_f1 >= {floor:.4f} ({len(pool)}/{len(rows)} seeds)"
        f"{', breach-free' if clean else ', NO BREACH-FREE SEED -- the penalty is failing'}"
        f", min flip exposure at margin {flip_margin:g}"
    )
    return winner, rule, pool


def report_selection(rows: list[dict], winner: dict, rule: str, pool: list[dict]) -> None:
    print(
        f"\n{'seed':>5}{'val_f1':>9}{'test_f1':>9}{'flip<0.2':>10}{'flip<0.3':>10}"
        f"{'breach':>8}{'pooler':>9}"
    )
    for r in sorted(rows, key=lambda r: -r["val_f1"]):
        mark = " <-" if r["seed"] == winner["seed"] else ""
        print(
            f"{r['seed']:>5}{r['val_f1']:>9.4f}{r['test_f1']:>9.4f}"
            f"{r['flip_exposure']['0.2']:>10}{r['flip_exposure']['0.3']:>10}"
            f"{r['pooler_breaches']:>8}{r['pooler_max']:>9.2f}{mark}"
        )
    # Only the pool's spread matters -- a wide-scaled seed outside the F1 band never competes.
    medians = [r["risk"]["validation"]["margin_median"] for r in pool]
    lo, hi = min(medians), max(medians)
    if hi > 1.2 * lo:
        print(
            f"\n  ⚠ val margin medians span {lo:.2f}-{hi:.2f} (>20%) among the {len(pool)} seeds "
            f"in the band: their flip counts are not comparable -- a larger logit scale inflates "
            f"every margin."
        )
    print(f"\nrule: {rule}")


def cmd_select(args: argparse.Namespace) -> None:
    with open(os.path.join(args.sweep_dir, "summary.json")) as f:
        summary = json.load(f)
    rows = summary["student_results"]
    winner, rule, pool = select_winner(rows, args.val_f1_band, args.flip_margin)
    report_selection(rows, winner, rule, pool)
    print(
        f"\n>>> seed {winner['seed']}  val_f1={winner['val_f1']:.4f} "
        f"test_f1={winner['test_f1']:.4f}  {winner['checkpoint']}"
    )
    export_student(torch.load(winner["checkpoint"], map_location="cpu"), args.student_out)


# --------------------------------------------------------------------- orchestrate
def _launch_workers(argv_lists: list[tuple[int, list[str], str]]) -> None:
    """Run one worker subprocess per (gpu, argv, logfile); barrier on all of them.

    Each worker is pinned to a single GPU via CUDA_VISIBLE_DEVICES so torch sees it
    as cuda:0 (no DataParallel). Raises if any worker exits non-zero.
    """
    procs = []
    for gpu, argv, logfile in argv_lists:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        lf = open(logfile, "w")
        print(f"  -> GPU {gpu}: {' '.join(argv)}  (log: {logfile})", flush=True)
        procs.append(
            (subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT), lf, gpu)
        )
    failed = []
    for p, lf, gpu in procs:
        rc = p.wait()
        lf.close()
        if rc != 0:
            failed.append((gpu, rc))
    if failed:
        raise RuntimeError(f"worker(s) failed: {failed} — inspect the per-GPU logs")


def cmd_orchestrate(args: argparse.Namespace) -> None:
    seeds = parse_seeds(args.seeds)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    cfg = Config.from_yaml(args.config)
    os.makedirs(args.sweep_dir, exist_ok=True)
    logs = os.path.join(args.sweep_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    py = sys.executable

    print(
        f"Seeds: {seeds}\nGPUs: {gpus}\nConfig: {args.config} "
        f"(depth={cfg.model.num_hidden_layers}, softmax={cfg.model.softmax}, "
        f"activation={cfg.model.activation})",
        flush=True,
    )

    chunks = split_chunks(seeds, len(gpus))
    # (gpu, [seeds]); may be fewer than #gpus if few seeds -> truncate to chunks
    pairs = list(zip(gpus, chunks, strict=False))

    # ---- Phase 1: teacher (parallel across GPUs) ----
    if args.skip_teacher:
        best_teacher, teacher_results = args.skip_teacher, []
        print(f"\nSkipping teacher phase; using {best_teacher}", flush=True)
    else:
        print("\n### Phase 1: teacher sweep ###", flush=True)
        jobs = []
        for gpu, chunk in pairs:
            out = os.path.join(args.sweep_dir, f"teacher_gpu{gpu}.json")
            argv = [
                py,
                "-m",
                MODULE,
                "teacher-worker",
                "--config",
                args.config,
                "--seeds",
                ",".join(map(str, chunk)),
                "--sweep-dir",
                args.sweep_dir,
                "--out",
                out,
            ]
            jobs.append((gpu, argv, os.path.join(logs, f"teacher_gpu{gpu}.log")))
        _launch_workers(jobs)

        teacher_results = []
        for gpu, _ in pairs:
            with open(os.path.join(args.sweep_dir, f"teacher_gpu{gpu}.json")) as f:
                teacher_results.extend(json.load(f))
        # Selection: lowest validation CE (best-calibrated soft targets), tie-break higher val F1.
        best_t = min(teacher_results, key=lambda r: (r["val_ce"], -r["val_f1"]))
        best_teacher = best_t["ckpt"]
        print(
            f"\n>>> best teacher: seed {best_t['seed']} "
            f"(val_ce={best_t['val_ce']:.4f}, val_f1={best_t['val_f1']:.4f}) -> {best_teacher}",
            flush=True,
        )

    # ---- Phase 2: student (parallel across GPUs, fixed teacher) ----
    print("\n### Phase 2: student sweep ###", flush=True)
    jobs = []
    for gpu, chunk in pairs:
        out = os.path.join(args.sweep_dir, f"student_gpu{gpu}.json")
        argv = [
            py,
            "-m",
            MODULE,
            "student-worker",
            "--config",
            args.config,
            "--seeds",
            ",".join(map(str, chunk)),
            "--sweep-dir",
            args.sweep_dir,
            "--teacher-ckpt",
            best_teacher,
            "--out",
            out,
        ]
        jobs.append((gpu, argv, os.path.join(logs, f"student_gpu{gpu}.log")))
    _launch_workers(jobs)

    student_results = []
    for gpu, _ in pairs:
        with open(os.path.join(args.sweep_dir, f"student_gpu{gpu}.json")) as f:
            student_results.extend(json.load(f)["results"])

    winner, rule, pool = select_winner(student_results, args.val_f1_band, args.flip_margin)
    report_selection(student_results, winner, rule, pool)
    print(
        f"\n>>> best student: seed {winner['seed']} "
        f"(val_f1={winner['val_f1']:.4f}, test_f1={winner['test_f1']:.4f})",
        flush=True,
    )
    export_student(torch.load(winner["checkpoint"], map_location="cpu"), args.student_out)

    # ---- summary ----
    summary = {
        "config": args.config,
        "seeds": seeds,
        "gpus": gpus,
        "selection": {"teacher": "min val CE", "student": rule},
        "val_f1_band": args.val_f1_band,
        "flip_margin": args.flip_margin,
        "fhe_max_seq_len": FHE_MAX_SEQ_LEN,
        "margin_thresholds": list(MARGIN_THRESHOLDS),
        "pooler_tanh_wall": POOLER_TANH_WALL,
        "best_teacher_ckpt": best_teacher,
        "best_student_seed": winner["seed"],
        "best_student_val_f1": winner["val_f1"],
        "best_student_test_f1": winner["test_f1"],
        "best_student_checkpoint": winner["checkpoint"],
        "teacher_results": teacher_results,
        "student_results": student_results,
    }
    with open(os.path.join(args.sweep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====", flush=True)
    if teacher_results:
        print("teacher (seed: val_ce / val_f1):")
        for r in sorted(teacher_results, key=lambda r: r["val_ce"]):
            print(f"  {r['seed']}: {r['val_ce']:.4f} / {r['val_f1']:.4f}")
    print(f"\nBest teacher: {best_teacher}")
    print(f"Best student: seed {winner['seed']} val_f1={winner['val_f1']:.4f}")
    print(f"Exported to: {args.student_out}")
    print(f"summary.json -> {os.path.join(args.sweep_dir, 'summary.json')}")


# ------------------------------------------------------------------------- parser
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command")

    # orchestrate is the default when no subcommand is given.
    def add_common(sp):
        sp.add_argument("--config", default="configs/bert/depth8_exact.yaml")
        sp.add_argument("--seeds", default="42-51", help="e.g. '42-51' or '42,43,44'")
        sp.add_argument("--sweep-dir", default="outputs/bert/seed_sweep")

    # The two judgement calls worth re-running against a finished sweep.
    def add_selection(sp):
        sp.add_argument("--val-f1-band", type=float, default=0.005)
        sp.add_argument("--flip-margin", type=float, default=0.3, choices=MARGIN_THRESHOLDS)

    po = sub.add_parser("orchestrate", help="parallel teacher+student sweep across GPUs")
    add_common(po)
    add_selection(po)
    po.add_argument("--gpus", default="2,3", help="comma-separated GPU ids")
    po.add_argument("--student-out", default="outputs/bert/student")
    po.add_argument(
        "--skip-teacher",
        metavar="CKPT",
        default=None,
        help="skip teacher phase; use this teacher checkpoint dir",
    )
    po.set_defaults(func=cmd_orchestrate)

    pl = sub.add_parser("select", help="re-select a winner from a finished sweep (no GPU)")
    add_selection(pl)
    pl.add_argument("--sweep-dir", default="outputs/bert/seed_sweep")
    pl.add_argument("--student-out", default="outputs/bert/student")
    pl.set_defaults(func=cmd_select)

    pt = sub.add_parser("teacher-worker", help="(internal) train+score teachers for a seed chunk")
    add_common(pt)
    pt.add_argument("--out", required=True)
    pt.set_defaults(func=cmd_teacher_worker)

    ps = sub.add_parser("student-worker", help="(internal) distil students for a seed chunk")
    add_common(ps)
    ps.add_argument("--teacher-ckpt", required=True)
    ps.add_argument("--out", required=True)
    ps.set_defaults(func=cmd_student_worker)

    args = p.parse_args(argv)
    if args.command is None:  # default to orchestrate with its defaults
        args = p.parse_args(["orchestrate", *(argv or sys.argv[1:])])
    args.func(args)


if __name__ == "__main__":
    main()
