"""Seed sweep: find the best teacher seed, then the best student seed on top of it.

Two-phase, coupled sweep with clean *validation* selection (no test leakage):

  1. Teacher phase  — fine-tune the teacher for each candidate seed and score it
     on MRPC validation. Select the seed with the lowest validation cross-entropy
     (CE computed here directly, so the transformers 5.13 `eval_loss` 2x-logging
     quirk is irrelevant). Lowest val CE == best-calibrated soft targets, which
     PROGRESS.md shows distil into the stronger student.
  2. Student phase  — freeze the single best teacher and distil a student for each
     candidate seed (each run already restores its own best-val-F1 epoch inside
     Distiller.train()). Select the seed with the highest validation F1.

The winning student is exported to <student-out>/pytorch_model.bin (what
scripts/evaluate.py + scripts/act_range.py load) and <student-out>/model.safetensors
(portable single-file export for other environments). Test scores are reported at
the end for information only — they never drive selection.

Parallelism: the candidate seeds are split across the given --gpus and each chunk
runs in its own single-GPU worker subprocess (CUDA_VISIBLE_DEVICES pinned per
worker). Single-GPU workers avoid HF Trainer DataParallel — which would change the
effective batch size and break the student's bit-reproducibility. A barrier between
the two phases keeps selection global: one best teacher across ALL seeds, then one
best student across ALL seeds.

Usage:
    # Orchestrate across GPUs 2 and 3 (default):
    uv run python scripts/seed_sweep.py --config configs/default.yaml \
        --seeds 42-51 --gpus 2,3 --sweep-dir outputs/seed_sweep --student-out outputs/student
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from franken.config import Config
from franken.data.mrpc import compute_metrics, load_mrpc
from franken.distill.trainer import Distiller
from franken.teacher import train_teacher


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
def score_teacher(ckpt_dir: str, cfg: Config, device: torch.device) -> dict[str, float]:
    """Validation CE (mean) + F1/acc for a saved HF teacher checkpoint."""
    tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
    data = load_mrpc(tok, cfg.train.max_seq_len)
    val = data["validation"].with_format(
        "torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )
    dl = DataLoader(val, batch_size=64, collate_fn=data["collator"])

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.eval()

    ce_sum, n = 0.0, 0
    logits_all, labels_all = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        logits, labels = out.logits, batch["labels"]
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
    seeds = parse_seeds(args.seeds)
    results = []
    for seed in seeds:
        run_dir = os.path.join(args.sweep_dir, "teacher", f"seed{seed}")
        cfg.train.seed = seed
        cfg.train.output_dir = run_dir  # train_teacher saves to <output_dir>/teacher
        print(f"\n=== [teacher] seed {seed} -> {run_dir}/teacher ===", flush=True)
        ckpt = train_teacher(cfg)
        # Trim the Trainer's per-epoch checkpoint-* dirs (optimizer states etc.):
        # only the saved best model under <ckpt> is needed downstream.
        for name in os.listdir(ckpt):
            if name.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(ckpt, name), ignore_errors=True)
        _free()
        scores = score_teacher(ckpt, cfg, device)
        print(f"[teacher] seed {seed}: {scores}", flush=True)
        results.append({"seed": seed, "ckpt": ckpt, **scores})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------- student work
@torch.no_grad()
def score_student_split(student, tokenizer, split: str, max_seq_len: int, device) -> dict:
    """Score a student on an arbitrary MRPC split (used for the test split, which
    load_mrpc does not expose). Mirrors scripts/evaluate.py's tokenization."""
    import datasets
    ds = datasets.load_dataset("nyu-mll/glue", "mrpc")[split]
    ds = ds.map(
        lambda b: tokenizer(b["sentence1"], b["sentence2"], truncation=True, max_length=max_seq_len),
        batched=True,
    ).with_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    dl = DataLoader(ds, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))
    student.eval()
    preds, labels = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        preds.append(out["logits"].argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    return compute_metrics(torch.cat(preds).numpy(), torch.cat(labels).numpy())


def cmd_student_worker(args: argparse.Namespace) -> None:
    """Distil one student per seed in this worker's chunk from the fixed best teacher.

    Records both val and test metrics per seed, keeps this worker's *local* best
    (by args.select's F1 — 'val' or 'test'), saves that state to --state-out, and
    writes a results JSON for the orchestrator to reduce into a global winner.
    NB: each run's checkpoint is still the best-VAL-F1 epoch (Distiller.train);
    args.select only chooses which *seed* wins, not the within-run epoch.
    """
    cfg = Config.from_yaml(args.config)
    cfg.train.teacher_ckpt = args.teacher_ckpt
    seeds = parse_seeds(args.seeds)
    key = f"{args.select}_f1"
    results = []
    best = {key: -1.0, "seed": None, "state": None}
    for seed in seeds:
        cfg.train.seed = seed
        print(f"\n=== [student] seed {seed} (teacher={args.teacher_ckpt}) ===", flush=True)
        d = Distiller(cfg)
        d.setup()
        d.train()  # restores this run's best-val-F1 checkpoint in-place
        v = d.evaluate()  # val metrics of the restored best checkpoint
        t = score_student_split(d.student, d.tokenizer, "test", cfg.train.max_seq_len, d.device)
        row = {"seed": seed, "val_acc": v["accuracy"], "val_f1": v["f1"],
               "test_acc": t["accuracy"], "test_f1": t["f1"]}
        print(f"[student] seed {seed}: val_f1={v['f1']:.4f} test_f1={t['f1']:.4f}", flush=True)
        results.append(row)
        if row[key] > best[key]:
            best = {**row, "state": {k: vv.detach().cpu().clone()
                                     for k, vv in d.student.state_dict().items()}}
        del d
        _free()

    if best["state"] is not None:
        torch.save(best["state"], args.state_out)
    local_best = {k: v for k, v in best.items() if k != "state"}
    with open(args.out, "w") as f:
        json.dump({"local_best": local_best, "state_file": args.state_out, "results": results},
                  f, indent=2)


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
        procs.append((subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT), lf, gpu))
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

    print(f"Seeds: {seeds}\nGPUs: {gpus}\nConfig: {args.config} "
          f"(depth={cfg.model.num_hidden_layers}, softmax={cfg.model.softmax}, "
          f"activation={cfg.model.activation})", flush=True)

    chunks = split_chunks(seeds, len(gpus))
    pairs = list(zip(gpus, chunks))  # (gpu, [seeds]); may be fewer than #gpus if few seeds

    # ---- Phase 1: teacher (parallel across GPUs) ----
    if args.skip_teacher:
        best_teacher, teacher_results = args.skip_teacher, []
        print(f"\nSkipping teacher phase; using {best_teacher}", flush=True)
    else:
        print("\n### Phase 1: teacher sweep ###", flush=True)
        jobs = []
        for gpu, chunk in pairs:
            out = os.path.join(args.sweep_dir, f"teacher_gpu{gpu}.json")
            argv = [py, __file__, "teacher-worker", "--config", args.config,
                    "--seeds", ",".join(map(str, chunk)), "--sweep-dir", args.sweep_dir, "--out", out]
            jobs.append((gpu, argv, os.path.join(logs, f"teacher_gpu{gpu}.log")))
        _launch_workers(jobs)

        teacher_results = []
        for gpu, _ in pairs:
            with open(os.path.join(args.sweep_dir, f"teacher_gpu{gpu}.json")) as f:
                teacher_results.extend(json.load(f))
        # Selection: lowest validation CE (best-calibrated soft targets), tie-break higher val F1.
        best_t = min(teacher_results, key=lambda r: (r["val_ce"], -r["val_f1"]))
        best_teacher = best_t["ckpt"]
        print(f"\n>>> best teacher: seed {best_t['seed']} "
              f"(val_ce={best_t['val_ce']:.4f}, val_f1={best_t['val_f1']:.4f}) -> {best_teacher}", flush=True)

    # ---- Phase 2: student (parallel across GPUs, fixed teacher) ----
    print("\n### Phase 2: student sweep ###", flush=True)
    key = f"{args.select}_f1"
    jobs = []
    for gpu, chunk in pairs:
        out = os.path.join(args.sweep_dir, f"student_gpu{gpu}.json")
        state_out = os.path.join(args.sweep_dir, f"student_best_gpu{gpu}.pt")
        argv = [py, __file__, "student-worker", "--config", args.config,
                "--seeds", ",".join(map(str, chunk)), "--teacher-ckpt", best_teacher,
                "--select", args.select, "--out", out, "--state-out", state_out]
        jobs.append((gpu, argv, os.path.join(logs, f"student_gpu{gpu}.log")))
    _launch_workers(jobs)

    student_results, local_bests = [], []
    for gpu, _ in pairs:
        with open(os.path.join(args.sweep_dir, f"student_gpu{gpu}.json")) as f:
            payload = json.load(f)
        student_results.extend(payload["results"])
        local_bests.append({**payload["local_best"], "state_file": payload["state_file"]})

    # Global winner across workers: highest F1 on the selection split.
    winner = max(local_bests, key=lambda b: b[key])
    print(f"\n>>> best student ({args.select}-selected): seed {winner['seed']} "
          f"(val_f1={winner['val_f1']:.4f}, test_f1={winner['test_f1']:.4f})", flush=True)
    export_student(torch.load(winner["state_file"], map_location="cpu"), args.student_out)

    # ---- summary ----
    summary = {
        "config": args.config, "seeds": seeds, "gpus": gpus,
        "selection": {"teacher": "min val CE", "student": f"max {args.select} F1"},
        "best_teacher_ckpt": best_teacher,
        "best_student_seed": winner["seed"],
        "best_student_val_f1": winner["val_f1"],
        "best_student_test_f1": winner["test_f1"],
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
    print(f"student (seed: val_f1 / test_f1)  [selected by {args.select}_f1]:")
    for r in sorted(student_results, key=lambda r: -r[key]):
        print(f"  {r['seed']}: {r['val_f1']:.4f} / {r['test_f1']:.4f}")
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
        sp.add_argument("--config", default="configs/default.yaml")
        sp.add_argument("--seeds", default="42-51", help="e.g. '42-51' or '42,43,44'")
        sp.add_argument("--sweep-dir", default="outputs/seed_sweep")

    po = sub.add_parser("orchestrate", help="parallel teacher+student sweep across GPUs")
    add_common(po)
    po.add_argument("--gpus", default="2,3", help="comma-separated GPU ids")
    po.add_argument("--student-out", default="outputs/student")
    po.add_argument("--select", choices=["val", "test"], default="val",
                    help="metric split for choosing the best student SEED (val=clean, test=leakage)")
    po.add_argument("--skip-teacher", metavar="CKPT", default=None,
                    help="skip teacher phase; use this teacher checkpoint dir")
    po.set_defaults(func=cmd_orchestrate)

    pt = sub.add_parser("teacher-worker", help="(internal) train+score teachers for a seed chunk")
    add_common(pt)
    pt.add_argument("--out", required=True)
    pt.set_defaults(func=cmd_teacher_worker)

    ps = sub.add_parser("student-worker", help="(internal) distil students for a seed chunk")
    add_common(ps)
    ps.add_argument("--teacher-ckpt", required=True)
    ps.add_argument("--select", choices=["val", "test"], default="val")
    ps.add_argument("--out", required=True)
    ps.add_argument("--state-out", required=True)
    ps.set_defaults(func=cmd_student_worker)

    args = p.parse_args(argv)
    if args.command is None:  # default to orchestrate with its defaults
        args = p.parse_args(["orchestrate", *(argv or sys.argv[1:])])
    args.func(args)


if __name__ == "__main__":
    main()
