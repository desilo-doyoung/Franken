# Generalize Franken — Progress Tracker

Temporary working doc (not committed). Tracks the model/task abstraction refactor.
Goal: `Distiller` + scripts operate through a `ModelBackend` + `Task` interface. BERT/MRPC
path stays behaviorally identical (regression baseline); Qwen3-Embedding backend + embedding
self-distill task land as inert stubs.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Step 1 — paths + config fields
- [x] `franken/paths.py`: `RunPaths` (base = output_dir, or output_dir/run_name)
- [x] `franken/config.py`: `ModelConfig.backend="bert"`; `TrainConfig.task="mrpc"`, `run_name=None`

## Step 2 — interfaces + registries
- [x] `franken/backends/base.py`: `ModelBackend` ABC (6 methods)
- [x] `franken/backends/__init__.py`: `BACKENDS`, `build_backend`
- [x] `franken/tasks/base.py`: `Task` ABC (compute_loss takes cfg for distill weights)
- [x] `franken/tasks/__init__.py`: `TASKS`, `build_task`

## Step 3 — BERT/MRPC impls + Qwen/embed stubs
- [x] `franken/backends/bert.py`: `BertBackend` (delegates to existing code)
- [x] `franken/tasks/mrpc.py`: `MrpcTask` (wraps load_mrpc / DistillationLoss / teacher.train_teacher)
- [x] `franken/backends/qwen3.py`: `Qwen3Backend` STUB (NotImplementedError + guidance)
- [x] `franken/tasks/embed.py`: `EmbedSelfDistillTask` STUB (NotImplementedError + guidance)

## Step 4 — route core through interfaces
- [x] `franken/distill/trainer.py`: `Distiller` uses backend + task; generic metric/loss-log
- [x] `franken/teacher.py`: output paths via `RunPaths`
- [x] `franken/cli.py`: `train-teacher` via `task.train_teacher`; save student via `RunPaths`
- [x] Import/registry smoke test passes (all configs parse; stubs raise; CLI parser builds)

## Step 5 — route scripts through interfaces
- [x] `scripts/evaluate.py`: `build_backend`, drop `is_hf`, paths/data via RunPaths/task
- [x] `scripts/act_range.py`: `backend.ffn_preact_modules`/`activation_ops`, data via task
- [x] `scripts/stage_distill.py`: default dirs via `RunPaths.subdir`
- [x] `scripts/seed_sweep.py`: teacher/student scoring via backend+task; per-seed dirs via RunPaths
- [x] ruff clean (format + check) across franken + scripts

## Verification
- [x] Registry smoke: build bert/mrpc ok; qwen3/embed construct, raise on use; bad names KeyError
- [x] BERT regression: `distill --config configs/default.yaml` → best val F1 **0.897** (healthy curve)
- [x] FHE regression: `distill --config configs/fhe_full.yaml` (cheb_gelu+cgf+range_penalty) → best val F1 **0.8915**; exercises backend hook path + range penalty; no requires_grad warning
- [x] `evaluate.py` on saved student: teacher F1 0.9003 / student F1 0.8915 (matches train-time eval); `is_hf` branch gone, RunPaths default ckpt works
- [x] act_range.py: shares the same backend hook accessors validated by fhe_full training (import-clean; not run standalone)
- [x] BERT output paths unchanged (`outputs/teacher`, `outputs/student`) with `run_name` unset
- [x] ruff clean across franken + scripts

## Step 6 — per-model file/dir organization (follow-up)
- [x] Models → per-model packages: `franken/models/{base.py,__init__ registry}` + `franken/models/bert/{backend.py, nn modules, loader.py}` + `franken/models/qwen3/backend.py` (folded old `franken/backends/` + flat `franken/model/`)
- [x] Configs → `configs/bert/*.yaml` (+ `configs/qwen3/` placeholder); `teacher_ckpt` → `outputs/bert/teacher`
- [x] Outputs → namespaced by model in `RunPaths` (`outputs/<run_name or backend>/...`); existing BERT outputs moved under `outputs/bert/`
- [x] thor symlink `thor/distilled-model` repointed → `../outputs/bert/student`
- [x] Task-specific scripts → `scripts/bert/{evaluate,act_range,seed_sweep}.py` (generic `stage_distill.py` stays at `scripts/`); repo-root path depth fixed
- [x] `cli.py eval` derives evaluator path from `cfg.model.backend` (`scripts/<backend>/evaluate.py`), not hardcoded
- [x] All default `--config`/output paths + docstrings updated; ruff clean; `main.py eval` end-to-end OK (teacher test F1 0.8807, student 0.8648)

## DONE — abstraction + per-model layout landed; BERT/MRPC behavior preserved; Qwen3-Embedding backend + embed task + configs/qwen3 + scripts/qwen3 are the stubs/slots to fill.

## Notes / decisions
- New model = Qwen3-**Embedding**-0.6B; task = embedding self-distill (cheap swap to fine-tuned later = Task swap only).
- Backend interface = ABC, 6 methods pinned to current `Distiller` usage; provisional.
- GPU constraint: use CUDA devices 2 and 3 only.
