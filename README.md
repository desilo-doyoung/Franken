# Franken

A configurable **knowledge-distillation framework** for producing HE-friendly transformer students.
Franken distills a full-precision teacher into a smaller student whose internal **softmax / activation
are swappable** via config, so you can measure the accuracy cost of making a transformer cheaper to
evaluate under homomorphic encryption / MPC. Three customizations are first-class and config-driven:

1. **Layer reduction** — fewer layers than the teacher (e.g. 12 → 8).
2. **Softmax approximation** — swap exact attention softmax for an HE-friendly op (`cgf`).
3. **Polynomial activation** — swap GELU for a low-degree polynomial (`cheb_gelu`, `quad`).

Two models are implemented: BERT ← `google-bert/bert-base-uncased` on GLUE **MRPC** (the reference
track), and **Qwen3-Embedding-0.6B** embedding self-distillation.

## Architecture

Two registries keep the core model-/task-agnostic, mirroring the op registry:

- **`ModelBackend`** (`franken/models/`) — builds the student (ops injected), loads/seeds the teacher, runs
  a normalized `forward -> {output, hidden_states}`, and exposes the FFN pre-activation / activation modules
  the range penalty hooks.
- **`Task`** (`franken/tasks/`) — tokenizer, dataset, distillation loss, checkpoint metric, teacher fine-tune.

`Distiller` (`franken/distill/trainer.py`) wires a backend + task together and names no model or task, so
adding a model = one `franken/models/<name>/` package + a registry entry (+ a task if the objective is new).

Each student is a from-scratch reimplementation whose parameter names mirror the HF model, so teacher weights
load by name — including a **strided copy** under layer reduction (student block `i` from teacher block
`layer_map[i]`). Classification loss (`tasks/mrpc.py::ClassificationDistillLoss`):

```
L = (1 - alpha) * CE(student, labels)              # hard-label
  + alpha * T^2  * KL(student/T, teacher/T)         # logit distillation
  + beta         * masked_MSE(student_h, teacher_h) # per-layer hidden-state match
```

The hidden term stays well-defined under layer reduction via a **uniform-stride layer map**
(`distill/layer_map.py`); `masked_mse_loss` is the shared, task-agnostic helper any task reuses.
`tasks/embed.py` swaps the CE/KL terms for `(1-cos)` on the pooled embedding.

For polynomial ops valid only on a bounded domain (`cheb_gelu`, or `quad` with a `domain` set),
`distill.range_penalty` squashes FFN pre-activations into `[-domain, domain]` during training, so the
deployed bare polynomial never sees out-of-range inputs.

## Layout

```
franken/
  config.py          dataclass config + YAML loader (model.backend, train.task, train.run_name)
  paths.py           RunPaths — outputs namespaced per model: outputs/<run_name or backend>/...
  ops/               swappable-op registry: softmax (exact|cgf), activation (exact|cheb_gelu|quad)
  models/            base.py = ModelBackend ABC + build_backend registry; bert/, qwen3/
  tasks/             base.py = Task ABC + build_task registry; mrpc.py, embed.py
  distill/           layer_map, masked_mse_loss, Distiller (backend + task driven)
  data/              mrpc.py, embed_corpus.py
  cli.py             train-teacher | distill | eval
configs/<model>/     e.g. configs/bert/{default,fhe_gelu,fhe_full,quad,quad_fhe,quad_cgf_fhe}.yaml
scripts/
  stage_distill.py   op-curriculum (staged op-replacement) distillation — model-agnostic
  bert/              MRPC-specific: evaluate.py, act_range.py, seed_sweep.py
  qwen3/             gates, eval, orchestration (see scripts/qwen3/README.md)
outputs/<model>/     teacher/, student/, stage*/ (gitignored)
```

## Usage

```bash
uv sync                 # Python >=3.11; installs torch (CUDA), transformers, datasets, sklearn

# 1. prepare the task's teacher (MRPC: fine-tune google-bert/bert-base-uncased)
uv run python main.py train-teacher --config configs/bert/default.yaml
# 2. distill a student (teacher_ckpt in the config points at outputs/bert/teacher)
uv run python main.py distill --config configs/bert/default.yaml
# 3. evaluate teacher + student (delegates to scripts/<backend>/evaluate.py)
uv run python main.py eval --config configs/bert/default.yaml --ckpt outputs/bert/student
```

Swap ops by editing the config (`model.softmax: cgf`, `model.activation: cheb_gelu` or `quad`, with per-op
`*_kwargs`) and re-running `distill`, or start from a recipe in `configs/bert/`.

```bash
# Op-curriculum: distill the easier op set first, then warm-start and swap in the harder op.
# Helps when two aggressive ops interact (e.g. quad GELU + cgf softmax); see PROGRESS.md.
uv run python scripts/stage_distill.py \
  --config-a configs/bert/quad_fhe.yaml --config-b configs/bert/quad_cgf_fhe.yaml

# Verify a polynomial-op student stays in-domain (FHE self-containment) + write a histogram.
uv run python scripts/bert/act_range.py --config configs/bert/quad_cgf_fhe.yaml \
  --student-ckpt outputs/bert/stageB_quad_cgf/pytorch_model.bin --out preact.png
```

Results live in `PROGRESS.md` (BERT/MRPC), `franken/models/qwen3/PROGRESS.md` (Qwen3), and
`thor/EXECUTION_NOTES.md` (running these students under FHE).
