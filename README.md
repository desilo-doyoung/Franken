# Franken

A configurable **knowledge-distillation framework** for producing HE-friendly transformer
students. Franken distills a full-precision teacher into a smaller student whose internal
**softmax / activation are swappable** via config, so you can measure the accuracy cost of
making a transformer cheaper to evaluate under homomorphic encryption / MPC.

The **model** and **task** are pluggable behind two small registries, so one
training/distillation core serves different models:

- **Reference (implemented):** BERT student ← `google-bert/bert-base-uncased` teacher, GLUE **MRPC**.
- **Stub (to implement):** **Qwen3-Embedding-0.6B**, embedding self-distillation.

Three student customizations are first-class and config-driven:

1. **Layer reduction** — fewer layers than the teacher (e.g. 12 → 8).
2. **Softmax approximation** — swap exact attention softmax for an HE-friendly op (`cgf`).
3. **Polynomial activation** — swap GELU for a low-degree polynomial (`cheb_gelu`, `quad`).

## Architecture

Two registries keep the core model-/task-agnostic (mirroring the op registry):

- **`ModelBackend`** (`franken/models/`) — builds the student (ops injected), loads/seeds the
  teacher, runs a normalized `forward -> {output, hidden_states}`, and exposes the FFN
  pre-activation / activation modules the range penalty hooks. One package per model:
  `models/bert/` (real), `models/qwen3/` (stub).
- **`Task`** (`franken/tasks/`) — owns the tokenizer, dataset, distillation loss, checkpoint
  metric, and teacher fine-tune. `tasks/mrpc.py` (real), `tasks/embed.py` (stub).

`Distiller` (`franken/distill/trainer.py`) just wires a backend + task together and names no
model or task. Adding a model = one `franken/models/<name>/` package + a registry entry (+ a
task if the objective is new); nothing in the trainer or scripts changes.

The BERT student is a from-scratch reimplementation whose parameter names mirror HF `BertModel`,
so teacher weights load by name — including a **strided copy** under layer reduction (student
block `i` from teacher block `layer_map[i]`). Softmax/activation are **injected ops** from
`franken/ops`, so swapping in an approximation is a config change.

Classification distillation loss (`franken/tasks/mrpc.py::ClassificationDistillLoss`):

```
L = (1 - alpha) * CE(student, labels)              # hard-label
  + alpha * T^2  * KL(student/T, teacher/T)         # logit distillation
  + beta         * masked_MSE(student_h, teacher_h) # per-layer hidden-state match
```

The hidden term stays well-defined under layer reduction via a **uniform-stride layer map**
(`franken/distill/layer_map.py`); `masked_mse_loss` (`franken/distill/loss.py`) is the shared,
task-agnostic helper any task reuses.

For polynomial ops valid only on a bounded domain (`cheb_gelu`, or `quad` with a `domain` set),
`distill.range_penalty` squashes FFN pre-activations into `[-domain, domain]` during training,
so the deployed bare polynomial never sees out-of-range inputs.

## Layout

```
franken/
  config.py          dataclass config + YAML loader (model.backend, train.task, train.run_name)
  paths.py           RunPaths — outputs namespaced per model: outputs/<run_name or backend>/...
  ops/               swappable-op registry: softmax (exact|cgf), activation (exact|cheb_gelu|quad)
  models/
    base.py          ModelBackend ABC; __init__ = build_backend registry
    bert/            from-scratch BERT student + backend (backend.py, bert.py, …, loader.py)
    qwen3/           Qwen3-Embedding backend (stub)
  tasks/
    base.py          Task ABC; __init__ = build_task registry
    mrpc.py          MRPC data + ClassificationDistillLoss + teacher fine-tune
    embed.py         embedding self-distillation (stub)
  distill/           layer_map, masked_mse_loss, Distiller (backend + task driven)
  data/mrpc.py       GLUE MRPC load / tokenize / metrics
  cli.py             train-teacher | distill | eval
configs/<model>/     e.g. configs/bert/{default,fhe_gelu,fhe_full,quad,quad_fhe,quad_cgf_fhe}.yaml
scripts/
  stage_distill.py   op-curriculum (staged op-replacement) distillation — model-agnostic
  bert/              MRPC-specific: evaluate.py, act_range.py, seed_sweep.py
outputs/<model>/     teacher/, student/, stage*/ (gitignored)
```

## Setup

```bash
uv sync                 # Python >=3.11; installs torch (CUDA), transformers, datasets, sklearn
```

## Usage

```bash
# 1. prepare the task's teacher (MRPC: fine-tune google-bert/bert-base-uncased)
uv run python main.py train-teacher --config configs/bert/default.yaml

# 2. distill a student (teacher_ckpt in the config points at outputs/bert/teacher)
uv run python main.py distill --config configs/bert/default.yaml

# 3. evaluate teacher + student (delegates to scripts/<backend>/evaluate.py)
uv run python main.py eval --config configs/bert/default.yaml --ckpt outputs/bert/student
```

Swap ops by editing the config (`model.softmax: cgf`, `model.activation: cheb_gelu` or `quad`, with
per-op `*_kwargs`) and re-running `distill`, or start from a ready-made recipe in `configs/bert/`.

### FHE-friendly extras

```bash
# Op-curriculum: distill the easier op set first, then warm-start and swap in the harder op.
# Helps when two aggressive ops interact (e.g. quad GELU + cgf softmax); see PROGRESS.md.
uv run python scripts/stage_distill.py \
  --config-a configs/bert/quad_fhe.yaml --config-b configs/bert/quad_cgf_fhe.yaml

# Verify a polynomial-op student stays in-domain (FHE self-containment) + write a histogram.
uv run python scripts/bert/act_range.py --config configs/bert/quad_cgf_fhe.yaml \
  --student-ckpt outputs/bert/stageB_quad_cgf/pytorch_model.bin --out preact.png
```
