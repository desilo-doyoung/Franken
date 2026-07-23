# Franken

A configurable **knowledge-distillation framework** for producing HE-friendly BERT
students. Franken distills `google-bert/bert-base-uncased` into a smaller student whose
internal operations can be swapped out via configuration, so you can study the accuracy
cost of making a transformer cheaper to evaluate under homomorphic encryption / MPC.

Three student customizations are first-class, all config-driven:

1. **Layer reduction** — fewer transformer layers than the teacher (e.g. 12 → 6).
2. **Softmax approximation** — replace exact attention softmax with an HE-friendly approximation.
3. **Polynomial GELU** — replace GELU with a low-degree polynomial.

## How it works

The student is a from-scratch BERT reimplementation whose module and parameter names mirror
HuggingFace's `BertModel`, so teacher weights load by name — including a **strided copy** under
layer reduction (student block `i` initialized from teacher block `layer_map[i]`). Attention and
FFN take their softmax / GELU as **injected ops** resolved from a registry, so swapping in an
approximation is a config change, not a code change.

Distillation uses the loss:

```
L = (1 - alpha) * CE(student, labels)              # hard-label
  + alpha * T^2  * KL(student/T, teacher/T)         # logit distillation
  + beta         * masked_MSE(student_h, teacher_h) # per-layer hidden-state match
```

The hidden-state term stays well-defined under layer reduction via a **uniform-stride layer map**
(teacher→student, overridable in config), since student and teacher no longer align 1:1.

For polynomial ops that are only valid on a bounded domain (e.g. `cheb_gelu`, or `quad` with a
`domain` set), `distill.range_penalty` adds a training-time term that squashes FFN pre-activations
into `[-domain, domain]`, so the deployed bare polynomial never sees out-of-range inputs.

## Layout

```
franken/
  config.py            dataclass config + YAML loader
  ops/                 swappable-op registry: softmax (exact|cgf), activation (exact|cheb_gelu|quad)
  model/               custom BERT: embeddings, attention, ffn, layer, encoder, bert, loader
  distill/             layer_map, loss, trainer (Distiller)
  data/mrpc.py         GLUE MRPC load / tokenize / metrics
  teacher.py           HF teacher fine-tune + frozen load
  cli.py               train-teacher | distill | eval
configs/               default.yaml + HE recipes (fhe_gelu, fhe_full, quad, quad_fhe, quad_cgf_fhe)
scripts/
  evaluate.py          score teacher + student on MRPC val & test
  stage_distill.py     op-curriculum (staged op-replacement) distillation
  act_range.py         FHE activation-range / self-containment check + histogram
```

## Setup

```bash
uv sync                 # Python >=3.11; installs torch (CUDA), transformers, datasets, sklearn
```

## Usage

```bash
# 1. fine-tune the teacher on MRPC
uv run python main.py train-teacher --config configs/bert/default.yaml

# 2. point configs/bert/default.yaml at the teacher checkpoint (train.teacher_ckpt: outputs/bert/teacher),
#    then distill a student
uv run python main.py distill --config configs/bert/default.yaml

# 3. evaluate — scores teacher + student on MRPC validation & test
#    (delegates to scripts/bert/evaluate.py; --ckpt defaults to <output_dir>/student)
uv run python main.py eval --config configs/bert/default.yaml --ckpt outputs/bert/student
```

Swap ops by editing the config (`model.softmax: cgf`, `model.activation: cheb_gelu` or `quad`, with
per-op `*_kwargs`) and re-running `distill`, or start from a ready-made recipe in `configs/`.

### FHE-friendly extras

```bash
# Op-curriculum: distill the easier op set first, then warm-start and swap in the harder op.
# Helps when two aggressive ops interact (e.g. quad GELU + cgf softmax); see PROGRESS.md.
uv run python scripts/stage_distill.py \
  --config-a configs/bert/quad_fhe.yaml --config-b configs/bert/quad_cgf_fhe.yaml \
  --stagea-dir outputs/bert/stageA_quad --stageb-dir outputs/bert/stageB_quad_cgf

# Verify a polynomial-op student stays in-domain (FHE self-containment) + write a histogram.
uv run python scripts/bert/act_range.py --config configs/bert/quad_cgf_fhe.yaml \
  --student-ckpt outputs/bert/stageB_quad_cgf/pytorch_model.bin --out preact.png
```
