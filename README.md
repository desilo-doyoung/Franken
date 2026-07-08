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

## Layout

```
franken/
  config.py            dataclass config + YAML loader
  ops/                 swappable-op registry: exact/approx softmax, exact/poly gelu
  model/               custom BERT: embeddings, attention, ffn, layer, encoder, bert, loader
  distill/             layer_map, loss, trainer (Distiller)
  data/mrpc.py         GLUE MRPC load / tokenize / metrics
  teacher.py           HF teacher fine-tune + frozen load
  cli.py               train-teacher | distill | eval
configs/default.yaml   example config exercising all three customizations
```

## Setup

```bash
uv sync                 # Python >=3.11; installs torch (CUDA), transformers, datasets, sklearn
```

## Usage

```bash
# 1. fine-tune the teacher on MRPC
python main.py train-teacher --config configs/default.yaml

# 2. point configs/default.yaml at the teacher checkpoint (train.teacher_ckpt: outputs/teacher),
#    then distill a student
python main.py distill --config configs/default.yaml

# 3. evaluate a saved student
python main.py eval --config configs/default.yaml --ckpt outputs/student
```

Swap ops by editing `configs/default.yaml` (`model.softmax: approx`, `model.gelu: poly`) and
re-running `distill`.
