# `scripts/qwen3` — tooling for the Qwen3-Embedding student

Every script takes `--config <yaml>` and reads the model/task/corpus from it, so the config is
the single source of truth. Training itself is `main.py distill`, not here.

| script | answers | run it when |
|---|---|---|
| `parity_gate.py` | is the from-scratch student still bit-equal to the teacher? | after touching any module under `franken/models/qwen3/` |
| `precision_gate.py` | is `precision: bf16` safe for this architecture? | before trusting a bf16 run; after touching `rope.py` or the residual path |
| `bench_step.py` | how fast is one training step? | before/after any change made for speed |
| `embed_eval.py` | how close to the teacher is a checkpoint? | to score a run (**recall@10** is the headline) |
| `retrieval_eval.py` | is the student absolutely worse, not just different? | end of a run, when the question is "good enough?" |
| `act_range.py` | what range do the FHE operators actually see? | before picking a polynomial `domain` |
| `run_experiments.py` | all of the above, over many configs | any batch of experiments |

## Correctness gates

**`parity_gate.py`** — exact ops + full depth + teacher weights ⇒ the student *is* the teacher.
Any gap is a module bug (RoPE/QK-norm order, `repeat_kv`, causal+pad mask, `hidden_states`
bookkeeping), not float noise. Passes at pooled cosine 1.0.

```
uv run python scripts/qwen3/parity_gate.py --config configs/qwen3/exact.yaml
```

- ⚠️ **Fails by design on FHE configs** (`cgf`, polynomial activation). It is an exact-op gate.
- ⚠️ **Judge hidden deltas in ULPs, not absolutes.** `|Δ|max` lands on layer 3's massive-activation
  channel where one fp32 ULP is already ~5e-4, so 9.8e-4 is 2 ULP of summation-order noise.
- The hidden comparison covers **real tokens only**; the raw pad-inclusive max is printed
  alongside. `attn_impl: sdpa_causal` leaves garbage at pad positions on purpose, and nothing
  reads them — but a broken pad mask still shows up in the raw column.

**`precision_gate.py`** — three assertions that bf16's safety argument actually holds: RoPE
cos/sin stay fp32 inside an autocast region, all 29 `hidden_states` stay fp32, and a bf16
teacher moves the targets by more than the comparison band.

```
uv run python scripts/qwen3/precision_gate.py --config configs/qwen3/depth28_control.yaml
```

- The third check **passes when the bf16 teacher DISAGREES**. That is the point: it proves
  keeping the teacher out of the autocast region is load-bearing rather than decorative.

## Measurement

**`bench_step.py`** — times the real hot path (teacher forward, student forward/backward, clip,
optimizer) after a warmup that absorbs autotuning and `torch.compile`.

```
uv run python scripts/qwen3/bench_step.py --config configs/qwen3/depth28_control.yaml
uv run python scripts/qwen3/bench_step.py --config ... --precision bf16 --compile
```

- `--precision` / `--compile` override the config, so a sweep needs no YAML edits.
- `tok/s` counts **real (non-pad)** tokens, matching the recipe table in `PROGRESS.md`.
- ⚠️ Does not attach the range-penalty hooks, so penalized configs read slightly fast.

**`act_range.py`** — per-layer range of what each approximated operator *receives*: `gate_proj`
output for the activation, attention scores for the softmax.

```
uv run python scripts/qwen3/act_range.py --config configs/qwen3/exact.yaml [--student-ckpt ...]
```

- ⚠️ Measure the **operator's input**, never hidden states. RMSNorm strips Qwen3's massive
  activations before `gate_proj`, so hidden-state stats overestimate the needed domain ~20×,
  and domain costs multiplicative depth.
- ⚠️ **The max matters, not the percentile.** There is no clamp at inference, and a high-degree
  polynomial explodes outside its domain. Maxima are exact — never subsample them.

## Scoring

**`embed_eval.py`** — recall@10 (headline), `embed_dist`, sim-rho, and STS-B as a delta.

```
uv run python scripts/qwen3/embed_eval.py --student-ckpt outputs/qwen3/student/pytorch_model.bin
```

- ⚠️ **recall@10 is already relative to the teacher — 1.0 is the ceiling, so quote it raw.**
  Do not normalize by a trained control; that double-normalizes.
- ⚠️ `embed_dist` **misranks** recall@10 and is logging only.
- With no `--student-ckpt` the student is seeded from the teacher, so recall ~1.0 and delta ~0
  are this script's self-test.

**`retrieval_eval.py`** — nDCG@10 against ground-truth relevance judgements: the only absolute,
top-k-sensitive metric here. recall@10 can only say the student retrieves *differently*; STS-B is
absolute but too coarse to see top-of-list damage.

```
uv run python scripts/qwen3/retrieval_eval.py --student-ckpt ... --tasks nfcorpus,scifact
```

- ⚠️ **Not comparable to the published MTEB table** — task subset, `max_seq_len` 128 (the FHE
  deployment condition) vs MTEB's 512, one generic instruction. Valid teacher-vs-student, invalid
  as a leaderboard figure. Report strictly as **nDCG@10**; MTEB also defines a "recall@10" and
  ours means teacher-neighbour agreement.

## Orchestration

**`run_experiments.py`** — distills each config, scores it with `embed_eval.py`, and prints one
markdown table that pastes into `PROGRESS.md`, plus every run's per-epoch trace.

```
uv run python scripts/qwen3/run_experiments.py --devices 2,3 configs/qwen3/depth19.yaml ...
uv run python scripts/qwen3/run_experiments.py --devices 2,3 --eval-only <configs>
```

- Configs are a **work queue** over the given cards, so N configs cost ⌈N/devices⌉ slots even
  when runs differ in length. For a batch of experiments this beats DDP outright: full GPU
  utilization, zero communication.
- Metrics come from `embed_eval --json`, never scraped from prose. A crashed run degrades to one
  `FAILED` row with its log tail; the batch and the table survive.
- ⚠️ Pass only cards you own — `--devices` becomes `CUDA_VISIBLE_DEVICES` per subprocess.

## Multi-GPU for a single run

Not a script here — `main.py distill` reads torchrun's environment (`franken/distill/dist.py`):

```
python main.py distill --config X                                # 1 GPU
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 main.py distill --config X
```

`train.distill.batch_size` is the **global** batch and is split across ranks, so steps/epoch and
the LR schedule are preserved. That means per-rank batch shrinks as cards are added — for 4 cards
set `batch_size: 128` (32/rank) and scale LR by **√batch**, not linearly.
