# `scripts/qwen3` — tooling for the Qwen3-Embedding student

Every script takes `--config <yaml>` and reads model/task/corpus from it, so the config is the single
source of truth. Training itself is `main.py distill`, not here.

| script | answers | run it when |
|---|---|---|
| `parity_gate.py` | is the from-scratch student still bit-equal to the teacher? | after touching any module under `franken/models/qwen3/` |
| `precision_gate.py` | is `precision: bf16` safe for this architecture? | before trusting a bf16 run; after touching `rope.py` or the residual path |
| `bench_step.py` | how fast is one training step? | before/after any change made for speed |
| `embed_eval.py` | how close to the teacher is a checkpoint? | to score a run (**recall@10** is the headline) |
| `retrieval_eval.py` | is the student absolutely worse, not just different? | end of a run, when the question is "good enough?" |
| `act_range.py` | what range do the FHE operators actually see? | before picking a polynomial `domain` |
| `run_experiments.py` | all of the above, over many configs | any batch of experiments |

```
uv run python scripts/qwen3/parity_gate.py     --config configs/qwen3/exact.yaml
uv run python scripts/qwen3/precision_gate.py  --config configs/qwen3/depth28_control.yaml
uv run python scripts/qwen3/bench_step.py      --config ... [--precision bf16] [--compile]
uv run python scripts/qwen3/act_range.py       --config ... [--student-ckpt ...]
uv run python scripts/qwen3/embed_eval.py      --student-ckpt outputs/qwen3/student/pytorch_model.bin
uv run python scripts/qwen3/retrieval_eval.py  --student-ckpt ... --tasks nfcorpus,scifact
uv run python scripts/qwen3/run_experiments.py --devices 2,3 [--eval-only] configs/qwen3/depth19.yaml ...
```

## Correctness gates

**`parity_gate.py`** — exact ops + full depth + teacher weights ⇒ the student *is* the teacher, so any gap
is a module bug (RoPE/QK-norm order, `repeat_kv`, causal+pad mask, `hidden_states` bookkeeping), not float
noise. Passes at pooled cosine 1.0.

- ⚠️ **Fails by design on FHE configs** (`cgf`, polynomial activation) — it is an exact-op gate.
- ⚠️ **Judge hidden deltas in ULPs.** `|Δ|max` lands on layer 3's massive-activation channel where one fp32
  ULP is ~5e-4, so 9.8e-4 is 2 ULP of summation-order noise.
- The comparison covers **real tokens only**, with the raw pad-inclusive max printed alongside:
  `attn_impl: sdpa_causal` leaves garbage at pad positions on purpose and nothing reads them, but a broken
  pad mask still shows up in the raw column.

**`precision_gate.py`** — three assertions that bf16's safety argument holds: RoPE cos/sin stay fp32 inside
an autocast region, all 29 `hidden_states` stay fp32, and a bf16 teacher moves the targets by more than the
comparison band. The third **passes when the bf16 teacher DISAGREES** — that proves keeping the teacher out
of the autocast region is load-bearing.

## Measurement

**`bench_step.py`** — times the real hot path (teacher forward, student forward/backward, clip, optimizer)
after a warmup absorbing autotuning and `torch.compile`. `--precision` / `--compile` override the config so
a sweep needs no YAML edits. `tok/s` counts **real (non-pad)** tokens, matching `PROGRESS.md`.
⚠️ Does not attach the range-penalty hooks, so penalized configs read slightly fast.

**`act_range.py`** — per-layer range of what each approximated operator *receives*: `gate_proj` output for
the activation, attention scores for the softmax.

- ⚠️ Measure the **operator's input**, never hidden states — RMSNorm strips Qwen3's massive activations
  before `gate_proj`, so hidden-state stats overestimate the needed domain ~20×, and domain costs
  multiplicative depth.
- ⚠️ **The max matters, not the percentile** (no clamp at inference; a high-degree polynomial explodes
  outside its domain). Maxima are exact — never subsample them.

## Scoring

**`embed_eval.py`** — recall@10 (headline), `embed_dist`, sim-rho, STS-B as a delta.

- ⚠️ **recall@10 is already relative to the teacher — 1.0 is the ceiling, so quote it raw.** Normalizing by a
  trained control double-normalizes.
- ⚠️ `embed_dist` **misranks** recall@10; logging only.
- With no `--student-ckpt` the student is seeded from the teacher, so recall ~1.0 / delta ~0 is the self-test.

**`retrieval_eval.py`** — nDCG@10 against ground-truth judgements: the only absolute, top-k-sensitive metric
here (recall@10 says only that the student retrieves *differently*; STS-B is too coarse for top-of-list
damage). ⚠️ **Not comparable to the published MTEB table** — task subset, `max_seq_len` 128 (the FHE
deployment condition) vs MTEB's 512, one generic instruction. Valid teacher-vs-student, invalid as a
leaderboard figure. Report strictly as **nDCG@10**; MTEB also defines a "recall@10" and ours means
teacher-neighbour agreement.

## Orchestration

**`run_experiments.py`** — distills each config, scores it with `embed_eval --json` (never scraped from
prose), and prints one markdown table that pastes into `PROGRESS.md` plus every run's per-epoch trace.

- Configs are a **work queue** over the given cards, so N configs cost ⌈N/devices⌉ slots even when runs
  differ in length. For a batch of experiments this beats DDP outright: full utilization, zero communication.
- A crashed run degrades to one `FAILED` row with its log tail; the batch and the table survive.
- ⚠️ Pass only cards you own — `--devices` becomes `CUDA_VISIBLE_DEVICES` per subprocess.

## Multi-GPU for a single run

`main.py distill` reads torchrun's environment (`franken/distill/dist.py`), either directly or
through the runner's `--ddp`, which wraps the same launch and still writes the table:

```
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 main.py distill --config X
uv run python scripts/qwen3/run_experiments.py --devices 0,1,2,3 --ddp configs/qwen3/X.yaml
```

Pick by what is scarce: the default queue maximizes **throughput** (a batch of configs, zero
communication), `--ddp` maximizes the compute **one** run can absorb. ⚠️ `compile: true` costs a
~6-minute CPU-bound warmup on every rank before the first epoch line appears — GPUs sit at 0%
util with memory resident, which reads exactly like a collective deadlock. Check `%CPU` first:
~90% and state `R` on every rank is Dynamo compiling, not a hang.

`train.distill.batch_size` is the **global** batch, split across ranks, so steps/epoch and the LR schedule
are preserved — per-rank batch therefore shrinks as cards are added. For 4 cards set `batch_size: 128`
(32/rank) and scale LR by **√batch**, not linearly.
