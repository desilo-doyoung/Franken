# `franken/scripts/qwen3` — tooling for the Qwen3-Embedding student

Every script takes `--config <yaml>` and reads model/task/corpus from it, so the config is the single
source of truth. Training itself is `main.py distill`, not here.

| script | answers |
|---|---|
| `corpus.py` | is the holdout sound, is every source scoreable, what is `corpus_size` — and build the cache |
| `eval.py` | how much did the student lose, and is it coverage or capacity? |
| `parity_gate.py` | is the from-scratch student still bit-equal to the teacher? |
| `precision_gate.py` | is `precision: bf16` safe for this architecture? |
| `act_range.py` | what range do the FHE operators actually see? |
| `search.py` | what does the student actually retrieve for this query, and where did it differ from the teacher? |
| `run_experiments.py` | all of the above over many configs, into one markdown table |
| `common.py` | shared: path bootstrap, flags, `load()` (teacher+student), nDCG scoring |

## The whole workflow

One command. It gates the corpus once per corpus, **builds the cache if there isn't one**, distills
each config on its own GPU, scores all three eval suites, and prints the table:

```bash
# depth28 FIRST under --ddp: it is the control, and --ddp finishes it before spending on the rest.
uv run python -m franken.scripts.qwen3.run_experiments --devices 0,1,2,3 --ddp \
  configs/qwen3/depth28_exact.yaml \
  configs/qwen3/depth19_exact.yaml \
  configs/qwen3/depth19_quad.yaml
```

⚠️ `distill.token_budget` is **per rank**, so the `multi_domain` configs are calibrated for **4 ranks**
(16,384 × 4 = 65,536 tokens/step → ~25.6k steps/epoch). On a different rank count the trainer still
derives a correct `lr` from the batch it actually assembles, but the step count moves and the
numbers stop being
comparable to the recorded ladder.

Or step by step for a single config, via the top-level CLI:

```bash
CFG=configs/qwen3/depth19_quad.yaml
uv run python main.py corpus  --config $CFG
uv run python main.py distill --config $CFG
uv run python main.py eval    --config $CFG [--ckpt outputs/<run>/student]
```

`--config` is **required**, never defaulted — the config decides what is measured, these scripts cost
minutes to hours, and a wrong default is silent. (`parity_gate` and `precision_gate` are the
exceptions: each is only meaningful for one config, so they keep theirs.)

Pass only cards you own — `--devices` becomes `CUDA_VISIBLE_DEVICES` per subprocess, and a co-tenant's
idle GPU is not free capacity. `--ddp` instead spreads one config across all devices; the default queue
maximizes throughput, `--ddp` maximizes what a single run can absorb. ⚠️ `compile: true` costs a
~6-minute CPU-bound warmup per rank before the first epoch line, which reads exactly like a collective
deadlock — check `%CPU` first (~90%, state `R` = Dynamo compiling).

## Smoke-test a fresh box first

Four checks, ~30 min total, before committing hours. Each one fails loudly and locally.

```bash
DEV=2                                    # a card you own
CFG=configs/qwen3/depth19_quad.yaml

# 1. env, CUDA, model download, and the student still bit-equal to the teacher  (~2 min)
uv run python -m franken.scripts.qwen3.parity_gate --config configs/qwen3/gate_parity.yaml

# 2. the real corpus gates: all 18 sources load, all are scoreable, tok/text measured (~40 min,
#    and its HF downloads are exactly the ones the real build reuses, so this is not wasted)
uv run python main.py corpus --config $CFG

# 3. the whole pipeline on 1/450th of the data: build -> distill -> checkpoint  (~10 min)
CUDA_VISIBLE_DEVICES=$DEV uv run python main.py distill \
  --config configs/qwen3/smoke.yaml

# 4. eval plumbing, with a known right answer: smoke is full depth + exact ops, so with no
#    --student-ckpt the student IS the teacher and every delta must be exactly 0.0000  (~10 min)
CUDA_VISIBLE_DEVICES=$DEV uv run python main.py eval \
  --config configs/qwen3/smoke.yaml \
  --suite corpus,external --sources gooaq,specter --tasks nfcorpus
```

```bash
# 5. only if the real run will use --ddp: exercise the collective path too (~10 min). A
#    single-GPU smoke cannot catch a DDP deadlock, and `token_budget` is PER RANK.
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run --nproc_per_node=2 \
  main.py distill --config configs/qwen3/smoke.yaml
```

Step 4 is the one to read carefully: **any non-zero delta means the eval is broken**, not that the
model is. Narrow `--sources` / `--tasks` on purpose — pool size is independent of `corpus_size`, so
the full suite would cost as much here as on a real run.

If all four pass, the only thing left untested is scale. Report: `parity_gate` verdict, the `CORPUS OK`
line plus the printed `corpus_size` (it must match the config), the distill run's
`token-budgeted batching: N steps/epoch/rank` line, and step 4's delta column.

## `corpus.py` — three stages, each gating the next

Datasets are declared **once**, as a `Source` in `franken/data/embed_corpus/registry.py`; the training
corpus and the eval pools both derive from that entry, and a source must be scoreable to exist. See
that package's README for the design.

1. **check** — holdout disjoint and representative, for the sources with upstream splits. Hash-split
   sources are disjoint by construction.
2. **measure** — every source loads, every source is scoreable, and `tok/text` is *measured*, printing
   the `corpus_size` to paste. The documented precedent is a 15% miss on an estimated mean.
3. **build** — builds the cache if it is missing, reports the realized mix off the stored `source`
   column, and reports the realized token count against `TOKEN_TARGET`.

Stage 3 does **not** gate. `corpus_size` converts the budget through a `tok/text` sampled from
`SAMPLE` texts per source, and that estimate's standard error is roughly the same couple of percent
the build lands off 2B — so a gate there rejects noise. It is a number to read, not a verdict, and it
prints on a cache hit as well as on a build: a corpus is reused across runs, and one that only
reported on the run that created it went unexamined on every run after.

`run_experiments.py` runs all three stages before any GPU time, so the build is not a separate manual
step. ⚠️ An HF retry thread has historically raised SIGABRT during interpreter finalization *after*
results print, so a successful build can exit **134** — `corpus.py` exits via `os._exit` to dodge
that, and the runner trusts the `CORPUS OK` verdict over the exit code.

`TOKEN_TARGET = 2e9` lives in exactly one place (`corpus.py`). `corpus_size` is measured once and
pasted into the config. `lr` is normally `null` in a config, and the trainer derives it by
sqrt-batch scaling from the batch the run actually assembles. A ladder once ran 5.6% hot because a corrected estimate never reached
the config that consumed it — which is what the stage-3 report is for.

## `eval.py` — three suites, and the subtraction between two of them

| suite | data | answers |
|---|---|---|
| `agreement` | the corpus's own held-out pool + STS-B | did the student track *this* teacher? |
| `corpus` | held-out rows of the **training** slices | quality lost where the corpus **has** coverage |
| `external` | benchmarks the corpus does not contain | quality lost in the wild — the number to quote |

`external − corpus` is the **coverage gap**, printed at the bottom: a small in-distribution deficit
with a large external one means the fix is data; both large means it is capacity and more data will not
help. This subtraction replaced a separate drift script that attempted the same read with weaker,
unlabelled metrics.

- ⚠️ **Two corpus macros, not one.** `MACRO-pair` is fully held out. `MACRO-qrels` is not — the
  judgements pick the gold documents, so each is ~96% likely to be a train row and only the
  distractors are held out. Averaging them together is the CORE-macro mistake.
- ⚠️ **recall@10 is already relative to the teacher** — 1.0 is the ceiling, so quote it raw; it comes
  from the same function that selected the checkpoint, and is **only comparable at a fixed pool size**
  (measured, at fixed damage: 1.000 at n=11, 0.110 at 500, 0.039 at 5000).
- ⚠️ `embed_dist` **misranks** recall@10; logging only. Report retrieval strictly as **nDCG@10** —
  MTEB also defines a "recall@10" and ours means teacher-neighbour agreement.
- ⚠️ **Not comparable to the published MTEB table**: task subset, the config's own `max_seq_len` (the
  FHE deployment condition) vs MTEB's 512, one generic instruction. Valid teacher-vs-student only.
- The external macro is **every** scored task. Two tasks ("CORE") read the depth-19 cut at +0.4% where
  five put it at −16.0% and inverted the ratio column too — it reversed the conclusion, not the value.
- With no `--student-ckpt` on a **full-depth** config the student *is* the teacher, so every delta must
  read ~0 — the self-test. Below full depth it is an untrained truncation and reads ~−100%.

`--suite`, `--sources` and `--tasks` narrow it; a narrower list is a different, non-comparable macro.

## Correctness gates

**`parity_gate.py`** — exact ops + full depth + teacher weights ⇒ the student *is* the teacher, so any
gap is a module bug (RoPE/QK-norm order, `repeat_kv`, causal+pad mask, `hidden_states` bookkeeping),
not float noise. Passes at pooled cosine 1.0.

- ⚠️ **Fails by design on FHE configs** (`cgf`, polynomial activation) — it is an exact-op gate.
- ⚠️ **Judge hidden deltas in ULPs.** `|Δ|max` lands on layer 3's massive-activation channel where one
  fp32 ULP is ~5e-4, so 9.8e-4 is 2 ULP of summation-order noise.
- The comparison covers **real tokens only**, with the raw pad-inclusive max printed alongside:
  `attn_impl: sdpa_causal` leaves garbage at pad positions on purpose and nothing reads them, but a
  broken pad mask still shows up in the raw column.

**`precision_gate.py`** — three assertions that bf16's safety argument holds: RoPE cos/sin stay fp32
inside an autocast region, all 29 `hidden_states` stay fp32, and a bf16 teacher moves the targets by
more than the comparison band. The third **passes when the bf16 teacher DISAGREES** — that proves
keeping the teacher out of the autocast region is load-bearing.

**`act_range.py`** — per-layer range of what each approximated operator *receives*: `gate_proj` output
for the activation, attention scores for the softmax.

- ⚠️ Measure the **operator's input**, never hidden states — RMSNorm strips Qwen3's massive activations
  before `gate_proj`, so hidden-state stats overestimate the needed domain ~20×, and domain costs
  multiplicative depth.
- ⚠️ **The max matters, not the percentile** (no clamp at inference; a high-degree polynomial explodes
  outside its domain). Maxima are exact — never subsample them.
- 🐛 Reports `0.0` for attention scores under `attn_impl: sdpa_causal` instead of saying so — actively
  misleading on a `cgf` config.

## `search.py` — eyeball the retrievals

`eval.py` gives you the aggregates; this gives you one query at a time. It prints the student's top-10
with the teacher's rank of each hit beside it, the teacher hits the student missed, and the gold
documents marked — under a banner carrying the pool-wide numbers, so an anecdote is always read
against the population it came from.

```bash
CUDA_VISIBLE_DEVICES=2 uv run python -m franken.scripts.qwen3.search --source gooaq \
  --config configs/qwen3/depth19_exact.yaml \
  --student-ckpt outputs/qwen3_depth19/student/pytorch_model.bin
```

Then type queries. A pool id (`q0`) runs that query with its judgements, `:r` draws a random one,
anything else is free-form; `--query` does the same non-interactively. Startup embeds the pool once
per model (the teacher's half is the cache `eval.py` already writes; the student's is recomputed) —
after that each query is instant.

`--worst N` prints the N queries with the lowest `agree@10` instead of waiting for input — the tail
the histogram shows but a hand-picked query almost never hits. Both it and `--query` skip the REPL.

- ⚠️ **The worst queries are usually the pool's, not the model's.** The same ids come top of the
  list for every checkpoint including a full-depth one, and they sit on a flat cosine plateau where
  ranks 6–10 differ by ~0.01 — so read them as hard queries before reading them as damage.

- **Three metrics, and they are not the same thing.** `recall@10 docs` is the tracker's number:
  teacher-neighbour agreement over the pool's *documents*, no queries and no judgements involved.
  `agree@10` is the same idea on the query side. `recall@10 gold` is the MTEB sense — did the judged
  document land in the top 10. A student can lose a lot of the first while barely moving the last.
- **Read the histogram, not just the mean.** An `agree@10` of 0.74 spread evenly is a different model
  from one that is perfect on half the queries and broken on the rest.
- A pool query is sent as `q_texts` **verbatim**, so it is byte-identical to what `eval.py` scored;
  a free-form query gets the source's own `Source.instruct`, and documents are never prefixed.
- ⚠️ nDCG is suppressed for the `scores_ndcg=False` sources (`wiki_*`, `specter`) exactly as in
  `eval.py`; `recall@10 gold` still prints, and on those sources it inherits the same caveat that the
  gold is one arbitrary member of an equally valid set.
- With no `--student-ckpt` on a **full-depth** config the student is the teacher: `agree@10` must read
  1.00 and the "missed" block must be empty. That is the self-test.

## Multi-GPU for a single run

`main.py distill` reads torchrun's environment (`franken/distill/dist.py`), either directly or through
the runner's `--ddp`:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 main.py distill --config X
```

⚠️ Under `distill.token_budget` the budget is **per rank**, so tokens/step scale with the device count
and steps/epoch is data-dependent — the trainer logs the real count at startup and derives `lr` from
it. `batch_size` is unused under `token_budget` except to size the eval loader.
