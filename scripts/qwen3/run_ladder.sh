#!/usr/bin/env bash
# Gates -> corpus build -> lr derivation -> the three-config ladder -> both eval suites, in one go.
#
# corpus_size and lr are DERIVED here, not hand-copied: the probe measures tok/text on this machine
# and the batch planner reports the real step count, so the config that trains is always consistent
# with the corpus that was built. The last ladder ran 5.6% hot on drift precisely because a
# corrected estimate never reached the config it fed.
#
# Run it detached -- this is a 12h+ job:
#     tmux new -s ladder
#     bash scripts/qwen3/run_ladder.sh 2>&1 | tee outputs/ladder.log
#
# Re-running is safe: the corpus cache hits, and the gates are quick on a warm HF cache.

# NOT `set -e`: build_corpus.py exits 134 on success (SIGABRT from an HF retry thread during
# interpreter shutdown, after the cache is written and the results printed).
set -uo pipefail

CONFIGS=(depth28_multi_domain depth19_multi_domain_exact depth19_multi_domain)
DEVICES=0,1,2,3
WORLD=4
TOKEN_TARGET=2000000000
DRIFT=7.69e-3        # the lr*sqrt(steps) invariant every run in the tracker is matched on
PROBE_JSON=outputs/corpus_probe.json

cd "$(dirname "$0")/../.." || exit 1
mkdir -p outputs
step() { printf '\n\033[1m=== %s ===\033[0m\n\n' "$1"; }
die() { printf '\n\033[1;31mSTOP: %s\033[0m\n' "$1"; exit 1; }

# Built explicitly: bash prefix expansion cannot add the .yaml suffix in the same substitution.
PATHS=()
for c in "${CONFIGS[@]}"; do PATHS+=("configs/qwen3/$c.yaml"); done
LAST="configs/qwen3/${CONFIGS[-1]}.yaml"   # any config resolves the shared corpus cache key

# --- 1. gates ------------------------------------------------------------------------------------
# Before the build, because a failed gate changes the mix and the cache is keyed on the mix.

step "1a. corpus probe (all sources stream + tok/text)"
uv run python scripts/qwen3/corpus_probe.py --config "$LAST" --json "$PROBE_JSON" 2>&1 | tee outputs/corpus_probe.log
[ "${PIPESTATUS[0]}" -eq 0 ] || die "a source failed to load -- see outputs/corpus_probe.log"

step "1b. split check (stability, uniformity, disjointness, CSN language mix)"
uv run python scripts/qwen3/split_check.py --config "$LAST" 2>&1 | tee outputs/split_check.log
[ "${PIPESTATUS[0]}" -eq 0 ] || die "split is not sound -- see outputs/split_check.log"

# --- 2. corpus_size from the measured tok/text ----------------------------------------------------

step "2. set corpus_size for ${TOKEN_TARGET} tokens/epoch"
uv run python - "$PROBE_JSON" "$TOKEN_TARGET" "${CONFIGS[@]}" <<'PY' || die "could not set corpus_size"
import json, re, sys
probe, target, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
tok = json.load(open(probe))["tok_per_text"]
size = round(target / tok / 1e5) * 100_000
print(f"measured {tok:.1f} tok/text -> corpus_size {size:,} "
      f"({size * tok / 1e9:.2f}B tokens/epoch)")
for n in names:
    p = f"configs/qwen3/{n}.yaml"
    s = open(p).read()
    s2, k = re.subn(
        r"^(\s*corpus_size:\s*)\d+.*$",
        rf"\g<1>{size} # {tok:.1f} measured tok/text x {size:,} = "
        rf"{size * tok / 1e9:.2f}B tokens/epoch (corpus_probe.py)",
        s, count=1, flags=re.M)
    assert k == 1, f"no corpus_size in {p}"
    open(p, "w").write(s2)
    print(f"  {p}")
PY

# --- 3. build the corpus once ---------------------------------------------------------------------
# All three configs share one cache key (same corpus/size/seq_len/tokenizer), so one build serves
# the set. Doing it here means the ranks cache-hit instead of each re-streaming the whole corpus.

step "3. build corpus (hours; ~10 GB tokenized + a larger HF download cache)"
uv run python scripts/qwen3/build_corpus.py --config "$LAST" 2>&1 | tee outputs/build_corpus.log
grep -q "token-passes at" outputs/build_corpus.log \
  || die "build did not finish -- exit 134 is normal, a missing 'token-passes' line is not"
grep -n "EXHAUSTED" outputs/build_corpus.log && printf '\n^ those weights did not take effect\n'

# --- 4. lr from the real step count ---------------------------------------------------------------
# plan_batches is deterministic and the trainer shards it, so this reproduces the logged
# "steps/epoch/rank" exactly rather than estimating it from padded-token arithmetic.

step "4. derive lr from steps/epoch"
uv run python - "$WORLD" "$DRIFT" "${CONFIGS[@]}" <<'PY' || die "could not derive lr"
import re, sys
import pyarrow.compute as pc
from franken.config import Config
from franken.distill.batching import plan_batches
from franken.tasks import build_task

world, drift, names = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
cfg = Config.from_yaml(f"configs/qwen3/{names[-1]}.yaml")
task = build_task(cfg.train.task)
ds = task.datasets(task.build_tokenizer(cfg), cfg, splits=("train",))["train"]
lengths = pc.list_value_length(ds.data.column("input_ids")).to_numpy(zero_copy_only=False)
opt = cfg.train.distill
per_rank = len(plan_batches(lengths, opt.token_budget, opt.max_seqs, cfg.train.seed)) // world
total = per_rank * opt.epochs
lr = drift / total**0.5
print(f"{per_rank:,} steps/epoch/rank x {opt.epochs} epochs = {total:,} steps -> lr {lr:.3e}")
print("  reconcile against the trainer's first line: 'token-budgeted batching: N steps/epoch/rank'")
for n in names:  # identical lr in all three, or depth stops being the only difference
    p = f"configs/qwen3/{n}.yaml"
    lines = open(p).read().split("\n")
    at = next(i for i, ln in enumerate(lines) if re.match(r"^  distill:\s*$", ln))
    j = next(i for i in range(at + 1, len(lines)) if re.match(r"^    lr:", lines[i]))
    lines[j] = f"    lr: {lr:.3e} # {drift} / sqrt({total:,} steps) -- run_ladder.sh"
    open(p, "w").write("\n".join(lines))
    print(f"  {p}")
PY

# --- 5. the ladder --------------------------------------------------------------------------------
# depth28 first: it is mathematically the teacher, so if it does not land near ceiling the setup
# regressed and the other two rows cannot be interpreted.

step "5. distill the ladder (~12h, depth28 first)"
uv run python scripts/qwen3/run_experiments.py --devices "$DEVICES" --ddp \
  "${PATHS[@]}" 2>&1 | tee outputs/ladder_train.log
printf "\nwatch: dynamo unique_graphs ~8 is healthy; a climbing count means eager fallback.\n"

# --- 6. both eval suites --------------------------------------------------------------------------
# retrieval_eval = the off-distribution canaries (comparable to the existing tables).
# corpus_eval    = held-out rows of the training corpus itself.
# Read them TOGETHER: small deficits on corpus_eval with large ones on retrieval_eval means
# coverage, not architecture. `--split validation` on purpose -- test is touched once, later.

step "6a. canary suite (retrieval_eval)"
uv run python scripts/qwen3/run_experiments.py --devices "$DEVICES" --eval-only \
  "${PATHS[@]}" 2>&1 | tee outputs/ladder_eval.log

step "6b. matched suite (corpus_eval, held-out corpus rows)"
for c in "${CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES="${DEVICES%%,*}" uv run python scripts/qwen3/corpus_eval.py \
    --config "configs/qwen3/$c.yaml" --split validation \
    --student-ckpt "outputs/qwen3_$c/student/pytorch_model.bin" \
    --json "outputs/experiments/corpus_eval_$c.json" 2>&1 | tee "outputs/corpus_eval_$c.log"
done

step "done"
printf 'results:  outputs/experiments/results.md  outputs/experiments/corpus_eval_*.json\n\n'
