# `franken.data.corpus`

The shared corpus layer: read a source's rows, measure them, and build a tokenized artifact from a
declared mix — from **one declaration per dataset**.

Declarations live per model, in `franken/data/<model>/registry.py`. `qwen3/` declares
`multi_domain` (pairs, judgements, instruction prefixes, for `task: embed`); `llama/` declares
`llama_web` (plain documents, for `task: lm`). `franken.data.corpus_sources(name)` resolves a
`train.corpus` string across both — one flat namespace, because the cache is one flat directory.

## How data reaches a student

```
configs/<model>/x.yaml      corpus, tokens_per_epoch, max_seq_len, pack
  ↓
Task.datasets()             franken/tasks/selfdistill.py
  ↓
load_corpus(tokenizer, name, corpus_sources(name), ...)
  ↓  cache hit?  ->  load_from_disk, done. Nothing below runs.
  ↓
_rows()   per source: records() -> corpus_texts() -> tokenize -> draw to the TOKEN quota
          pack:   best-fit blocks of exactly max_seq_len, padded (see "Anatomy" below)
          plain:  one row per document, truncated to max_seq_len
  ↓
Dataset.from_generator -> shuffle(seed=0) -> save_to_disk (tmp + rename)
outputs/corpus_cache/v<N>-<mix>-<split>-<size>-<cap>-<tokenizer>[-packed]
  ↓
Distiller: with_format("torch") -> plan_batches(token budget) -> DataCollatorWithPadding
  ↓
Task.model_inputs()  packed: derives position_ids and drops attention_mask from the forward
```

Everything above the cache line runs once and is expensive (hours at the 2B budget); everything
below runs every epoch and is cheap. **Tokenization happens at build time, not at train time.**

Three things are worth knowing about that path.

**The build draws to a quota; it does not plan.** Each source is streamed until it has contributed
`budget × weight` tokens. It used to estimate tokens/doc from a sample, plan a document count, then
stream again — which tokenized everything twice and left realized shares ±3pp off declared. Drawing
to a quota removes the estimate, so **`measure.py` cannot influence an artifact**: it is reporting
only. That is why `build.py` does not import it.

**`Source.weight` is a TOKEN share.** Documents differ in length by more than 10× within a mix
(arxiv abstracts ~180 tokens against codeparrot files ~2,500), so a text-count share silently
produced a different mix than the one declared.

**Rows are stored ragged; batches are built by token budget.** Nothing is padded to `max_seq_len` on
the unpacked path. `DataCollatorWithPadding` pads each batch to *its own* longest row, and
`plan_batches` (`franken/distill/batching.py`) length-buckets within a 12,800-row window before
filling to `distill.tokens_per_step`, which keeps occupancy around 98.5%.

## Anatomy of a packed block (`train.pack: true`)

Packing exists so that no token is thrown away: unpacked, a document longer than `max_seq_len` is
**truncated** and its tail is lost. Packed, `truncation` is off and every token reaches the artifact.

### What the build does

1. **Tokenize whole.** `add_special_tokens=True`, so each document already begins with **BOS**.
2. **Terminate it.** `_rows` appends **EOS**: `document = ids + [eos]`. EOS means exactly one thing —
   *this document ended*. It is not used for padding, so counting EOS counts documents.
3. **Fragment only if forced.** A document longer than `max_seq_len` cannot fit any block, so it is
   cut into `max_seq_len`-sized pieces. Anything that fits stays whole.
4. **Best-fit-decreasing into blocks.** Fragments accumulate into a buffer of `_PACK_BUFFER_DOCS`,
   get sorted longest-first, and each is placed into the *tightest* block that still holds it
   (`_bin_pack`). The buffer is then flushed whole, so no half-full block survives into the next one.
5. **Pad to exactly `max_seq_len`** with the tokenizer's **pad token**, and record
   `attention_mask` = 1 on real tokens, 0 on padding.

Packing is **per source**, so the `source` column stays exact — a block spanning sources could only
record a majority.

### What a block looks like

Two documents of 5 and 7 tokens in a block of 16:

```
input_ids       BOS a1 a2 a3 EOS  BOS b1 b2 b3 b4 b5 EOS  PAD PAD PAD PAD
attention_mask    1  1  1  1   1    1  1  1  1  1  1   1    0   0   0   0
position_ids      0  1  2  3   4    0  1  2  3  4  5   6    0   1   2   3   <- derived at train time
segment           0  0  0  0   0    1  1  1  1  1  1   1    2   2   2   2
```

### How EOS becomes document isolation

`position_ids` is **not stored**; `franken/distill/packing.py::doc_positions` derives it from
`input_ids` at train time. The rule is one line: **a position restarts at index 0, or at any index
whose predecessor is EOS.** That is exactly where the build put the boundaries, so the artifact needs
no extra column.

From those positions, `doc_ids` recovers the segment index as `(diff(position_ids) != 1).cumsum()`,
and the model masks attention to *within* a segment. Consequences:

- **Documents cannot see each other**, even though they share a block. An isolated block computes
  bit-identically to running its documents separately (`test_isolated_block_equals_separate_forwards`).
- **Padding is its own segment.** The first pad follows the last document's EOS, so it restarts —
  and because pads are not EOS, the rest continue from it as *one* segment rather than many.
- **Real tokens never see padding anyway**, because padding is right-side and attention is causal.
  The segment is what keeps the pad region cheap under `attn_impl: flex`, not what makes it correct.
- **A fragment of an over-long document restarts too.** It has no prefix inside the block, so it is
  presented as a fresh sequence. Honest, but its opening tokens are a weaker distillation signal —
  the teacher is predicting mid-document text with its context removed.

### What the teacher sees

`SelfDistillTask.model_inputs` sends the packed forward **`input_ids` + `position_ids`, and no
`attention_mask`**. That omission is load-bearing rather than an optimization: HF derives the
teacher's document-isolation mask from `position_ids` **only** when `attention_mask is None` *and*
`past_key_values is None` (`transformers/masking_utils.py`). Send the mask, or leave `use_cache` on,
and the teacher silently keeps cross-document attention while the student isolates — no error, just
a wrong target. `load_teacher` sets `config.use_cache = False` for the same reason.

The loss still masks padding: it reads `batch["attention_mask"]`, which the collator kept, so only
the *forward* drops it.

## Modules

| module | owns |
|---|---|
| `source.py` | `Source` / `Qrels` — the declaration schema, plus `normalized()` |
| `spec.py` | `Record`, the instruction wire format (`instruct`), the split policy (`split_of`, `SPLIT_PCT`), and `corpus_texts` / `eval_pair` |
| `adapters.py` | one `row -> Record \| None` per dataset *shape*: `pair`, `triplet`, `marco`, `titled`, `paragraphs`, `whole`, `wikitext` |
| `read.py` | `records()` — a source's rows for one split, hash-split and qrels-holdout applied — and `source_texts()` |
| `measure.py` | `profile()` per source, `describe()` / `realized_mix()` / `real_tokens()` on a built artifact. **Reporting only** |
| `build.py` | `_rows` (the draw), `_bin_pack`, the cache key, `load_corpus` |
| `evalset.py` | `Pool` (documents, queries, judgements) from a source's held-out rows |

Imports run one way: `build → read` and `measure → read`, never between `build` and `measure`.

## Using it

```python
from franken.data import corpus_sources
from franken.data.corpus import load_corpus, pool
from franken.data.qwen3 import mix

name = "multi_domain"
data = load_corpus(tokenizer, name, corpus_sources(name), tokens_per_epoch=2e9, max_seq_len=1024)
data["train"]        # input_ids, attention_mask, source (uint8 index into the source list)
data["collator"]     # DataCollatorWithPadding

for src in mix(name):
    p = pool(src, "validation", name)   # cached to outputs/corpus_pool_cache
```

⚠️ **Anything that means "tokens" must read the mask, not the row length.** A padded block is not a
full one. `measure.real_tokens()` exists for this; `describe()` and `realized_mix()` both use it, so
the realized-mix table is a genuine token share on both tracks.

## Adding a dataset

One `Source` in your model's `registry.py`. Pick the adapter matching its row shape (or add one if
the shape is new), set `key` to the column the split hashes on — or leave it `None` if the dataset
ships its own splits — and give it a relative token `weight`; weights are normalised at import.

⚠️ **Prefer a hash `key` over the dataset's own splits.** With `whole` there is no short side to
average against, so an upstream length skew shows through: CodeSearchNet's train/val/test means read
985 / 1354 / 1210 characters and tripped the holdout gate. A hash key makes the splits both disjoint
and identically distributed.

⚠️ **Test that a source still loads before declaring it.** `datasets` 5.0 removed script loaders, so
`codeparrot/github-code-clean` and several `bigcode/*` repos fail or are gated.

**Every source in a *retrieval* mix must be scoreable** — it yields `(query, positive)` records, or
it declares `Qrels`. `franken/scripts/qwen3/corpus.py` fails the gate before a build is paid for. An
LM mix is exempt: `adapters.whole` yields documents only, and logit KD scores perplexity, not
ranking. An unscoreable slice is a permanent blind spot — that is how `code_apps` −53.9% turned out
to be measuring corpus coverage rather than the depth cut.

`titled` and `wikitext` yield documents only, so any source using them in a retrieval mix needs
`Qrels`.

## Contracts worth knowing

**Splits.** `split_of(key)` is a pure function of a stable key (blake2b — never `hash()`, which
Python salts per process), so the three splits are disjoint however a stream is read and identical
text cannot straddle them. A row's whole yield goes to one split, which keeps a paragraph with its
article and an anchor with its positive. `key` is declared explicitly because `evalset` re-derives
membership from it: the two must hash the same string, or the eval silently scores trained rows.

`SPLIT_PCT` is **1% validation / 4% test**, sized to what each split is *for*. **Validation selects,
test reports.** Validation draws `VAL_POOL` = 500 documents — the pool `Distiller.train` scores
`recall@10` on — at the same token composition as train, by over-drawing and trimming after the
shuffle. Test fills a 500×5,000 retrieval pool *per source*, and at 2% three sources could not reach
it (`codefeedback` 2,900, `glaive_code` 2,726, `stackexchange` 4,365 documents), which biased
`MACRO-pair` by averaging tasks of unequal difficulty. `eval.py --split` therefore defaults to
`test`, so the reported in-distribution number does not share rows with the pool that selected the
model.

**Instructions.** `Source.instruct` is a *task description*; `spec.instruct()` wraps it in the wire
format verified against Qwen3-Embedding's `config_sentence_transformers.json` (no space after
`Query:`). `None` leaves the query bare, correct for a symmetric task. Documents never take one.
The prefix is baked into the cached text, so changing it needs a `_CACHE_VERSION` bump. Llama's mix
sets it nowhere — a base LM has no such protocol.

**The quota counts tokens PLACED, per fragment.** Counting tokens *read* let a single 36k-token
Hindi article blow `finewiki_hi`'s 63k quota by 57%; counting per *document* reintroduces exactly
that bug, since a document is buffered all at once. Per fragment bounds the overshoot at one block.

**Packing is a different artifact, never a reinterpreted one.** `train.pack` adds its own cache-key
suffix, so turning it on cannot silently re-read an existing build. Under packing `describe`'s
`truncated` counts blocks that needed **no padding**, which is why the report relabels it `full@cap`
and adds a `pad=%` column.

**Cache invalidation is manual.** `build._CACHE_VERSION` must be bumped when a source, a weight or
an adapter changes — the key covers the *request* (mix, split, size, cap, tokenizer, packing), not
the recipe that answered it. Deliberate: a content digest would throw away an hours-long build on a
cosmetic edit. The cost is that forgetting to bump serves stale text silently.

🚨 **`Dataset.from_generator` fingerprints its `gen_kwargs`, not the generator's code.** A logic
change to `_rows` does not invalidate HF's own cache, so it will happily republish a stale draw —
once observed reporting `CORPUS OK` in 0.0 min with the draw lines missing from the log. `build.py`
therefore points it at a scratch dir deleted in a `finally`, leaving `_CACHE_VERSION` as the single
invalidation story. Do not remove that.

## Which source supports which metric

**`recall@10` works on any source** — it needs no queries, no judgements and no labels.
`recall_at_k` masks the diagonal of two self-similarity matrices and intersects the top-10s, so the
gold set is whatever *the teacher* ranks highest, and the teacher scores 1.0 by construction. It
measures **fidelity, never quality**, and compares only at a fixed pool size (at fixed per-vector
damage it reads 1.000 at n=11, 0.110 at 500, 0.039 at 5,000).

**`nDCG@10` needs judgements, and they come in three tiers that are not equivalent:**

| tier | sources | what the number means |
|---|---|---|
| graded | `nfcorpus` (0–2), `xpqa_cmn` | genuine graded ranking |
| real binary | `scifact`, `fiqa`, `code_apps`, `nq_passage`, `hotpotqa_passage` | real judgements, several golds possible, no grades |
| synthesized single-gold | every pair source | one binary gold ⇒ IDCG = 1, so nDCG@10 is **exactly** `1/log2(rank+2)` — MRR with a log discount |

Tier 3 is not weak for lacking grades; `1/log2(rank+2)` is sensitive (1.000 / 0.631 / 0.500 at ranks
0–2). It is weak where the gold is **one arbitrary member of an equally valid set**, because the
alternatives sit in the same pool as unjudged false negatives and the metric then scores a
tie-break. `Source.scores_ndcg = False` marks those: the `wiki_*` slices (`paragraphs` promotes one
sibling and sends the rest to `docs`) and `specter` (title → *a* related title). They still yield
pairs, so the scoreability gate is untouched and their `recall@10` is reported; only the nDCG cells
are blank, and they leave `MACRO-pair`.

The criterion is the adapter **shape**, not the teacher's score. Teacher scores merely agree with it
(`specter` 0.2921, `wiki_ja` 0.4864 vs `glaive_code` 0.9970) — picking it from a score threshold is
how `magicoder` was once retired on a single datum with no student delta ever measured.

⚠️ In `multi_domain` those slices are ~24% of the corpus by weight, which carries **fidelity only**.
A pre-existing blind spot made visible, not a new one: four of the five non-English wiki languages
have no external twin either.

## Caveats

- **Qrels-derived pools are weaker than pair-derived ones.** The judgements pick the gold documents,
  so each lands in `train` with ~96% probability: the document side is *seen*, and only the
  distractors are held out. `eval.py` keeps them in a separate macro (`MACRO-qrels`) for this reason.
- **A source that runs dry does not take effect at its declared weight.** The build prints
  requested-vs-delivered tokens per source; anything flagged `EXHAUSTED` is a weight that did not
  apply. An earlier mix asked for 20% queries and delivered 5.2%, unnoticed, through a whole tracker.
- **A source that fails mid-draw raises.** `profile()` tolerates a dead loader because the gate must
  name every one; the build makes the opposite choice, because a silently short corpus trains a
  different experiment than the one declared.
- **Adapters keep the gold count small on purpose.** Every positive becomes gold, and many golds per
  query makes nDCG@10 trivially satisfiable — which is why `paragraphs` promotes one sibling
  paragraph and sends the rest to `docs`.
- **A green test suite is not a usable artifact.** `_rows` once dropped the `attention_mask` column:
  every test passed, the gate passed, the realized-mix table was exact, and `torch_columns()` could
  not load the result. Load the artifact through the real task path.
