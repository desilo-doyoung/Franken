# `franken.data.embed_corpus`

The training corpus for embedding self-distillation, and the eval pools that score it — from **one
declaration per dataset**.

## Why it looks like this

The distillation loss is pointwise teacher-matching (`1 - cos` on the pooled embedding + masked
hidden MSE, see `franken/tasks/embed.py`). No labels, no logits, no in-batch negatives. So
**training needs only text plus one bit of role**: a query gets the instruction prefix, a document
gets none. Whether a dataset is query→answer, symmetric-pairwise or a triplet is irrelevant to the
gradient.

Pair structure earns its place for exactly two reasons:

1. both sides of a pair must land on the same side of the train/val/test split, or an eval task
   built from held-out rows has its answer in the training set;
2. the eval pools are built from it.

That is why the shared dataclass is `Record` at the **row** level, not a per-dataset config. One
adapter per dataset shape feeds both the corpus and the eval, through two functions that are the
whole sync point: `corpus_texts()` and `eval_pair()`. Before this, every dataset was declared twice
— once as a mix entry, once as a retrieval task in the eval script — and each column rename had to
land in both.

> Qwen3-Embedding itself is trained with contrastive InfoNCE over weakly-supervised then supervised
> pairs with hard negatives. That recipe governs *which pairs are worth collecting*, not this
> objective. Adding a contrastive term would need a different loss and large negative batches.

## Modules

| module | owns |
|---|---|
| `spec.py` | `Record`, the instruction format (`instruct`), the split policy (`split_of`, `SPLIT_PCT`), and `corpus_texts` / `eval_pair` |
| `adapters.py` | one `row -> Record \| None` function per dataset *shape* (`pair`, `triplet`, `marco`, `titled`, `paragraphs`, `wikitext`) |
| `registry.py` | `Source` / `Qrels` and the mix — **the only place a dataset is named** |
| `build.py` | streaming, weighted mixing, tokenizing, on-disk cache, `load_embed_corpus` |
| `evalset.py` | `Pool` (documents, queries, judgements) from a source's held-out rows |

## Using it

```python
from franken.data.embed_corpus import load_embed_corpus, mix, pool

data = load_embed_corpus(tokenizer, "multi_domain", size=9_000_000, max_seq_len=1024)
data["train"]        # input_ids, attention_mask, source (uint8 index into data["sources"])
data["collator"]     # DataCollatorWithPadding

for src in mix("multi_domain"):
    p = pool(src, "validation", "multi_domain")   # cached to outputs/corpus_pool_cache
```

`source` keeps provenance on the artifact: it is what lets the realized mix be verified after a
build rather than trusted from the build log.

## Adding a dataset

One `Source` in `registry.py`. Pick the adapter matching its row shape (or add one if the shape is
new), set `key` to the column the split hashes on — or leave it `None` if the dataset ships its own
splits — and give it a relative `weight`; weights are normalised at import, so no rebalancing.

**Every source must be scoreable.** It yields `(query, positive)` records, or it declares `Qrels`.
There is no escape hatch, and `scripts/qwen3/corpus.py` fails the gate before a build is paid for.
An unscoreable slice is a permanent blind spot — that is how `code_apps` −53.9% turned out to be
measuring corpus coverage rather than the depth cut.

`titled` and `wikitext` yield documents only, so any source using them needs `Qrels`.

## Two contracts worth knowing

**Splits.** `split_of(key)` is a pure function of a stable key (blake2b — never `hash()`, which
Python salts per process), so the three splits are disjoint however a stream is read and identical
text cannot straddle them. A row's whole yield goes to one split, which is what keeps a paragraph
from being separated from its article or an anchor from its positive. `key` is declared explicitly
because `evalset` re-derives membership from it: the two must hash the same string, or the eval
silently scores trained rows.

`SPLIT_PCT` is **1% validation / 4% test**, sized to what each split is *for* rather than split
evenly. **Validation selects, test reports.** Validation only ever draws `build.VAL_POOL` = 500
texts, the pool `Distiller.train` scores `recall@10` on to pick the checkpoint — 1% is ~50× that.
Test fills a 500×5,000 retrieval pool *per source*, and at 2% three sources could not reach it
(`codefeedback` 2,900, `glaive_code` 2,726, `stackexchange` 4,365 documents), which biased
`MACRO-pair` by averaging tasks of unequal difficulty. `eval.py --split` therefore defaults to
`test`, so the reported in-distribution number does not share rows with the pool that selected the
model.

**Instructions.** `Source.instruct` is a *task description*; `spec.instruct()` wraps it in the wire
format verified against this checkpoint's `config_sentence_transformers.json` (no space after
`Query:`). `None` leaves the query bare, which is correct for a symmetric task — there is no
query/document asymmetry to instruct. Documents never take one. The prefix is baked into the cached
corpus text, so changing a source's instruction needs a `build._CACHE_VERSION` bump.

**Cache invalidation is manual.** `build._CACHE_VERSION` must be bumped when a source, a weight or
an adapter changes — the cache key covers the *request* (preset, split, size, cap, tokenizer), not
the recipe that answered it. Deliberate: a content digest over the adapters would throw away an
hours-long build on a cosmetic edit. The cost is that forgetting to bump serves stale text silently.

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
tie-break. `Source.scores_ndcg = False` marks those: the six `wiki_*` (`paragraphs` promotes one
sibling and sends the rest to `docs`) and `specter` (title → *a* related title). They still yield
pairs, so `corpus.py`'s scoreability gate is untouched and their `recall@10` is reported; only the
nDCG cells are blank, and they leave `MACRO-pair`.

The criterion is the adapter **shape**, not the teacher's score. Teacher scores merely agree with it
(`specter` 0.2921, `wiki_ja` 0.4864 vs `glaive_code` 0.9970, `pubmed` 0.9877) — picking it from a
score threshold is how `magicoder` was once retired on a single datum with no student delta ever
measured.

⚠️ Those seven are ~24% of the corpus by weight, which now carries **fidelity only**. That is a
pre-existing blind spot made visible, not a new one: four of the five non-English wiki languages
have no external twin either.

## Caveats

- **Qrels-derived pools are weaker than pair-derived ones.** The judgements pick the gold documents,
  so each lands in `train` with ~96% probability: the document side is *seen*, and only the
  distractors are held out. `eval.py` keeps them in a separate macro (`MACRO-qrels`) for this reason.
- **A source that runs dry does not take effect at its declared weight.** `build` prints
  requested-vs-delivered per source; anything flagged `EXHAUSTED` is a weight that did not apply.
  An earlier mix asked for 20% queries and delivered 5.2%, unnoticed, through a whole tracker.
- **Adapters keep the gold count small on purpose.** Every positive becomes gold, and many golds per
  query makes nDCG@10 trivially satisfiable — which is why `paragraphs` promotes one sibling
  paragraph and sends the rest to `docs`.
