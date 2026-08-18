"""The mix: one ``Source`` per dataset, and nothing else declares a dataset.

**Every source is scoreable** — it yields (query, positive) records or declares ``Qrels``. No escape
hatch; `franken/scripts/qwen3/corpus.py` enforces it before a build is paid for. An unscoreable
slice
is a permanent blind spot, which is how `code_apps` -53.9% turned out to be measuring coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from franken.data.embed_corpus import adapters
from franken.data.embed_corpus.spec import WEB_SEARCH, Record


@dataclass(frozen=True)
class Qrels:
    """Real judgements, for a source whose rows hold no pair. Layouts are not inferrable: BeIR ships
    query-id/corpus-id in `test`, C-MTEB ships qid/pid in `dev` with queries under config `default`.

    Judged documents are held out of training by `build._judged` (v8): `evalset._from_qrels`
    force-adds every gold to the pool whatever it hashes to, so `split_of` alone left most of them
    in the draw. Still weaker than a pair task — the query side is unjudged text the pool inherits.
    """

    repo: str
    split: str = "test"
    queries: tuple[str | None, str] = ("queries", "queries")  # (config, split)
    cols: tuple[str, str, str] = ("query-id", "corpus-id", "score")


@dataclass(frozen=True)
class Source:
    name: str
    domain: str
    repo: str
    config: str | None
    adapt: Callable[[dict], Record | None]
    weight: float
    # Column the split is hashed from; None means the dataset ships its own splits. Explicit rather
    # than derived, because `evalset` must hash the identical string or it scores trained rows.
    key: str | None = None
    hf_split: str = "train"  # the single upstream split to draw from; unused when key is None
    split_map: Mapping[str, str] = field(default_factory=dict)  # ours -> upstream, when key is None
    # Task description for the query-side instruction; None leaves the query bare. Baked into the
    # cached corpus text, so changing it needs a `build._CACHE_VERSION` bump.
    instruct: str | None = None
    qrels: Qrels | None = None
    # False where the gold is one arbitrary member of an equally valid set (a sibling paragraph, a
    # related title): the unpromoted siblings sit in the pool as unjudged false negatives, so nDCG
    # scores an arbitrary tie-break. Such a source still yields pairs -- recall@10 and the split
    # guarantee both need them -- so this governs REPORTING only, not `corpus.py`'s gate.
    scores_ndcg: bool = True


_WIKI_LANGS = ("zh", "ja", "ar", "ru", "es")  # scripts that move the teacher activation range

# Relative weights, normalised below. Coverage is the motivation, not volume: training text was web
# + encyclopedia prose while nDCG is measured on biomedical and scientific benchmarks.
_MULTI_DOMAIN = [
    Source(
        "msmarco",
        "english_prose",
        "microsoft/ms_marco",
        "v2.1",
        adapters.marco,
        0.209,
        instruct=WEB_SEARCH,
    ),
    # `titled`, not a pair: nq packs 35.3 passages per article, so keyed on `_id` the title
    # straddles splits 60.7% of the time and leaks the query side. Scored on judgements instead.
    Source(
        "nq_passage",
        "english_prose",
        "BeIR/nq",
        "corpus",
        adapters.titled,
        0.06,
        key="_id",
        hf_split="corpus",
        qrels=Qrels("BeIR/nq-qrels"),
    ),
    Source(
        "hotpotqa_passage",
        "english_prose",
        "BeIR/hotpotqa",
        "corpus",
        adapters.titled,
        0.05,
        key="_id",
        hf_split="corpus",
        qrels=Qrels("BeIR/hotpotqa-qrels"),
    ),
    # `paragraphs` promotes one sibling to gold and sends the rest to `docs`, where they become
    # unjudged false negatives -- so nDCG here scores which arbitrary sibling won. Fidelity only.
    Source(
        "wiki_en",
        "english_prose",
        "wikimedia/wikipedia",
        "20231101.en",
        adapters.paragraphs,
        0.06,
        key="id",
        scores_ndcg=False,
    ),
    # Informal prose: the `fiqa` domain, -13.5% at depth 19.
    # Every instructed source carries WEB_SEARCH, measured rather than assumed: task-matching
    # strings written from the dataset cards came in at +0.0017 to -0.0067 against it, i.e. noise or
    # worse, on all seven. The model keys on the canonical string it was trained with, not on
    # semantic fit -- so tailoring is only worth it where it beat web by more than the ~0.005 floor,
    # which happened on exactly two EXTERNAL tasks (see franken.data.external) and no corpus source.
    Source(
        "gooaq",
        "informal",
        "sentence-transformers/gooaq",
        None,
        adapters.pair("question", "answer"),
        0.1,
        key="question",
        instruct=WEB_SEARCH,
    ),
    Source(
        "eli5",
        "informal",
        "sentence-transformers/eli5",
        None,
        adapters.pair("question", "answer"),
        0.03,
        key="question",
        instruct=WEB_SEARCH,  # a forum-question string measured -0.0067
    ),
    Source(
        "stackexchange",
        "informal",
        "sentence-transformers/stackexchange-duplicates",
        "post-post-pair",
        adapters.pair("post1", "post2"),
        0.03,
        key="post1",
    ),
    # Science bulk, and the query side is a TITLE, not a question. Chosen over a PubMed corpus
    # because nfcorpus IS PubMed: overlap here measured at 3 rows in 400,000, exact-hash only.
    Source(
        "arxiv",
        "science",
        "gfissore/arxiv-abstracts-2021",
        None,
        adapters.pair("title", "abstract"),
        0.06,
        key="id",
        instruct=WEB_SEARCH,
    ),
    # NOT "citation sentence -> cited abstract": the card pairs an abstract with "an excerpt or
    # passage from a cited paper", and both sides measure ~217 / ~280 tokens. Abstract -> abstract
    # is SYMMETRIC, so there is no query side to instruct -- and the gate agrees, -0.0088 for the
    # web string, the only negative among the corpus sources.
    Source(
        "s2orc",
        "science",
        "sentence-transformers/s2orc",
        "abstract-citation-pair",
        adapters.pair("abstract", "citation"),
        0.08,
        key="abstract",
    ),
    # Title -> *a* related title: many papers are equally related, so the promoted one is arbitrary.
    # Teacher 0.2921 is the symptom, not the criterion -- the shape is.
    Source(
        "specter",
        "science",
        "sentence-transformers/specter",
        "triplet",
        adapters.triplet,
        0.03,
        key="anchor",
        scores_ndcg=False,
    ),
    # CSN is library functions; the instruction sets are problem -> solution, i.e. APPS's task shape
    # without being APPS, so they close that genre gap while `code_apps` stays a clean canary.
    Source(
        "code",
        "code",
        "code-search-net/code_search_net",
        "python",
        adapters.pair("func_documentation_string", "whole_func_string"),
        0.04,
        instruct=WEB_SEARCH,
    ),
    Source(
        "codefeedback",
        "code",
        "m-a-p/CodeFeedback-Filtered-Instruction",
        None,
        adapters.pair("query", "answer"),
        0.015,
        key="query",
        instruct=WEB_SEARCH,
    ),
    Source(
        "glaive_code",
        "code",
        "glaiveai/glaive-code-assistant",
        None,
        adapters.pair("question", "answer"),
        0.013,
        key="question",
        instruct=WEB_SEARCH,
    ),
] + [
    # `datasets` 5.0 removed script loaders, killing miracl/mr-tydi/MLDR; wikipedia is
    # parquet-native with 323 language configs, so multilingual coverage comes from here.
    Source(
        f"wiki_{lang}",
        "multilingual",
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        adapters.paragraphs,
        0.026,
        key="id",
        scores_ndcg=False,  # same arbitrary-sibling gold as wiki_en
    )
    for lang in _WIKI_LANGS
]


def _normalized(sources: list[Source]) -> list[Source]:
    # Declared relative so dropping a source is a one-line delete, not a rebalance of 18 numbers.
    total = sum(s.weight for s in sources)
    return [replace(s, weight=s.weight / total) for s in sources]


# Scoreable mixes only — this is what the gates and the eval iterate.
MIXES: dict[str, list[Source]] = {"multi_domain": _normalized(_MULTI_DOMAIN)}

# Buildable presets. `smoke` is a pipeline proof, never a result, so it sits outside MIXES rather
# than weakening the scoreability rule for everything else.
PRESETS: dict[str, list[Source]] = {
    "smoke": [
        Source(
            "wikitext2",
            "english_prose",
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            adapters.wikitext,
            1.0,
        )
    ],
    # The original three-source recipe, kept only so the pre-1024 configs naming it still run. Not
    # scoreable and not in MIXES: its sources carry no eval pair. Its declared weights never took
    # effect either -- all three exhaust before 2.1M, so the realized mix was 5.2 / 42.9 / 51.8%.
    "mixed": [
        Source(
            "msmarco_query",
            "query",
            "microsoft/ms_marco",
            "v1.1",
            adapters.marco_side("query"),
            0.2,
            instruct=WEB_SEARCH,
        ),
        Source(
            "msmarco_passage",
            "english_prose",
            "microsoft/ms_marco",
            "v1.1",
            adapters.marco_side("passage"),
            0.4,
        ),
        Source(
            "wikitext103",
            "english_prose",
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            adapters.wikitext,
            0.4,
        ),
    ],
    **MIXES,
}


def mix(name: str) -> list[Source]:
    if name not in MIXES:
        raise KeyError(
            f"No source registry for corpus {name!r}; registered: {sorted(MIXES)}. "
            "('smoke' is a single-source pipeline proof and is not scoreable by design.)"
        )
    return MIXES[name]
