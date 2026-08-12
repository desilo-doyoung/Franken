"""Which query instruction is best per task? Teacher only.

Blocking before a corpus rebuild: a corpus instruction is baked into the cached text, so adopting
one costs a `build._CACHE_VERSION` bump and hours. This is the cheap way to check first -- no
training, no build, and the document side is instruction-independent, so only the 500 queries are
re-embedded per variant.

Four variants per task:
    bare      no instruction -- what a symmetric task correctly uses
    web       the one generic web-search string
    guessed   round-1 candidate, written from the adapter's shape alone
    adopted   what `Source.instruct` / `EXTERNAL` holds now, written after reading the dataset card
              and real query/gold samples

`guessed` vs `adopted` is the point: it measures whether reading the data bought anything, which
round 1 could not answer because it never had a second string to compare against.

⚠️ A teacher gain shows the instruction fits the task SHAPE. It cannot show the student trains
better -- the teacher already knows the instructed mode, while a student has to learn it from the
~10% of training text that carries a prefix. Only a run answers that, so do not report a win here
as a student-side result.

⚠️ Four corpus sources sit at 0.97-0.996 teacher, where this has no power to discriminate between
strings. Read those rows as "no objection", not as evidence.

Adoption is manual: edit `Source.instruct` (corpus) or the `EXTERNAL` entry (external), then bump
`evalset._CACHE_VERSION` -- the pool key does not cover the instruction.

    uv run python scripts/qwen3/instruct_gate.py --config configs/qwen3/depth28_multi_domain.yaml
"""

from __future__ import annotations

import json
from dataclasses import replace

import common
import datasets
from common import K, _embed_texts, load, ndcg_pool
from eval import EXTERNAL
from franken.data.embed_corpus import WEB_SEARCH, instruct, mix, pool

# Corpus sources with a genuine query side. The 6 wikis, `stackexchange` and `specter` are
# symmetric, `s2orc` turned out to be too (abstract -> cited-paper text, both 200+ tokens), and
# nq / hotpotqa emit documents only -- none of them have a query to instruct. `s2orc` stays here as
# the control: it is the one source where the web string measured NEGATIVE (-0.0088).
CORPUS_UNDER_TEST = (
    "msmarco",
    "gooaq",
    "eli5",
    "pubmed",
    "s2orc",
    "code",
    "codefeedback",
    "glaive_code",
)

# Round-1 candidates, written from the adapter shape before anyone looked at a row or a card. Kept
# ONLY as the control for the adopted strings; never adopt from here. Three were wrong about what
# the gold even is: `s2orc` is not a citation sentence, and neither code source returns bare code.
GUESSED = {
    "gooaq": "Given a question, retrieve the answer that best responds to it",
    "eli5": "Given an open-ended question, retrieve a long-form explanation that answers it",
    "pubmed": "Given a biomedical article title, retrieve its abstract",
    "s2orc": "Given a citation sentence, retrieve the abstract of the cited paper",
    "code": "Given a docstring, retrieve the function that implements it",
    "codefeedback": "Given a programming instruction, retrieve the code that satisfies it",
    "glaive_code": "Given a programming question, retrieve the code that answers it",
    "nfcorpus": "Given a medical question, retrieve documents that best answer it",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "xpqa_cmn": "Given a product question, retrieve passages that answer the question",
    "code_apps": "Given a programming problem, retrieve the code that solves it",
}


def _score(m, p, d_emb, task: str | None, raw: list[str]) -> float:
    q_emb = _embed_texts(
        m.backend, m.teacher, m.tokenizer, m.cfg, [instruct(task, q) for q in raw], m.device
    )
    return ndcg_pool(p, d_emb, q_emb)


def _variant(m, p, d_emb, raw, scored: dict, text: str | None) -> float:
    """`None` and WEB_SEARCH are already in `scored`; only a distinct string costs another pass."""
    if text is None:
        return scored["bare"]
    return scored["web"] if text == WEB_SEARCH else _score(m, p, d_emb, text, raw)


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--split", default="test", choices=("validation", "test"))
    args = p.parse_args(argv)

    datasets.disable_progress_bars()
    m = load(args)
    sources = {s.name: s for s in mix(m.cfg.train.corpus)}

    def corpus_pool(src):
        # BARE and UNCACHED, so `raw` is the unprefixed query: a source that now carries an
        # instruction would hand back prefixed queries and every variant would double-wrap. Uncached
        # means re-streaming the source -- the reason this is a gate and not a suite.
        return pool(replace(src, instruct=None), args.split, m.cfg.train.corpus, cache=False)

    jobs = [("external", n, (lambda ld=ld: ld(None)), task) for n, (ld, task) in EXTERNAL.items()]
    jobs += [
        ("corpus", n, (lambda s=sources[n]: corpus_pool(s)), sources[n].instruct)
        for n in CORPUS_UNDER_TEST
        if n in sources
    ]

    print(
        f"\n== instruction gate -- TEACHER nDCG@{K}, {m.cfg.train.teacher_model} "
        f"@{m.cfg.train.max_seq_len}, split={args.split} ==\n"
        f"   adopted = registry / EXTERNAL as it stands; guessed = the pre-dataset-card candidate\n"
        f"   corpus pools rebuild uncached to recover raw queries; expect minutes per source\n"
        f"{'suite':>9} {'task':>18} {'bare':>8} {'web':>8} {'guessed':>8} {'adopted':>8} "
        f"{'best':>8}  adopted vs best",
        flush=True,
    )

    out = {}
    for suite, name, build, adopted in jobs:
        try:
            pl = build()
        except Exception as e:  # one dead loader must not cost the other eleven
            print(f"{suite:>9} {name:>18}   FAILED {type(e).__name__}: {e}"[:110], flush=True)
            continue
        if not pl:
            print(f"{suite:>9} {name:>18}   no queries", flush=True)
            continue
        raw = list(pl.q_texts)
        d_emb = _embed_texts(m.backend, m.teacher, m.tokenizer, m.cfg, pl.d_texts, m.device)

        s = {"bare": _score(m, pl, d_emb, None, raw), "web": _score(m, pl, d_emb, WEB_SEARCH, raw)}
        s["guessed"] = _variant(m, pl, d_emb, raw, s, GUESSED.get(name))
        s["adopted"] = _variant(m, pl, d_emb, raw, s, adopted)
        best = max(s, key=s.get)
        print(
            f"{suite:>9} {name:>18} {s['bare']:>8.4f} {s['web']:>8.4f} {s['guessed']:>8.4f} "
            f"{s['adopted']:>8.4f} {best:>8}  {s['adopted'] - s[best]:>+.4f}",
            flush=True,
        )
        out[name] = {"suite": suite, "adopted_text": adopted, **s, "best": best}

    if out:
        gained = [r["adopted"] - r["guessed"] for r in out.values()]
        print(
            f"\nreading the data: adopted - guessed, median {sorted(gained)[len(gained) // 2]:+.4f}"
            f", worst {min(gained):+.4f}, best {max(gained):+.4f}"
        )
    if regret := {n: r for n, r in out.items() if r["best"] != "adopted"}:
        print("ADOPTED IS NOT BEST:")
        for n, r in regret.items():
            print(f"  {n:<18} adopted {r['adopted']:.4f}  best {r['best']} {r[r['best']]:.4f}")
    print(
        "\nA gap inside ~0.005 is the measured noise floor -- leave it alone. Corpus changes land "
        "WITH the rebuild and need an `evalset._CACHE_VERSION` bump; external ones stand alone.\n"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
