"""Does a task-matching instruction beat the generic web-search one? Teacher only.

Blocking before a corpus rebuild: a corpus-side instruction is baked into the cached text, so
adopting one costs a `build._CACHE_VERSION` bump and hours. This is the cheap way to find out
first -- no training, no build, and the document side never changes, so only the 500 queries are
re-embedded per candidate.

⚠️ A teacher gain shows the instruction fits the task SHAPE. Whether the student trains better is
a question only a run answers, so do not report a win here as a student-side result.

Adoption is manual and explicit: move the winning string into `Source.instruct` (corpus) or the
`EXTERNAL` entry (external). Nothing here writes a config.

    uv run python scripts/qwen3/instruct_gate.py --config configs/qwen3/depth28_multi_domain.yaml
"""

from __future__ import annotations

import json

import common
import datasets
from common import K, _embed_texts, load, ndcg_pool
from eval import EXTERNAL
from franken.data.embed_corpus import WEB_SEARCH, instruct, mix, pool

# Corpus sources whose query side is genuinely a retrieval query, and the task each one retrieves.
# The 6 wikis, stackexchange and specter are symmetric -- no asymmetry to instruct -- and nq /
# hotpotqa emit documents only, so none of them are candidates. msmarco already carries WEB_SEARCH.
CORPUS_CANDIDATES = {
    "gooaq": "Given a question, retrieve the answer that best responds to it",
    "eli5": "Given an open-ended question, retrieve a long-form explanation that answers it",
    "pubmed": "Given a biomedical article title, retrieve its abstract",
    "s2orc": "Given a citation sentence, retrieve the abstract of the cited paper",
    "code": "Given a docstring, retrieve the function that implements it",
    "codefeedback": "Given a programming instruction, retrieve the code that satisfies it",
    "glaive_code": "Given a programming question, retrieve the code that answers it",
}


def _score(m, p, d_emb, task: str | None, raw: list[str]) -> float:
    q_emb = _embed_texts(
        m.backend, m.teacher, m.tokenizer, m.cfg, [instruct(task, q) for q in raw], m.device
    )
    return ndcg_pool(p, d_emb, q_emb)


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--split", default="test", choices=("validation", "test"))
    args = p.parse_args(argv)

    datasets.disable_progress_bars()
    m = load(args)
    cfg_sources = {s.name: s for s in mix(m.cfg.train.corpus)}

    # Pools built BARE, so `raw` is the unprefixed query and every candidate is applied on top.
    # Corpus pools are already cached bare (only msmarco carries an instruction today).
    jobs = [("external", n, EXTERNAL[n][0](None), EXTERNAL[n][1]) for n in EXTERNAL]
    jobs += [
        ("corpus", n, pool(cfg_sources[n], args.split, m.cfg.train.corpus), cand)
        for n, cand in CORPUS_CANDIDATES.items()
        if n in cfg_sources
    ]

    print(
        f"\n== instruction gate -- TEACHER nDCG@{K}, {m.cfg.train.teacher_model} "
        f"@{m.cfg.train.max_seq_len} ==\n"
        f"   bare = no instruction (what 17 of 18 corpus sources use today)\n"
        f"   web  = the one generic string ({WEB_SEARCH[:40]}...)\n"
        f"{'suite':>9} {'task':>18} {'bare':>8} {'web':>8} {'task-fit':>9} {'best':>9}  verdict",
        flush=True,
    )

    out = {}
    for suite, name, pl, cand in jobs:
        if not pl:
            print(f"{suite:>9} {name:>18}   no queries", flush=True)
            continue
        raw = list(pl.q_texts)  # bare, because the pool was built with task=None
        d_emb = _embed_texts(m.backend, m.teacher, m.tokenizer, m.cfg, pl.d_texts, m.device)
        s = {
            "bare": _score(m, pl, d_emb, None, raw),
            "web": _score(m, pl, d_emb, WEB_SEARCH, raw),
            "task": _score(m, pl, d_emb, cand, raw),
        }
        best = max(s, key=s.get)
        # Against what that suite uses TODAY: external is on `web`, corpus sources are on `bare`.
        baseline = "web" if suite == "external" else "bare"
        print(
            f"{suite:>9} {name:>18} {s['bare']:>8.4f} {s['web']:>8.4f} {s['task']:>9.4f} "
            f"{best:>9}  {s[best] - s[baseline]:+.4f} vs {baseline}",
            flush=True,
        )
        out[name] = {"suite": suite, "candidate": cand, "baseline": baseline, **s, "best": best}

    print(
        "\nAdopt where `task-fit` wins by more than noise; leave the rest. Corpus adoptions must "
        "land WITH the rebuild (they change training text); external ones can land alone.\n"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
