"""Type a query, see what the student actually retrieves — and how the teacher ranked the same docs.

The tracker's `recall@10` is teacher-neighbour agreement over the pool's DOCUMENTS; it takes no
queries and no judgements, so it cannot be eyeballed. This prints the three things a single query
can support, each named separately, under a banner carrying the population numbers so an anecdote
is always read against the aggregate it came from.

    CUDA_VISIBLE_DEVICES=2 uv run python scripts/qwen3/search.py --source gooaq
        --config configs/qwen3/depth19_exact.yaml
        --student-ckpt outputs/qwen3_depth19/student/pytorch_model.bin

    CUDA_VISIBLE_DEVICES=2 uv run python scripts/qwen3/search.py --source specter
        --config configs/qwen3/smoke.yaml --query "graph neural networks"

`--worst N` picks the queries for you -- the N the student agrees with the teacher least on, which
is where a lever's damage shows if it has any. Both it and `--query` skip the REPL.

With no --student-ckpt at full depth the student IS the teacher: agree@10 must read 1.00 and the
"missed" block must be empty. That is the self-test.
"""

from __future__ import annotations

import random
import unicodedata

import common
import datasets
import torch
from common import K, embed_pool, ndcg_at_k, ndcg_pool, teacher_cache
from franken.data.embed_corpus import instruct, mix, pool
from franken.paths import RunPaths
from franken.tasks.embed import recall_at_k

SNIPPET = 58
LINES = 3  # rows of document text per hit; 1 restores the old one-line snippet
BINS = 11
PAD = " " * 26  # the rank/cos/t@/id columns, for a continuation row


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _snip(text: str) -> list[str]:
    # Collapse first: code and abstracts carry newlines that would break every column below.
    s = " ".join(text.split())
    rows = []
    while s and len(rows) < LINES:
        last = len(rows) == LINES - 1
        budget = SNIPPET - 3 if last else SNIPPET
        take, w = 0, 0
        for c in s:
            cw = 2 if unicodedata.east_asian_width(c) in "WF" else 1
            if w + cw > budget:
                break
            take, w = take + 1, w + cw
        # Prefer a word break, but only a late one -- CJK sources carry no spaces at all.
        if not last and take < len(s) and (brk := s.rfind(" ", 0, take + 1)) > budget * 0.6:
            take = brk
        row, s = s[:take].rstrip(), s[take:].lstrip()
        if last and s:
            row += "..."
        rows.append(row + " " * (SNIPPET - _width(row)))
    return rows or [" " * SNIPPET]


def _recall(ranked_ids: list[str], gold: dict[str, float]) -> float:
    # min(K, |gold|): a query with 40 judged docs could never exceed 0.25 otherwise, which reads as
    # model damage rather than the ceiling it is.
    return sum(d in gold for d in ranked_ids[:K]) / min(K, len(gold))


def _top(q_vec, d_emb):
    sims = (q_vec @ d_emb.T).squeeze(0)
    top = sims.topk(min(K, sims.numel()))
    return top.indices.tolist(), top.values.tolist()


@torch.no_grad()
def _aggregate(p, td, tq, sd, sq):
    agree, gold_t, gold_s = [], [], []
    for i in range(0, len(p.q_ids), 256):
        t_top = (tq[i : i + 256] @ td.T).topk(K, dim=-1).indices
        s_top = (sq[i : i + 256] @ sd.T).topk(K, dim=-1).indices
        for tr, sr, qid in zip(t_top, s_top, p.q_ids[i : i + 256], strict=True):
            t_idx, s_idx = tr.tolist(), sr.tolist()
            agree.append(len(set(t_idx) & set(s_idx)) / K)
            gold_t.append(_recall([p.d_ids[j] for j in t_idx], p.qrels[qid]))
            gold_s.append(_recall([p.d_ids[j] for j in s_idx], p.qrels[qid]))
    return agree, sum(gold_t) / len(gold_t), sum(gold_s) / len(gold_s)


def _histogram(agree: list[float]) -> None:
    print(f"\n  agree@10 over {len(agree)} pool queries -- a mean hides whether it is bimodal")
    counts = [0] * BINS
    for a in agree:
        counts[round(a * (BINS - 1))] += 1
    scale = max(counts) or 1
    for b in range(BINS - 1, -1, -1):
        bar = "#" * round(40 * counts[b] / scale)
        print(f"    {b / (BINS - 1):.1f} {bar:<40} {counts[b]:>4}")


def _banner(p, td, tq, sd, sq, name, split, ndcg_ok):
    agree, gold_t, gold_s = _aggregate(p, td, tq, sd, sq)
    mean_agree = sum(agree) / len(agree)
    pad = " " * 31
    print(f"\n== {name}/{split} -- the reported numbers, before you eyeball anything ==")
    print(
        f"  recall@10  docs    {recall_at_k(sd, td, K):.4f}    student vs teacher NEIGHBOURS over"
        f" {len(p.d_ids):,} docs --\n{pad}what the tracker quotes. Teacher 1.0 by construction."
    )
    print(f"  agree@10   queries {mean_agree:.4f}    the same, query side")
    print(
        f"  recall@10  gold    t {gold_t:.4f}  s {gold_s:.4f}  {gold_s - gold_t:+.4f}"
        f"    |top-10 & qrels| / min(10, |qrels|)"
    )
    if ndcg_ok:
        nt, ns = ndcg_pool(p, td, tq), ndcg_pool(p, sd, sq)
        print(f"  nDCG@10            t {nt:.4f}  s {ns:.4f}  {ns - nt:+.4f}")
    else:
        print("  nDCG@10            -    gold is one arbitrary member of an equally valid set")
    _histogram(agree)
    return mean_agree, agree


@torch.no_grad()
def _show(m, p, td, sd, sent, qid, mean_agree, ndcg_ok):
    gold = p.qrels.get(qid, {})
    tq = common._embed_texts(m.backend, m.teacher, m.tokenizer, m.cfg, [sent], m.device)
    sq = common._embed_texts(m.backend, m.student, m.tokenizer, m.cfg, [sent], m.device)
    t_idx, t_cos = _top(tq, td)
    s_idx, s_cos = _top(sq, sd)
    t_rank = {j: r + 1 for r, j in enumerate(t_idx)}

    print("\n" + "-" * 96)
    origin = f"{qid}   pool query, {len(gold)} gold" if qid else "free-form, no judgements"
    print(f"query   {origin}")
    for i, line in enumerate(sent.split("\n")):
        print(f"{'sent' if i == 0 else '':<7} {line}")

    print("\nSTUDENT top-10        t@ = the teacher's rank of the same doc, - = outside its top-10")
    print(f"{'rank':>4} {'cos':>6} {'t@':>4}  {'id':<8}{'document':<{SNIPPET}} mark")
    for r, (j, c) in enumerate(zip(s_idx, s_cos, strict=True), 1):
        mark = "=" if j in t_rank else "x"
        mark += "  G" if p.d_ids[j] in gold else ""
        head, *rest = _snip(p.d_texts[j])
        print(f"{r:>4} {c:>6.4f} {t_rank.get(j, '-'):>4}  {p.d_ids[j]:<8}{head} {mark}")
        for row in rest:
            print(f"{PAD}{row}")

    missed = [
        (r, j, c) for r, (j, c) in enumerate(zip(t_idx, t_cos, strict=True), 1) if j not in s_idx
    ]
    if missed:
        print(f"\nTEACHER top-10 the student missed{'':<32}cos is the TEACHER's")
        for r, j, c in missed:
            g = "  G" if p.d_ids[j] in gold else ""
            head, *rest = _snip(p.d_texts[j])
            print(f"{r:>4} {c:>6.4f} {'':>4}  {p.d_ids[j]:<8}{head}   {g}")
            for row in rest:
                print(f"{PAD}{row}")

    hits = len(set(t_idx) & set(s_idx))
    print(
        f"\nagree@10    {hits / K:.2f}   {hits} of the teacher's top-10"
        f"   (pool mean {mean_agree:.2f})"
    )
    if gold:
        s_ids = [p.d_ids[j] for j in s_idx]
        t_ids = [p.d_ids[j] for j in t_idx]
        got, t_got = sum(d in gold for d in s_ids), sum(d in gold for d in t_ids)
        print(
            f"recall@10   {_recall(s_ids, gold):.2f}   {got} of {len(gold)} gold"
            f"          (teacher {_recall(t_ids, gold):.2f}, {t_got} of {len(gold)})"
        )
        if ndcg_ok:
            ns, nt = ndcg_at_k(s_ids, gold), ndcg_at_k(t_ids, gold)
            print(f"nDCG@10     s {ns:.4f}   t {nt:.4f}   {ns - nt:+.4f}")
    else:
        print("recall@10   --     no judgements for a free-form query")


def _sent(p, src, text):
    # A pool query already carries its instruction; re-wrapping it would double the prefix.
    if text in p.q_ids:
        return p.q_texts[p.q_ids.index(text)], text
    return instruct(src.instruct, text), None


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__, json=False)
    p.add_argument("--source", default="gooaq", help="a source of the config's corpus mix")
    p.add_argument("--split", default="test", choices=("validation", "test"))
    p.add_argument("--query", action="append", help="one-shot, repeatable; omit for the REPL")
    p.add_argument(
        "--worst", type=int, default=0, help="show the N queries the student agrees least on"
    )
    args = p.parse_args(argv)

    cfg = common.config(args)
    sources = {s.name: s for s in mix(cfg.train.corpus)}
    if args.source not in sources:
        raise SystemExit(f"Unknown source {args.source!r}; available: {', '.join(sources)}")
    src = sources[args.source]

    datasets.disable_progress_bars()
    m = common.load(args)
    if args.student_ckpt and args.student_ckpt != RunPaths(cfg).student_bin():
        # max_seq_len is not recorded in a bare state_dict, so a foreign ckpt is the only signal
        # that this config's length and corpus may not be the ones it was distilled under.
        print(
            f"\nWARN  this ckpt is not {RunPaths(cfg).student_bin()}, the path this\n"
            f"      config writes. It is being run at max_seq_len {cfg.train.max_seq_len}."
        )
    if torch.cuda.is_available():
        print(f"\ndevice  {m.device}  {torch.cuda.get_device_name(m.device)}")

    # Uncached pools re-stream from HF for minutes, which otherwise reads as a hang.
    print(f"pool    {args.source}/{args.split}, uncached means minutes of HF streaming", flush=True)
    pl = pool(src, args.split, cfg.train.corpus)
    if not pl:
        raise SystemExit(f"{args.source}/{args.split} has no held-out queries")
    print(f"pool    {len(pl.q_ids):,} queries    {len(pl.d_ids):,} docs")
    if src.instruct:
        print(f"prefix  {src.instruct}\n        on the query only; documents are always sent bare")
    else:
        print("prefix  none -- symmetric source, the query is sent unprefixed")

    print("embed   teacher...", flush=True)
    td, tq = embed_pool(m, m.teacher, pl, teacher_cache("corpus", args.source, cfg))
    print("embed   student (recomputed every run)...", flush=True)
    sd, sq = embed_pool(m, m.student, pl)

    mean_agree, agree = _banner(pl, td, tq, sd, sq, args.source, args.split, src.scores_ndcg)

    worst = sorted(range(len(agree)), key=agree.__getitem__)[: args.worst]
    if worst:
        print("\nworst by agree@10  " + "  ".join(f"{pl.q_ids[i]} {agree[i]:.1f}" for i in worst))
    for i in worst:
        _show(m, pl, td, sd, pl.q_texts[i], pl.q_ids[i], mean_agree, src.scores_ndcg)
    for text in args.query or ():
        sent, qid = _sent(pl, src, text)
        _show(m, pl, td, sd, sent, qid, mean_agree, src.scores_ndcg)
    if worst or args.query:
        return

    print("\ntype a query, a pool id (q0..), :r for a random pool query, :q to quit")
    while True:
        try:
            text = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text in (":q", ""):
            return
        if text == ":r":
            text = random.choice(pl.q_ids)
        sent, qid = _sent(pl, src, text)
        _show(m, pl, td, sd, sent, qid, mean_agree, src.scores_ndcg)


if __name__ == "__main__":
    main()
