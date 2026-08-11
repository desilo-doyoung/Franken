"""One function per dataset shape: ``row -> Record | None`` (None drops the row).

Replaced two parallel sets of extractors, the second existing only to undo the first. Texts stay
natural units (a paragraph, a query) — an embedding model is deployed on whole passages.
"""

from __future__ import annotations

from collections.abc import Callable

from franken.data.embed_corpus.spec import Record

_MIN_DOC = 32  # below this a "document" is a fragment
_MIN_PARAGRAPH = 64  # Wikipedia's short lines are section stubs and list items


def _clean(row, col: str) -> str:
    return (row[col] or "").strip()


def pair(a: str, b: str) -> Callable[[dict], Record | None]:
    """``a`` is the query side, ``b`` the document side. A row missing one side still contributes
    the other as corpus text; it just yields no eval pair."""

    def adapt(row) -> Record | None:
        query, doc = _clean(row, a), _clean(row, b)
        if not query and not doc:
            return None
        return Record(query=query, positives=(doc,) if doc else ())

    return adapt


def triplet(row) -> Record | None:
    # No length floor: NLI-style text is short by nature, and short is the regime CGF normalizes
    # differently — a mode worth covering rather than filtering.
    query, pos, neg = (_clean(row, c) for c in ("anchor", "positive", "negative"))
    if not (query or pos or neg):
        return None
    return Record(query=query, positives=(pos,) if pos else (), negatives=(neg,) if neg else ())


def marco(row) -> Record | None:
    """A query plus 10 passages, the relevant ones flagged — one row is a whole retrieval task, so
    no split can separate a query from its positive. Every flagged passage is a positive."""
    query = row["query"].strip()
    texts = [p.strip() for p in row["passages"]["passage_text"]]
    flags = row["passages"]["is_selected"]
    positives = tuple(t for t, f in zip(texts, flags, strict=True) if f and t)
    negatives = tuple(t for t, f in zip(texts, flags, strict=True) if not f and t)
    if not (query or positives or negatives):
        return None
    return Record(query=query, positives=positives, negatives=negatives)


def titled(row) -> Record | None:
    """A corpus dump: title + body, no query, so the source MUST declare `Qrels`. Space, not ". ",
    matching the f"{title} {text}" shape `eval.py` builds every external document with."""
    title, text = _clean(row, "title"), _clean(row, "text")
    if len(text) < _MIN_DOC:
        return None
    return Record(docs=(f"{title} {text}" if title else text,))


def paragraphs(row) -> Record | None:
    """Wikipedia rows are whole articles (median 1,040 tokens zh, 1,764 ru), so taken whole ~93% of
    every row is discarded and the slice is nothing but lead paragraphs.

    Two paragraphs of one article are related by construction. Only ONE becomes gold: promoting
    every sibling would hand a single query ~35 golds and make nDCG@10 trivially satisfiable.
    """
    paras = [p.strip() for p in row["text"].split("\n") if len(p.strip()) >= _MIN_PARAGRAPH]
    if not paras:
        return None
    if len(paras) == 1:
        return Record(docs=(paras[0],))
    return Record(query=paras[0], positives=(paras[1],), docs=tuple(paras[2:]))


def wikitext(row) -> Record | None:
    # Blank lines and " = = Heading = = " rows ship as records of their own. Smoke preset only.
    text = row["text"].strip()
    if len(text) < _MIN_DOC or text.startswith("="):
        return None
    return Record(docs=(text,))
