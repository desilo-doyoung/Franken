"""One function per dataset shape: ``row -> Record | None``. Texts stay natural units, since an
embedding model is deployed on whole passages."""

from __future__ import annotations

from collections.abc import Callable

from franken.data.corpus.spec import Record

_MIN_DOC = 32  # below this a "document" is a fragment
_MIN_PARAGRAPH = 64  # Wikipedia's short lines are section stubs and list items


def _clean(row, col: str) -> str:
    return (row[col] or "").strip()


def pair(a: str, b: str) -> Callable[[dict], Record | None]:
    """``a`` is the query side. A row missing one side still yields the other as corpus text."""

    def adapt(row) -> Record | None:
        query, doc = _clean(row, a), _clean(row, b)
        if not query and not doc:
            return None
        return Record(query=query, positives=(doc,) if doc else ())

    adapt.shape = f"{a} -> {b}"
    return adapt


def triplet(row) -> Record | None:
    # No length floor: short text is the regime CGF normalizes differently, worth covering.
    query, pos, neg = (_clean(row, c) for c in ("anchor", "positive", "negative"))
    if not (query or pos or neg):
        return None
    return Record(query=query, positives=(pos,) if pos else (), negatives=(neg,) if neg else ())


triplet.shape = "anchor -> positive (+hard negative)"


def marco(row) -> Record | None:
    """One row is a whole retrieval task, so no split can separate a query from its positive."""
    query = row["query"].strip()
    texts = [p.strip() for p in row["passages"]["passage_text"]]
    flags = row["passages"]["is_selected"]
    positives = tuple(t for t, f in zip(texts, flags, strict=True) if f and t)
    negatives = tuple(t for t, f in zip(texts, flags, strict=True) if not f and t)
    if not (query or positives or negatives):
        return None
    return Record(query=query, positives=positives, negatives=negatives)


marco.shape = "web query -> selected passage (+9 near-misses)"


def titled(row) -> Record | None:
    """Title + body, no query, so the source MUST declare `Qrels`. Space join matches external."""
    title, text = _clean(row, "title"), _clean(row, "text")
    if len(text) < _MIN_DOC:
        return None
    return Record(docs=(f"{title} {text}" if title else text,))


titled.shape = "no pair in the row -- needs Qrels"


def paragraphs(row) -> Record | None:
    """Wikipedia rows are whole articles, so taken whole the slice is nothing but lead paragraphs.
    Two paragraphs are related by construction; only ONE becomes gold, or nDCG is trivial."""
    paras = [p.strip() for p in row["text"].split("\n") if len(p.strip()) >= _MIN_PARAGRAPH]
    if not paras:
        return None
    if len(paras) == 1:
        return Record(docs=(paras[0],))
    return Record(query=paras[0], positives=(paras[1],), docs=tuple(paras[2:]))


paragraphs.shape = "paragraph 1 -> paragraph 2 of the same article"


def wikitext(row) -> Record | None:
    # Headings and blank lines ship as records of their own. Smoke preset only.
    text = row["text"].strip()
    if len(text) < _MIN_DOC or text.startswith("="):
        return None
    return Record(docs=(text,))


def whole(col: str) -> Callable[[dict], Record | None]:
    """The row IS the training text -- no pair to mine, so a mix using this scores nothing."""

    def adapt(row) -> Record | None:
        text = _clean(row, col)
        return Record(docs=(text,)) if len(text) >= _MIN_DOC else None

    adapt.shape = f"no pair in the row -- {col} taken whole"
    return adapt


def marco_side(kind: str) -> Callable[[dict], Record | None]:
    """Legacy `mixed` preset: one side only, since `marco` yields both and would change its
    measured proportions."""

    def adapt(row) -> Record | None:
        if kind == "query":
            query = row["query"].strip()
            return Record(query=query) if query else None
        texts = tuple(p.strip() for p in row["passages"]["passage_text"] if p.strip())
        return Record(docs=texts) if texts else None

    adapt.shape = f"no pair in the row -- {kind} side only"
    return adapt


wikitext.shape = "no pair in the row -- smoke only"
