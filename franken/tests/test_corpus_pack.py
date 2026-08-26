import pytest

from franken.data.corpus import adapters, build
from franken.data.corpus.source import Source

CAP = 8


class _Tok:
    """One token per whitespace-separated word; EOS is a value no document produces."""

    eos_token_id = 99
    pad_token_id = 98
    name_or_path = "fake"

    def __call__(self, batch, truncation=False, max_length=None):
        cap = max_length if truncation and max_length else 10**9
        return {"input_ids": [[int(t.split()[0])] * min(len(t.split()), cap) for t in batch]}


def _req(per_source_docs, pack=True, tokens=10_000):
    """Source i emits documents whose every token is `i`, so a block's provenance is checkable from
    its VALUES rather than only from the column that claims it."""
    n = len(per_source_docs)
    sources = [Source(f"s{i}", "d", "r", None, adapters.whole("text"), 1 / n) for i in range(n)]
    lengths = dict(enumerate(per_source_docs))

    def batches(src, split, size=1000):
        i = int(src.name[1:])
        for n in lengths[i]:
            yield [f"{i} " * n]

    return (
        sources,
        batches,
        build._Build(
            name="t",
            sources=sources,
            split="train",
            tokenizer=_Tok(),
            max_seq_len=CAP,
            pack=pack,
            tokens=tokens,
        ),
    )


def _rows(monkeypatch, per_source_docs, **kw):
    sources, batches, req = _req(per_source_docs, **kw)
    monkeypatch.setattr(build, "_batches", batches)
    return list(build._rows(req))


def test_every_block_is_exactly_the_cap(monkeypatch):
    rows = _rows(monkeypatch, [[20, 13], [31]])
    assert {len(r["input_ids"]) for r in rows} == {CAP}


def test_the_mask_marks_exactly_the_padding(monkeypatch):
    # `torch_columns()` requires the column; omitting it broke the artifact while the build still
    # printed CORPUS OK. Under best fit a bin can end short, so ones-then-zeros, never ragged.
    for pack in (True, False):
        for r in _rows(monkeypatch, [[20, 13], [31]], pack=pack):
            mask = r["attention_mask"]
            assert len(mask) == len(r["input_ids"])
            assert mask == sorted(mask, reverse=True)
            assert set(r["input_ids"][mask.count(1) :]) <= {_Tok.pad_token_id}


def test_every_block_comes_from_one_source(monkeypatch):
    # The provenance invariant: `source` is a uint8 index `realized_mix` reads, so a block spanning
    # sources could only record a majority. Drawing per source is what keeps it honest.
    rows = _rows(monkeypatch, [[20, 13], [31], [17]])
    for r in rows:
        assert set(r["input_ids"]) - {_Tok.eos_token_id, _Tok.pad_token_id} == {r["source"]}


def test_short_documents_share_a_block(monkeypatch):
    # Two 3-token docs (+EOS each) exactly fill one cap-8 bin: packing, not one block apiece.
    rows = _rows(monkeypatch, [[3, 3]])
    assert len(rows) == 1
    assert rows[0]["attention_mask"] == [1] * CAP


def test_a_document_that_fits_is_never_split(monkeypatch):
    # The whole point of best fit over concatenate-and-chop: a boundary must not fall inside a
    # document that would have fit a bin. Every row here starts a document, so none is a
    # continuation.
    rows = _rows(monkeypatch, [[5, 5, 5, 5]])
    for r in rows:
        real = r["input_ids"][: r["attention_mask"].count(1)]
        # each document contributes 5 payload tokens then its EOS, so runs are 5 long
        runs = "".join("x" if t == _Tok.eos_token_id else "." for t in real).split("x")
        assert all(len(run) == 5 for run in runs if run)


def test_a_source_totalling_under_one_block_is_padded_not_dropped(monkeypatch):
    # Best fit pads rather than discards, so a tiny source still reaches the artifact; dropping it
    # would silently bias the mix toward whatever packs tightly.
    rows = _rows(monkeypatch, [[3], [CAP * 3]])
    assert {r["source"] for r in rows} == {0, 1}
    tiny = next(r for r in rows if r["source"] == 0)
    assert tiny["attention_mask"] == [1, 1, 1, 1] + [0] * (CAP - 4)


def test_unpacked_yields_documents_truncated_to_the_cap(monkeypatch):
    rows = _rows(monkeypatch, [[20, 3]], pack=False)
    assert [len(r["input_ids"]) for r in rows] == [CAP, 3]


def test_the_draw_stops_at_the_quota(monkeypatch):
    # Unpacked, where emitted tokens ARE the drawn tokens. The quota is checked mid-batch, so a
    # source overshoots by at most one document rather than by a whole batch.
    doc = 4
    sources, batches, req = _req([[doc] * 1000], pack=False, tokens=100)
    monkeypatch.setattr(build, "_batches", batches)
    drawn = sum(len(r["input_ids"]) for r in build._rows(req))
    assert 100 <= drawn <= 100 + doc


@pytest.mark.parametrize("pack", [True, False])
def test_the_generator_is_deterministic(monkeypatch, pack):
    a = _rows(monkeypatch, [[20, 13], [31]], pack=pack)
    b = _rows(monkeypatch, [[20, 13], [31]], pack=pack)
    assert a == b


def test_one_huge_document_cannot_blow_the_quota(monkeypatch):
    # A single 36k-token Hindi article overshot finewiki_hi's 63k quota by 57% when the quota
    # counted tokens READ. Counting tokens EMITTED bounds the overshoot at one block.
    sources, batches, req = _req([[500, 500]], tokens=100)
    monkeypatch.setattr(build, "_batches", batches)
    emitted = sum(len(r["input_ids"]) for r in build._rows(req))
    assert emitted <= 100 + CAP
