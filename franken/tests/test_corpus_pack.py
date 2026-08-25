import pytest

from franken.data.corpus import adapters, build
from franken.data.corpus.source import Source

CAP = 8


class _Tok:
    """One token per whitespace-separated word; EOS is a value no document produces."""

    eos_token_id = 99
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


def test_every_row_carries_an_all_ones_mask(monkeypatch):
    # `torch_columns()` requires the column and packing pads nothing; omitting it broke the
    # artifact while the build still printed CORPUS OK.
    for pack in (True, False):
        for r in _rows(monkeypatch, [[20, 13], [31]], pack=pack):
            assert r["attention_mask"] == [1] * len(r["input_ids"])


def test_every_block_comes_from_one_source(monkeypatch):
    # The provenance invariant: `source` is a uint8 index `realized_mix` reads, so a block spanning
    # sources could only record a majority. Drawing per source is what keeps it honest.
    rows = _rows(monkeypatch, [[20, 13], [31], [17]])
    for r in rows:
        assert set(r["input_ids"]) - {_Tok.eos_token_id} == {r["source"]}


def test_short_documents_concatenate_rather_than_pad(monkeypatch):
    # Two 7-token docs at cap 8 make one block (+EOS each), not two padded ones -- the point of
    # packing.
    rows = _rows(monkeypatch, [[CAP - 1, CAP - 1]])
    assert len(rows) == 2


def test_a_source_totalling_under_one_block_contributes_nothing(monkeypatch):
    # The trailing partial is dropped, never padded, so a tiny source vanishes rather than
    # smuggling pad tokens into the mix.
    rows = _rows(monkeypatch, [[3], [CAP * 3]])
    assert {r["source"] for r in rows} == {1}


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
