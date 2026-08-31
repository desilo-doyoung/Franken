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


def test_a_packed_row_carries_no_mask_and_an_unpacked_one_does(monkeypatch):
    # `torch_columns()` follows the artifact; omitting the column while it still asked for one
    # broke the artifact with the build still printing CORPUS OK.
    for r in _rows(monkeypatch, [[20, 13], [31]], pack=True):
        assert set(r) == {"input_ids", "source"}
    for r in _rows(monkeypatch, [[20, 13], [31]], pack=False):
        assert r["attention_mask"] == [1] * len(r["input_ids"])


def test_every_block_comes_from_one_source(monkeypatch):
    # The provenance invariant: `source` is a uint8 index `realized_mix` reads, so a block spanning
    # sources could only record a majority. Drawing per source is what keeps it honest.
    rows = _rows(monkeypatch, [[20, 13], [31], [17]])
    for r in rows:
        assert set(r["input_ids"]) - {_Tok.eos_token_id} == {r["source"]}


def test_short_documents_share_a_block(monkeypatch):
    # Two 3-token docs (+EOS each) exactly fill one cap-8 block: packing, not one block apiece.
    rows = _rows(monkeypatch, [[3, 3]])
    assert len(rows) == 1
    assert len(rows[0]["input_ids"]) == CAP


def test_a_document_straddling_a_boundary_continues_on_the_next_row(monkeypatch):
    # Deliberately the reverse of best fit, which never split a document that would have fit. Two
    # 5-token docs (+EOS = 6 each) do not tile a cap-8 block, so the second one is cut.
    rows = _rows(monkeypatch, [[5, 5, 5, 5]])
    assert [len(r["input_ids"]) for r in rows] == [CAP] * 3
    # No token is reordered: concatenation is in stream order, so flattening rebuilds the stream.
    flat = [t for r in rows for t in r["input_ids"]]
    assert flat == ([0] * 5 + [_Tok.eos_token_id]) * 4


def test_the_tail_of_a_split_document_reads_as_a_fresh_sequence(monkeypatch):
    # The cross-module claim that makes chopping safe: `doc_positions` restarts at index 0, so a
    # continuation is presented as its own document rather than glued to its neighbour.
    import torch

    from franken.distill.packing import doc_ids, doc_positions

    rows = _rows(monkeypatch, [[5, 5, 5, 5]])
    block = torch.tensor([rows[1]["input_ids"]])  # opens mid-document
    pos = doc_positions(block, _Tok.eos_token_id)
    assert pos[0, 0].item() == 0
    assert doc_ids(pos)[0, 0].item() == 0


def test_a_source_that_cannot_fill_one_block_is_reported_not_hidden(monkeypatch, capsys):
    # Chopping drops the trailing partial, so a sub-block source reaches NO row. Silently that
    # biases the mix toward whatever fills a block, hence the loud line.
    rows = _rows(monkeypatch, [[3], [CAP * 3]])
    assert {r["source"] for r in rows} == {1}
    assert "PRODUCED NO BLOCKS" in capsys.readouterr().out


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


def test_the_quota_is_exact_not_merely_bounded(monkeypatch):
    # Cutting the last document AT the remaining room, rather than appending it whole and stopping
    # after, makes the draw exact: a 36k-token article once blew a 63k quota by 57%.
    sources, batches, req = _req([[500, 500]], tokens=100)
    monkeypatch.setattr(build, "_batches", batches)
    emitted = sum(len(r["input_ids"]) for r in build._rows(req))
    assert emitted == (100 // CAP) * CAP  # whole blocks only; the partial tail is dropped


def test_chopping_reorders_nothing(monkeypatch):
    # The blocks are a partition of the stream, so no token is duplicated or moved.
    docs = [7, 2, 11, 4]
    rows = _rows(monkeypatch, [docs])
    stream = [t for n in docs for t in [0] * n + [_Tok.eos_token_id]]
    flat = [t for r in rows for t in r["input_ids"]]
    assert flat == stream[: len(flat)]
    assert len(flat) % CAP == 0


def test_exhausted_is_judged_on_the_draw_not_on_the_dropped_tail(monkeypatch, capsys):
    # Emitted tokens are always a whole number of blocks, so testing THOSE against the quota flags
    # every source. EXHAUSTED has to keep meaning "ran dry": a mix once asked for 20% queries and
    # delivered 5.2%, unnoticed, through a whole tracker.
    _rows(monkeypatch, [[CAP * 20]], tokens=40)  # plenty of text for a 40-token quota
    assert "EXHAUSTED" not in capsys.readouterr().out

    _rows(monkeypatch, [[CAP, CAP]], tokens=10_000)  # nowhere near the quota
    assert "EXHAUSTED" in capsys.readouterr().out


def _blocks(doc, n_docs, cap=CAP):
    """`n_docs` copies of `doc`, concatenated and chopped into cap-wide rows like `_rows`."""
    import datasets

    flat = doc * n_docs
    rows = [flat[i : i + cap] for i in range(0, len(flat) - cap + 1, cap)]
    return datasets.Dataset.from_list([{"input_ids": r} for r in rows])


BOS, EOS = 90, _Tok.eos_token_id


def test_split_doc_share_counts_documents_not_rows():
    # The ROW share reads ~98% at any realistic block size and means nothing -- a row holds several
    # documents and almost never ends on one. The DOCUMENT share is what compares against the
    # 16.6% that best-fit packing avoided.
    from franken.data.corpus.measure import split_doc_share

    # 6-token docs over cap-8 rows: rows 1 and 2 open mid-document, 4 documents in total.
    ds = _blocks([BOS, 1, 1, 1, 1, EOS], 4)
    assert [r["input_ids"][0] for r in ds] == [BOS, 1, 1]
    assert split_doc_share(ds, BOS, EOS) == 2 / 4


def test_split_doc_share_is_zero_when_documents_tile_the_block():
    # Calibrates the test above: 4-token docs tile a cap-8 row exactly, so nothing is cut.
    from franken.data.corpus.measure import split_doc_share

    ds = _blocks([BOS, 1, 1, EOS], 4)
    assert split_doc_share(ds, BOS, EOS) == 0.0
