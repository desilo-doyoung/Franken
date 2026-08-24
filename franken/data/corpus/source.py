"""What a dataset IS -- where its rows come from, how they split, what text they yield. The mixes
themselves are declared per model, in `franken/data/<model>/registry.py`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from franken.data.corpus.spec import Record


@dataclass(frozen=True)
class Qrels:
    """Real judgements, for a source whose rows hold no pair. Layouts are not inferrable, hence the
    explicit repo/split/cols. Judged documents are held out of training by `build._judged`."""

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
    # Column the split is hashed from; None means the dataset ships its own splits. `evalset` must
    # hash the identical string or it scores trained rows.
    key: str | None = None
    hf_split: str = "train"  # the single upstream split to draw from; unused when key is None
    split_map: Mapping[str, str] = field(default_factory=dict)  # ours -> upstream, when key is None
    # Baked into the cached corpus text, so changing it needs a `build._CACHE_VERSION` bump.
    instruct: str | None = None
    qrels: Qrels | None = None
    # False where the gold is one arbitrary member of an equally valid set, so nDCG would score a
    # tie-break. Governs REPORTING only -- the source still yields pairs for recall@10.
    scores_ndcg: bool = True


def normalized(sources: list[Source]) -> list[Source]:
    # `_mix` draws `round(n * weight)`, so these must sum to 1; declaring them relative is what
    # makes dropping a source a one-line delete.
    total = sum(s.weight for s in sources)
    return [replace(s, weight=s.weight / total) for s in sources]
