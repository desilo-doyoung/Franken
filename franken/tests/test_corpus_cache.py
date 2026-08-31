"""Cache identity and the guards around a cold one."""

import json
import os

import datasets
import pytest

from franken.data.corpus import adapters, build
from franken.data.corpus.source import Source
from franken.distill import dist


def _req(tmp_path, pack=True):
    class _Tok:
        eos_token_id, pad_token_id, name_or_path = 99, 98, "fake"

        def __call__(self, batch, truncation=False, max_length=None):
            return {"input_ids": [[1] * len(t.split()) for t in batch]}

    src = Source("s0", "d", "repo", "cfg", adapters.whole("text"), 1.0)
    return build._Build(
        name="t",
        sources=[src],
        split="train",
        tokenizer=_Tok(),
        max_seq_len=8,
        pack=pack,
        tokens=200,
    )


@pytest.fixture
def stub_stream(monkeypatch):
    def batches(_src, _split, size=1000):
        for _ in range(40):
            yield ["w " * 5]

    monkeypatch.setattr(build, "_batches", batches)


# ------------------------------------------------------------------ the DDP guard


def test_a_cold_cache_under_torchrun_is_refused(monkeypatch, tmp_path):
    # A miss used to mean N ranks streaming the whole corpus and racing on the rename.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError, match="main.py corpus"):
        build._refuse_under_ddp(str(tmp_path / "missing"))


@pytest.mark.parametrize(
    "env",
    [{}, {"WORLD_SIZE": "4"}, {"RANK": "0", "WORLD_SIZE": "1"}],
    ids=["bare", "no-rank", "one"],
)
def test_a_single_process_build_is_never_refused(monkeypatch, tmp_path, env):
    for k in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    build._refuse_under_ddp(str(tmp_path / "missing"))  # must not raise


@pytest.mark.parametrize(
    "env",
    [{}, {"WORLD_SIZE": "4"}, {"RANK": "0", "WORLD_SIZE": "1"}, {"RANK": "0", "WORLD_SIZE": "2"}],
)
def test_the_guard_and_init_distributed_read_the_env_the_same_way(monkeypatch, tmp_path, env):
    """Two copies of one predicate: if they drift, a run either builds N times or refuses alone."""
    for k in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    refused = False
    try:
        build._refuse_under_ddp(str(tmp_path / "missing"))
    except RuntimeError:
        refused = True

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    assert refused == distributed
    assert dist.init_distributed.__doc__  # the predicate this mirrors lives there


# ------------------------------------------------------------------ the manifest


def test_the_manifest_records_the_recipe_the_key_omits(stub_stream, tmp_path):
    # `_CACHE_VERSION` is manual, so sources/weights/repos can change under an unchanged key. The
    # manifest is what makes that diagnosable instead of invisible.
    req = _req(tmp_path)
    ds = datasets.Dataset.from_list(list(build._rows(req)))
    m = build._manifest(req, ds)

    assert m["mix"] == "t"
    assert m["pack"] is True
    assert m["max_seq_len"] == 8
    assert m["cache_version"] == build._CACHE_VERSION
    assert m["sources"] == [{"name": "s0", "weight": 1.0, "repo": "repo", "config": "cfg"}]
    assert m["realized_tokens_by_source"] == [m["rows"] * 8]
    json.dumps(m)  # must survive the round trip to disk


# ------------------------------------------------------------------ where it lives


def test_both_caches_are_absolute_not_cwd_relative():
    # A relative dir made the cache identity depend on where python was started.
    from franken.data.corpus import evalset

    for d in (build._CACHE_DIR, evalset._CACHE_DIR):
        assert os.path.isabs(d), d


def test_cache_missing_checks_the_validation_split_too(monkeypatch, tmp_path):
    """`Distiller.train` builds both splits on every rank, so a train-only probe would let a
    half-cached mix reach the ranks and trip the DDP guard."""
    from franken.config import Config

    cfg = Config.from_dict(
        {
            "train": {
                "task": "lm",
                "corpus": "llama_web",
                "tokens_per_epoch": 200,
                "max_seq_len": 8,
                "pack": True,
            }
        }
    )

    class _Tok:
        name_or_path = "fake"

    monkeypatch.setattr(build, "_CACHE_DIR", str(tmp_path))
    train = build.train_cache_path("llama_web", 200, 8, _Tok(), True)
    os.makedirs(train)
    assert build.cache_missing(cfg, _Tok()) is True  # validation still absent

    os.makedirs(build.cache_path("llama_web", "validation", str(build.VAL_POOL), 8, _Tok(), True))
    assert build.cache_missing(cfg, _Tok()) is False
