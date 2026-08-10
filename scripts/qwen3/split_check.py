"""Verify the train/validation/test split before paying for a build.

Membership is baked into the corpus cache, so a wrong split is only discoverable after hours of
streaming and a full re-distill. Three properties are checked, in increasing cost:

1. **Stability across processes.** `_split_of` must be a pure function of the key. Python salts
   `str.__hash__` per process, so a `hashlib` -> `hash()` regression would silently redraw the
   split on every run and every machine — and every earlier checkpoint would become untrustworthy
   without anything failing. Checked by recomputing in a subprocess.
2. **Uniformity.** The hash must spread keys evenly, or the realized fractions drift from
   VAL_PCT/TEST_PCT and the holdout is not the size it claims.
3. **Disjointness of the upstream-split sources.** Only the three datasets that ship their own
   train/test/validation are streamed — for a hashed source, disjointness follows from membership
   being a pure function of the key, and (2) already exercises that over 200k keys. The
   CodeSearchNet language histogram rides along: its stream is grouped by language with python
   first, so a split that reads 100% python means the shuffle is not doing its job.

(1) and (2) need no network and run in seconds; (3) streams, so prefer running this on the machine
that will build the corpus — it needs the download cache anyway.

    uv run python scripts/qwen3/split_check.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datasets  # noqa: E402
from franken.config import Config  # noqa: E402
from franken.data.embed_corpus import MIXES, TEST_PCT, VAL_PCT, _split_of  # noqa: E402

SAMPLE = 100  # texts per split per source; enough for an intersection to be meaningful
KEYS = 200_000  # synthetic keys for the uniformity check
SPLITS = ("train", "validation", "test")


def _stability() -> bool:
    keys = [f"key-{i}" for i in range(64)]
    here = [_split_of(k) for k in keys]
    code = (
        f"import sys; sys.path.insert(0, {sys.path[0]!r})\n"
        "from franken.data.embed_corpus import _split_of\n"
        "print(' '.join(_split_of(f'key-{i}') for i in range(64)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    ok = out == here
    print(f"stability across processes: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("  ⚠️  _split_of is not deterministic — is it using hash() instead of hashlib?")
    return ok


def _uniformity() -> bool:
    counts = {s: 0 for s in SPLITS}
    for i in range(KEYS):
        counts[_split_of(f"row-{i}")] += 1
    got = {s: 100 * counts[s] / KEYS for s in SPLITS}
    want = {"validation": VAL_PCT, "test": TEST_PCT - VAL_PCT, "train": 100 - TEST_PCT}
    ok = all(abs(got[s] - want[s]) < 0.5 for s in SPLITS)
    detail = "  ".join(f"{s} {got[s]:.2f}% (want {want[s]}%)" for s in SPLITS)
    print(f"uniformity: {'PASS' if ok else 'FAIL'}   {detail}", flush=True)
    return ok


def _disjoint(mix) -> bool:
    """Only the sources with UPSTREAM splits are worth streaming.

    For a `_hashed` source, disjointness is a theorem, not an observation: membership is a pure
    function of the key, so two splits cannot share a row however the stream is read — and
    `_uniformity` already exercises that function over 200k keys. Confirming it over the network
    for 13 sources costs an hour to verify arithmetic. What is *not* guaranteed is the three
    datasets whose splits come from upstream, where disjointness is the dataset's promise.
    """
    native = [(n, d, s, w) for n, d, s, w in mix if s.meta["native"]]
    print(
        f"\ndisjointness, upstream-split sources only ({len(native)} of {len(mix)}; "
        "hashed sources are disjoint by construction)"
    )
    print(f"{'source':<16} {'train':>7} {'val':>6} {'test':>6}  overlap", flush=True)
    ok = True
    for name, _domain, source, _weight in native:
        try:
            drawn = {s: set(source(s, SAMPLE)) for s in SPLITS}
        except Exception as e:
            print(f"{name:<16}  FAILED {type(e).__name__}: {e}"[:110], flush=True)
            ok = False
            continue
        pairs = [
            (a, b, len(drawn[a] & drawn[b]))
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ]
        bad = [f"{a}/{b}={n}" for a, b, n in pairs if n]
        ok = ok and not bad
        print(
            f"{name:<16} {len(drawn['train']):>7} {len(drawn['validation']):>6} "
            f"{len(drawn['test']):>6}  {', '.join(bad) if bad else 'none'}",
            flush=True,
        )
    return ok


def _csn_languages() -> None:
    print("\nCodeSearchNet language mix per split (python-only ⇒ shuffle is not working):")
    for split in SPLITS:
        ds = datasets.load_dataset(
            "code-search-net/code_search_net", "all", split=split, streaming=True
        ).shuffle(seed=0, buffer_size=10_000)
        seen: dict[str, int] = {}
        for i, row in enumerate(ds):
            seen[row["language"]] = seen.get(row["language"], 0) + 1
            if i >= 999:
                break
        print(f"  {split:<11} {dict(sorted(seen.items(), key=lambda x: -x[1]))}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/depth19_multi_domain.yaml")
    args = p.parse_args(argv)

    datasets.disable_progress_bars()
    mix = MIXES[Config.from_yaml(args.config).train.corpus]

    ok = _stability() and _uniformity()
    ok = _disjoint(mix) and ok
    _csn_languages()

    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED — do not build'}\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
