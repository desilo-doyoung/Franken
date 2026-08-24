"""Dataset loading, tokenization, and metrics.

`corpus/` is the shared machinery; each model declares its own mixes in `franken/data/<model>/`.
"""

from franken.data.mrpc import compute_metrics, load_mrpc

__all__ = ["load_mrpc", "compute_metrics", "corpus_sources"]

# Every registry that can name a corpus. `cache_path` has ONE flat directory, so a name reused
# across two registries would serve the other's text -- hence one lookup, not a per-model one.
_REGISTRIES = ("franken.data.qwen3.registry", "franken.data.llama.registry")


def corpus_sources(name: str):
    """Resolve `train.corpus` to its source list. Imported lazily: bert never loads a registry."""
    import importlib

    found = [
        mod.PRESETS[name]
        for mod in (importlib.import_module(m) for m in _REGISTRIES)
        if name in mod.PRESETS
    ]
    if len(found) != 1:
        known = sorted(n for m in _REGISTRIES for n in importlib.import_module(m).PRESETS)
        what = "declared twice" if found else "unknown"
        raise KeyError(f"Corpus {name!r} is {what}; registered: {known}")
    return found[0]
