import importlib
import inspect

import pytest

from franken.cli import _EVALUATOR

SCRIPTS = [
    # backend-agnostic, so they live at the scripts root
    "franken.scripts.stage_distill",
    "franken.scripts.parity_gate",
    "franken.scripts.precision_gate",
    "franken.scripts.bert.act_range",
    "franken.scripts.bert.evaluate",
    "franken.scripts.bert.seed_sweep",
    "franken.scripts.qwen3.act_range",
    "franken.scripts.qwen3.corpus",
    "franken.scripts.qwen3.eval",
    "franken.scripts.qwen3.run_experiments",
    "franken.scripts.qwen3.search",
]

# Reached through `cli._delegate`, which calls `main(argv)`.
DELEGATED = ["franken.scripts.qwen3.corpus"] + [
    f"franken.scripts.{b}.{s}" for b, s in _EVALUATOR.items()
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_imports_and_exposes_main(name):
    assert callable(importlib.import_module(name).main)


@pytest.mark.parametrize("name", DELEGATED)
def test_delegated_script_takes_argv(name):
    main = importlib.import_module(name).main
    assert inspect.signature(main).parameters, f"{name}.main() must accept argv"


def test_every_backend_with_an_evaluator_resolves():
    from franken.models import BACKENDS

    assert set(_EVALUATOR) <= set(BACKENDS)
