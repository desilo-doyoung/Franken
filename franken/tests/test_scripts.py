import importlib
import inspect

import pytest

from franken.cli import _CORPUS, _EVALUATOR, _TASK_EVALUATOR

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
    "franken.scripts.llama.lm_corpus",
    "franken.scripts.llama.lm_eval",
    "franken.scripts.llama.run_experiments",
]

# Reached through `cli._delegate`, which calls `main(argv)`. The tables hold fully-qualified
# module paths, so a scorer needs no `franken/scripts/<backend>/` shim to be reachable.
DELEGATED = sorted({*_CORPUS.values(), *_EVALUATOR.values(), *_TASK_EVALUATOR.values()})


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
