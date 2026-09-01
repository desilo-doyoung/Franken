import inspect

import pytest

from franken.config import Config
from franken.models import BACKENDS, build_backend
from franken.models.base import ModelBackend

ACCESSORS = ["ffn_preact_modules", "activation_ops", "softmax_ops"]


@pytest.mark.parametrize("name", ACCESSORS)
def test_accessor_is_abstract_on_the_interface(name):
    # A backend that forgets one must fail at construction, not when a script walks the tree.
    assert name in ModelBackend.__abstractmethods__


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("name", ACCESSORS)
def test_every_backend_implements_the_accessor(backend, name):
    assert callable(getattr(build_backend(backend), name))


def test_bert_accessors_return_one_module_per_layer():
    cfg = Config.from_dict({"model": {"backend": "bert", "num_hidden_layers": 4}})
    backend = build_backend("bert")
    student = backend.build_student(cfg)
    for name in ACCESSORS:
        mods = getattr(backend, name)(student)
        assert len(mods) == 4, name


def test_softmax_ops_returns_the_injected_op():
    cfg = Config.from_dict({"model": {"backend": "bert", "num_hidden_layers": 2, "softmax": "cgf"}})
    backend = build_backend("bert")
    ops = backend.softmax_ops(backend.build_student(cfg))
    assert {type(o).__name__ for o in ops} == {"CGFSoftmax"}


def test_accessors_are_not_walking_the_module_tree_in_scripts():
    # The leak this replaced: act_range reached into `layer.self_attn.softmax` directly.
    from franken.scripts.qwen3 import act_range

    assert "self_attn.softmax" not in inspect.getsource(act_range)


def test_pooler_accessor_is_optional_and_defaults_to_empty():
    # Deliberately NOT abstract, unlike ACCESSORS: only bert has a pooler, and a decoder backend
    # must not be forced to implement an accessor for a module it does not have.
    assert "pooler_preact_modules" not in ModelBackend.__abstractmethods__
    cfg = Config.from_dict({"model": {"backend": "bert", "num_hidden_layers": 4}})
    bert = build_backend("bert")
    assert len(bert.pooler_preact_modules(bert.build_student(cfg))) == 1
    for name in sorted(BACKENDS):
        if name != "bert":
            assert build_backend(name).pooler_preact_modules(None) == []
