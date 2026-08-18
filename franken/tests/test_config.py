import glob
import os

import pytest

from franken.config import Config, OptimConfig
from franken.models.bert.config import BertModelConfig
from franken.models.qwen3.config import Qwen3ModelConfig
from franken.paths import RunPaths

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True))


def test_configs_are_discovered():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.relpath(p, ROOT) for p in CONFIGS])
def test_every_shipped_config_loads(path):
    cfg = Config.from_yaml(path)
    assert cfg.model.num_hidden_layers > 0
    assert cfg.train.task


@pytest.mark.parametrize(
    ("backend", "expected"), [("bert", BertModelConfig), ("qwen3", Qwen3ModelConfig)]
)
def test_backend_selects_its_model_config_subclass(backend, expected):
    cfg = Config.from_dict({"model": {"backend": backend}})
    assert type(cfg.model) is expected


def test_unknown_key_in_a_block_is_rejected():
    with pytest.raises(ValueError, match="num_hidden_layerss"):
        Config.from_dict({"model": {"num_hidden_layerss": 6}})


def test_backend_specific_key_is_rejected_by_the_other_backend():
    Config.from_dict({"model": {"backend": "qwen3", "attn_impl": "manual"}})
    with pytest.raises(ValueError, match="attn_impl"):
        Config.from_dict({"model": {"backend": "bert", "attn_impl": "manual"}})


def test_nested_optim_blocks_parse_independently():
    cfg = Config.from_dict(
        {"train": {"teacher": {"epochs": 5}, "distill": {"epochs": 2, "lr": None}}}
    )
    assert (cfg.train.teacher.epochs, cfg.train.distill.epochs) == (5, 2)
    assert cfg.train.distill.lr is None
    assert cfg.train.teacher.lr == OptimConfig.lr


def test_unknown_key_in_an_optim_block_is_rejected():
    with pytest.raises(ValueError, match="epoch"):
        Config.from_dict({"train": {"distill": {"epoch": 2}}})


def test_run_name_namespaces_the_output_tree():
    cfg = Config.from_dict({"model": {"backend": "qwen3"}, "train": {"run_name": "d19_quad"}})
    assert RunPaths(cfg).base == os.path.join("outputs", "d19_quad")


def test_output_namespace_defaults_to_the_backend():
    cfg = Config.from_dict({"model": {"backend": "qwen3"}})
    paths = RunPaths(cfg)
    assert paths.base == os.path.join("outputs", "qwen3")
    assert paths.student_bin() == os.path.join("outputs", "qwen3", "student", "pytorch_model.bin")
