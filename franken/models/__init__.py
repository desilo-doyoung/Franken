"""Model backend registry.

Each model lives in its own subpackage (``franken.models.bert``,
``franken.models.qwen3``, ...) holding its nn.Module implementation and a
``ModelBackend`` adapter. This module maps a name -> backend class, mirroring
``franken.ops.build_softmax`` / ``build_activation``. Add a model = one subpackage
+ one dict entry.
"""

from __future__ import annotations

from franken.models.base import ModelBackend
from franken.models.bert.backend import BertBackend
from franken.models.qwen3.backend import Qwen3Backend

BACKENDS: dict[str, type[ModelBackend]] = {
    "bert": BertBackend,
    "qwen3": Qwen3Backend,
}


def build_backend(name: str) -> ModelBackend:
    if name not in BACKENDS:
        raise KeyError(f"Unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
