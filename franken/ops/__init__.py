"""Swappable ops — the flexibility mechanism.

``ModelConfig.softmax`` / ``ModelConfig.gelu`` are just names (+ optional kwargs)
resolved here into ``nn.Module`` instances. Attention / FFN modules receive the
built module and never hardcode ``F.softmax`` / ``F.gelu``, so swapping in an
HE-friendly approximation is a config change. Add a new op = add one class and
one dict entry.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


# --- softmax ops: forward(scores, dim=-1) -> attention weights ---

class ExactSoftmax(nn.Module):
    """Standard numerically-stable softmax."""

    def forward(self, scores, dim=-1):
        return F.softmax(scores, dim=dim)


class ApproxSoftmax(nn.Module):
    """HE-friendly softmax approximation (to be implemented in the tutorial)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, scores, dim=-1):
        raise NotImplementedError("ApproxSoftmax is a stub — implement the approximation.")


# --- gelu ops: forward(x) -> x ---

class ExactGELU(nn.Module):
    """Reference GELU (matches HF BERT)."""

    def forward(self, x):
        return F.gelu(x)


class PolyGELU(nn.Module):
    """Low-degree polynomial GELU approximation (to be implemented)."""

    def __init__(self, degree: int = 2, **kwargs):
        super().__init__()
        self.degree = degree
        self.kwargs = kwargs

    def forward(self, x):
        raise NotImplementedError(f"PolyGELU(degree={self.degree}) is a stub.")


SOFTMAX_OPS = {"exact": ExactSoftmax, "approx": ApproxSoftmax}
GELU_OPS = {"exact": ExactGELU, "poly": PolyGELU}


def build_softmax(name: str, **kwargs) -> nn.Module:
    if name not in SOFTMAX_OPS:
        raise KeyError(f"Unknown softmax op {name!r}; available: {sorted(SOFTMAX_OPS)}")
    return SOFTMAX_OPS[name](**kwargs)


def build_gelu(name: str, **kwargs) -> nn.Module:
    if name not in GELU_OPS:
        raise KeyError(f"Unknown gelu op {name!r}; available: {sorted(GELU_OPS)}")
    return GELU_OPS[name](**kwargs)
