"""Swappable ops — the flexibility mechanism.

``ModelConfig.softmax`` / ``ModelConfig.activation`` are just names (+ optional
kwargs) resolved here into ``nn.Module`` instances. Attention / FFN modules
receive the built module and never hardcode ``F.softmax`` / ``F.gelu``, so
swapping in an HE-friendly approximation is a config change. Add a new op = add
one class and one dict entry.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- softmax ops: forward(scores, mask=None, dim=-1) -> attention weights ---
# `scores` are RAW (unmasked); `mask` is the additive attention mask
# (0 = visible, large-negative = masked). Ops apply the mask themselves.


class ExactSoftmax(nn.Module):
    """Standard numerically-stable softmax (adds the additive mask if given)."""

    def forward(self, scores, mask=None, dim=-1):
        if mask is not None:
            scores = scores + mask
        return F.softmax(scores, dim=dim)


class CGFSoftmax(nn.Module):
    """CGF softmax (HE-friendly): approximate log-sum-exp by its 2nd-order
    cumulant, so ``softmax_i ~= exp(x_i - mu - var/2 - log n_vis)`` with mu/var
    over visible positions (binary mask). Only masked multiply/add/square/exp —
    no ciphertext division or max-subtraction. Unnormalized by design;
    distillation adapts to it.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, scores, mask=None, dim=-1):
        m = (mask == 0).to(scores.dtype) if mask is not None else torch.ones_like(scores)
        n_vis = m.sum(dim=dim, keepdim=True)
        x_vis = scores * m  # zero out masked positions before taking statistics
        mu = x_vis.sum(dim=dim, keepdim=True) / n_vis
        var = (x_vis**2).sum(dim=dim, keepdim=True) / n_vis - mu**2
        logits = scores - mu - 0.5 * var - torch.log(n_vis)
        return torch.exp(logits) * m


# --- activation ops: forward(x) -> x ---


class ExactGELU(nn.Module):
    """Reference GELU (matches HF BERT); the exact-ops baseline."""

    def forward(self, x):
        return F.gelu(x)


class ChebyshevGELU(nn.Module):
    """GELU as a single Chebyshev polynomial on ``u = x / domain``, fit once over
    ``[-domain, domain]`` (a fixed function -> task-independent, no refit). The
    Chebyshev basis keeps intermediates in ``[-1, 1]`` -> FHE-stable (the monomial
    basis' ``x**k`` would explode); FHE eval is Paterson-Stockmeyer at mult-depth
    ~``ceil(log2 degree)``.

    ⚠️ Outside ``[-domain, domain]`` the polynomial explodes. Training clamps the
    input to ``[-1, 1]`` (scaffold, so init doesn't NaN on the teacher's ~±150
    outliers); inference does NOT clamp (min/max is costly in FHE), so the bare
    poly is safe only while activations stay in-domain — an empirical, per-dataset
    property to verify, NOT a guarantee. Widen ``domain`` (costs depth) for margin.
    """

    def __init__(self, degree: int = 52, domain: float = 32.0, **kwargs):
        super().__init__()
        self.degree = degree
        self.domain = float(domain)
        # Least-squares fit of GELU over [-domain, domain] in the Chebyshev basis
        # (numerically stable over wide domains, unlike a monomial fit).
        xs = np.linspace(-self.domain, self.domain, max(8001, int(self.domain * 400)))
        xt = torch.from_numpy(xs)
        y = (0.5 * xt * (1.0 + torch.erf(xt / 2.0**0.5))).numpy()
        coef = np.polynomial.chebyshev.Chebyshev.fit(
            xs, y, degree, domain=[-self.domain, self.domain]
        ).coef
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))

    def _eval_poly(self, u):
        """``sum_k c_k T_k(u)``, basis built by ``T_k = 2 T_{k//2} T_{k-k//2} - T_|.|``
        at mult-depth ``ceil(log2 degree)``. (FHE would use Paterson-Stockmeyer:
        same depth, ~2*sqrt(degree) mults vs the ~degree here.)"""
        c = self.coef
        n = c.numel() - 1
        T = [torch.ones_like(u)]  # T_0
        if n >= 1:
            T.append(u)  # T_1
        for k in range(2, n + 1):
            a, b = k // 2, k - k // 2  # a + b = k, |a - b| in {0, 1}
            T.append(2.0 * T[a] * T[b] - T[abs(a - b)])
        out = c[0] * T[0]
        for k in range(1, n + 1):
            out = out + c[k] * T[k]
        return out

    def forward(self, x):
        u = x / self.domain
        if self.training:
            u = u.clamp(-1.0, 1.0)  # scaffold; no clamp at inference
        if self.training and u.requires_grad:
            return checkpoint(self._eval_poly, u, use_reentrant=False)  # else OOM at high degree
        return self._eval_poly(u)


class QuadGELU(nn.Module):
    """MPCFormer's quadratic GELU replacement: ``0.125 x^2 + 0.25 x + 0.5``. A
    degree-2 activation (FHE mult-depth 1) evaluated everywhere — NOT a
    domain-limited approximation, so it never explodes. But the ``x^2`` term
    amplifies large activations, so (a) its output range is ~5x wider than exact
    GELU (a dynamic-range cost for FHE, not bounded by this op), and (b) it needs
    heavy hidden-state alignment to train — a plain single-stage KD (beta=1) gets
    stuck; set a large ``distill.beta`` (e.g. 10). See configs/bert/quad.yaml.

    ``domain`` (optional): if set, it's exposed so ``distill.range_penalty`` squashes
    pre-activations into ``[-domain, domain]`` during training, bounding the output to
    ~``0.125*domain^2`` (the FHE dynamic-range lever). None = unbounded output."""

    def __init__(self, domain: float | None = None, **kwargs):
        super().__init__()
        self.domain = domain

    def forward(self, x):
        return 0.125 * x * x + 0.25 * x + 0.5


SOFTMAX_OPS = {"exact": ExactSoftmax, "cgf": CGFSoftmax}
ACTIVATION_OPS = {"exact": ExactGELU, "cheb_gelu": ChebyshevGELU, "quad": QuadGELU}


def build_softmax(name: str, **kwargs) -> nn.Module:
    if name not in SOFTMAX_OPS:
        raise KeyError(f"Unknown softmax op {name!r}; available: {sorted(SOFTMAX_OPS)}")
    return SOFTMAX_OPS[name](**kwargs)


def build_activation(name: str, **kwargs) -> nn.Module:
    if name not in ACTIVATION_OPS:
        raise KeyError(f"Unknown activation op {name!r}; available: {sorted(ACTIVATION_OPS)}")
    return ACTIVATION_OPS[name](**kwargs)
