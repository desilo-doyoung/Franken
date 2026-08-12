"""Swappable ops: ``ModelConfig.softmax``/``activation`` are names resolved here into modules,
so attention/FFN never hardcode ``F.softmax``/``F.gelu``. Add an op = one class + one dict entry.
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
    """HE-friendly: log-sum-exp approximated by its 2nd-order cumulant, so
    ``softmax_i ~= exp(x_i - mu - var/2 - log n_vis)`` over visible positions. Only masked
    mul/add/square/exp — no ciphertext division or max-subtraction. Unnormalized by design;
    distillation adapts to it."""

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
    """GELU as one Chebyshev polynomial on ``u = x / domain``; the basis keeps intermediates in
    ``[-1, 1]`` (the monomial ``x**k`` would explode). FHE mult-depth ~``ceil(log2 degree)``.

    ⚠️ Outside ``[-domain, domain]`` the polynomial explodes. Training clamps ``u`` (scaffold, so
    init doesn't NaN on the teacher's ~±150 outliers); inference does NOT clamp — min/max is costly
    in FHE — so safety is an empirical, per-dataset property to VERIFY, not a guarantee. Widening
    ``domain`` buys margin at the cost of depth.
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
        # sum_k c_k T_k(u), basis by T_k = 2 T_{k//2} T_{k-k//2} - T_|.|. FHE would use
        # Paterson-Stockmeyer: same depth, ~2*sqrt(degree) mults vs the ~degree here.
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
    """MPCFormer's ``0.125 x^2 + 0.25 x + 0.5``. Degree 2 everywhere (FHE mult-depth 1), so unlike
    a Chebyshev fit it never explodes — but ``x^2`` amplifies, leaving the output range ~5x wider
    than exact GELU, and it needs heavy hidden-state alignment: beta=1 gets stuck, use beta~10.

    ``domain`` only exposes the op to ``distill.range_penalty``, bounding output to
    ~``0.125*domain^2``. None = unbounded."""

    def __init__(self, domain: float | None = None, **kwargs):
        super().__init__()
        self.domain = domain

    def forward(self, x):
        return 0.125 * x * x + 0.25 * x + 0.5


class ExactSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class QuadSiLU(nn.Module):
    """``a x^2 + b x + c`` fitted to **SiLU**: same FHE cost as ``quad`` (mult-depth 1) at 4.2x
    lower error, because ``quad`` fits *GELU* and isn't even good there (``quad(0)=0.5``,
    SiLU(0)=0). Coefficients are free in FHE, so fitting the right function is a pure win.

    Defaults: least-squares over the **measured** Qwen3-Embedding-0.6B gate_proj distribution on
    ``|x| <= 16`` (38M samples; bulk RMSE 0.222 vs 0.924 for ``quad``). Fitting the full observed
    range (max 319) is dragged nearly-linear by the tails and gets *worse* in the bulk. The fit
    domain is insensitive though — RMSE 0.221/0.222/0.292 on ``|x| <= 8/16/32`` — so no refit is
    needed when the deployed ``domain`` moves inside that band.

    ⚠️ Degree 2 everywhere, so it never explodes: ``domain`` is a purely FHE-side output-range
    requirement and can only COST accuracy, since ``range_penalty`` spends capacity forcing
    activations where they do not naturally go. Set it from the ciphertext scale budget and prefer
    the loosest the scheme tolerates. Unpenalized, real data drives output to ~9000 under either
    fit (input tails dominate, not coefficients); the penalty bounds it to ~``a*domain^2``.
    """

    def __init__(
        self,
        a: float = 0.0752,
        b: float = 0.4313,
        c: float = 0.1970,
        domain: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.domain = domain

    def forward(self, x):
        return (self.a * x + self.b) * x + self.c  # 1 square + 1 mult, Horner form


SOFTMAX_OPS = {"exact": ExactSoftmax, "cgf": CGFSoftmax}
ACTIVATION_OPS = {
    "exact": ExactGELU,
    "cheb_gelu": ChebyshevGELU,
    "quad": QuadGELU,
    "silu": ExactSiLU,
    "quad_silu": QuadSiLU,
}


def build_softmax(name: str, **kwargs) -> nn.Module:
    if name not in SOFTMAX_OPS:
        raise KeyError(f"Unknown softmax op {name!r}; available: {sorted(SOFTMAX_OPS)}")
    return SOFTMAX_OPS[name](**kwargs)


def build_activation(name: str, **kwargs) -> nn.Module:
    if name not in ACTIVATION_OPS:
        raise KeyError(f"Unknown activation op {name!r}; available: {sorted(ACTIVATION_OPS)}")
    return ACTIVATION_OPS[name](**kwargs)
