"""Swappable ops, so attention/FFN never hardcode ``F.softmax``/``F.gelu``."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# softmax ops: forward(raw_scores, additive_mask=None, dim=-1); the op applies the mask itself.


class ExactSoftmax(nn.Module):
    def forward(self, scores, mask=None, dim=-1):
        if mask is not None:
            scores = scores + mask
        return F.softmax(scores, dim=dim)


class CGFSoftmax(nn.Module):
    """log-sum-exp by its 2nd-order cumulant: only mul/add/square/exp, no ciphertext division or
    max-subtraction. Unnormalized by design; distillation adapts to it."""

    def forward(self, scores, mask=None, dim=-1):
        m = (mask == 0).to(scores.dtype) if mask is not None else torch.ones_like(scores)
        n_vis = m.sum(dim=dim, keepdim=True)
        x_vis = scores * m  # zero out masked positions before taking statistics
        mu = x_vis.sum(dim=dim, keepdim=True) / n_vis
        var = (x_vis**2).sum(dim=dim, keepdim=True) / n_vis - mu**2
        logits = scores - mu - 0.5 * var - torch.log(n_vis)
        return torch.exp(logits) * m


class ExactGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class ChebyshevGELU(nn.Module):
    """GELU as one Chebyshev polynomial on ``u = x / domain``; the basis keeps intermediates in
    ``[-1, 1]``. Explodes outside the domain, and inference does not clamp -- verify the range."""

    def __init__(self, degree: int = 52, domain: float = 32.0, **kwargs):
        super().__init__()
        self.degree = degree
        self.domain = float(domain)
        # Chebyshev basis, not monomial: numerically stable over wide domains.
        xs = np.linspace(-self.domain, self.domain, max(8001, int(self.domain * 400)))
        xt = torch.from_numpy(xs)
        y = (0.5 * xt * (1.0 + torch.erf(xt / 2.0**0.5))).numpy()
        coef = np.polynomial.chebyshev.Chebyshev.fit(
            xs, y, degree, domain=[-self.domain, self.domain]
        ).coef
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))

    def _eval_poly(self, u):
        # sum_k c_k T_k(u). FHE would use Paterson-Stockmeyer: same depth, fewer mults.
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
    """MPCFormer's ``0.125 x^2 + 0.25 x + 0.5``. Never explodes, but needs heavy hidden-state
    alignment (beta~10). ``domain`` only exposes the op to ``distill.range_penalty``."""

    def __init__(self, domain: float | None = None, **kwargs):
        super().__init__()
        self.domain = domain

    def forward(self, x):
        return 0.125 * x * x + 0.25 * x + 0.5


class ExactSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class QuadSiLU(nn.Module):
    """``a x^2 + b x + c`` fitted to SiLU: same FHE cost as ``quad`` at 4.2x lower error, since
    ``quad`` fits GELU. Defaults are least-squares over the measured gate_proj bulk (``|x| <= 16``).

    ``domain`` is an FHE output-range requirement and can only COST accuracy, so prefer the loosest
    the scheme tolerates.
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
