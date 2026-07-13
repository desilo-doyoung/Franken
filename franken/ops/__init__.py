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


class ApproxSoftmax(nn.Module):
    """CGF (cumulant-generating-function) softmax — HE-friendly.

    Replaces the log-sum-exp normalizer with its 2nd-order cumulant (Gaussian)
    approximation: ``log(sum_j exp x_j) ~= mu + var/2 + log(n_vis)``, where mu
    and var are the mean/variance of the *visible* scores. So
    ``softmax_i ~= exp(x_i - mu - var/2 - log n_vis)``. Statistics use a binary
    plaintext mask (visible positions only); the ops are plaintext-mask
    multiply, add, square, and exp — no ciphertext division or max-subtraction.
    Output is unnormalized-by-design (no reciprocal); distillation adapts to it.
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


class PolySiLU(nn.Module):
    """Degree-3 (default) polynomial approximation of SiLU (x * sigmoid(x)).

    The FFN activation must run under homomorphic encryption, which supports
    only additions and multiplications — so this op uses *only* those: no
    exp/sigmoid, no clamping, no data-dependent branches. Coefficients are a
    density-weighted least-squares fit to SiLU over the empirical BERT FFN
    pre-activation domain [-12, 5] (see PROGRESS.md). Degree 3 is the default:
    same multiplicative depth as degree 4 but a far better fit than degree 2
    (a parabola cannot track SiLU's linear rise on firing tokens).

    Evaluated in the power basis (x2 = x*x, x3 = x2*x, x4 = x2*x2) so the
    multiplicative depth is ceil(log2(degree)) = 2 for degrees 2-4, versus
    Horner's `degree`. The coeff * power products are plaintext * ciphertext.
    """

    # density-weighted fits to SiLU over [-12, 5], high -> low order.
    _FITTED = {
        2: (4.3665e-02, 2.2396e-01, 2.9755e-02),
        3: (9.3172e-03, 1.2586e-01, 3.6631e-01, 3.3845e-02),
        4: (6.3491e-04, 1.7505e-02, 1.4519e-01, 3.5259e-01, 3.5574e-04),
    }

    def __init__(self, degree: int = 3, coeffs=None, learnable: bool = False, **kwargs):
        super().__init__()
        if degree not in (2, 3, 4):
            raise ValueError(f"PolySiLU supports degrees 2-4, got {degree}.")
        if coeffs is None:
            coeffs = self._FITTED[degree]
        elif len(coeffs) != degree + 1:
            raise ValueError(f"degree {degree} needs {degree + 1} coeffs, got {len(coeffs)}.")

        self.degree = degree
        c = torch.tensor(coeffs, dtype=torch.float32)  # high -> low order
        # Fixed constants at HE inference; `learnable` lets distillation refine
        # them from this fit, after which they are frozen (still just constants).
        if learnable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    def forward(self, x):
        c = self.coeffs
        x2 = x * x
        if self.degree == 2:
            return c[0] * x2 + c[1] * x + c[2]
        x3 = x2 * x
        if self.degree == 3:
            return c[0] * x3 + c[1] * x2 + c[2] * x + c[3]
        x4 = x2 * x2
        return c[0] * x4 + c[1] * x3 + c[2] * x2 + c[3] * x + c[4]


class ChebyshevGELU(nn.Module):
    """GELU as a single Chebyshev polynomial on ``u = x / domain``, evaluated by
    the Clenshaw recurrence so all intermediate values stay O(1) — FHE-stable,
    unlike the monomial basis whose ``x**k`` terms explode. Fit once to GELU over
    ``[-domain, domain]``; the coefficients approximate a fixed function and are
    task-independent (no refit per dataset).

    ⚠️ Domain / blow-up (read this). A polynomial diverges outside its fit
    interval, so for ``|x| > domain`` the polynomial explodes. During *training*
    the input is clamped to ``[-1, 1]`` — a numerical scaffold so distillation
    does not NaN on the teacher's ~±150 outlier activations at init. At
    *inference* the clamp is gone (a clamp is min/max, expensive under FHE), so
    the deployed op is the bare polynomial. It is therefore safe **only while
    activations stay in ``[-domain, domain]``** — an *empirical, per-dataset*
    property you must verify with an activation-range check. This is NOT a hard
    guarantee: an out-of-domain activation on unseen data will blow up and
    cascade through later layers. Widen ``domain`` (costs FHE depth) to buy more
    escape margin.
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

    def _clenshaw(self, u):
        # Sum_k c_k T_k(u) via Clenshaw; T_k(u) in [-1,1] for u in [-1,1].
        b1 = torch.zeros_like(u)
        b2 = torch.zeros_like(u)
        c = self.coef
        for k in range(c.numel() - 1, 0, -1):
            b0 = 2.0 * u * b1 - b2 + c[k]
            b2, b1 = b1, b0
        return u * b1 - b2 + c[0]

    def forward(self, x):
        u = x / self.domain
        if self.training:
            u = u.clamp(-1.0, 1.0)  # numerical scaffold; bare (no clamp) at inference
        if self.training and u.requires_grad:
            # Checkpoint the degree-length recurrence: otherwise autograd stores
            # `degree` intermediates per layer and OOMs at high degree.
            return checkpoint(self._clenshaw, u, use_reentrant=False)
        return self._clenshaw(u)


SOFTMAX_OPS = {"exact": ExactSoftmax, "approx": ApproxSoftmax}
ACTIVATION_OPS = {"exact": ExactGELU, "poly": PolySiLU, "cheb_gelu": ChebyshevGELU}


def build_softmax(name: str, **kwargs) -> nn.Module:
    if name not in SOFTMAX_OPS:
        raise KeyError(f"Unknown softmax op {name!r}; available: {sorted(SOFTMAX_OPS)}")
    return SOFTMAX_OPS[name](**kwargs)


def build_activation(name: str, **kwargs) -> nn.Module:
    if name not in ACTIVATION_OPS:
        raise KeyError(f"Unknown activation op {name!r}; available: {sorted(ACTIVATION_OPS)}")
    return ACTIVATION_OPS[name](**kwargs)
