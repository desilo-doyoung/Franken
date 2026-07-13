"""Swappable ops — the flexibility mechanism.

``ModelConfig.softmax`` / ``ModelConfig.activation`` are just names (+ optional
kwargs) resolved here into ``nn.Module`` instances. Attention / FFN modules
receive the built module and never hardcode ``F.softmax`` / ``F.gelu``, so
swapping in an HE-friendly approximation is a config change. Add a new op = add
one class and one dict entry.
"""

from __future__ import annotations

import torch
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


SOFTMAX_OPS = {"exact": ExactSoftmax, "approx": ApproxSoftmax}
ACTIVATION_OPS = {"exact": ExactGELU, "poly": PolySiLU}


def build_softmax(name: str, **kwargs) -> nn.Module:
    if name not in SOFTMAX_OPS:
        raise KeyError(f"Unknown softmax op {name!r}; available: {sorted(SOFTMAX_OPS)}")
    return SOFTMAX_OPS[name](**kwargs)


def build_activation(name: str, **kwargs) -> nn.Module:
    if name not in ACTIVATION_OPS:
        raise KeyError(f"Unknown activation op {name!r}; available: {sorted(ACTIVATION_OPS)}")
    return ACTIVATION_OPS[name](**kwargs)
