"""LM logit distillation: match the teacher's next-token distribution at every position.

Label-free like `embed`, but supervised everywhere instead of at one pooled vector. Nothing here
pools, so a base decoder is distilled as what it is rather than as an embedding model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from franken.config import Config, DistillConfig
from franken.distill.loss import PER_LAYER, layerwise_hidden_loss
from franken.models.base import ModelBackend
from franken.tasks.selfdistill import SelfDistillTask

# Positions per KL chunk. A 128k-vocab log-softmax is ~1 GB per 2048 positions and backward
# needs its own output, so slicing alone still retains every chunk -- hence `checkpoint`.
# COST: the head projection runs twice, ~10% of a 16k-token step. To trade it away, drop the
# checkpoint and lower FRANKEN_MAX_TOKENS_PER_RANK, or fuse linear+KL. Top-k changes the loss.
_CHUNK = 2048


def _kl_sum(s_hidden, t_hidden, s_w, t_w, temperature: float):
    # Projected in here, not by the caller, so `checkpoint` can discard the logits.
    s = (s_hidden @ s_w.T) / temperature
    t = (t_hidden @ t_w.T) / temperature
    # input=student, target=teacher => sum p*(log p - log q) = KL(teacher || student), the
    # mode-covering direction. `log_target` keeps the teacher in log space, no separate softmax.
    return F.kl_div(
        F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1), reduction="sum", log_target=True
    )


def logit_kl(student_out: dict, teacher_out: dict, attention_mask, temperature: float = 1.0):
    """Mean per-position KL. No shift: at position i both models predict token i+1, so reading
    them at the same position already compares two predictions of the same thing."""
    for out, who in ((student_out, "student"), (teacher_out, "teacher")):
        if "lm_head_weight" not in out:
            raise ValueError(
                f"logit distillation needs `lm_head_weight` from the {who} backend's forward; "
                "this backend has no output projection to distil."
            )
    s_w, t_w = student_out["lm_head_weight"], teacher_out["lm_head_weight"]
    s_h, t_h = student_out["hidden_states"][-1], teacher_out["hidden_states"][-1]

    # Dropped BEFORE the projection: a padded position's state is meaningless under right padding,
    # and projecting it to 128k vocab would be pure waste.
    keep = attention_mask.reshape(-1).bool()
    s_flat = s_h.reshape(-1, s_h.size(-1))[keep]
    t_flat = t_h.reshape(-1, t_h.size(-1))[keep]
    n = s_flat.size(0)
    if n == 0:
        return s_h.sum() * 0.0

    total = s_h.new_zeros(())
    for i in range(0, n, _CHUNK):
        s_c, t_c = s_flat[i : i + _CHUNK], t_flat[i : i + _CHUNK]
        if torch.is_grad_enabled() and s_c.requires_grad:
            total = total + checkpoint(
                _kl_sum, s_c, t_c, s_w, t_w, temperature, use_reentrant=False
            )
        else:
            total = total + _kl_sum(s_c, t_c, s_w, t_w, temperature)
    return total / n


class LogitDistillLoss(nn.Module):
    """``alpha * T^2 * KL(vocab) + beta * masked_MSE(hidden)``; see `DistillConfig`."""

    def __init__(self, cfg: DistillConfig):
        super().__init__()
        if cfg.hidden_loss not in PER_LAYER:
            raise ValueError(
                f"Unknown hidden_loss {cfg.hidden_loss!r}; use {' | '.join(PER_LAYER)}"
            )
        self.cfg = cfg
        self.per_layer = PER_LAYER[cfg.hidden_loss]

    def forward(self, student_out: dict, teacher_out: dict, attention_mask):
        T = self.cfg.temperature
        # T^2 restores the gradient scale the softening removed (Hinton); applied here so
        # `logit_kl` stays a statement about the divergence itself.
        kl = logit_kl(student_out, teacher_out, attention_mask, T) * (T**2)

        # Skipped, not scaled by zero: one masked MSE per layer over (B, S, H), each difference
        # retained until backward, so an ablation only measures anything if it never runs.
        if not self.cfg.beta:
            return self.cfg.alpha * kl, kl, kl.new_zeros(())

        # The KL constrains only the last layer; under a depth cut this is what supervises the
        # interior, and what anchors an aggressive op early enough for the KL to inform it.
        hidden = layerwise_hidden_loss(
            student_out["hidden_states"],
            teacher_out["hidden_states"],
            attention_mask,
            self.per_layer,
            self.cfg.hidden_layer_map,
            self.cfg.hidden_layer_mode,
        )
        return self.cfg.alpha * kl + self.cfg.beta * hidden, kl, hidden


@torch.no_grad()
def _nll_and_agreement(student_out: dict, teacher_out: dict, batch: dict):
    """Shifted NLL per model, plus top-1 agreement. Perplexity DOES shift -- position i predicts
    token i+1 -- where the KL does not, since there both models are read at the same position."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    # Scorable only where the state is real AND its target is real.
    valid = (mask[:, :-1] * mask[:, 1:]).reshape(-1).bool()
    target = ids[:, 1:].reshape(-1)[valid]

    def flat(out):
        h = out["hidden_states"][-1][:, :-1]
        return h.reshape(-1, h.size(-1))[valid]

    s_h, t_h = flat(student_out), flat(teacher_out)
    n = int(valid.sum())
    s_nll = t_nll = hits = 0.0
    for i in range(0, n, _CHUNK):
        s_logits = s_h[i : i + _CHUNK] @ student_out["lm_head_weight"].T
        t_logits = t_h[i : i + _CHUNK] @ teacher_out["lm_head_weight"].T
        tgt = target[i : i + _CHUNK]
        s_nll += float(F.cross_entropy(s_logits, tgt, reduction="sum"))
        t_nll += float(F.cross_entropy(t_logits, tgt, reduction="sum"))
        hits += float((s_logits.argmax(-1) == t_logits.argmax(-1)).sum())
    return s_nll, t_nll, hits, n


_TOTALS = ("s_nll", "t_nll", "hits", "scored", "kl", "positions")


@torch.no_grad()
def score_totals(backend: ModelBackend, task, model, teacher, loader, device) -> dict:
    """Unnormalized sums over one loader. Left unnormalized so a caller scoring a source at a time
    can add the parts and get exactly what a single pass would have read."""
    totals = dict.fromkeys(_TOTALS, 0.0)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = task.model_inputs(batch)
        s_out, t_out = backend.forward(model, inputs), backend.forward(teacher, inputs)

        n = int(batch["attention_mask"].sum())
        totals["kl"] += float(logit_kl(s_out, t_out, batch["attention_mask"])) * n
        totals["positions"] += n

        s_nll, t_nll, hits, scored = _nll_and_agreement(s_out, t_out, batch)
        totals["s_nll"] += s_nll
        totals["t_nll"] += t_nll
        totals["hits"] += hits
        totals["scored"] += scored
    return totals


def metrics(totals: dict) -> dict:
    scored, positions = max(totals["scored"], 1), max(totals["positions"], 1)
    return {
        "agreement": totals["hits"] / scored,
        "kl": totals["kl"] / positions,
        "ppl": math.exp(totals["s_nll"] / scored),
        "teacher_ppl": math.exp(totals["t_nll"] / scored),
    }


def sum_totals(parts) -> dict:
    return {k: sum(p[k] for p in parts) for k in _TOTALS}


class LMDistillTask(SelfDistillTask):
    def __init__(self):
        self._loss_fn: LogitDistillLoss | None = None

    def compute_loss(self, student_out: dict, teacher_out: dict, batch: dict, cfg: Config) -> tuple:
        if self._loss_fn is None:
            self._loss_fn = LogitDistillLoss(cfg.distill)
        total, kl, hidden = self._loss_fn(student_out, teacher_out, batch["attention_mask"])
        return total, {"kl": kl.detach(), "hidden": hidden.detach()}

    def select_metric(self) -> tuple[str, bool]:
        # Fidelity selects, quality reports -- the same division as the embed track's recall@10 vs
        # nDCG. `ppl` is the teacher's own quality and moves for reasons the student cannot control.
        return ("agreement", True)

    @torch.no_grad()
    def evaluate(
        self, backend: ModelBackend, model, tokenizer, cfg: Config, split="validation", teacher=None
    ) -> dict:
        _ds, loader = self.eval_loader(tokenizer, cfg, split, teacher)
        model.eval()
        device = next(model.parameters()).device
        return metrics(score_totals(backend, self, model, teacher, loader, device))

    # train_teacher inherits the base no-op (pretrained checkpoint is the teacher).
