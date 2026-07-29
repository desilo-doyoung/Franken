"""Generic, task-agnostic distillation loss helpers.

Only the reusable pieces live here so any task can share them. Task-specific
losses (e.g. the classification CE + logit-KL used by MRPC) live in their task
module — see ``franken.tasks.mrpc.ClassificationDistillLoss``.
"""


def masked_mse_loss(student_hidden, teacher_hidden, attention_mask):
    diff = (student_hidden - teacher_hidden) ** 2  # (B, S, H)
    mask = attention_mask.unsqueeze(-1).to(diff.dtype)  # (B, S, 1)
    return (diff * mask).sum() / (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)


def masked_relative_mse_loss(student_hidden, teacher_hidden, attention_mask, eps: float = 1e-6):
    """Exactly ``masked_mse_loss`` divided by the teacher layer's own masked mean square, so every
    layer contributes on equal terms regardless of its activation scale. Same term, same layers,
    same mask — only the per-layer normalization is new.

    Raw MSE lets whichever layers happen to have large activations own the gradient. Measured on a
    seeded depth-14 Qwen3 student, the teacher's per-layer mean square spans **11,085x** (0.07 to
    808), loss share rises monotonically with depth, and — worst of all — the FINAL layer, the one
    the pooled embedding is computed from, gets only **2.7%** of the weight while carrying the
    largest *relative* error (relMSE 1.12 vs 0.05-0.15 elsewhere), because the final RMSNorm leaves
    its magnitude small. Normalizing fixes that inversion.

    It also rescales the term from O(10) to O(0.1), which is what makes ``distill.beta`` a real
    knob: under raw MSE the hidden term outweighed the embedding term ~59:1 at init, so beta
    changed almost nothing.

    NB this fixes cross-layer scale spread only. Within-layer outlier dominance turned out NOT to
    be a problem — for the layers carrying the loss, the top 1% of error elements hold just 21-29%.
    """
    mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)  # (B, S, 1)
    denom = (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)
    mse = (((student_hidden - teacher_hidden) ** 2) * mask).sum() / denom
    scale = ((teacher_hidden**2) * mask).sum() / denom
    return mse / (scale + eps)
