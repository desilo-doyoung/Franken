"""Task-agnostic distillation loss helpers. Task-specific losses (MRPC's CE + logit-KL) live in
their task module."""


def masked_mse_loss(student_hidden, teacher_hidden, attention_mask):
    diff = (student_hidden - teacher_hidden) ** 2  # (B, S, H)
    mask = attention_mask.unsqueeze(-1).to(diff.dtype)  # (B, S, 1)
    return (diff * mask).sum() / (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)


def masked_relative_mse_loss(student_hidden, teacher_hidden, attention_mask, eps: float = 1e-6):
    """``masked_mse_loss`` divided by the teacher layer's own masked mean square, so layers
    contribute on equal terms whatever their activation scale.

    Raw MSE lets large-activation layers own the gradient: on a depth-14 Qwen3 student the
    teacher's per-layer mean square spans 11,085x (0.07 to 808), and the FINAL layer -- the one the
    embedding comes from -- gets only 2.7% of the weight while carrying the worst relative error
    (relMSE 1.12 vs 0.05-0.15). Normalizing also rescales the term O(10) -> O(0.1), which is what
    makes `beta` a real knob (raw, the hidden term outweighed the embedding term ~59:1 at init).

    Fixes cross-layer spread only; within-layer outliers were measured NOT to be a problem
    (top 1% of error elements hold just 21-29%).
    """
    mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)  # (B, S, 1)
    denom = (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)
    mse = (((student_hidden - teacher_hidden) ** 2) * mask).sum() / denom
    scale = ((teacher_hidden**2) * mask).sum() / denom
    return mse / (scale + eps)
