"""Task-agnostic hidden-state losses; task-specific terms live in the task module."""


def masked_mse_loss(student_hidden, teacher_hidden, attention_mask):
    diff = (student_hidden - teacher_hidden) ** 2  # (B, S, H)
    mask = attention_mask.unsqueeze(-1).to(diff.dtype)  # (B, S, 1)
    return (diff * mask).sum() / (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)


def masked_relative_mse_loss(student_hidden, teacher_hidden, attention_mask, eps: float = 1e-6):
    """Normalized by the teacher layer's own mean square, so layers contribute equally whatever
    their activation scale -- raw MSE lets large-activation layers own the gradient."""
    mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)  # (B, S, 1)
    denom = (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)
    mse = (((student_hidden - teacher_hidden) ** 2) * mask).sum() / denom
    scale = ((teacher_hidden**2) * mask).sum() / denom
    return mse / (scale + eps)
