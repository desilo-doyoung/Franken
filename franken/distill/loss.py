"""Task-agnostic hidden-state losses; task-specific terms live in the task module."""

from franken.distill.layer_map import resolve_layer_map


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


PER_LAYER = {"mse": masked_mse_loss, "relative": masked_relative_mse_loss}


def layerwise_hidden_loss(
    student_hidden, teacher_hidden, attention_mask, per_layer, layer_map, layer_mode="stride"
):
    """Mean per-layer hidden match under the student's layer map. `hidden_states[0]` is the
    embedding output on every backend, so block i lives at index i+1."""
    resolved = resolve_layer_map(
        len(teacher_hidden) - 1, len(student_hidden) - 1, layer_map, layer_mode
    )
    total = 0.0
    for s_block, t_block in enumerate(resolved):
        total = total + per_layer(
            student_hidden[s_block + 1], teacher_hidden[t_block + 1], attention_mask
        )
    return total / len(resolved)
