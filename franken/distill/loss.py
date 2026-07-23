"""Generic, task-agnostic distillation loss helpers.

Only the reusable pieces live here so any task can share them. Task-specific
losses (e.g. the classification CE + logit-KL used by MRPC) live in their task
module — see ``franken.tasks.mrpc.ClassificationDistillLoss``.
"""


def masked_mse_loss(student_hidden, teacher_hidden, attention_mask):
    diff = (student_hidden - teacher_hidden) ** 2  # (B, S, H)
    mask = attention_mask.unsqueeze(-1).to(diff.dtype)  # (B, S, 1)
    return (diff * mask).sum() / (mask.sum() * student_hidden.size(-1)).clamp_min(1.0)
