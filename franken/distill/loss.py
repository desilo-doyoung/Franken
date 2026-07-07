from torch import nn
import torch.nn.functional as F

from franken.distill.layer_map import resolve_layer_map
from franken.config import DistillConfig

def masked_mse_loss(student_hidden, teacher_hidden, attention_mask):
    diff = (student_hidden - teacher_hidden) ** 2 # (B, S, H)
    mask = attention_mask.unsqueeze(-1).to(diff.dtype) # (B, S, 1)
    return (diff * mask).sum() / (mask.sum() * student_hidden.size(-1)).clamp_min(1.0) #

class DistillationLoss(nn.Module):
    def __init__(self, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, student_logits, teacher_logits, labels, student_hidden, teacher_hidden, attention_mask):
        # 1. hard-label loss
        ce = F.cross_entropy(student_logits, labels)

        # 2. logit KD
        T = self.cfg.temperature
        kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        num_studets = len(student_hidden) - 1
        num_teachers = len(teacher_hidden) - 1
        layer_map = resolve_layer_map(num_teachers, num_studets, self.cfg.hidden_layer_map)

        hidden = 0.0
        for s_block, t_block in enumerate(layer_map, start=1):   # student blocks 1..num_student
            hidden += masked_mse_loss(student_hidden[s_block], teacher_hidden[t_block], attention_mask)
        hidden = hidden / len(layer_map)

        total = (1 - self.cfg.alpha) * ce + self.cfg.alpha * kl + self.cfg.beta * hidden

        return total, ce, kl, hidden
