import torch
from torch import nn


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)),
            persistent=False,
        )

    def forward(self, x, position_ids):
        # Broadcast multiply, not einsum: autocast intercepts einsum by name and casts to bf16
        # despite the .float(), corrupting the angles. This is an outer product, so it is
        # bit-identical.
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
