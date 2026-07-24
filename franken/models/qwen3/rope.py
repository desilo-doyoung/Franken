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
        freqs = torch.einsum("bi, j -> bij", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # unsqueeze to the head dimension to match the shape of q and k
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
