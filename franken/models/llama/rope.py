import math

import torch
from torch import nn


def llama3_scale(inv_freq, factor, low_freq_factor, high_freq_factor, old_context_len):
    """Llama 3's piecewise NTK smoothing of inv_freq. Position-independent, so it moves the
    angles at EVERY sequence length -- not only past the original context."""
    low_wavelen = old_context_len / low_freq_factor
    high_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # 1. for low frequencies, scale down heavily
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    # 2. smoothe the transition between low and high frequencies
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim,
        theta,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer(
            "inv_freq",
            llama3_scale(
                inv_freq,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            ),
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
