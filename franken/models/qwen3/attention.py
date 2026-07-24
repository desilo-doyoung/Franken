import torch
from torch import nn

from franken.models.qwen3.config import Qwen3ModelConfig
from franken.ops import build_softmax

from .rope import apply_rotary_pos_emb


def repeat_kv(x, repeat):
    B, num_kv_heads, S, head_dim = x.size()
    if repeat == 1:
        return x
    x = x[:, :, None, :, :].expand(B, num_kv_heads, repeat, S, head_dim)
    return x.reshape(B, num_kv_heads * repeat, S, head_dim)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.bias = config.attention_bias

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.softmax = build_softmax(config.softmax, **config.softmax_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, H = hidden_states.size()  # Batch size, Sequence length, Hidden size

        def _split_heads(x, num_heads):
            return x.view(B, S, num_heads, self.head_dim).transpose(
                1, 2
            )  # (B, num_heads, S, head_dim)

        q = self.q_norm(_split_heads(self.q_proj(hidden_states), self.num_heads))
        k = self.k_norm(_split_heads(self.k_proj(hidden_states), self.num_kv_heads))
        v = _split_heads(self.v_proj(hidden_states), self.num_kv_heads)

        (cos, sin) = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = repeat_kv(v, self.num_heads // self.num_kv_heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        probs = self.softmax(scores, attention_mask, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        # TODO: return probs for distillation purposes
        return self.o_proj(context)
