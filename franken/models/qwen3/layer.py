import torch
from torch import nn

from franken.models.qwen3.attention import Qwen3Attention
from franken.models.qwen3.config import Qwen3ModelConfig
from franken.models.qwen3.mlp import Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps

        # names are kept for compatibility with the original implementation
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=self.eps)
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, eps=self.eps)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, attention_mask=None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
