import torch
from torch import nn

from franken.models.llama.attention import LlamaAttention
from franken.models.llama.config import LlamaModelConfig
from franken.models.llama.mlp import LlamaMLP


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps

        # names are kept for compatibility with the original implementation
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
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
