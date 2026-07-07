import torch

from torch import nn
from franken.config import ModelConfig
from franken.ops import build_softmax

class BertSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads # 64

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = build_softmax(config.softmax, **config.softmax_kwargs)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, S, H = hidden_states.size()  # Batch size, Sequence length, Hidden size

        def _split_heads(x):
            return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)

        q = _split_heads(self.query(hidden_states))
        k = _split_heads(self.key(hidden_states))
        v = _split_heads(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, S, S)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = self.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)  # (B, num_heads, S, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, S, H)  # (B, S, H)

        # return probs for distillation purposes
        return context, probs


class BertSelfOutput(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        self_output, attn_probs = self.self(hidden_states, attention_mask)
        hidden_states = self.output(self_output, hidden_states)
        return hidden_states, attn_probs
