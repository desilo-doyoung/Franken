import torch
from torch import nn

from franken.models.qwen3.config import Qwen3ModelConfig
from franken.models.qwen3.rope import Qwen3RotaryEmbedding
from franken.models.qwen3.layer import Qwen3DecoderLayer


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = Qwen3RotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None) -> dict:
        hidden_states = self.embed_tokens(input_ids)
        B, S, _ = hidden_states.shape
        position_ids = (
            torch.arange(S, device=hidden_states.device)
            .unsqueeze(0)
            .expand(B, S)
        )
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        mask = self._causal_mask(attention_mask, S, hidden_states)

        all_hidden_states = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, (cos, sin), mask)
        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        return dict(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    # additive mask where 0 is visible and -inf is masked
    def _causal_mask(self, attention_mask, S, hidden_states):
        dtype = hidden_states.dtype
        device = hidden_states.device
        min_val = torch.finfo(dtype).min

        mask  = torch.full((S, S), min_val, device=device, dtype=dtype).triu(diagonal=1)[None, None]
        if attention_mask is not None:
            pad = (1 - attention_mask[:, None, None, :].to(dtype=dtype)) * min_val
            mask = mask + pad
        return mask
