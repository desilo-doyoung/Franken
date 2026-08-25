import torch
from torch import nn

from franken.distill.packing import doc_ids
from franken.models.qwen3.config import Qwen3ModelConfig
from franken.models.qwen3.layer import Qwen3DecoderLayer
from franken.models.qwen3.rope import Qwen3RotaryEmbedding


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

    def forward(self, input_ids, attention_mask=None, position_ids=None) -> dict:
        hidden_states = self.embed_tokens(input_ids)
        B, S, _ = hidden_states.shape
        # Only a caller-supplied position_ids can carry document boundaries; the arange
        # fallback has none, so it must not pay for the segment comparison.
        packed = position_ids
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, S)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        mask = self._causal_mask(attention_mask, S, hidden_states, packed)

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
    def _causal_mask(self, attention_mask, S, hidden_states, position_ids=None):
        if position_ids is None and self.config.attn_impl == "sdpa_causal":
            # Nothing the kernel's is_causal does not already express: under right padding, causal
            # masking hides pads from every real row. Building the mask would only waste it.
            return None

        dtype = hidden_states.dtype
        device = hidden_states.device
        min_val = torch.finfo(dtype).min

        mask = torch.full((S, S), min_val, device=device, dtype=dtype).triu(diagonal=1)[None, None]
        if attention_mask is not None:
            pad = (1 - attention_mask[:, None, None, :].to(dtype=dtype)) * min_val
            mask = mask + pad
        if position_ids is not None:
            # Packed blocks: a query never sees a neighbouring document. Same segment rule the HF
            # teacher applies to the identical position_ids, so the two masks agree by construction.
            doc = doc_ids(position_ids)
            cross = (doc[:, :, None] != doc[:, None, :])[:, None].to(dtype=dtype)
            mask = mask + cross * min_val
        return mask
