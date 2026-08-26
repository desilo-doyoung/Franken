import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, create_block_mask, create_mask

from franken.distill.packing import doc_ids
from franken.models.llama.config import LlamaModelConfig
from franken.models.llama.layer import LlamaDecoderLayer
from franken.models.llama.rope import LlamaRotaryEmbedding


def _causal_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _same_document_mod(doc):
    def mod(b, h, q_idx, kv_idx):
        return doc[b, q_idx] == doc[b, kv_idx]

    return mod


# Eager create_block_mask costs 5.5 ms/batch against 0.2 ms compiled.
_block_mask = torch.compile(create_block_mask, dynamic=False)


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            config.rope_scaling_factor,
            config.rope_low_freq_factor,
            config.rope_high_freq_factor,
            config.rope_original_max_position_embeddings,
        )
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
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

    # A BlockMask under flex, else additive where 0 is visible and -inf is masked.
    def _causal_mask(self, attention_mask, S, hidden_states, position_ids=None):
        if position_ids is not None:
            return self._packed_mask(S, hidden_states, position_ids)

        if self.config.attn_impl == "sdpa_causal":
            # Under right padding, is_causal already hides pads from every real row.
            return None

        dtype = hidden_states.dtype
        min_val = torch.finfo(dtype).min
        mask = torch.full((S, S), min_val, device=hidden_states.device, dtype=dtype)
        mask = mask.triu(diagonal=1)[None, None]
        if attention_mask is not None:
            pad = (1 - attention_mask[:, None, None, :].to(dtype=dtype)) * min_val
            mask = mask + pad
        return mask

    # Causal AND same-document, from the segment rule the HF teacher applies to the same
    # position_ids -- so the two agree by construction rather than by coincidence.
    def _packed_mask(self, S, hidden_states, position_ids):
        doc = doc_ids(position_ids)
        B = doc.shape[0]
        mod = and_masks(_causal_mod, _same_document_mod(doc))
        device = hidden_states.device

        if self.config.attn_impl == "flex":
            return _block_mask(mod, B, None, S, S, device=device)

        keep = create_mask(mod, B, None, S, S, device=device)
        dtype = hidden_states.dtype
        # Filled in the model's dtype, not accumulated: summed min_vals saturate to -inf, and a
        # float32 default cannot hold finfo(float64).min.
        mask = torch.zeros(keep.shape, dtype=dtype, device=device)
        return mask.masked_fill_(~keep, torch.finfo(dtype).min)
