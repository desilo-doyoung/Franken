import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from franken.models.llama.config import ATTN_IMPLS, LlamaModelConfig
from franken.ops import build_softmax

from .rope import apply_rotary_pos_emb


def repeat_kv(x, repeat):
    B, num_kv_heads, S, head_dim = x.size()
    if repeat == 1:
        return x
    x = x[:, :, None, :, :].expand(B, num_kv_heads, repeat, S, head_dim)
    return x.reshape(B, num_kv_heads * repeat, S, head_dim)


# Compiled once: under packing every block is exactly max_seq_len, so there is one shape.
_flex = torch.compile(flex_attention, dynamic=False)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.bias = config.attention_bias

        # No q_norm/k_norm: Qwen3 normalizes each head before RoPE, Llama does not.
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.bias)
        self.softmax = build_softmax(config.softmax, **config.softmax_kwargs)

        self.attn_impl = config.attn_impl
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"Unknown attn_impl {self.attn_impl!r}; use {' | '.join(ATTN_IMPLS)}")
        if self.attn_impl in ("sdpa_causal", "flex") and config.softmax != "exact":
            raise ValueError(
                f"attn_impl {self.attn_impl!r} fuses the softmax into the kernel, so it cannot run "
                f"softmax={config.softmax!r}. Approximate softmaxes need attn_impl 'manual'."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, H = hidden_states.size()

        def _split_heads(x, num_heads):
            return x.view(B, S, num_heads, self.head_dim).transpose(1, 2)

        q = _split_heads(self.q_proj(hidden_states), self.num_heads)
        k = _split_heads(self.k_proj(hidden_states), self.num_kv_heads)
        v = _split_heads(self.v_proj(hidden_states), self.num_kv_heads)

        (cos, sin) = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.attn_impl == "flex":
            # block_mask carries causality and document isolation; GQA is native, no repeat_kv.
            context = _flex(q, k, v, block_mask=attention_mask, enable_gqa=True)
        elif self.attn_impl == "sdpa_causal":
            if attention_mask is None:
                # No attn_mask: a float mask is what disqualifies the flash backend, and under
                # right padding causal masking already hides pads from every real row. Pad rows
                # compute garbage that nothing reads (masked in the loss, excluded by pooling).
                context = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                # Packed blocks: is_causal cannot say document isolation. The mask is already
                # causal, and in fp32 + GQA a 4D mask reaches only the MATH backend.
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, enable_gqa=True
                )
        else:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)

            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
            probs = self.softmax(scores, attention_mask, dim=-1)
            context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        return self.o_proj(context)
