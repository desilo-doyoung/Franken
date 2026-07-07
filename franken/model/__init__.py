"""Custom BERT student package.

Intentionally left empty — the student model is built interactively in the
tutorial session so the architecture is understood piece by piece. Ops
(softmax/GELU) are resolved from franken.ops so they stay swappable via config.
Planned modules:

  embeddings.py  word + position + token-type embeddings, LayerNorm
  attention.py   multi-head self-attention; softmax op injected from the registry
  ffn.py         feed-forward; GELU op injected from the registry
  layer.py       one transformer block (attention + FFN, residual + LayerNorm)
  encoder.py     stack of num_hidden_layers blocks; collects hidden states + attentions
  bert.py        BertModel + classification head; returns logits + hidden_states + attentions
  loader.py      strided teacher -> student weight initialization under layer reduction
"""
