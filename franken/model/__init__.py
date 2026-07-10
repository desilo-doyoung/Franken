"""Custom BERT student package.

A from-scratch BERT re-implementation whose ops (softmax/GELU) resolve from
franken.ops so they stay swappable via config. Modules:

  embeddings.py  word + position + token-type embeddings, LayerNorm
  attention.py   multi-head self-attention; softmax op injected from the registry
  ffn.py         feed-forward; GELU op injected from the registry
  layer.py       one transformer block (attention + FFN, residual + LayerNorm)
  encoder.py     stack of num_hidden_layers blocks; collects hidden states + attentions
  bert.py        BertModel + classification head; returns logits + hidden_states + attentions
  loader.py      strided teacher -> student weight initialization under layer reduction
"""
