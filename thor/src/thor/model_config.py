"""Model-specific configuration for the HE BERT inference pipeline.

The active model is the 8-layer distilled BERT (``distilled-model/``): a student
distilled from a 12-layer teacher and fine-tuned on MRPC (test acc/F1 =
0.8348/0.8789). It is a standard HF ``BertForSequenceClassification`` with 8
encoder layers, a pooler, and a 2-class ``classifier`` head; its weight keys are
HF-name-matched, so it loads directly with ``strict=False`` (no missing/unexpected).

The "wide" layer sets below select higher-range polynomial approximations for
the few layers whose activations exceed the default approximation domains. They
are MODEL-SPECIFIC: swapping the model requires re-measuring per-layer magnitudes
and updating these sets. The two quantities that matter are the max absolute
attention score (softmax) and the max per-token variance of the second LayerNorm
input (layernorm), both measured over valid tokens on the MRPC validation set.

Measured for ``distilled-model/`` (default approximation domains in he.py):
  - softmax: layer 1 has max|score| ~= 54 (all others <= 18). ``he_softmax1``
    covers only [-27, 22], so layer 1 uses ``he_softmax2`` ([-70, 70]).
  - layernorm: layer 6 has max ln2 variance ~= 1476 (all others <= 100).
    ``he_layernorm2`` covers var <= 150, so layer 6 uses ``he_layernorm3``
    (var <= 2500).

For reference, the previous 12-layer ``finetuned_models/mrpc`` model needed
WIDE_SOFTMAX_LAYERS = {2} and WIDE_LAYERNORM_LAYERS = {9, 10}.
"""

NUM_LAYERS = 8

# Encoder layers that need the wide-range softmax (he_softmax2) instead of he_softmax1.
WIDE_SOFTMAX_LAYERS = frozenset({1})

# Encoder layers that need the wide-range output layernorm (he_layernorm3) instead of he_layernorm2.
WIDE_LAYERNORM_LAYERS = frozenset({6})

MODEL_DIR = "./distilled-model"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
