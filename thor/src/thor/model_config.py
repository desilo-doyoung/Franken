"""Model-specific configuration for the HE BERT inference pipeline.

The active model is the 8-layer distilled BERT (``distilled-model/``): a student
distilled from a 12-layer teacher and fine-tuned on MRPC. Its FFN activation is
the MPCFormer quadratic GELU (``0.125 x^2 + 0.25 x + 0.5``); ``he.stage_13_gelu``
computes that quad directly (config.json has ``"activation": "quad"``), and the
plaintext reference is patched to match (``utils.load_model``). It is a standard
HF ``BertForSequenceClassification`` with 8 encoder layers, a pooler, and a
2-class ``classifier`` head (quad test acc/F1 = 0.8249/0.8733); its weight keys
are HF-name-matched, so it loads with ``strict=False`` (no missing/unexpected).

The "wide" layer sets below select higher-range polynomial approximations for
the few layers whose activations exceed the default approximation domains. They
are MODEL-SPECIFIC: swapping the model (or its activation) requires re-measuring
per-layer magnitudes and updating these sets. Use ``thor/measure_ranges.py``. The
two quantities that matter are the pre-softmax attention score range (softmax) and
the max per-token variance of the second-LayerNorm input (layernorm), both over
valid tokens on the MRPC validation set.

Measured for the quad ``distilled-model/`` (default domains in he.py; via
``measure_ranges.py``):
  - softmax: layers 1/2/4 breach ``he_softmax1``'s [-27, 22] box (scores reach
    [-39, 38] / +26 / +24), so they use ``he_softmax2`` ([-70, 70]). All other
    layers stay within [-27, 22].
  - layernorm: layers 3 and 6 have max ln2 variance ~= 164 and ~= 1188 (all
    others <= 91). ``he_layernorm2`` covers var <= 150, so 3 and 6 use
    ``he_layernorm3`` (var <= 2500; the quad output is range-penalty-bounded so
    the peak 1188 stays under 2500 — plain unbounded quad would not).

History: the exact-GELU 8-layer student needed WIDE_SOFTMAX_LAYERS = {1} and
WIDE_LAYERNORM_LAYERS = {6}; the earlier 12-layer ``finetuned_models/mrpc`` model
needed WIDE_SOFTMAX_LAYERS = {2} and WIDE_LAYERNORM_LAYERS = {9, 10}.
"""

NUM_LAYERS = 8

# Encoder layers that need the wide-range softmax (he_softmax2) instead of he_softmax1.
WIDE_SOFTMAX_LAYERS = frozenset({1, 2, 4})

# Encoder layers that need the wide-range output layernorm (he_layernorm3) instead of he_layernorm2.
WIDE_LAYERNORM_LAYERS = frozenset({3, 6})

MODEL_DIR = "./distilled-model"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
