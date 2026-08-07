"""Model-specific configuration for the HE BERT inference pipeline.

The active model is the 6-layer distilled BERT (``distilled-model/``): a student
distilled from a 12-layer teacher and fine-tuned on MRPC. Its FFN activation is
the MPCFormer quadratic GELU (``0.125 x^2 + 0.25 x + 0.5``); ``he.stage_13_gelu``
computes that quad directly (config.json has ``"activation": "quad"``), and the
plaintext reference is patched to match (``utils.load_model``). It is a standard
HF ``BertForSequenceClassification`` with 6 encoder layers, a pooler, and a
2-class ``classifier`` head (quad test acc/F1 = 0.8041/0.8638); its weight keys
are HF-name-matched, so it loads with ``strict=False`` (no missing/unexpected).

The "wide" layer sets below select higher-range polynomial approximations for
the few layers whose activations exceed the default approximation domains. They
are MODEL-SPECIFIC: swapping the model (or its activation) requires re-measuring
per-layer magnitudes and updating these sets. Use ``thor/measure_ranges.py``. The
two quantities that matter are the pre-softmax attention score range (softmax) and
the max per-token variance of the second-LayerNorm input (layernorm), both over
valid tokens on the MRPC validation set.

Measured for the quad ``distilled-model/`` over validation AND test (default
domains in he.py; via ``measure_ranges.py``):
  - softmax: only layer 1 breaches ``he_softmax1``'s [-27, 22] box (scores reach
    +27.5 on test), so it uses ``he_softmax2`` ([-70, 70]) with 42 units of slack.
    Layer 3 is the tightest of the rest (+20.7, margin 1.06).
  - layernorm: layers 2 and 4 have max ln2 variance ~= 311 and ~= 1694 (all others
    <= 38). ``he_layernorm2`` covers var <= 150, so 2 and 4 use ``he_layernorm3``
    (var <= 2500). L4's 1694 leaves 1.48x headroom; the range penalty is what keeps
    it there (quad output |max| 142.7, vs ~700 unpenalized).

History: the 8-layer quad student needed SOFTMAX2_LAYERS = {1, 2},
SOFTMAX3_LAYERS = {4} and WIDE_LAYERNORM_LAYERS = {3, 6}; the exact-GELU 8-layer
student needed SOFTMAX2_LAYERS = {1} and WIDE_LAYERNORM_LAYERS = {6}; the earlier
12-layer ``finetuned_models/mrpc`` model needed SOFTMAX2_LAYERS = {2} and
WIDE_LAYERNORM_LAYERS = {9, 10}.
"""

NUM_LAYERS = 6

# Layers dispatched to he_softmax2 (wide exp domain he_exp2, [-70,70], vs he_softmax1's
# [-27,22]). Disjoint from SOFTMAX3_LAYERS below.
SOFTMAX2_LAYERS = frozenset({1})

# Subset also needing he_softmax3 (= he_softmax2 + exp_scale): its sum-of-exps is too small
# for he_inv's Goldschmidt, so 1/Sum overshoots and detonates (sftmx_out ~1e+86). Checked
# first in stage_07_softmax; same domain/encoding as softmax2, so no re-encode.
# Not predictable from ranges — populate from a forward run that detonates.
SOFTMAX3_LAYERS = frozenset()

# Union = all wide-exp-domain layers. Drives encoding scale (1/1024 vs 1/512), stage_07
# level handling, and the plot rescale -- NOT dispatch (that uses the two sets above).
WIDE_SOFTMAX_LAYERS = SOFTMAX2_LAYERS | SOFTMAX3_LAYERS

# Encoder layers that need the wide-range output layernorm (he_layernorm3) instead of he_layernorm2.
WIDE_LAYERNORM_LAYERS = frozenset({2, 4})

MODEL_DIR = "./distilled-model"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
