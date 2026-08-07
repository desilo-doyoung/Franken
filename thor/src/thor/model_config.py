"""Model-specific configuration for the HE BERT inference pipeline.

The active model is the 8-layer distilled BERT (``distilled-model/``): a student
distilled from a 12-layer teacher and fine-tuned on MRPC. Its FFN activation is
the MPCFormer quadratic GELU (``0.125 x^2 + 0.25 x + 0.5``); ``he.stage_13_gelu``
computes that quad directly (config.json has ``"activation": "quad"``), and the
plaintext reference is patched to match (``utils.load_model``). It is a standard
HF ``BertForSequenceClassification`` with 8 encoder layers, a pooler, and a
2-class ``classifier`` head (quad test acc/F1 = 0.8249/0.8733); its weight keys
are HF-name-matched, so it loads with ``strict=False`` (no missing/unexpected).

The per-layer sets below select the polynomial approximation each layer uses. They are
MODEL-SPECIFIC: swapping the model or its ops requires re-measuring with
``thor/measure_ranges.py`` (over ALL 128 token slots, on val AND test) and updating them.
"""

NUM_LAYERS = 8

# Layers using the CGF (cumulant) softmax instead of THOR's exact-softmax poly. CGF is
# unnormalized/division-free, so he_inv/update_inv_D are bypassed (EXECUTION_NOTES.md 8).
CGF_SOFTMAX_LAYERS = frozenset(range(NUM_LAYERS))

# Exact-softmax dispatch for layers NOT in CGF_SOFTMAX_LAYERS: he_softmax2 (wide he_exp2 [-70,70])
# and its he_inv-stable subset he_softmax3.
SOFTMAX2_LAYERS = frozenset({1, 2})
SOFTMAX3_LAYERS = frozenset({4})

# Wide layers -> K-weight softmax_scale 1/1024 (stored 1/128 -> s_u = 1/64) in encode_weights, plus
# the stage_07 level handling and forward.py's plot rescale. A union, not a choice: CGF hardcodes
# s_u = 1/64 so every CGF layer is wide, and an exact layer is wide iff it needs he_exp2's range.
WIDE_SOFTMAX_LAYERS = CGF_SOFTMAX_LAYERS | SOFTMAX2_LAYERS | SOFTMAX3_LAYERS

# he_layernorm variant per LayerNorm call: 0|1|2|3, costing 6/5/6/7 he_invsqrt iterations. LN1
# (stage_11) and LN2 (stage_16) get their own map because their ranges differ ~100x. Admissible
# var_x per variant is measure_ranges.LAYERNORM_VARIANTS -- paste both maps from that script after
# any model change. LN1 needs ln0's wide band: no stock variant holds 0.035..12.1 (EXECUTION_NOTES 8.4).
LN1_VARIANT = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
LN2_VARIANT = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 2}

MODEL_DIR = "./distilled-model"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
