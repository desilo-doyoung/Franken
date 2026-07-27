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

History: the exact-GELU 8-layer student needed SOFTMAX2_LAYERS = {1} and
WIDE_LAYERNORM_LAYERS = {6}; the earlier 12-layer ``finetuned_models/mrpc`` model
needed SOFTMAX2_LAYERS = {2} and WIDE_LAYERNORM_LAYERS = {9, 10}.
"""

NUM_LAYERS = 8

# Layers whose attention softmax uses the CGF (cumulant) approximation instead of THOR's
# exact-softmax poly. CGF is unnormalized/division-free, so he_inv/update_inv_D (and
# SOFTMAX2/3) are bypassed -- see he_softmax_cgf and EXECUTION_NOTES.md 8. The distilled
# quad+cgf student uses CGF on every layer.
CGF_SOFTMAX_LAYERS = frozenset(range(NUM_LAYERS))

# Non-CGF (exact-softmax) dispatch: he_softmax2 (wide he_exp2 [-70,70]) and its he_inv-stable
# subset he_softmax3. Ignored on the CGF path.
SOFTMAX2_LAYERS = frozenset({1, 2})
SOFTMAX3_LAYERS = frozenset({4})

# Wide-softmax layers -> K-weight softmax_scale 1/1024 (stored 1/128 -> s_u=1/64) in
# encode_weights.py, plus the stage_07 plot rescale / level handling. CGF assumes s_u=1/64
# everywhere (he_softmax_cgf / he_exp_cgf), so all CGF layers must be wide. Non-CGF models
# fall back to the measured SOFTMAX2|SOFTMAX3 set.
WIDE_SOFTMAX_LAYERS = CGF_SOFTMAX_LAYERS if CGF_SOFTMAX_LAYERS else (SOFTMAX2_LAYERS | SOFTMAX3_LAYERS)

# Layers using the wide layernorm (he_layernorm3, var<=2500) vs he_layernorm2 (var<=150),
# for BOTH stage_11 (LN1) and stage_16 (LN2). Unnormalized CGF inflates the LayerNorm-input
# variance, so more layers qualify than the exact model's {3,6}. Measured (measure_ranges.py,
# CGF, val+test): LN1 breaches {0,2,3}, LN2 breaches {3,6}. NB these are input-dependent and
# only fit target-idx 0 -- see EXECUTION_NOTES.md 8 (range-penalty re-distill is the real fix).
WIDE_LAYERNORM_LAYERS = frozenset({0, 2, 3, 6})

MODEL_DIR = "./distilled-model"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
