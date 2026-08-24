"""Per-deployment configuration for the HE BERT inference pipeline.

The layer sets below select higher-range polynomial approximations for the few layers whose
activations exceed the default domains. They are a property of the CHECKPOINT, not of the depth:
the exact-GELU 8L student needed softmax2={1} / wide_layernorm={6} where the quad 8L student needs
{1,2} / {3,6}, and the earlier 12-layer ``finetuned_models/mrpc`` needed {2} / {9,10}. So _BUILDS is
keyed by the franken recipe that trained the checkpoint, and NUM_LAYERS derives from the build rather
than being set alongside it (they used to drift apart).

Re-measure with ``thor/measure_ranges.py`` after swapping the model -- it prints a ready Build(...).
Measured ranges and the FHE verification per build: EXECUTION_NOTES.md §0 and §9.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Build:
    num_layers: int
    softmax2: frozenset[int]  # he_softmax2: wide exp domain [-70,70] vs he_softmax1's [-27,22]
    softmax3: frozenset[int]  # he_softmax2 + exp_scale: sum-of-exps too small for he_inv (§5)
    wide_layernorm: frozenset[int]  # he_layernorm3 (var<=2500) instead of he_layernorm2 (<=150)

    def __post_init__(self):
        # stage_07_softmax tests softmax3 first, so an overlap silently dead-codes the softmax2 entry.
        if overlap := self.softmax2 & self.softmax3:
            raise ValueError(f"softmax2 and softmax3 must be disjoint; both contain {sorted(overlap)}")


_BUILDS = {
    "depth8_quad_dom32": Build(8, frozenset({1, 2}), frozenset({4}), frozenset({3, 6})),
    "depth6_quad_dom32": Build(6, frozenset(), frozenset({1}), frozenset({2, 4})),
}

# The one line to edit when deploying a different student. Must match distilled-model/ --
# utils.load_model asserts num_hidden_layers against NUM_LAYERS.
DEPLOYMENT = "depth6_quad_dom32"

_build = _BUILDS[DEPLOYMENT]

NUM_LAYERS = _build.num_layers
SOFTMAX2_LAYERS = _build.softmax2
SOFTMAX3_LAYERS = _build.softmax3
WIDE_LAYERNORM_LAYERS = _build.wide_layernorm

# Union = every wide-exp-domain layer. Drives the encoding scale (1/1024 vs 1/512), stage_07 level
# handling and the plot rescale -- NOT dispatch, which uses the two sets above.
WIDE_SOFTMAX_LAYERS = SOFTMAX2_LAYERS | SOFTMAX3_LAYERS

MODEL_DIR = "./distilled-model"
MODEL_PATH = f"{MODEL_DIR}/model.safetensors"
