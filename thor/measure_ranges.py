"""Measure the per-layer activation ranges an FHE consumer needs to pick approximation domains.

For the model in --model-dir, over the MRPC splits, report per encoder layer:
  - softmax:   min/max pre-softmax attention score (Q.K^T / sqrt(d), pre-mask)
  - layernorm: max per-token variance of BOTH LayerNorm inputs (LN1 and LN2 dispatch
               independently and their variances differ by ~1000x)
  - gelu:      min/max FFN pre-activation -- the only measurement of stripe's composed-tanh
               divergence wall that exists anywhere
plus the pooler's pre-tanh distribution, and a suggested per-layer build for each consumer.

⚠️ Padding: masking is applied AFTER exp and LayerNorm normalizes ALL token slots, so the
domains must cover PAD positions too (EXECUTION_NOTES.md §4, §8.4 -- the 1861-vs-4.1 lesson).
This pads to --max-seq-len like the runtimes do. `--valid-only` reproduces the old, wrong
valid-tokens-only numbers for A/B.

Standalone (no thor/desilofhe import); run with any env that has torch+transformers+datasets:
    python thor/measure_ranges.py --model-dir outputs/bert_exact6l_nopen/student \
        --max-seq-len 64 --splits validation test --consumer stripe --device 2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

# --- Franken thor/ (src/thor/he.py) -------------------------------------------------------
# he_softmax1 -> he_exp1, he_softmax2/3 -> he_exp2. Franken KEEPS the `max_x < 30` dispatch,
# so these labels really do select the polynomial.
THOR_SOFTMAX1_BOX = (-27.2493, 21.72692)
THOR_SOFTMAX2_BOX = (-70.0, 70.0)

# --- beyond-THOR stripe_bert_standalone.py ------------------------------------------------
# ⚠️ These assume stripe:1582-1585's `max_x < 30` dispatch is UN-COMMENTED. While it is commented
# every call lands on he_exp2, and softmax1's l=2 then yields exp(z/2) instead of exp(z) -- see the
# 8*n*l note in report_stripe(). min_x/max_x are inert to the domain either way: they are consumed
# only as mid_x = (min+max)/2, which is 0 for both variants, so they select the polynomial and
# nothing else. Both boxes were measured, not read off the labels.
#
# Measured on the COMPOSED path (he_exp poly, its n squarings, and update_inv_D's l squarings),
# as abs error over the max target:
#   exp1 (l=2, u=z/32): 8e-05 at |score|<=27, 5e-04 at 30, 2e-03 at 32.  slope exactly 1.0000
#   exp2 (l=4, u=z/64): 6.7e-02 already at |score|<=17.                  slope 0.998 (!)
# So exp1 is ~750x more accurate AND exp2 has a 0.2% error in the exponent itself -- it computes
# exp(0.998 z), a systematic temperature error. Prefer softmax1 wherever the scores fit; he.py's
# [-27.25, 21.73] label is conservative, and exceeding it is NOT a reason to move to softmax2.
# Both degrade gracefully, unlike the GELU and pooler fits below, which are cliffs.
STRIPE_EXP1_BOX = (-30.0, 30.0)
STRIPE_SOFTMAX_MAX = 70.0

# he_tanh_single_for_gelu (deg-31 o deg-27 on u = z/64), measured by bisection. This is a
# DIVERGENCE CLIFF, not a fidelity slope: err 0.0105 at z=-70, 61.8 at -71, 3.6e+11 at -72.
STRIPE_GELU_SAFE = (-70.4803, 151.0799)

# he_tanh_single_for_pooler on z = pre_tanh/40. One out-of-domain slot blows the ciphertext scale
# and the next bootstrap. beyond-THOR's stripe_forward.py refit this (deg-15 o deg-19, fit on
# |z| <= 1.0): the wall moved out to 41.5 and the in-domain error fell 101x. Franken's own
# thor/src/thor/he.py still carries the old deg-15 o deg-15 fit, whose wall is 39.9146 -- so this
# constant is the STRICTER of the two, and stays valid for both consumers.
POOLER_TANH_WALL = 39.9146

# he_layernorm{1,2,3} (min_var, max_var). he_invsqrt consumes 2*iters-1 levels and iters follows
# from min_var/max_var: 5/6/7 iters -> 9/11/13 levels (verified by replaying the k=np.roots loop).
LN_PARAMS = {1: (0.15, 10.0), 2: (0.2, 150.0), 3: (0.75, 2500.0)}
LN_W_BUFFER = 1.05


def ln_band(variant: int) -> tuple[float, float]:
    """Admissible plaintext per-token var_x, taken as the raw (min_var, max_var) args.

    A tempting looser reading -- the mask scales by 1/sqrt(1.05*max_var*n^2) and ln2/ln3 halve it
    again, so the ciphertext variance is 4x smaller and the band 4x wider -- is NOT used: it does
    not reproduce the one build known to work. `bert_quad6l` runs wide_ln={2,4} on `main`, and L2's
    variance is 310.6, which the 4x band (ceiling 630) would leave on ln2. Over the ceiling
    he_invsqrt overshoots and its squaring amplifies catastrophically, so the margin is not ours
    to spend on an unverified derivation.
    """
    lo, hi = LN_PARAMS[variant]
    return lo, hi


# The constraint is ONE-SIDED. Over the ceiling he_invsqrt overshoots and its squaring amplifies
# catastrophically; under the floor it merely under-converges -- §8.5 measured a student below
# ln1's floor on ~100% of tokens with 0 divergences. Pick by ceiling, accept floor breaches.
HEADROOM_OK = 3.0  # PASS below this ratio of the ceiling; MARGINAL above it


@dataclass
class LayerStats:
    score_min: float = math.inf
    score_max: float = -math.inf
    ln1_var: float = 0.0
    ln2_var: float = 0.0
    pre_min: float = math.inf
    pre_max: float = -math.inf

    def merge(self, o: "LayerStats") -> None:
        self.score_min = min(self.score_min, o.score_min)
        self.score_max = max(self.score_max, o.score_max)
        self.ln1_var = max(self.ln1_var, o.ln1_var)
        self.ln2_var = max(self.ln2_var, o.ln2_var)
        self.pre_min = min(self.pre_min, o.pre_min)
        self.pre_max = max(self.pre_max, o.pre_max)


class _QuadGELU(nn.Module):
    """MPCFormer quadratic GELU: 0.125 x^2 + 0.25 x + 0.5 (an nn.Module because HF
    stores intermediate_act_fn as a child module)."""

    def forward(self, x):
        return 0.125 * x * x + 0.25 * x + 0.5


def load_model(model_dir: Path):
    raw = json.loads((model_dir / "config.json").read_text())
    cfg = BertConfig(
        num_hidden_layers=raw["num_hidden_layers"],
        hidden_size=raw["hidden_size"],
        num_attention_heads=raw["num_attention_heads"],
        intermediate_size=raw["intermediate_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        vocab_size=raw["vocab_size"],
        type_vocab_size=raw["type_vocab_size"],
        num_labels=raw.get("num_labels", 2),
        pad_token_id=raw.get("pad_token_id", 0),
        layer_norm_eps=raw.get("layer_norm_eps", 1e-12),
        output_hidden_states=True,
    )
    model = BertForSequenceClassification(cfg)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(model_dir / "model.safetensors")))
    model.eval()
    act = raw.get("activation", "exact")
    # Silently falling through to HF's default gelu is right for `exact` and WRONG for anything
    # else; an unmeasured activation would report another model's ranges.
    if act == "quad":
        for layer in model.bert.encoder.layer:
            layer.intermediate.intermediate_act_fn = _QuadGELU()
    elif act != "exact":
        raise SystemExit(
            f"activation {act!r} has no plaintext replica here; only 'exact' and 'quad' do. "
            "Add one before trusting any number this script prints."
        )
    return model, cfg, act


@torch.no_grad()
def layer_ranges(model, hidden_states, layer_idx, attn_mask, valid_only: bool) -> LayerStats:
    L = model.bert.encoder.layer[layer_idx]
    a = L.attention.self

    def heads(x):
        return x.view(*x.shape[:-1], a.num_attention_heads, a.attention_head_size).transpose(1, 2)

    q, k, v = heads(a.query(hidden_states)), heads(a.key(hidden_states)), heads(a.value(hidden_states))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a.attention_head_size)  # pre-mask

    ext = model.get_extended_attention_mask(attn_mask, hidden_states.shape)
    probs = torch.softmax(scores + ext, dim=-1)
    ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(*hidden_states.shape[:-1], a.all_head_size)

    ln1_in = L.attention.output.dense(ctx) + hidden_states
    ln1 = L.attention.output.LayerNorm(ln1_in)
    preact = L.intermediate.dense(ln1)
    inter = L.intermediate.intermediate_act_fn(preact)
    ln2_in = L.output.dense(inter) + ln1

    var1 = ln1_in.var(dim=-1, unbiased=False)
    var2 = ln2_in.var(dim=-1, unbiased=False)
    pre, sc = preact, scores
    if valid_only:  # the old, wrong behaviour: PAD slots dropped
        keep = attn_mask.bool()
        var1, var2, pre = var1[keep], var2[keep], pre[keep]
        sc = scores * keep[:, None, None, :] * keep[:, None, :, None]

    return LayerStats(
        score_min=sc.amin().item(),
        score_max=sc.amax().item(),
        ln1_var=var1.amax().item(),
        ln2_var=var2.amax().item(),
        pre_min=pre.amin().item(),
        pre_max=pre.amax().item(),
    )


def _verdict(value: float, ceiling: float) -> tuple[str, float]:
    ratio = ceiling / abs(value) if value else math.inf
    if abs(value) > ceiling:
        return "FAIL", ratio
    return ("PASS" if ratio >= HEADROOM_OK else "MARGINAL"), ratio


def pick_ln(var: float) -> int | None:
    for variant in (1, 2, 3):
        if var <= ln_band(variant)[1]:
            return variant
    return None


def report_thor(stats: list[LayerStats]) -> None:
    print("\n=== consumer: Franken thor/ (src/thor/he.py) ===")
    print("  he_exp1/he_exp2 dispatch is LIVE here, so the softmax label selects the polynomial.")
    wide_sm, wide_ln = [], []
    for i, s in enumerate(stats):
        narrow = THOR_SOFTMAX1_BOX[0] <= s.score_min and s.score_max <= THOR_SOFTMAX1_BOX[1]
        box = THOR_SOFTMAX1_BOX if narrow else THOR_SOFTMAX2_BOX
        if not narrow:
            wide_sm.append(i)
        worst = max(abs(s.score_min) / abs(box[0]), abs(s.score_max) / abs(box[1]))
        sm = "FAIL" if worst > 1 else "PASS"
        lnv = pick_ln(s.ln2_var)
        if lnv and lnv > 2:
            wide_ln.append(i)
        ln = "FAIL" if lnv is None else f"ln{lnv} ({ln_band(lnv)[1] / max(s.ln2_var, 1e-9):.1f}x)"
        print(f"  L{i}  softmax{'1' if narrow else '2'} {sm} ({1 / worst:.2f}x)   LN2 {ln}")
    print(f"\n  Build({len(stats)}, softmax2=frozenset({set(wide_sm) or ''}), "
          f"softmax3=frozenset(), wide_ln=frozenset({set(wide_ln) or ''}))")
    print("  softmax3 starts EMPTY: he_inv overshoot is not predictable from ranges (§5) and only")
    print("  shows up across a BATCH. Move any layer that detonates at runtime into it.")


def report_stripe(stats: list[LayerStats]) -> None:
    print("\n=== consumer: beyond-THOR stripe_bert_standalone.py ===")
    print(f"  softmax: softmax1 (l=2, he_exp1, box {STRIPE_EXP1_BOX}) where the scores fit -- it runs one")
    print(f"           update_inv_D round instead of two. Else softmax2 (l=4, he_exp2, max {STRIPE_SOFTMAX_MAX:g}).")
    print("           REQUIRES un-commenting stripe:1582-1585; the min_x/max_x labels stay inert either way.")
    print(f"  gelu:    composed-tanh fit is safe only on ({STRIPE_GELU_SAFE[0]:.2f}, {STRIPE_GELU_SAFE[1]:.2f}) "
          "-- a divergence cliff.")
    ln1s, ln2s, sms = [], [], []
    for i, s in enumerate(stats):
        narrow = STRIPE_EXP1_BOX[0] <= s.score_min and s.score_max <= STRIPE_EXP1_BOX[1]
        sms.append(1 if narrow else 2)
        sm, sm_r = _verdict(s.score_max, STRIPE_EXP1_BOX[1] if narrow else STRIPE_SOFTMAX_MAX)
        lo_r = STRIPE_GELU_SAFE[0] / s.pre_min if s.pre_min < 0 else math.inf
        hi_r = STRIPE_GELU_SAFE[1] / s.pre_max if s.pre_max > 0 else math.inf
        g = "FAIL" if min(lo_r, hi_r) < 1 else ("PASS" if min(lo_r, hi_r) >= HEADROOM_OK else "MARGINAL")
        v1, v2 = pick_ln(s.ln1_var), pick_ln(s.ln2_var)
        ln1s.append(v1 or 0)
        ln2s.append(v2 or 0)
        ln2_txt = (f"LN2 ln{v2} ({ln_band(v2)[1] / max(s.ln2_var, 1e-9):.1f}x)" if v2
                   else f"LN2 FAIL (var {s.ln2_var:.0f}, no variant covers it)")
        print(f"  L{i}  softmax{sms[i]} {sm:8s} ({sm_r:5.2f}x)   "
              f"GELU {g:8s} (neg {lo_r:5.2f}x, pos {hi_r:5.2f}x)   "
              f"LN1 ln{v1 or '?'}  {ln2_txt}")
    print(f"\n  Build(num_layers={len(stats)}, softmax={tuple(sms)},")
    print(f"        exp_scale=({', '.join(['1.0'] * len(stats))},),")
    print(f"        ln1={tuple(ln1s)}, ln2={tuple(ln2s)})")
    print("  The variant is pinned by ARITHMETIC, not precision: he_exp's poly has slope 8, doubled")
    print("  once per squaring, and update_inv_D squares the running exp l times, so the total is")
    print("  8*n*l. exp1 (u = z/32) needs 32 -> l=2; exp2 (u = z/64) needs 64 -> l=4. Pairing a")
    print("  variant with the other polynomial silently halves or doubles the softmax exponent.")
    print("  exp_scale defaults to 1.0 and is NOT range-predictable -- see §5/§9, it needs a batch.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="thor/distilled-model")
    p.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--consumer", choices=["thor", "stripe", "both"], default="both")
    p.add_argument("--valid-only", action="store_true", help="reproduce the old, PAD-blind numbers")
    p.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    args = p.parse_args()

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")
    model_dir = Path(args.model_dir)
    model, cfg, act = load_model(model_dir)
    model.to(device)
    pad = "valid-only (OLD, WRONG)" if args.valid_only else f"max_length={args.max_seq_len}"
    print(f"model: {model_dir}  layers={cfg.num_hidden_layers}  activation={act}  device={device}")
    print(f"splits={args.splits}  padding={pad}  batch={args.batch_size}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    nL = cfg.num_hidden_layers
    stats = [LayerStats() for _ in range(nL)]
    pooler_abs = []

    for split in args.splits:
        ds = load_dataset("nyu-mll/glue", "mrpc")[split]
        for start in range(0, len(ds), args.batch_size):
            chunk = ds[start : start + args.batch_size]
            batch = tok(
                chunk["sentence1"], chunk["sentence2"],
                truncation=True, max_length=args.max_seq_len,
                padding="max_length", return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model.bert(**batch)
                for li in range(nL):
                    stats[li].merge(
                        layer_ranges(model, out.hidden_states[li], li, batch["attention_mask"], args.valid_only)
                    )
                pooler_abs.append(model.bert.pooler.dense(out.last_hidden_state[:, 0]).abs().flatten().cpu())

    print(f"\n{'layer':>5}{'score min':>11}{'score max':>11}{'ln1 var':>10}{'ln2 var':>11}"
          f"{'pre min':>10}{'pre max':>10}")
    print("-" * 68)
    for i, s in enumerate(stats):
        print(f"{i:>5}{s.score_min:>11.2f}{s.score_max:>11.2f}{s.ln1_var:>10.2f}{s.ln2_var:>11.1f}"
              f"{s.pre_min:>10.2f}{s.pre_max:>10.2f}")

    pa = torch.cat(pooler_abs)
    over = int((pa > POOLER_TANH_WALL).sum())
    print(f"\npooler pre-tanh |z|: p50 {pa.median():.2f}  p90 {pa.quantile(0.9):.2f}  max {pa.max():.2f}"
          f"   over {POOLER_TANH_WALL:.2f}: {over}/{pa.numel()}")
    if over:
        print(f"  🚨 {over} slot(s) past the pooler fit's wall -- z/40 = 1.034 gives 4.1e+13, which blows")
        print("     the ciphertext scale and the following bootstrap, corrupting token 0 with it.")

    if args.consumer in ("thor", "both"):
        report_thor(stats)
    if args.consumer in ("stripe", "both"):
        report_stripe(stats)


if __name__ == "__main__":
    main()
