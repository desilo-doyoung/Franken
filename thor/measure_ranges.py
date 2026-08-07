"""Measure the per-layer activation ranges THOR needs to pick approximation domains.

For the model in --model-dir, over an MRPC split, report per encoder layer the ranges that
decide THOR's approximation domains, and suggest WIDE_SOFTMAX_LAYERS / LN1_VARIANT /
LN2_VARIANT for thor/src/thor/model_config.py.

Everything is measured over ALL 128 token slots, not just the valid ones: THOR pads every example
to 128 and its he_layernorm / exp run on every slot, masking only afterwards, so a PAD slot that
breaches he_invsqrt's ceiling detonates the layer. A valid-token-only measurement hid exactly that
(4.1 valid vs 1861 all-slot). See EXECUTION_NOTES.md 8.3/8.4.

Needs torch+transformers+datasets and thor/src on the path (thor.utils pulls no desilofhe):
    python thor/measure_ranges.py --model-dir thor/distilled-model --device 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, str(Path(__file__).parent / "src"))
from thor.utils import load_model as load_thor_model  # noqa: E402

# Exp-poly domains in he.py: the exact path feeds raw scores to he_exp1/2, the CGF path feeds
# CGF logits to he_exp_cgf.
SOFTMAX1_DOMAIN = (-27.2493, 21.72692)
SOFTMAX2_DOMAIN = (-70.0, 70.0)
CGF_EXP_DOMAIN = (-26.0, 6.0)
# Admissible plaintext var_x per he_layernorm variant, cheapest first, and its he_invsqrt iteration
# count. These are NOT he.py's (min_var, max_var) args: the mask scales x by
# M = 1/sqrt(1.05*max_var*n^2) (and by M/2 when halve_mask), so he_invsqrt sees
# var_x/(1.05*max_var), /4 more with halve_mask, and needs it in [min_var/max_var, 1].
# NOT nested -- ln3 has the highest floor too, so a wide band can fit none of them.
LAYERNORM_VARIANTS = ((1, 0.158, 10.5), (0, 0.032, 31.5), (2, 0.84, 630.0), (3, 3.15, 10500.0))
# One-sided: over the ceiling he_invsqrt diverges catastrophically, under the floor it merely
# under-converges. So demand 3x headroom on the ceiling and treat the floor as advisory.
HE_VAR_INFLATION = 3.0


def ln_variant(var_lo: float, var_hi: float) -> tuple[int | None, bool]:
    """(cheapest variant whose CEILING covers var_hi with headroom, whether its floor is met).
    Floors rise with ceilings, so the first ceiling-fitting variant is also the floor answer."""
    for variant, min_var, max_var in LAYERNORM_VARIANTS:
        if var_hi * HE_VAR_INFLATION <= max_var:
            return variant, var_lo >= min_var
    return None, False


def load_model(model_dir: Path):
    """Reuses thor.utils.load_model so the measured model IS the one THOR references -- it installs
    the quad GELU and, for softmax=cgf, the CGF attention. Measuring a stock-softmax forward of a
    CGF-trained model silently reports another model's ranges (EXECUTION_NOTES.md 1)."""
    raw = json.loads((model_dir / "config.json").read_text())
    model = load_thor_model("mrpc", str(model_dir / "model.safetensors"))
    model.config.output_hidden_states = True
    return model, model.config, raw.get("activation", "exact"), raw.get("softmax", "exact")


@torch.no_grad()
def layer_ranges(model, hidden_states, layer_idx, attn_mask, device, softmax="cgf"):
    """Per-layer exp-input range, max softmax row-sum, and min/max LN1/LN2-input variance.

    ``softmax`` picks the op for the row-sum / LayerNorm measurements: ``cgf`` for a CGF-distilled
    student, ``exact`` for one trained with the exact softmax (the row-sum == 1 reference).

    The CGF-logit range, NOT the raw score range, is what he_exp_cgf sees: CGF recenters each row by
    mu + var/2 + log n_vis, so the top logit sits near log(max prob) ~ 0."""
    n = int(attn_mask.sum().item())  # valid length (mask is 1s then 0s after collation)
    L = model.bert.encoder.layer[layer_idx]
    a = L.attention.self

    def heads(x):
        return x.view(*x.shape[:-1], a.num_attention_heads, a.attention_head_size).transpose(1, 2)

    q = heads(a.query(hidden_states))
    k = heads(a.key(hidden_states))
    v = heads(a.value(hidden_states))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a.attention_head_size)  # (1,H,S,S) pre-mask
    # ALL query rows x visible keys: FHE evaluates the exp on every one of the 128 query slots
    # (pad rows are only masked afterwards), so pad rows are part of the exp domain too.
    rows = scores[..., :n]
    score_min, score_max = rows.min().item(), rows.max().item()

    # CGF logits (franken CGFSoftmax; population var over visible keys, n_vis = n)
    mu = rows.mean(dim=-1, keepdim=True)
    var = rows.var(dim=-1, unbiased=False, keepdim=True)
    cgf_logits = rows - mu - 0.5 * var - math.log(n)
    cgf_min, cgf_max = cgf_logits.min().item(), cgf_logits.max().item()

    if softmax == "cgf":
        probs = torch.zeros_like(scores)
        probs[..., :n] = cgf_logits.exp()  # unnormalized, the op the student was distilled with
    else:
        ext = model.get_extended_attention_mask(attn_mask, hidden_states.shape).to(device)
        probs = torch.softmax(scores + ext, dim=-1)
    rowsum_max = probs.sum(dim=-1).max().item()  # all query rows, incl. pad (see above)
    ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(*hidden_states.shape[:-1], a.all_head_size)
    att_dense = L.attention.output.dense(ctx)
    ln1_in = att_dense + hidden_states  # input to the 1st LayerNorm (stage_11)
    ln1 = L.attention.output.LayerNorm(ln1_in)
    inter = L.intermediate.intermediate_act_fn(L.intermediate.dense(ln1))
    ln2_in = L.output.dense(inter) + ln1  # input to the 2nd LayerNorm (stage_16)
    # min as well as max: too small a variance also leaves he_invsqrt's band (see LAYERNORM_VARIANTS)
    v1 = ln1_in[0].var(dim=-1, unbiased=False)
    v2 = ln2_in[0].var(dim=-1, unbiased=False)
    return {
        "score": (score_min, score_max),
        "cgf": (cgf_min, cgf_max),
        "rowsum": rowsum_max,
        "ln1": (v1.min().item(), v1.max().item()),
        "ln2": (v2.min().item(), v2.max().item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="thor/distilled-model")
    p.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--split", default="validation", help="MRPC split to measure (validation|test)")
    p.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    args = p.parse_args()

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")
    model_dir = Path(args.model_dir)
    model, cfg, act, sftmx = load_model(model_dir)
    model.to(device)
    print(f"model: {model_dir}  layers={cfg.num_hidden_layers}  activation={act}  softmax={sftmx}  device={device}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("nyu-mll/glue", "mrpc")[args.split]
    ds = ds.map(
        lambda b: tok(b["sentence1"], b["sentence2"], truncation=True, max_length=args.max_seq_len), batched=True
    ).with_format("torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    # max_length, not batch-max: THOR always encodes 128 slots, and batch-max padding on these
    # 1-example batches pads nothing at all.
    coll = DataCollatorWithPadding(tok, padding="max_length", max_length=args.max_seq_len)

    nL = cfg.num_hidden_layers
    smin = [0.0] * nL  # raw pre-softmax score range: the exp domain on the EXACT path
    smax = [0.0] * nL
    cmin = [0.0] * nL  # CGF-logit range: the exp domain on the CGF path
    cmax = [0.0] * nL
    rsum = [0.0] * nL  # max softmax row-sum (1 for the exact softmax; unbounded for CGF)
    var1 = [0.0] * nL  # layernorm-1 input variance (stage_11), max over tokens
    var2 = [0.0] * nL  # layernorm-2 input variance (stage_16), max over tokens
    var1_lo = [float("inf")] * nL  # ... and min: he_layernorm also diverges BELOW min_var
    var2_lo = [float("inf")] * nL
    for ex in ds:
        batch = coll([{k: ex[k] for k in ("input_ids", "token_type_ids", "attention_mask")}])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.bert(**batch)  # output_hidden_states=True via config
        for li in range(nL):
            r = layer_ranges(model, out.hidden_states[li], li, batch["attention_mask"], device, sftmx)
            smin[li] = min(smin[li], r["score"][0])
            smax[li] = max(smax[li], r["score"][1])
            cmin[li] = min(cmin[li], r["cgf"][0])
            cmax[li] = max(cmax[li], r["cgf"][1])
            rsum[li] = max(rsum[li], r["rowsum"])
            var1_lo[li] = min(var1_lo[li], r["ln1"][0])
            var1[li] = max(var1[li], r["ln1"][1])
            var2_lo[li] = min(var2_lo[li], r["ln2"][0])
            var2[li] = max(var2[li], r["ln2"][1])

    elo, ehi = (cmin, cmax) if sftmx == "cgf" else (smin, smax)
    exp_domain = CGF_EXP_DOMAIN if sftmx == "cgf" else SOFTMAX1_DOMAIN
    wide_softmax = [li for li in range(nL) if ehi[li] > exp_domain[1] or elo[li] < exp_domain[0]]
    over_exp = [li for li in wide_softmax if ehi[li] > SOFTMAX2_DOMAIN[1] or elo[li] < SOFTMAX2_DOMAIN[0]]
    # LN1 (stage_11) and LN2 (stage_16) are separate he_layernorm calls whose bands differ by orders
    # of magnitude (attention output vs FFN/quad output), so each gets its own variant.
    ln1 = [ln_variant(var1_lo[li], var1[li]) for li in range(nL)]
    ln2 = [ln_variant(var2_lo[li], var2[li]) for li in range(nL)]

    print(f"\nsplit={args.split}  softmax={sftmx}  (LN1=stage_11 input var, LN2=stage_16 input var)")
    print(
        f"{'layer':>5}{'exp lo':>9}{'exp hi':>9}{'rowsum':>9}"
        f"{'LN1 var (min..max)':>21}{'ln?':>5}{'LN2 var (min..max)':>21}{'ln?':>5}"
    )
    print("-" * 86)
    for li in range(nL):
        cells = [f"{'X' if v is None else v}{'' if ok else '*'}" for v, ok in (ln1[li], ln2[li])]
        print(
            f"{li:>5}{elo[li]:>9.2f}{ehi[li]:>9.2f}{rsum[li]:>9.2f}"
            f"{var1_lo[li]:>10.3f}..{var1[li]:<10.1f}{cells[0]:>5}"
            f"{var2_lo[li]:>10.3f}..{var2[li]:<10.1f}{cells[1]:>5}" + ("  OVER-EXP!" if li in over_exp else "")
        )

    print("\n# suggested thor/src/thor/model_config.py  ('*' = under the variant's soft floor)")
    print(f"WIDE_SOFTMAX_LAYERS   = frozenset({set(wide_softmax) if wide_softmax else set()})")
    for tag, lnx in (("LN1", ln1), ("LN2", ln2)):
        print(f"{tag}_VARIANT = {dict(enumerate(v for v, _ in lnx))}")
    if any(v is None for v, _ in ln1 + ln2):
        print("# WARNING: 'None' above means the variance exceeds even he_layernorm3's ceiling (10500).")


if __name__ == "__main__":
    main()
