"""Measure the per-layer activation ranges THOR needs to pick approximation domains.

For the model in --model-dir, over the MRPC validation set (valid tokens only),
report per encoder layer:
  - softmax:   max |pre-softmax attention score| (Q.K^T / sqrt(d), pre-mask)
  - layernorm: max per-token variance of the 2nd-LayerNorm input (output_dense +
               layernorm_1_output), across the hidden dim, per valid token

and suggest SOFTMAX2_LAYERS / WIDE_LAYERNORM_LAYERS for thor/src/thor/model_config.py.

Domain references (thor/src/thor/he.py):
  he_softmax1 [-27, 22]   he_softmax2 [-70, 70]
  he_layernorm2 var<=150  he_layernorm3 var<=2500

Standalone (no thor/desilofhe import); run with any env that has torch+transformers+datasets:
    python thor/measure_ranges.py --model-dir thor/distilled-model --device 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification, DataCollatorWithPadding

# Exact domains the fixed exp polynomials in he.py are fit to (he_softmax1/he_softmax2).
SOFTMAX1_DOMAIN = (-27.2493, 21.72692)  # he_softmax1 -> he_exp1
SOFTMAX2_DOMAIN = (-70.0, 70.0)  # he_softmax2 -> he_exp2
LAYERNORM2_MAX_VAR = 150.0  # he_layernorm2 ceiling; above -> he_layernorm3 (<=2500)
LAYERNORM3_MAX_VAR = 2500.0


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
    if act == "quad":
        for layer in model.bert.encoder.layer:
            layer.intermediate.intermediate_act_fn = _QuadGELU()
    return model, cfg, act


@torch.no_grad()
def layer_ranges(model, hidden_states, layer_idx, attn_mask, device):
    """(score min/max, CGF-logit min/max, max per-token LN2-input variance) for one
    layer, valid tokens only.

    The CGF-logit range is the domain the FHE exp (he_exp1/he_exp2) actually sees for
    the distilled CGF student -- NOT the raw score range. CGF recenters each row by
    mu + var/2 + log n_vis (the log-sum-exp estimate), so the top logit sits near
    log(max prob) ~ 0 and the range is the score spread shifted down. This is the
    quantity that must fit he_exp1's [-27.25, 21.73] / he_exp2's [-70, 70]."""
    n = int(attn_mask.sum().item())  # valid length (mask is 1s then 0s after collation)
    L = model.bert.encoder.layer[layer_idx]
    a = L.attention.self

    def heads(x):
        return x.view(*x.shape[:-1], a.num_attention_heads, a.attention_head_size).transpose(1, 2)

    q = heads(a.query(hidden_states))
    k = heads(a.key(hidden_states))
    v = heads(a.value(hidden_states))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a.attention_head_size)  # (1,H,S,S) pre-mask
    valid = scores[..., :n, :n]  # valid query x valid key (all keys visible => n_vis = n)
    score_min, score_max = valid.min().item(), valid.max().item()

    # CGF logits over valid positions (franken CGFSoftmax; population var, n_vis = n)
    mu = valid.mean(dim=-1, keepdim=True)
    var = valid.var(dim=-1, unbiased=False, keepdim=True)
    cgf_logits = valid - mu - 0.5 * var - math.log(n)
    cgf_min, cgf_max = cgf_logits.min().item(), cgf_logits.max().item()

    ext = model.get_extended_attention_mask(attn_mask, hidden_states.shape).to(device)
    # CGF softmax (unnormalized) for the downstream LN2-variance measurement
    m = (ext == 0).to(scores.dtype)
    x_vis = scores * m
    n_vis = m.sum(dim=-1, keepdim=True)
    mu_f = x_vis.sum(dim=-1, keepdim=True) / n_vis
    var_f = (x_vis**2).sum(dim=-1, keepdim=True) / n_vis - mu_f**2
    probs = torch.exp(scores - mu_f - 0.5 * var_f - torch.log(n_vis)) * m
    ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(*hidden_states.shape[:-1], a.all_head_size)
    att_dense = L.attention.output.dense(ctx)
    ln1_in = att_dense + hidden_states  # input to the 1st LayerNorm (stage_11)
    var_ln1 = ln1_in[0, :n].var(dim=-1, unbiased=False).max().item()
    ln1 = L.attention.output.LayerNorm(ln1_in)
    inter = L.intermediate.intermediate_act_fn(L.intermediate.dense(ln1))
    ln2_in = L.output.dense(inter) + ln1  # input to the 2nd LayerNorm (stage_16)
    var_ln2 = ln2_in[0, :n].var(dim=-1, unbiased=False).max().item()
    return score_min, score_max, cgf_min, cgf_max, var_ln1, var_ln2


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
    model, cfg, act = load_model(model_dir)
    model.to(device)
    print(f"model: {model_dir}  layers={cfg.num_hidden_layers}  activation={act}  device={device}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("nyu-mll/glue", "mrpc")[args.split]
    ds = ds.map(
        lambda b: tok(b["sentence1"], b["sentence2"], truncation=True, max_length=args.max_seq_len), batched=True
    ).with_format("torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    coll = DataCollatorWithPadding(tok)

    nL = cfg.num_hidden_layers
    smin = [0.0] * nL
    smax = [0.0] * nL
    cmin = [0.0] * nL
    cmax = [0.0] * nL
    var1 = [0.0] * nL  # layernorm-1 input variance (stage_11)
    var2 = [0.0] * nL  # layernorm-2 input variance (stage_16)
    for ex in ds:
        batch = coll([{k: ex[k] for k in ("input_ids", "token_type_ids", "attention_mask")}])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.bert(**batch)  # output_hidden_states=True via config
        for li in range(nL):
            lo, hi, clo, chi, v1, v2 = layer_ranges(model, out.hidden_states[li], li, batch["attention_mask"], device)
            smin[li] = min(smin[li], lo)
            smax[li] = max(smax[li], hi)
            cmin[li] = min(cmin[li], clo)
            cmax[li] = max(cmax[li], chi)
            var1[li] = max(var1[li], v1)
            var2[li] = max(var2[li], v2)

    wide_softmax = [li for li in range(nL) if cmax[li] > SOFTMAX1_DOMAIN[1] or cmin[li] < SOFTMAX1_DOMAIN[0]]
    over_sm2 = [li for li in wide_softmax if cmax[li] > SOFTMAX2_DOMAIN[1] or cmin[li] < SOFTMAX2_DOMAIN[0]]
    # a layer needs the wide layernorm (he_layernorm3) if EITHER of its two LayerNorm
    # inputs exceeds he_layernorm2's var<=150. Unnormalized CGF inflates these vs the
    # exact softmax, so both LN1 (stage_11) and LN2 (stage_16) must be checked.
    vmax = [max(var1[li], var2[li]) for li in range(nL)]
    wide_ln = [li for li in range(nL) if vmax[li] > LAYERNORM2_MAX_VAR]
    over_ln = [li for li in range(nL) if vmax[li] > LAYERNORM3_MAX_VAR]

    print(f"\nsplit={args.split}  (CGF softmax; LN1=stage_11 input var, LN2=stage_16 input var)")
    print(
        f"{'layer':>5}{'cgf lo':>9}{'cgf hi':>9}{'softmax':>10}"
        f"{'LN1 var':>10}{'LN2 var':>10}{'layernorm':>14}"
    )
    print("-" * 72)
    for li in range(nL):
        sfx = "cgf2" if li in wide_softmax else "cgf1"
        lnx = "he_layernorm3" if li in wide_ln else "he_layernorm2"
        flags = ("  OVER-SM!" if li in over_sm2 else "") + ("  OVER-LN2500!" if li in over_ln else "")
        print(
            f"{li:>5}{cmin[li]:>9.2f}{cmax[li]:>9.2f}{sfx:>10}"
            f"{var1[li]:>10.1f}{var2[li]:>10.1f}{lnx:>14}{flags}"
        )

    print("\n# suggested thor/src/thor/model_config.py (CGF path)")
    print(f"WIDE_SOFTMAX_LAYERS   = frozenset({set(wide_softmax) if wide_softmax else set()})")
    print(f"WIDE_LAYERNORM_LAYERS = frozenset({set(wide_ln) if wide_ln else set()})")
    if over_ln:
        print(f"# WARNING: layers {over_ln} exceed he_layernorm3's var<=2500 — a wider layernorm poly is needed.")


if __name__ == "__main__":
    main()
