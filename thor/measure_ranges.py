"""Measure the per-layer activation ranges THOR needs to pick approximation domains.

For the model in --model-dir, over the MRPC validation set (valid tokens only),
report per encoder layer:
  - softmax:   max |pre-softmax attention score| (Q.K^T / sqrt(d), pre-mask)
  - layernorm: max per-token variance of the 2nd-LayerNorm input (output_dense +
               layernorm_1_output), across the hidden dim, per valid token

and suggest WIDE_SOFTMAX_LAYERS / WIDE_LAYERNORM_LAYERS for thor/src/thor/model_config.py.

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
from torch import nn
from datasets import load_dataset
from transformers import (AutoTokenizer, BertConfig, BertForSequenceClassification,
                          DataCollatorWithPadding)

# Exact domains the fixed exp polynomials in he.py are fit to (he_softmax1/he_softmax2).
SOFTMAX1_DOMAIN = (-27.2493, 21.72692)   # he_softmax1 -> he_exp1
SOFTMAX2_DOMAIN = (-70.0, 70.0)          # he_softmax2 -> he_exp2
LAYERNORM2_MAX_VAR = 150.0        # he_layernorm2 ceiling; above -> he_layernorm3 (<=2500)
LAYERNORM3_MAX_VAR = 2500.0


class _QuadGELU(nn.Module):
    """MPCFormer quadratic GELU: 0.125 x^2 + 0.25 x + 0.5 (an nn.Module because HF
    stores intermediate_act_fn as a child module)."""

    def forward(self, x):
        return 0.125 * x * x + 0.25 * x + 0.5


def load_model(model_dir: Path):
    raw = json.loads((model_dir / "config.json").read_text())
    cfg = BertConfig(
        num_hidden_layers=raw["num_hidden_layers"], hidden_size=raw["hidden_size"],
        num_attention_heads=raw["num_attention_heads"], intermediate_size=raw["intermediate_size"],
        max_position_embeddings=raw["max_position_embeddings"], vocab_size=raw["vocab_size"],
        type_vocab_size=raw["type_vocab_size"], num_labels=raw.get("num_labels", 2),
        pad_token_id=raw.get("pad_token_id", 0), layer_norm_eps=raw.get("layer_norm_eps", 1e-12),
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
    """(max|score|, max per-token LN2-input variance) for one layer, valid tokens only."""
    n = int(attn_mask.sum().item())  # valid length (mask is 1s then 0s after collation)
    L = model.bert.encoder.layer[layer_idx]
    a = L.attention.self

    def heads(x):
        return x.view(*x.shape[:-1], a.num_attention_heads, a.attention_head_size).transpose(1, 2)

    q = heads(a.query(hidden_states)); k = heads(a.key(hidden_states)); v = heads(a.value(hidden_states))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a.attention_head_size)  # (1,H,S,S) pre-mask
    valid = scores[..., :n, :n]
    score_min, score_max = valid.min().item(), valid.max().item()

    ext = model.get_extended_attention_mask(attn_mask, hidden_states.shape).to(device)
    probs = torch.softmax(scores + ext, dim=-1)
    ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(*hidden_states.shape[:-1], a.all_head_size)
    att_dense = L.attention.output.dense(ctx)
    ln1 = L.attention.output.LayerNorm(att_dense + hidden_states)
    inter = L.intermediate.intermediate_act_fn(L.intermediate.dense(ln1))
    ln2_in = L.output.dense(inter) + ln1  # input to the 2nd LayerNorm
    per_token_var = ln2_in[0, :n].var(dim=-1, unbiased=False)  # variance across hidden dim per token
    return score_min, score_max, per_token_var.max().item()


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
    ds = ds.map(lambda b: tok(b["sentence1"], b["sentence2"], truncation=True, max_length=args.max_seq_len),
                batched=True).with_format("torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    coll = DataCollatorWithPadding(tok)

    nL = cfg.num_hidden_layers
    smin = [0.0] * nL
    smax = [0.0] * nL
    max_var = [0.0] * nL
    for ex in ds:
        batch = coll([{k: ex[k] for k in ("input_ids", "token_type_ids", "attention_mask")}])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.bert(**batch)  # output_hidden_states=True via config
        for li in range(nL):
            lo, hi, var = layer_ranges(model, out.hidden_states[li], li, batch["attention_mask"], device)
            smin[li] = min(smin[li], lo)
            smax[li] = max(smax[li], hi)
            max_var[li] = max(max_var[li], var)

    # he_softmax1 covers [-27, 22] (asymmetric): a layer needs he_softmax2 only if its
    # scores actually exceed that box on either side.
    wide_softmax = [li for li in range(nL) if smax[li] > SOFTMAX1_DOMAIN[1] or smin[li] < SOFTMAX1_DOMAIN[0]]
    over_sm2 = [li for li in wide_softmax if smax[li] > SOFTMAX2_DOMAIN[1] or smin[li] < SOFTMAX2_DOMAIN[0]]
    wide_ln = [li for li in range(nL) if max_var[li] > LAYERNORM2_MAX_VAR]
    over_ln = [li for li in range(nL) if max_var[li] > LAYERNORM3_MAX_VAR]

    def sm_margin(li):
        lo, hi = SOFTMAX2_DOMAIN if li in wide_softmax else SOFTMAX1_DOMAIN
        return min(smax[li] - lo, hi - smax[li], smin[li] - lo, hi - smin[li])  # min distance to either wall

    print(f"\nsplit={args.split}")
    print(f"{'layer':>5}{'score min':>11}{'score max':>11}{'softmax(domain)':>22}{'margin':>9}{'ln2 var':>10}{'layernorm':>14}")
    print("-" * 82)
    for li in range(nL):
        wide = li in wide_softmax
        sfx = f"he_softmax2[-70,70]" if wide else "he_softmax1[-27,22]"
        lnx = "he_layernorm3" if li in wide_ln else "he_layernorm2"
        flags = ("  OVER-SM!" if li in over_sm2 else "") + ("  OVER-LN2500!" if li in over_ln else "")
        print(f"{li:>5}{smin[li]:>11.2f}{smax[li]:>11.2f}{sfx:>22}{sm_margin(li):>9.2f}{max_var[li]:>10.1f}{lnx:>14}{flags}")

    sm_ok = not over_sm2
    print(f"\nSOFTMAX verdict: {'PASS' if sm_ok else 'FAIL'} — every layer's score range fits its assigned exp domain"
          + ("" if sm_ok else f"; layers {over_sm2} exceed he_softmax2's [-70,70]!"))
    print("\n# suggested thor/src/thor/model_config.py")
    print(f"WIDE_SOFTMAX_LAYERS   = frozenset({set(wide_softmax) if wide_softmax else set()})")
    print(f"WIDE_LAYERNORM_LAYERS = frozenset({set(wide_ln) if wide_ln else set()})")
    if over_ln:
        print(f"# WARNING: layers {over_ln} exceed he_layernorm3's var<=2500 — no domain covers them.")


if __name__ == "__main__":
    main()
