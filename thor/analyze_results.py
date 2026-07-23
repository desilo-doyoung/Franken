#!/usr/bin/env python3
"""Summarize THOR forward-batch results (HE vs PT vs ground truth).

Reads every ``forward-batch-results/*/result.json`` (relative to this file, so it
runs from anywhere) and prints:
  - HE accuracy vs ground-truth labels
  - PT (plaintext reference) accuracy vs ground-truth labels
  - divergent samples (HE forward blew up numerically, |he_logit| > DIVERGENCE_ABS),
    the logit MAE over the sane rest, and the raw logits of the divergent samples.

No arguments. Run it with:  python thor/analyze_results.py
"""

import glob
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "forward-batch-results"

# A sample is "divergent" when the HE forward blew up numerically, not merely when
# its argmax flips. Sane HE logits track PT (~+-5); benign argmax-flips are all
# |he| < 1, while the smallest real explosion seen is ~27, so 10 cleanly separates.
DIVERGENCE_ABS = 10.0


def main() -> None:
    files = sorted(glob.glob(str(RESULTS_DIR / "*" / "result.json")))
    if not files:
        print(f"No result.json found under {RESULTS_DIR}")
        return

    rows = [json.load(open(f)) for f in files]
    rows.sort(key=lambda r: r["target_idx"])

    idx = np.array([r["target_idx"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    ppred = np.array([r["plain_pred"] for r in rows])
    label = np.array([r["label"] for r in rows])
    he = np.array([r["he_logits"] for r in rows], dtype=float)
    pt = np.array([r["plain_logits"] for r in rows], dtype=float)
    n = len(rows)

    he_acc = (pred == label).mean()
    pt_acc = (ppred == label).mean()
    he_correct = int((pred == label).sum())
    pt_correct = int((ppred == label).sum())

    div = np.abs(he).max(axis=1) > DIVERGENCE_ABS
    keep = ~div
    div_idx = idx[div]
    mae = float(np.abs(he[keep] - pt[keep]).mean()) if keep.any() else float("nan")

    print("=== 정확도 (HE vs Ground Truth) ===")
    print(f"Accuracy           : {he_acc:.6f} ({he_acc * 100:.2f}% = {he_correct} / {n})")
    print()
    print("=== 정확도 (PT vs Ground Truth, 참고용) ===")
    print(f"Accuracy           : {pt_acc:.6f} ({pt_acc * 100:.2f}% = {pt_correct} / {n})")
    print()
    print("=== 정밀도 (HE vs PT) ===")
    arr = np.array2string(div_idx, separator=" ", max_line_width=200)
    print(f"발산 샘플          : {arr} (Total {int(div.sum())} samples)")
    print(f"발산 샘플 제외 MAE : {mae:.12f} (Total {int(keep.sum())} samples)")
    print("발산 샘플 값들 (index, he_a, he_b, pt_a, pt_b)")
    for i in np.where(div)[0]:
        print(f"  {idx[i]:4d}  {he[i, 0]:14.4f} {he[i, 1]:14.4f}  {pt[i, 0]:9.4f} {pt[i, 1]:9.4f}")


if __name__ == "__main__":
    main()
