"""reparse_summaries.py — recompute 10%-BLER summaries from existing CSVs.

Fixes the crossing-detector artifact: when an adjacent SNR cell has BLER
exactly 0 (small-sample saturation), the original log-linear detector
required both endpoints > 0 and bailed to "unreachable" even when the
crossing was clearly between adjacent cells. The fix replaces 0 BLER
with a one-sided floor of 0.5/n_blocks (≈ half a codeword) for the
log10 interpolation.

Operates on existing per-Δ_max CSVs in a results dir; rewrites the
summary CSV with corrected values. No GPU work.

Usage:
    python reparse_summaries.py results/r5_inference_a [results/...]
"""

import csv
import math
import os
import sys

NAMES = ["0", "ln2", "ln4", "ln8", "ln16", "unbounded"]


def find_irr_in_dir(d):
    for f in os.listdir(d):
        if f.startswith("r5inf_bler_dmax0_irr") and f.endswith(".csv"):
            return f.split("_irr")[1].split(".csv")[0]
    return "25"


def reparse(d):
    irr = find_irr_in_dir(d)
    print(f"\n=== {d}  (IRR={irr}) ===")
    print(f"{'Δ_max':>12}  {'value':>10}  {'10%-BLER':>10}  {'floor':>10}")
    out_rows = []
    for n in NAMES:
        path = os.path.join(d, f"r5inf_bler_dmax{n}_irr{irr}.csv")
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        snrs = [float(r["snr_db"]) for r in rows]
        bler = [float(r["bler_mean"]) for r in rows]
        n_tti = int(rows[0]["n_tti"])
        n_runs = int(rows[0]["n_runs"])
        n_blocks = n_tti * n_runs
        # Floor for log10 interpolation when BLER hits 0.
        eps = 0.5 / n_blocks
        # Find first descent below 10%.
        snr10 = None
        for i in range(1, len(snrs)):
            if bler[i-1] >= 0.1 > bler[i]:
                b1 = max(bler[i-1], eps)
                b2 = max(bler[i], eps)
                lb1, lb2 = math.log10(b1), math.log10(b2)
                if lb1 == lb2:
                    snr10 = (snrs[i-1] + snrs[i]) / 2.0
                else:
                    snr10 = snrs[i-1] + (math.log10(0.1) - lb1) * \
                            (snrs[i] - snrs[i-1]) / (lb2 - lb1)
                break
        floor = min([b for b in bler if b > 0], default=0.0)
        # Compute value
        try:
            v = float(n) if n not in ("ln2", "ln4", "ln8", "ln16",
                                       "unbounded") else \
                math.log(int(n[2:])) if n != "unbounded" else float("inf")
        except Exception:
            v = float("nan")
        s10s = f"{snr10:.4f}" if snr10 is not None else "unreachable"
        print(f"{n:>12}  {v:>10.4f}  {s10s:>10}  {floor:>10.3e}")
        out_rows.append((n, v, s10s, floor))
    # Rewrite summary
    summary_path = os.path.join(d, "r5inf_dmax_sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_max_name", "delta_max_value", "snr10_db",
                    "floor_bler"])
        for n, v, s10s, floor in out_rows:
            w.writerow([n, v, s10s, floor])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Reparse all known R5 inference result dirs.
        for d in ["results/r5_inference_a",
                  "results/r5_inference_v2_240tti",
                  "results/r5_inference_irr20",
                  "results/r5_inference_irr35",
                  "results/r5_inference_cdlD",
                  "results/r5_inference_cdlE"]:
            if os.path.isdir(d):
                reparse(d)
    else:
        for d in sys.argv[1:]:
            reparse(d)
