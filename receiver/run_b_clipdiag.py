"""run_b_clipdiag.py — Run B clip diagnostics, offline on the seed-42 dumps.

Computes the clip-saturation / safety-envelope metric.
Pure post-processing: no GPU, no decoder. Definitions:

  clip event      |L_R3 - L_L2| > Delta_max  (record upper/lower direction)
  clip fraction   clipped bits / total bits, per Delta_max x condition
  residual dist   quantiles {50,90,99,99.9} of |L_R3 - L_L2| + histogram
  confidence bins clip fraction within quartile bins of |L_L2|
  wrong-direction among clipped bits, share where sign(L_R3) disagrees with the
                  true coded bit (positive LLR == bit 1) vs right-sign-
                  oversized-magnitude; L_L2 sign-correctness on the same bits

Gates:
  B-G1  Delta_max = inf  => clip fraction exactly 0, every condition
  B-G2  at every finite Delta_max the clip-event mask is IDENTICAL to the
        (R5 != R3) mask -- masks, not counts. Corollary at Delta_max = 0:
        clip fraction == the R3-vs-L2 disagreement fraction.
"""
import csv, json, math, os, sys
import numpy as np

DUMPS = "results/run_c_ci/dumps/seed42"
OUT   = "results/run_b_clipdiag"
CONDS = ["cdlC_irr25", "irr20", "irr35"]          # scope: the IRR conditions
DMAX  = [("0", 0.0), ("ln2", math.log(2)), ("ln4", math.log(4)),
         ("ln8", math.log(8)), ("ln16", math.log(16)), ("unbounded", math.inf)]
CHUNK = 128                                        # blocks per chunk

# log-spaced histogram, exact and deterministic (no sampling)
NB = 20000
EDGES = np.concatenate([[0.0], np.logspace(-6, 4, NB), [np.inf]])

def snr_tags(d):
    m = json.load(open(os.path.join(d, "index.json")))
    return m, [f"snr{s:.1f}" for s in m["snr_db"]]

def hist_quantiles(counts, edges, qs):
    tot = counts.sum()
    if tot == 0: return [float("nan")] * len(qs)
    c = np.cumsum(counts)
    return [float(edges[min(int(np.searchsorted(c, q * tot)), len(edges) - 2)])
            for q in qs]

def main():
    os.makedirs(OUT, exist_ok=True)
    rows, gates = [], []
    for cond in CONDS:
        d = os.path.join(DUMPS, cond)
        meta, tags = snr_tags(d)
        print(f"\n=== {cond} ({meta['n_blocks']} blocks x {len(tags)} SNR) ===",
              flush=True)

        # ---- pass 1: histograms of |dL| and |L_L2| ----
        h_d = np.zeros(len(EDGES) - 1, dtype=np.int64)
        h_l2 = np.zeros(len(EDGES) - 1, dtype=np.int64)
        for t in tags:
            l2 = np.load(f"{d}/llr_l2_{t}.npy", mmap_mode="r")
            r3 = np.load(f"{d}/llr_r3_{t}.npy", mmap_mode="r")
            for lo in range(0, l2.shape[0], CHUNK):
                a = np.asarray(l2[lo:lo+CHUNK], dtype=np.float64)
                b = np.asarray(r3[lo:lo+CHUNK], dtype=np.float64)
                h_d  += np.histogram(np.abs(b - a), bins=EDGES)[0]
                h_l2 += np.histogram(np.abs(a),     bins=EDGES)[0]
        q = hist_quantiles(h_d, EDGES, [0.5, 0.9, 0.99, 0.999])
        l2q = hist_quantiles(h_l2, EDGES, [0.25, 0.5, 0.75])
        print(f"  |dL| quantiles p50/p90/p99/p99.9 = "
              f"{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f} / {q[3]:.4f}")
        print(f"  |L_L2| quartile edges = {l2q[0]:.3f} / {l2q[1]:.3f} / {l2q[2]:.3f}")
        np.save(f"{OUT}/hist_absdiff_{cond}.npy", h_d)
        np.save(f"{OUT}/hist_absl2_{cond}.npy", h_l2)
        np.save(f"{OUT}/hist_edges.npy", EDGES)

        # ---- pass 2: per-Delta_max clip stats ----
        for name, dm in DMAX:
            n_tot = n_clip = n_up = n_lo = 0
            n_wrong = n_right_over = 0
            n_l2_right_on_clipped = 0
            bins_tot = np.zeros(4, dtype=np.int64)
            bins_clip = np.zeros(4, dtype=np.int64)
            mask_mismatch = 0
            n_disagree = 0          # for the Delta_max=0 corollary
            for t in tags:
                l2 = np.load(f"{d}/llr_l2_{t}.npy", mmap_mode="r")
                r3 = np.load(f"{d}/llr_r3_{t}.npy", mmap_mode="r")
                bc = np.load(f"{d}/bits_coded_{t}.npy", mmap_mode="r")
                for lo in range(0, l2.shape[0], CHUNK):
                    a = np.asarray(l2[lo:lo+CHUNK], dtype=np.float64)
                    b = np.asarray(r3[lo:lo+CHUNK], dtype=np.float64)
                    tb = np.asarray(bc[lo:lo+CHUNK])
                    diff = b - a
                    clip = np.abs(diff) > dm if np.isfinite(dm) else np.zeros_like(diff, bool)
                    r5 = np.clip(b, a - dm, a + dm) if np.isfinite(dm) else b
                    # B-G2: clip-event mask must equal the (R5 != R3) mask
                    mask_mismatch += int(np.count_nonzero(clip != (r5 != b)))
                    n_tot += diff.size
                    n_clip += int(clip.sum())
                    n_up += int((clip & (diff > 0)).sum())
                    n_lo += int((clip & (diff < 0)).sum())
                    # sign convention: positive LLR encodes bit value 1
                    r3_sign_bit = (b > 0).astype(np.uint8)
                    l2_sign_bit = (a > 0).astype(np.uint8)
                    r3_wrong = r3_sign_bit != tb
                    n_wrong       += int((clip & r3_wrong).sum())
                    n_right_over  += int((clip & ~r3_wrong).sum())
                    n_l2_right_on_clipped += int((clip & (l2_sign_bit == tb)).sum())
                    # confidence bins on |L_L2|
                    idx = np.digitize(np.abs(a), l2q)      # 0..3
                    for k in range(4):
                        sel = idx == k
                        bins_tot[k]  += int(sel.sum())
                        bins_clip[k] += int((sel & clip).sum())
                    if dm == 0.0:
                        # "R3-vs-L2 disagreement" here means VALUE
                        # disagreement (L_R3 != L_L2), which is what the
                        # Delta_max=0 clip event tests: |L_R3 - L_L2| > 0.
                        # Sign disagreement is a different (much smaller)
                        # quantity and is reported separately as the
                        # wrong-direction share.
                        n_disagree += int(np.count_nonzero(b != a))
            frac = n_clip / n_tot
            rows.append(dict(
                condition=cond, delta_max=name, delta_max_value=dm,
                clip_fraction=frac, clip_upper=n_up / n_tot,
                clip_lower=n_lo / n_tot,
                resid_p50=q[0], resid_p90=q[1], resid_p99=q[2], resid_p999=q[3],
                wrong_direction_share=(n_wrong / n_clip) if n_clip else 0.0,
                right_sign_oversized_share=(n_right_over / n_clip) if n_clip else 0.0,
                l2_sign_correct_on_clipped=(n_l2_right_on_clipped / n_clip) if n_clip else 0.0,
                conf_bin1=bins_clip[0] / max(bins_tot[0], 1),
                conf_bin2=bins_clip[1] / max(bins_tot[1], 1),
                conf_bin3=bins_clip[2] / max(bins_tot[2], 1),
                conf_bin4=bins_clip[3] / max(bins_tot[3], 1),
                n_bits=n_tot))
            print(f"  dmax={name:<10} clip={frac:.6f}  up/lo={n_up/n_tot:.6f}/"
                  f"{n_lo/n_tot:.6f}  wrongdir={(n_wrong/n_clip if n_clip else 0):.4f}"
                  f"  maskmismatch={mask_mismatch}", flush=True)
            if name == "unbounded":
                gates.append(("B-G1", cond, "clip fraction == 0 at inf",
                              "PASS" if n_clip == 0 else f"FAIL ({n_clip})"))
            else:
                gates.append(("B-G2", f"{cond}/{name}", "clip mask == (R5!=R3) mask",
                              "PASS" if mask_mismatch == 0 else f"FAIL ({mask_mismatch})"))
            if dm == 0.0:
                ok = (n_disagree == n_clip)
                gates.append(("B-G2-corollary", cond,
                              "clip frac at 0 == R3-vs-L2 disagreement",
                              "PASS" if ok else f"FAIL ({n_clip} vs {n_disagree})"))

    with open(f"{OUT}/clipdiag_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/gates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gate", "scope", "criterion", "result"])
        w.writerows(gates)
    print("\n=== RUN B GATES ===")
    for g in gates: print(f"  {g[0]:<16} {g[1]:<22} {g[3]}")
    bad = [g for g in gates if not g[3].startswith("PASS")]
    print(f"\nRUN B: {'ALL GATES PASS' if not bad else f'{len(bad)} FAILURE(S)'}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
