"""run_a_offline.py — Run A offline parts: endpoint gates + the F2/F3 grids.

CPU-only. The fusion BLER sweeps need the LDPC decoder (GPU) and are NOT run
here; this establishes the two things that must exist before those sweeps are
worth running:

  1. Endpoint gates A-G1..A-G4 -- exact LLR equality, max |diff| = 0, the
     Theorem-1 verification standard.
  2. The swept-parameter grids for F2 and F3, which are quantiles of in-dist
     statistics and must be pinned before any condition is swept.

Methods:
  F1 convex        L = lam*L_R3 + (1-lam)*L_L2
  F2 confidence    L = L_R3 where |L_R3| >= tau else L_L2
  F3 block gating  per block: R3 if mean|L_R3-L_L2| <= g else L2
  F4 fixed clip    L = clip(L_R3, +-C)   (absolute, NOT anchored to L2)

RECORDED CHOICE: the F2/F3 quantiles are computed per SNR POINT on the
in-dist condition and reused at the same SNR point for every other condition.
The method definition says "computed in-dist, held fixed across conditions"
without fixing the SNR treatment; pooling across SNR would mix regimes whose LLR scales differ
by orders of magnitude, which would make one grid meaningless at both ends.
"""
import csv, json, math, os, sys
import numpy as np

DUMPS = "results/run_c_ci/dumps/seed42"
OUT   = "results/run_a_fusion"
CONDS = ["cdlC_irr25", "irr20", "irr35", "cdlD", "cdlE"]
INDIST = "cdlC_irr25"
CHUNK = 128
HUGE = 1e6            # F4 "huge"; asserted below to exceed max |L_R3|

def tags_of(d):
    m = json.load(open(os.path.join(d, "index.json")))
    return m, [f"snr{s:.1f}" for s in m["snr_db"]]

def fuse_F1(a, b, lam):  return lam * b + (1.0 - lam) * a
def fuse_F2(a, b, tau):  return np.where(np.abs(b) >= tau, b, a)
def fuse_F3(a, b, g):
    stat = np.abs(b - a).mean(axis=1, keepdims=True)      # per block
    return np.where(stat <= g, b, a)
def fuse_F4(a, b, C):    return np.clip(b, -C, C)

def main():
    os.makedirs(OUT, exist_ok=True)
    gates, grids = [], []

    # ---------- 1. F2 / F3 grids from the in-dist dump, per SNR point -------
    d = os.path.join(DUMPS, INDIST); meta, tags = tags_of(d)
    print(f"=== F2/F3 grids from {INDIST}, per SNR point ===", flush=True)
    max_abs_r3 = 0.0
    for t, snr in zip(tags, meta["snr_db"]):
        l2 = np.load(f"{d}/llr_l2_{t}.npy", mmap_mode="r")
        r3 = np.load(f"{d}/llr_r3_{t}.npy", mmap_mode="r")
        absr3, blkstat = [], []
        for lo in range(0, l2.shape[0], CHUNK):
            a = np.asarray(l2[lo:lo+CHUNK], dtype=np.float32)
            b = np.asarray(r3[lo:lo+CHUNK], dtype=np.float32)
            absr3.append(np.abs(b).ravel()[::37])          # deterministic stride
            blkstat.append(np.abs(b - a).mean(axis=1))
            max_abs_r3 = max(max_abs_r3, float(np.abs(b).max()))
        absr3 = np.concatenate(absr3); blkstat = np.concatenate(blkstat)
        tq = np.percentile(absr3, [50, 75, 90])
        gq = np.percentile(blkstat, [50, 75, 90])
        grids.append(dict(snr_db=snr,
                          tau_p50=tq[0], tau_p75=tq[1], tau_p90=tq[2],
                          g_p50=gq[0], g_p75=gq[1], g_p90=gq[2]))
        print(f"  snr={snr:+.1f}  tau p50/p75/p90 = {tq[0]:.3f}/{tq[1]:.3f}/{tq[2]:.3f}"
              f"   g p50/p75/p90 = {gq[0]:.4f}/{gq[1]:.4f}/{gq[2]:.4f}", flush=True)
    with open(f"{OUT}/fusion_param_grids.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grids[0].keys())); w.writeheader()
        w.writerows(grids)
    print(f"  max |L_R3| observed in-dist = {max_abs_r3:.2f}; HUGE = {HUGE:g}")
    assert HUGE > max_abs_r3, "F4 'huge' must exceed max |L_R3| for A-G4"

    # ---------- 2. endpoint gates, every condition --------------------------
    print("\n=== endpoint gates A-G1..A-G4 (exact, max|diff| must be 0) ===",
          flush=True)
    for cond in CONDS:
        d = os.path.join(DUMPS, cond); meta, tags = tags_of(d)
        worst = {k: 0.0 for k in
                 ["G1_lam1_R3", "G1_lam0_L2", "G2_tau0_R3", "G2_tauinf_L2",
                  "G3_ginf_R3", "G3_g0_L2", "G4_huge_R3"]}
        for t in tags:
            l2 = np.load(f"{d}/llr_l2_{t}.npy", mmap_mode="r")
            r3 = np.load(f"{d}/llr_r3_{t}.npy", mmap_mode="r")
            for lo in range(0, l2.shape[0], CHUNK):
                a = np.asarray(l2[lo:lo+CHUNK], dtype=np.float32)
                b = np.asarray(r3[lo:lo+CHUNK], dtype=np.float32)
                u = lambda k, v: worst.__setitem__(k, max(worst[k], float(v)))
                u("G1_lam1_R3",  np.abs(fuse_F1(a, b, 1.0) - b).max())
                u("G1_lam0_L2",  np.abs(fuse_F1(a, b, 0.0) - a).max())
                u("G2_tau0_R3",  np.abs(fuse_F2(a, b, 0.0) - b).max())
                u("G2_tauinf_L2",np.abs(fuse_F2(a, b, np.inf) - a).max())
                u("G3_ginf_R3",  np.abs(fuse_F3(a, b, np.inf) - b).max())
                u("G3_g0_L2",    np.abs(fuse_F3(a, b, 0.0) - a).max())
                u("G4_huge_R3",  np.abs(fuse_F4(a, b, HUGE) - b).max())
        print(f"  {cond}")
        for k, v in worst.items():
            status = "PASS" if v == 0.0 else "FAIL"
            print(f"    {k:<16} max|diff|={v:.3e}  {status}")
            gates.append((k.split("_")[0].replace("G", "A-G"), cond, k, v,
                          status))
    with open(f"{OUT}/endpoint_gates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gate", "condition", "check", "max_abs_diff", "result"])
        w.writerows(gates)
    bad = [g for g in gates if g[4] != "PASS"]
    print(f"\nRUN A endpoint gates: "
          f"{'ALL PASS' if not bad else f'{len(bad)} FAILURE(S)'}")
    print("Fusion BLER sweeps NOT run (need the decoder / GPU).")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
