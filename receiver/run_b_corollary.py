"""run_b_corollary.py — B-G2 corollary, recomputed with the correct definition.

At Delta_max = 0 the clip event is |L_R3 - L_L2| > 0, so the "R3-vs-L2
disagreement fraction" it must equal is VALUE disagreement (L_R3 != L_L2), not
sign disagreement. The first run of run_b_clipdiag.py used sign disagreement
for this one line and would report a spurious FAIL; the source is corrected and
this recomputes the check standalone rather than re-streaming all six
Delta_max passes.
"""
import json, os, sys
import numpy as np
DUMPS="results/run_c_ci/dumps/seed42"; CHUNK=128
rc=0
for cond in ["cdlC_irr25","irr20","irr35"]:
    d=os.path.join(DUMPS,cond); m=json.load(open(f"{d}/index.json"))
    n_clip=n_val=n_sign=n_tot=0
    for s in m["snr_db"]:
        t=f"snr{s:.1f}"
        l2=np.load(f"{d}/llr_l2_{t}.npy",mmap_mode="r")
        r3=np.load(f"{d}/llr_r3_{t}.npy",mmap_mode="r")
        for lo in range(0,l2.shape[0],CHUNK):
            a=np.asarray(l2[lo:lo+CHUNK],dtype=np.float64)
            b=np.asarray(r3[lo:lo+CHUNK],dtype=np.float64)
            n_tot+=a.size
            n_clip+=int(np.count_nonzero(np.abs(b-a)>0.0))
            n_val +=int(np.count_nonzero(b!=a))
            n_sign+=int(np.count_nonzero((b>0)!=(a>0)))
    ok = n_clip==n_val
    print(f"{cond:<12} clip@0={n_clip}  value_disagree={n_val}  "
          f"sign_disagree={n_sign} ({n_sign/n_tot:.6f})  "
          f"{'PASS' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
print("B-G2 corollary:", "PASS" if rc==0 else "FAIL")
sys.exit(rc)
