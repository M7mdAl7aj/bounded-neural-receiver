"""ag0_cpu_runner.py — gate A-G0 on CPU, sharded by (condition, Delta_max).

The LDPC decoder barely benefits from the GPU (measured 0.121 s/block on CPU vs
0.078 on GPU), so A-G0 runs well on a many-core CPU and leaves the GPU free
for other sweeps. Proven bit-exact first: on
cdlC_irr25 / SNR 20 the CPU decoder reproduced the GPU sweep's INTEGER error
counts exactly on all six ladders (276/191/141/108/97/98).

Comparison is on integer error counts, not floats. The online CSV stores a
mean formed as (e0/960 + e1/960)/2 while the offline side computes
(e0+e1)/1920; those differ in the last float bit for some values even when the
decoding is identical, and that artifact is device-independent. Counting
errors sidesteps it.

  shard:     ag0_cpu_runner.py --shard --condition C --only-dmax ln4
  aggregate: ag0_cpu_runner.py --aggregate
"""
import argparse, csv, importlib.util, json, math, os, sys
import numpy as np
import tensorflow as tf

DM = [("0",0.0),("ln2",math.log(2)),("ln4",math.log(4)),
      ("ln8",math.log(8)),("ln16",math.log(16)),("unbounded",math.inf)]
CONDS = ["cdlC_irr25","irr20","irr35","cdlD","cdlE"]
DUMPS = "results/run_c_ci/dumps/seed42"
ONLINE = "results/run_c_ci/seed42"
SHARDS = "results/run_c_ci/ag0/shards"

_here = os.path.dirname(os.path.abspath(__file__))
def _load(n, path):
    sp = importlib.util.spec_from_file_location(n, path)
    m = importlib.util.module_from_spec(sp); sys.modules[n] = m
    sp.loader.exec_module(m); return m

def run_shard(cond, dname, batch):
    lad = _load("ra4_offline_ladder", os.path.join(_here, "ra4_offline_ladder.py"))
    d = os.path.join(DUMPS, cond)
    meta = json.load(open(os.path.join(d, "index.json")))
    dval = dict(DM)[dname]
    sim = lad.build_decoder(meta); tb_dec = sim.tb_dec
    out = {"condition": cond, "delta_max": dname, "n_blocks": meta["n_blocks"],
           "snr_db": meta["snr_db"], "errors": []}
    for s in meta["snr_db"]:
        t = f"snr{s:.1f}"
        l2 = np.load(f"{d}/llr_l2_{t}.npy", mmap_mode="r")
        r3 = np.load(f"{d}/llr_r3_{t}.npy", mmap_mode="r")
        bi = np.load(f"{d}/bits_info_{t}.npy", mmap_mode="r")
        errs = 0
        for lo in range(0, l2.shape[0], batch):
            hi = min(lo+batch, l2.shape[0]); nb = hi-lo
            a = tf.constant(np.asarray(l2[lo:hi]), dtype=tf.float32)
            b = tf.constant(np.asarray(r3[lo:hi]), dtype=tf.float32)
            llr = b if math.isinf(dval) else (a if dval==0.0
                  else tf.clip_by_value(b, a-dval, a+dval))
            bh, _ = tb_dec(tf.reshape(llr, [nb,1,1,-1]))
            bh = tf.reshape(bh, [nb,-1]).numpy().astype(np.uint8)
            errs += int(np.any(bh != np.asarray(bi[lo:hi]), axis=-1).sum())
        out["errors"].append(errs)
        print(f"{cond} {dname} snr={s:+.1f} errors={errs}", flush=True)
    os.makedirs(SHARDS, exist_ok=True)
    json.dump(out, open(f"{SHARDS}/{cond}__{dname}.json","w"))

def aggregate(tol):
    lad = _load("ra4_offline_ladder", os.path.join(_here, "ra4_offline_ladder.py"))
    overall = 0
    print(f"{'condition':<12}{'Δ_max':>11}{'online':>10}{'offline':>10}"
          f"{'Δ dB':>9}{'int mismatches':>16}")
    verdicts = {}
    for cond in CONDS:
        meta = json.load(open(f"{DUMPS}/{cond}/index.json"))
        irr = int(meta["eval_irr_db"]); N = meta["n_blocks"]
        worst_c, mism_c = 0.0, 0
        for dname,_ in DM:
            f = f"{SHARDS}/{cond}__{dname}.json"
            if not os.path.exists(f):
                print(f"{cond:<12}{dname:>11}   MISSING SHARD"); overall=1; continue
            sh = json.load(open(f))
            off_b = [e/N for e in sh["errors"]]
            rows = list(csv.DictReader(open(
                f"{ONLINE}/{cond}/r5inf_bler_dmax{dname}_irr{irr}.csv")))
            on_s = [float(r["snr_db"]) for r in rows]
            on_b = [float(r["bler_mean"]) for r in rows]
            mism = sum(1 for i in range(len(on_b))
                       if round(on_b[i]*N) != sh["errors"][i])
            mism_c += mism
            c_on = lad.crossing_10pct(on_s, on_b)
            c_off = lad.crossing_10pct(sh["snr_db"], off_b)
            if c_on is None and c_off is None: dd = 0.0; sc="unreach/unreach"
            elif c_on is None or c_off is None: dd = float("inf"); sc="REACHABILITY"
            else: dd = abs(c_on-c_off); sc=f"{c_on:.3f}/{c_off:.3f}"
            worst_c = max(worst_c, dd)
            print(f"{cond:<12}{dname:>11}{sc:>21}{dd:>9.4f}{mism:>16}")
        ok = (mism_c == 0) and (worst_c <= tol)
        verdicts[cond] = ("PASS" if ok else "FAIL", worst_c, mism_c)
        print(f"  -> {cond}: {'PASS' if ok else 'FAIL'} "
              f"(max |Δ| {worst_c:.4f} dB, {mism_c} integer mismatches)\n")
        if not ok: overall = 1
    print("=== A-G0 (CPU) SUMMARY ===")
    for c,(v,w,m) in verdicts.items():
        print(f"  {c:<12} GATE A-G0: {v}   max|Δ|={w:.4f} dB   int_mismatches={m}")
    n_pass = sum(1 for v,_,_ in verdicts.values() if v=="PASS")
    print(f"\nA-G0 CPU: {n_pass}/5 PASS")
    return overall

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--condition"); p.add_argument("--only-dmax")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.05)
    a = p.parse_args()
    if a.shard: run_shard(a.condition, a.only_dmax, a.batch_size); sys.exit(0)
    if a.aggregate: sys.exit(aggregate(a.tol))
    p.error("choose --shard or --aggregate")
