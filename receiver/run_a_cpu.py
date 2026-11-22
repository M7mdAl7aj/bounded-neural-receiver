"""run_a_cpu.py — run A fusion BLER sweeps on CPU (the GPU stays free).

CUDA_VISIBLE_DEVICES="" is a HARD requirement and is asserted at startup, with
the visible-device list printed so every job log carries the proof.

The endpoint check runs FIRST on CPU. F1 at lambda=1 is identically R3, so
decoding it must reproduce the unbounded-ladder error counts already recorded by
A-G0. Tolerance is the measured CPU block-flip budget (~2e-6), not zero -- the
A-G0 run found 1 borderline block in 518,400.

  gate:  run_a_cpu.py --endpoint-gate --condition cdlC_irr25
  shard: run_a_cpu.py --shard --condition C --method F1 --param 0.5
"""
import argparse, csv, importlib.util, json, math, os, sys
import numpy as np

if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    sys.exit("REFUSING TO RUN: CUDA_VISIBLE_DEVICES must be set to the empty "
             "string so this cannot touch the GPU.")
import tensorflow as tf
_gpus = tf.config.list_physical_devices("GPU")
print(f"[startup] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} "
      f"visible_GPUs={_gpus} -> {'CPU-ONLY OK' if not _gpus else 'GPU VISIBLE - ABORT'}",
      flush=True)
if _gpus: sys.exit("GPU visible despite the guard; aborting.")

_here = os.path.dirname(os.path.abspath(__file__))
def _load(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m; sp.loader.exec_module(m); return m
lad=_load("ra4_offline_ladder", os.path.join(_here,"ra4_offline_ladder.py"))
rao=_load("run_a_offline",      os.path.join(_here,"run_a_offline.py"))

DUMPS="results/run_c_ci/dumps/seed42"; OUT="results/run_a_fusion"
SHARDS=f"{OUT}/shards"; CHUNK=8

def decode_stream(tb_dec, l2, r3, bi, fuse, n_blocks):
    errs=0
    for lo in range(0,n_blocks,CHUNK):
        hi=min(lo+CHUNK,n_blocks); nb=hi-lo
        a=np.asarray(l2[lo:hi],dtype=np.float32); b=np.asarray(r3[lo:hi],dtype=np.float32)
        llr=tf.constant(fuse(a,b),dtype=tf.float32)
        bh,_=tb_dec(tf.reshape(llr,[nb,1,1,-1]))
        errs+=int(np.any(tf.reshape(bh,[nb,-1]).numpy().astype(np.uint8)!=np.asarray(bi[lo:hi]),axis=-1).sum())
    return errs

def arrays(cond,tag):
    d=os.path.join(DUMPS,cond)
    return (np.load(f"{d}/llr_l2_{tag}.npy",mmap_mode="r"),
            np.load(f"{d}/llr_r3_{tag}.npy",mmap_mode="r"),
            np.load(f"{d}/bits_info_{tag}.npy",mmap_mode="r"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint-gate",action="store_true")
    p.add_argument("--shard",action="store_true")
    p.add_argument("--condition",required=True)
    p.add_argument("--method"); p.add_argument("--param")
    a=p.parse_args()
    meta=json.load(open(f"{DUMPS}/{a.condition}/index.json"))
    tags=[f"snr{s:.1f}" for s in meta["snr_db"]]; N=meta["n_blocks"]
    sim=lad.build_decoder(meta); tb=sim.tb_dec

    if a.endpoint_gate:
        # F1(lambda=1) must reproduce A-G0's unbounded counts exactly.
        ref=json.load(open(f"results/run_c_ci/ag0/shards/{a.condition}__unbounded.json"))["errors"]
        print(f"[gate] endpoint check on CPU: F1(lambda=1) vs A-G0 unbounded, {a.condition}")
        bad=0; tot=0
        for i,t in enumerate(tags):
            l2,r3,bi=arrays(a.condition,t)
            e=decode_stream(tb,l2,r3,bi,lambda x,y: rao.fuse_F1(x,y,1.0),N)
            tot+=N
            flag="" if e==ref[i] else f"  <-- differs by {e-ref[i]:+d}"
            if e!=ref[i]: bad+=abs(e-ref[i])
            print(f"  {t}: F1(l=1)={e}  A-G0 unbounded={ref[i]}{flag}",flush=True)
        rate=bad/tot
        print(f"[gate] block mismatches {bad}/{tot} = {rate:.2e}  (budget 2e-6)")
        ok = rate<=2e-6
        print(f"[gate] RUN-A CPU ENDPOINT GATE: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    # F2/F3 sweep QUANTILES of in-dist statistics, which are per-SNR-point
    # (recorded choice in run_a_offline.py). A scalar param would silently
    # apply one SNR's threshold everywhere, so named params are resolved per
    # SNR from the pinned grid file.
    GRID={}
    for r in csv.DictReader(open(f"{OUT}/fusion_param_grids.csv")):
        GRID[float(r["snr_db"])]=r
    def resolve(method,param,snr):
        if param in ("0","inf","-inf"): return float(param)
        if param.startswith("p"):
            key=("tau_" if method=="F2" else "g_")+param
            return float(GRID[snr][key])
        return float(param)
    MK={"F1":rao.fuse_F1,"F2":rao.fuse_F2,"F3":rao.fuse_F3,"F4":rao.fuse_F4}[a.method]
    out={"condition":a.condition,"method":a.method,"param":a.param,
         "n_blocks":N,"snr_db":meta["snr_db"],"errors":[],"resolved":[]}
    for t,snr in zip(tags,meta["snr_db"]):
        pv=resolve(a.method,a.param,snr); out["resolved"].append(pv)
        l2,r3,bi=arrays(a.condition,t)
        e=decode_stream(tb,l2,r3,bi,lambda x,y,_p=pv: MK(x,y,_p),N); out["errors"].append(e)
        print(f"{a.condition} {a.method} p={a.param}(={pv:.4g}) {t} errors={e}",flush=True)
    os.makedirs(SHARDS,exist_ok=True)
    json.dump(out,open(f"{SHARDS}/{a.condition}__{a.method}__{a.param}.json","w"))
    return 0

if __name__=="__main__": sys.exit(main())
