"""r5_frontend_eval.py - the evaluation driver of the LMMSE-fed network (R3 whose channel-estimate
input is the practical-prior LMMSE estimate instead of LS).

  TF_USE_LEGACY_KERAS=1 TF_CUDNN_DETERMINISTIC=1 python r5_frontend_eval.py --front-end {ls,lmmse-practical} \
      [--prior results/run_p/prior_practical.npz] [--prior-sha results/run_p/prior_practical.sha256] \
      <every r5_simo_inference.py argument, unchanged>

r5_simo_inference.py is not edited, so the --front-end option lives here (subclass in a new file,
injection after construction):
  - --front-end ls installs NOTHING: the sweep is r5_simo_inference.py's own, byte for byte.
    It reproduces the recorded R3'' in-distribution seed-42 crossing (19.4091).
  - --front-end lmmse-practical subclasses R5InferenceSim and, after its __init__ (before the
    first traced forward), replaces the inner R3 sim's ls_est, R3's channel-estimate input only,
    by run P's practical-prior LMMSE estimator (lmmse_practical.make_estimator), loaded only
    if the prior's sha256 equals the recorded one. The classical L2-real arm (chan_est on the
    IQ-imbalance-compensated grid, the IQI fit's own estimator) is untouched.
Writes the sidecar front_end.json (front end, prior sha, evaluation seed, R3 weights,
argv) into --out-dir before the sweep starts.
"""
import argparse
import hashlib
import json
import logging
import os
import sys

_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--front-end", choices=["ls", "lmmse-practical"], required=True)
pre.add_argument("--prior", default=os.path.join(_PKG, "results/run_p/prior_practical.npz"))
pre.add_argument("--prior-sha", default=os.path.join(_PKG, "results/run_p/prior_practical.sha256"))
fe, rest = pre.parse_known_args()
peek = argparse.ArgumentParser(add_help=False)
peek.add_argument("--out-dir", default=None)
peek.add_argument("--seed", type=int, default=42)
peek.add_argument("--r3-weights", default="./results/r3_mirror/final")
pk, _ = peek.parse_known_args(rest)

import r5_simo_inference as r5   # noqa: E402  (unchanged module; importing runs nothing)

sha = None
if fe.front_end == "lmmse-practical":
    import lmmse_practical as lp   # noqa: E402
    import lmmse_errvar_floor as lf   # noqa: E402
    want = open(fe.prior_sha).read().split()[0]
    sha = hashlib.sha256(open(fe.prior, "rb").read()).hexdigest()
    if sha != want:
        raise SystemExit(f"REFUSED: prior {fe.prior} sha256 {sha[:16]} != recorded {want[:16]}")
    R_FREQ, R_TIME, _META = lp.load_prior(fe.prior)

    class R5FrontEndSim(r5.R5InferenceSim):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            inner = self._inner
            stock = inner.ls_est                   # the stock LS estimator (linear interpolation)
            est = lp.make_estimator(inner.pdsch, R_FREQ, R_TIME, inner.cfg.lmmse_order)
            lf.install(est)   # floor on the frequency step's error variance (lmmse_errvar_floor.py)
            # per-slot LS fallback where the LMMSE estimate is non-finite; counted, logged per call, totals at exit
            od = pk.out_dir
            inner.ls_est = lf.with_fallback(est, stock, log_path=os.path.join(od, "fallback_log.txt") if od else None,
                                            sidecar=os.path.join(od, "front_end_fallback.json") if od else None)
            logging.info(f"FRONT END lmmse-practical: R3 input estimate from {fe.prior} "
                         f"sha256[:16]={sha[:16]} order={inner.cfg.lmmse_order}")

    r5.R5InferenceSim = R5FrontEndSim

side = {"front_end": fe.front_end, "prior": fe.prior if sha else None, "prior_sha256": sha,
        "eval_seed": pk.seed, "r3_weights": pk.r3_weights, "script": os.path.basename(__file__),
        "wraps": "r5_simo_inference.py (unchanged)", "argv": sys.argv}
if sha:
    side["errvar_floor_rel"] = lf.FLOOR_REL   # error-variance floor
    side["slot_fallback"] = "per-slot stock LS estimate where the LMMSE estimate is non-finite; totals in front_end_fallback.json"
if pk.out_dir:
    os.makedirs(pk.out_dir, exist_ok=True)
    with open(os.path.join(pk.out_dir, "front_end.json"), "w") as f:
        json.dump(side, f, indent=1)
print(f"CONFIG front_end={fe.front_end} prior_sha256={sha} eval_seed={pk.seed} r3_weights={pk.r3_weights} "
      f"out_dir={pk.out_dir}", flush=True)

sys.argv = [os.path.join(_PKG, "r5_simo_inference.py")] + rest
r5.main()
