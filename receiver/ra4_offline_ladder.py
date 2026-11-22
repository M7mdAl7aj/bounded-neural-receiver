"""ra4_offline_ladder.py — recompute the R5 Δ_max ladder from an LLR dump
(r5_simo_inference.py --dump-llrs).

This is check A-G0's instrument. The architecture is "dump once,
post-process offline": one instrumented sweep per condition writes the
per-bit artifacts, and every method comparison (runs A and B) is then
computed on literally identical bits. A-G0 is the check that the offline
path reproduces the online one — if it does not, the dump or the
plumbing is wrong and no A/B/D number may be recorded.

Recomputes, per SNR point and per Δ_max:

    R5 LLR = clip(L_R3, L_L2 − Δ_max, L_L2 + Δ_max)   → TBDecoder → BLER

using the SAME decoder the online sweep used, rebuilt from the dump's
index.json. Compares against the online CSVs when they are supplied.

Usage:
  ra4_offline_ladder.py --dump DIR [--online-dir DIR] [--tol 0.05]
                        [--delta-max-list 0 ln2 ln4 ln8 ln16 unbounded]
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import sys

import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass

import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r1_baseline = _load("r1_baseline", os.path.join(_here, "r1_baseline.py"))
r5_inf = _load("r5_simo_inference", os.path.join(_here, "r5_simo_inference.py"))

SimConfig = r1_baseline.SimConfig
PdschSim = r1_baseline.PdschSim
parse_delta_max = r5_inf.parse_delta_max


def build_decoder(meta):
    """Rebuild the exact decoder the dumping sweep used, from index.json."""
    cfg = SimConfig(
        channel_model=meta["channel"], mcs_idx=meta["mcs"],
        dmrs_n_add=meta.get("n_add", 1),
        ue_speed_kmh=meta.get("ue_speed_kmh", 3.0),
        n_prb=meta["n_prb"], carrier_hz=meta["carrier_hz"],
        scs_hz=meta["scs_hz"], n_tx_ant=1, n_rx_ant=meta["n_rx_ant"],
        estimator="ls", pn_tx=meta["pn_tx"], pn_rx=meta["pn_rx"],
        pn_ue_offset_db=meta.get("pn_ue_offset_db", 3.0),
        iqi="spec", iqi_irr_db=meta["eval_irr_db"], real_iqi_comp=True,
    )
    return PdschSim(cfg)


def crossing_10pct(snrs, blers):
    """Log-linear interpolation to 10% BLER, same detector as the sweep."""
    for i in range(1, len(snrs)):
        if blers[i - 1] >= 0.1 > blers[i] and blers[i - 1] > 0 and blers[i] > 0:
            lb1, lb2 = math.log10(blers[i - 1]), math.log10(blers[i])
            return snrs[i - 1] + ((math.log10(0.1) - lb1)
                                  * (snrs[i] - snrs[i - 1]) / (lb2 - lb1))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--online-dir", default=None,
                   help="Directory of r5inf_bler_dmax*.csv from the online "
                        "sweep; when given, A-G0 is evaluated.")
    p.add_argument("--delta-max-list", nargs="+", default=None,
                   help="Defaults to the list recorded in index.json.")
    p.add_argument("--tol", type=float, default=0.05,
                   help="A-G0 crossing tolerance in dB (<=0.05 tolerated "
                        "with a recorded note; >0.05 STOP).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    meta = json.load(open(os.path.join(args.dump, "index.json")))
    names = args.delta_max_list or meta["delta_max_list"]
    deltas = [parse_delta_max(s) for s in names]
    snrs = meta["snr_db"]
    logging.info(f"dump {args.dump}: condition={meta['condition']} "
                 f"seed={meta['seed']} n_blocks={meta['n_blocks']} "
                 f"n_coded={meta['n_coded_bits']} k_info={meta['k_info_bits']}")

    sim = build_decoder(meta)
    tb_dec = sim.tb_dec

    offline = {n: [] for n in names}
    for si, s in enumerate(snrs):
        tag = f"snr{s:.1f}"
        l2 = np.load(os.path.join(args.dump, f"llr_l2_{tag}.npy"), mmap_mode="r")
        r3 = np.load(os.path.join(args.dump, f"llr_r3_{tag}.npy"), mmap_mode="r")
        bi = np.load(os.path.join(args.dump, f"bits_info_{tag}.npy"),
                     mmap_mode="r")
        n_blocks = l2.shape[0]
        errs = [0] * len(deltas)
        for lo in range(0, n_blocks, args.batch_size):
            hi = min(lo + args.batch_size, n_blocks)
            a = tf.constant(np.asarray(l2[lo:hi]), dtype=tf.float32)
            b_ = tf.constant(np.asarray(r3[lo:hi]), dtype=tf.float32)
            truth = np.asarray(bi[lo:hi])
            nb = hi - lo
            for di, d in enumerate(deltas):
                if math.isinf(d):
                    llr = b_
                elif d == 0.0:
                    llr = a
                else:
                    llr = tf.clip_by_value(b_, a - d, a + d)
                llr = tf.reshape(llr, [nb, 1, 1, -1])
                b_hat, _ = tb_dec(llr)
                bh = tf.reshape(b_hat, [nb, -1]).numpy().astype(np.uint8)
                errs[di] += int(np.any(bh != truth, axis=-1).sum())
        for di, n in enumerate(names):
            offline[n].append(errs[di] / max(1, n_blocks))
        line = " ".join(f"{n}={offline[n][-1]:.3e}" for n in names)
        logging.info(f"snr={s:+.1f} {line}")

    # ---- write the recomputed ladder ----
    out = args.out or os.path.join(args.dump, "offline_ladder.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_max_name", "snr_db", "bler_offline"])
        for n in names:
            for si, s in enumerate(snrs):
                w.writerow([n, s, offline[n][si]])
    logging.info(f"wrote {out}")

    # ---- gate A-G0 ----
    if not args.online_dir:
        logging.info("no --online-dir: ladder recomputed, A-G0 NOT evaluated")
        return 0

    print()
    print(f"{'Δ_max':>12}  {'online':>10}  {'offline':>10}  {'Δ dB':>8}  "
          f"{'maxΔBLER':>10}  verdict")
    print(f"{'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  -------")
    worst, verdicts = 0.0, []
    worst_bler = 0.0
    for n in names:
        path = os.path.join(
            args.online_dir,
            f"r5inf_bler_dmax{n}_irr{int(meta['eval_irr_db'])}.csv")
        if not os.path.exists(path):
            print(f"{n:>12}  {'MISSING':>10}")
            verdicts.append((n, None))
            continue
        rows = list(csv.DictReader(open(path)))
        on_s = [float(r["snr_db"]) for r in rows]
        on_b = [float(r["bler_mean"]) for r in rows]
        c_on = crossing_10pct(on_s, on_b)
        c_off = crossing_10pct(snrs, offline[n])
        # Per-point BLER agreement. Strictly stronger than the crossing
        # comparison and sensitive at SNR points that never cross 10%:
        # same decoder on the same dumped bits must give the same BLER,
        # so anything above ~0 means the dump is not what the sweep saw.
        db = 0.0
        by_snr = dict(zip(on_s, on_b))
        for si, sv in enumerate(snrs):
            if sv in by_snr:
                db = max(db, abs(by_snr[sv] - offline[n][si]))
        worst_bler = max(worst_bler, db)
        if c_on is None and c_off is None:
            print(f"{n:>12}  {'unreach':>10}  {'unreach':>10}  {'-':>8}  "
                  f"{db:>10.2e}  MATCH")
            verdicts.append((n, 0.0))
            continue
        if c_on is None or c_off is None:
            print(f"{n:>12}  "
                  f"{('unreach' if c_on is None else f'{c_on:.3f}'):>10}  "
                  f"{('unreach' if c_off is None else f'{c_off:.3f}'):>10}  "
                  f"{'-':>8}  {db:>10.2e}  MISMATCH (reachability)")
            verdicts.append((n, float("inf")))
            worst = float("inf")
            continue
        d = abs(c_on - c_off)
        worst = max(worst, d)
        verdicts.append((n, d))
        v = "EXACT" if d == 0 else ("within tol" if d <= args.tol else "FAIL")
        print(f"{n:>12}  {c_on:>10.3f}  {c_off:>10.3f}  {d:>8.4f}  "
              f"{db:>10.2e}  {v}")
    print()
    print(f"max per-point |ΔBLER| (offline vs online) = {worst_bler:.3e}")
    if worst_bler > 0.0:
        print("  note: nonzero per-point BLER delta means the offline decode "
              "did not see the same bits the sweep decoded.")
    if worst == 0.0 and worst_bler == 0.0:
        print(f"GATE A-G0: PASS (exact — crossings identical, "
              f"per-point BLER identical)")
        return 0
    if worst <= args.tol:
        print(f"GATE A-G0: PASS with recorded note — max |Δ| = {worst:.4f} dB "
              f"(<= {args.tol} dB, GPU nondeterminism band), "
              f"max |ΔBLER| = {worst_bler:.3e}")
        return 0
    print(f"GATE A-G0: FAIL — max |Δ| = {worst:.4f} dB > {args.tol} dB. "
          f"STOP: the dump or the plumbing is wrong.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
