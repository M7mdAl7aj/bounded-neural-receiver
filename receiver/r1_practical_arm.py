"""r1_practical_arm.py - driver for the LMMSE classical arms of R1 (run P).
r1_baseline.py is NOT edited: this file replaces r1_baseline.run_sweep at runtime and calls r1_baseline.main() with the
usual run-F flags, so argument parsing, seed_all(--seed) and SimConfig are the
oracle runs' own. The sweep loop below is r1_baseline.run_sweep line for line
except where the estimator is built: it is INJECTED from a prior.

Modes (own flags, before `--`; everything after `--` goes to r1_baseline):
  --mode oracle     harvest the oracle prior with lmmse_practical.oracle_prior
                    (the oracle path's own computation), save it, reload it,
                    inject it. Must reproduce run F (16.9073 dB, seed 42).
  --mode practical  inject the saved practical prior (--prior).
  --mode harvest    build the practical prior (--prior is written), no sweep.
  --runs i [j ...]  Monte-Carlo run indices to execute (default: all). Each run
                    reseeds at its start (cell_seed(seed, run_idx)), so runs can
                    be split across processes and merged.
Example (oracle arm, CPU):
  CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 TF_CUDNN_DETERMINISTIC=1 \
  python r1_practical_arm.py --mode oracle --runs 0 --threads 40 -- \
    --channel C --mcs 27 --n-prb 51 --n-tx-ant 1 --n-rx-ant 2 --pn-tx on --pn-rx on \
    --iqi spec --iqi-irr-db 25 --real-iqi-comp --estimator lmmse-empirical \
    --snr-min 16 --snr-max 24 --snr-step 1 --num-tti 960 --num-mc-runs 2 --seed 42 --out-dir OUT
"""
import argparse, csv, hashlib, json, logging, os, sys, time

own = argparse.ArgumentParser(add_help=True)
own.add_argument("--mode", choices=["oracle", "practical", "harvest"], required=True)
own.add_argument("--prior", default=None, help="practical prior .npz (read in practical, written in harvest)")
own.add_argument("--runs", nargs="+", type=int, default=None)
own.add_argument("--threads", type=int, default=0, help="TF intra-op threads (0 = TF default)")
own.add_argument("--harvest-steps", type=int, default=1400)
if "--" not in sys.argv:
    sys.exit("usage: r1_practical_arm.py <own flags> -- <r1_baseline flags>")
cut = sys.argv.index("--")
A = own.parse_args(sys.argv[1:cut])
sys.argv = [sys.argv[0]] + sys.argv[cut + 1:]

import numpy as np
import tensorflow as tf
if A.threads > 0:
    tf.config.threading.set_intra_op_parallelism_threads(A.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
import r1_baseline as rb
import lmmse_practical as lp


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def injected_run_sweep(cfg, snr_range, num_tti, batch_size, num_mc_runs, out_dir, base_seed=42):
    os.makedirs(out_dir, exist_ok=True)
    if A.mode == "harvest":
        import r3_deeprx                      # the training SNR sampler
        assert cfg.estimator == "ls" and cfg.real_iqi_comp, "harvest needs --estimator ls --real-iqi-comp"
        sim = rb.PdschSim(cfg)
        R_freq, R_time, stats, snrs = lp.harvest_practical(sim, batch_size, A.harvest_steps,
                                                          r3_deeprx.sample_ebno_db)
        meta = dict(kind="practical", harvest_seed=base_seed, channel=cfg.channel_model,
                    iqi_irr_db=cfg.iqi_irr_db, real_iqi_comp=True, sampler="r3_deeprx.sample_ebno_db mean",
                    R_time="static all-ones 14x14", **stats)
        lp.save_prior(A.prior, R_freq, R_time, meta)
        np.save(os.path.splitext(A.prior)[0] + "_step_snr.npy", snrs)
        json.dump(dict(meta, prior_sha256_16=sha16(A.prior)), open(os.path.join(out_dir, "harvest_stats.json"), "w"), indent=1)
        logging.info(f"practical prior written: {A.prior} sha256[:16]={sha16(A.prior)} stats={stats}")
        return

    sim = rb.PdschSim(cfg)
    if A.mode == "oracle":
        R_freq, R_time = lp.oracle_prior(sim, batch_size, cfg.lmmse_warmup_slots)
        prior_path = os.path.join(out_dir, f"prior_oracle_seed{base_seed}.npz")
        lp.save_prior(prior_path, R_freq, R_time, dict(kind="oracle", seed=base_seed,
                      warmup_batches=cfg.lmmse_warmup_slots, batch_size=batch_size,
                      warmup_channel=cfg.lmmse_warmup_channel or cfg.channel_model))
    else:
        prior_path = A.prior
    R_freq, R_time, meta = lp.load_prior(prior_path)
    sim.chan_est = lp.make_estimator(sim, R_freq, R_time, cfg.lmmse_order)
    logging.info(f"injected estimator from {prior_path} (sha256[:16]={sha16(prior_path)}, kind={meta.get('kind')})")

    # ---- from here: r1_baseline.run_sweep, unchanged except the run subset ----
    _ = sim(batch_size, tf.constant(10.0))
    n_snr = len(snr_range)
    runs = A.runs if A.runs is not None else list(range(num_mc_runs))
    bler_all = np.zeros((num_mc_runs, n_snr))
    t0 = time.time()
    for run_idx in runs:
        seed = rb.cell_seed(base_seed, run_idx)
        rb.seed_all(seed)
        for i, snr_db in enumerate(snr_range):
            total_blocks = 0
            total_errs = 0
            for _ in range(max(1, num_tti // batch_size)):
                b, b_hat = sim(batch_size, tf.constant(float(snr_db)))
                block_err = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
                total_errs += int(tf.reduce_sum(tf.cast(block_err, tf.int32)))
                total_blocks += batch_size
            bler = total_errs / max(1, total_blocks)
            bler_all[run_idx, i] = bler
            logging.info(f"run={run_idx} snr={snr_db:+.1f}dB bler={bler:.3e} blocks={total_blocks} "
                         f"errs={total_errs} elapsed={(time.time() - t0) / 60:.1f}min")
    tag = "lmmse-empirical" if A.mode == "oracle" else "lmmse-practical"
    json.dump(dict(runs=runs, snr=[float(s) for s in snr_range], num_tti=num_tti, batch_size=batch_size,
                   bler={str(r): [float(x) for x in bler_all[r]] for r in runs}, prior=prior_path,
                   prior_sha256_16=sha16(prior_path), mode=A.mode, seed=base_seed),
              open(os.path.join(out_dir, f"runs_{'_'.join(map(str, runs))}.json"), "w"), indent=1)
    if sorted(runs) == list(range(num_mc_runs)):
        write_csv(out_dir, cfg, snr_range, bler_all, num_mc_runs, num_tti, tag)


def write_csv(out_dir, cfg, snr_range, bler_all, num_mc_runs, num_tti, tag):
    """r1_baseline.run_sweep's CSV, byte for byte (estimator tag aside)."""
    mean = bler_all.mean(axis=0)
    sem = bler_all.std(axis=0, ddof=1) / np.sqrt(num_mc_runs)
    name = (f"bler_cdl{cfg.channel_model}_mcs{cfg.mcs_idx}_"
            f"nadd{cfg.dmrs_n_add}_v{int(cfg.ue_speed_kmh)}_{tag}.csv")
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "bler_mean", "bler_sem", "n_runs", "n_blocks_per_run"])
        for i, s in enumerate(snr_range):
            w.writerow([s, mean[i], sem[i], num_mc_runs, num_tti])
    logging.info(f"wrote {path}")


rb.run_sweep = injected_run_sweep
rb.main()
