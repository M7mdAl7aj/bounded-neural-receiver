"""r5_simo_inference.py — Inference-time bounded certifiable receiver.

R5 = clip(R3_with_mirror_LLR,  L2_real_LLR ± Δ_max), per-LLR.

The trained neural receiver (R3 with mirror) is projected into the
certified safety region of half-width Δ_max around the classical L2-real
baseline at inference. NO retraining; the Δ_max sweep is a pure
inference-time operation on the already-trained R3 model.

Theorem 1 (trivial by construction):
  Δ_max → 0   ⇒  R5 LLR = L2-real LLR exactly.
  Δ_max → ∞   ⇒  R5 LLR = R3 LLR exactly.
Both endpoints are exact, not approximate.

This replaces an earlier trained-bounded-residual approach (not part of
this release), which failed because focal-BCE on the combined LLR gave the residual
head near-zero gradient when L2 was already correct on most bits
(silent residual: |residual|/|L2 LLR| ≈ 0.1% at unbounded). Inference-
time clipping avoids the gradient-vanishing problem entirely.

Framing:
  "R5 is the trained neural receiver projected into a certified safety
   region around the classical baseline. The projection is applied at
   inference; a trained bounded residual is future work."

The Δ_max sweep produces two figures:
  - BLER-vs-SNR per Δ_max (one curve per bound)
  - 10%-BLER-vs-Δ_max (the gain-vs-safety curve)
"""

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
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
from sionna.phy import config as sionna_config


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(os.path.abspath(__file__))
r1_baseline = _load("r1_baseline", os.path.join(_here, "r1_baseline.py"))
r3_deeprx = _load("r3_deeprx", os.path.join(_here, "r3_deeprx.py"))

SimConfig = r1_baseline.SimConfig
MCS_TABLE = r1_baseline.MCS_TABLE
PdschSim = r1_baseline.PdschSim
seed_all = r1_baseline.seed_all          # shared deterministic seeding
cell_seed = r1_baseline.cell_seed        # per-cell CRN seed from --seed
NeuralReceiver = r3_deeprx.NeuralReceiver
R3TrainingSim = r3_deeprx.R3TrainingSim


# ----------------------------------------------------------------------------
# Δ_max enumeration
# ----------------------------------------------------------------------------

DELTA_MAX_NAMED = {
    "0": 0.0,
    "ln2": math.log(2.0),
    "ln4": math.log(4.0),
    "ln8": math.log(8.0),
    "ln16": math.log(16.0),
    "unbounded": float("inf"),
}


def parse_delta_max(s: str) -> float:
    if s in DELTA_MAX_NAMED:
        return DELTA_MAX_NAMED[s]
    return float(s)


# ----------------------------------------------------------------------------
# Per-bit LLR artifact dump (--dump-llrs)
# ----------------------------------------------------------------------------

def _condition_token(channel, irr_db):
    """Condition name of a dump folder: cdlC_irr25, irr20, irr35, cdlD,
    cdlE (and e.g. cdlD_irr20 for a joint channel + IRR shift)."""
    ch = str(channel).upper()
    irr = int(round(float(irr_db)))
    if ch == "C":
        return "cdlC_irr25" if irr == 25 else f"irr{irr}"
    return f"cdl{ch}" if irr == 25 else f"cdl{ch}_irr{irr}"


class _LlrDump:
    """Writes the per-bit artifacts needed to recompute the Δ_max ladder
    offline (L_R3, L_L2-real, ground-truth bits, TTI/block boundaries;
    float32 is mandatory — a float16 dump would be 25.4 GB but cannot
    reproduce the recorded error counts exactly).

    One directory per (condition, seed), passed via --dump-llrs. Per SNR
    point, four arrays:

      llr_l2_snr<s>.npy      float32 [n_blocks, n_coded]  L_L2-real
      llr_r3_snr<s>.npy      float32 [n_blocks, n_coded]  L_R3
      bits_coded_snr<s>.npy  uint8   [n_blocks, n_coded]  c — run B compares
                                                          sign(LLR) to this
      bits_info_snr<s>.npy   uint8   [n_blocks, k_info]   b — BLER ground truth

    The row index IS the block boundary. Rows [j*blocks_per_run,
    (j+1)*blocks_per_run) are MC run j; the boundaries go into index.json
    explicitly so an offline reader never infers them.

    Shapes are discovered from the first batch rather than derived, so a
    change in TBS or grid size cannot silently produce a mis-shaped dump.
    The disk preflight runs at that same moment, before a single array is
    allocated: a run that cannot fit its own dump must die immediately, not
    at 80% through a multi-hour sweep.
    """

    _MARGIN_BYTES = 20 * (1024 ** 3)   # 20 GB free-space margin

    def __init__(self, out_dir, snr_range, blocks_per_run, n_mc_runs, meta):
        self.dir = out_dir
        self.snr_range = list(snr_range)
        self.blocks_per_run = int(blocks_per_run)
        self.n_mc_runs = int(n_mc_runs)
        self.n_blocks = self.blocks_per_run * self.n_mc_runs
        self.meta = dict(meta)
        self._arrays = {}          # snr_idx -> dict(name -> memmap)
        self._cursor = {}          # (run_idx, snr_idx) -> blocks written
        self._shapes = None
        os.makedirs(self.dir, exist_ok=True)

    @staticmethod
    def _tag(s):
        return f"snr{s:.1f}"

    def _preflight(self, n_coded, k_info):
        per_block = 4 * n_coded * 2 + n_coded + k_info
        total = per_block * self.n_blocks * len(self.snr_range)
        free = shutil.disk_usage(self.dir).free
        logging.info(
            f"LLR dump sizing: {per_block/1e6:.3f} MB/slot x {self.n_blocks} "
            f"blocks x {len(self.snr_range)} SNR points = {total/1e9:.2f} GB; "
            f"free {free/1e9:.1f} GB")
        if total + self._MARGIN_BYTES > free:
            raise SystemExit(
                f"LLR dump needs {total/1e9:.2f} GB + 20 GB margin but only "
                f"{free/1e9:.1f} GB is free at {self.dir}. Refusing to start a "
                f"sweep that would die partway through. Clear space or drop "
                f"--dump-llrs.")
        self.meta["bytes_per_block"] = per_block
        self.meta["estimated_bytes"] = total

    def _open(self, snr_idx, n_coded, k_info):
        if self._shapes is None:
            self._shapes = (n_coded, k_info)
            self._preflight(n_coded, k_info)
        elif self._shapes != (n_coded, k_info):
            raise SystemExit(
                f"LLR dump: shape changed mid-sweep, {self._shapes} -> "
                f"{(n_coded, k_info)}. Refusing to write a ragged dump.")
        tag = self._tag(self.snr_range[snr_idx])
        specs = {
            "llr_l2": (np.float32, (self.n_blocks, n_coded)),
            "llr_r3": (np.float32, (self.n_blocks, n_coded)),
            "bits_coded": (np.uint8, (self.n_blocks, n_coded)),
            "bits_info": (np.uint8, (self.n_blocks, k_info)),
        }
        arrs = {}
        for name, (dt, shape) in specs.items():
            path = os.path.join(self.dir, f"{name}_{tag}.npy")
            arrs[name] = np.lib.format.open_memmap(
                path, mode="w+", dtype=dt, shape=shape)
        self._arrays[snr_idx] = arrs
        return arrs

    def write(self, run_idx, snr_idx, b, c, llr_l2, llr_r3):
        b = np.asarray(b).reshape(np.asarray(b).shape[0], -1)
        c = np.asarray(c).reshape(np.asarray(c).shape[0], -1)
        l2 = np.asarray(llr_l2).reshape(np.asarray(llr_l2).shape[0], -1)
        r3 = np.asarray(llr_r3).reshape(np.asarray(llr_r3).shape[0], -1)
        n_batch, n_coded = l2.shape
        k_info = b.shape[1]
        arrs = self._arrays.get(snr_idx) or self._open(snr_idx, n_coded, k_info)
        key = (run_idx, snr_idx)
        cur = self._cursor.get(key, 0)
        lo = run_idx * self.blocks_per_run + cur
        hi = lo + n_batch
        if hi > (run_idx + 1) * self.blocks_per_run:
            raise SystemExit(
                f"LLR dump: run {run_idx} wrote past its {self.blocks_per_run}"
                f"-block region. Loop counts and dump geometry disagree.")
        arrs["llr_l2"][lo:hi] = l2.astype(np.float32, copy=False)
        arrs["llr_r3"][lo:hi] = r3.astype(np.float32, copy=False)
        arrs["bits_coded"][lo:hi] = c.astype(np.uint8, copy=False)
        arrs["bits_info"][lo:hi] = b.astype(np.uint8, copy=False)
        self._cursor[key] = cur + n_batch

    def close(self):
        for arrs in self._arrays.values():
            for a in arrs.values():
                a.flush()
        n_coded, k_info = self._shapes if self._shapes else (None, None)
        written = {f"{self._tag(self.snr_range[i])}": sum(
            self._cursor.get((r, i), 0) for r in range(self.n_mc_runs))
            for i in range(len(self.snr_range))}
        short = None
        try:
            short = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        index = {
            "schema": "ra4-llr-dump/1",
            "written_by": "r5_simo_inference.py --dump-llrs",
            "git_commit": short,
            "snr_db": [float(x) for x in self.snr_range],
            "n_blocks": self.n_blocks,
            "blocks_per_run": self.blocks_per_run,
            "n_mc_runs": self.n_mc_runs,
            "mc_boundaries": [j * self.blocks_per_run
                              for j in range(self.n_mc_runs + 1)],
            "n_coded_bits": n_coded,
            "k_info_bits": k_info,
            "dtypes": {"llr_l2": "float32", "llr_r3": "float32",
                       "bits_coded": "uint8", "bits_info": "uint8"},
            "llr_sign_convention": (
                "positive LLR encodes bit value 1 "
                "(L = log P(b=1)/P(b=0)); measured 1.0000 agreement"),
            "blocks_written_per_snr": written,
            "files_per_snr": ["llr_l2", "llr_r3", "bits_coded", "bits_info"],
        }
        index.update(self.meta)
        path = os.path.join(self.dir, "index.json")
        with open(path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        logging.info(f"LLR dump complete: {self.dir} (index.json written)")
        short_of = [k for k, v in written.items() if v != self.n_blocks]
        if short_of:
            raise SystemExit(
                f"LLR dump: these SNR points are short of {self.n_blocks} "
                f"blocks: {short_of}. The dump is incomplete; an offline "
                f"recount would run on partial data.")


# ----------------------------------------------------------------------------
# R5 sim
# ----------------------------------------------------------------------------

class R5InferenceSim(tf.keras.Model):
    """Runs one shared TX→channel→y forward, then both:
      - L2-real classical chain → L2 LLR sequence
      - R3-with-mirror neural   → R3 LLR sequence
    and combines:
      R5 LLR = L2 LLR + clip(R3 LLR − L2 LLR, −Δ_max, +Δ_max)
             = clip(R3 LLR, L2 LLR − Δ_max, L2 LLR + Δ_max)
    decoded through the same LDPC decoder.

    real_iqi_comp=True is required on the underlying PdschSim so the L2
    LLRs come from the IQI-compensated chain (the deployable baseline).
    """

    def __init__(self, cfg: SimConfig, r3_net: NeuralReceiver,
                 num_conv_channels: int, num_blocks: int):
        super().__init__()
        self.cfg = cfg
        self.r3_net = r3_net
        self._inner = R3TrainingSim(
            cfg, r3_net,
            num_conv_channels=num_conv_channels,
            num_blocks=num_blocks,
        )

    @tf.function(jit_compile=False)
    def forward(self, batch_size, ebno_db):
        """Returns (b, c, llr_l2_seq, llr_r3_seq, no).

        Both LLR streams are aligned to the TBEncoder's coded-bit
        ordering and are produced from the SAME y/h/noise — so the
        per-LLR difference (R3 − L2) is meaningful.

        `c` (the coded bits) is returned for the LLR artifact dump: run
        B's would-have-been-wrong metric compares sign(LLR) against the
        coded bit, and `b` alone cannot supply that. It was already
        computed here before the dump existed, so exposing it draws no extra RNG and
        leaves the numerical path unchanged.
        """
        inner = self._inner
        # ---- TX ----
        b = inner.pdsch.source([batch_size, 1, 1, inner.pdsch.tb_size])
        c = inner.pdsch.tb_enc(b)
        x = inner.pdsch.mapper(c)
        x_rg = inner.pdsch.rg_mapper(x)
        no = r1_baseline.ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=inner.pdsch.q_m,
            coderate=inner.pdsch.code_rate,
            resource_grid=inner.pdsch.rg,
        )
        # ---- Impaired channel ----
        if inner.pdsch._td_active:
            x_t_sym = tf.signal.ifft(x_rg)
            x_t = tf.reshape(x_t_sym,
                             [batch_size, 1, 1, inner.pdsch._slot_len])
            if inner.pdsch.pn_tx is not None:
                phi_tx = inner.pdsch.pn_tx((batch_size, 1, 1))
                x_t = x_t * tf.exp(tf.complex(0.0, phi_tx))
            x_t_sym = tf.reshape(x_t, [batch_size, 1, 1,
                                       self.cfg.n_symbols,
                                       self.cfg.fft_size])
            x_rg = tf.signal.fft(x_t_sym)
        y, _h_true = inner.pdsch.ofdm_channel(x_rg, no)
        if inner.pdsch._td_active:
            y_t_sym = tf.signal.ifft(y)
            y_t = tf.reshape(y_t_sym,
                             [batch_size, 1, self.cfg.n_rx_ant,
                              inner.pdsch._slot_len])
            if inner.pdsch.pn_rx is not None:
                phi_rx = inner.pdsch.pn_rx(
                    (batch_size, 1, self.cfg.n_rx_ant))
                y_t = y_t * tf.exp(tf.complex(0.0, phi_rx))
            if inner.pdsch.iqi is not None:
                y_t = inner.pdsch.iqi(y_t)
            y_t_sym = tf.reshape(y_t, [batch_size, 1, self.cfg.n_rx_ant,
                                       self.cfg.n_symbols,
                                       self.cfg.fft_size])
            y = tf.signal.fft(y_t_sym)

        # ---- L2-real IQI compensation (modifies y) ----
        if self.cfg.real_iqi_comp and inner.pdsch.iqi is not None:
            y_l2 = self._apply_l2_real(y, no)
        else:
            y_l2 = y

        # ---- L2 LLRs (classical chain on IQI-compensated y) ----
        h_hat_l2, err_var_l2 = inner.pdsch.chan_est(y_l2, no)
        x_hat_l2, no_eff_l2 = inner.pdsch.equalizer(
            y_l2, h_hat_l2, err_var_l2, no)
        llr_l2_grid = inner.pdsch.demapper(x_hat_l2, no_eff_l2)
        B = tf.shape(c)[0]
        llr_l2_seq = tf.reshape(llr_l2_grid, [B, 1, 1, -1])

        # ---- R3 LLRs (neural receiver on RAW y, same as training) ----
        # R3 was trained on raw y (not L2-compensated y); use raw.
        h_hat_r3, _ev_r3 = inner.ls_est(y, no)
        h_hat_r3 = tf.squeeze(h_hat_r3, axis=[3, 4])
        pilot_mask = inner._build_pilot_mask()
        no_b = tf.broadcast_to(tf.reshape(no, [-1]), [batch_size])
        r3_llr_grid = self.r3_net(
            (y, h_hat_r3, pilot_mask, no_b), training=False)
        llr_r3_seq = inner._grid_to_seq(r3_llr_grid)

        return b, c, llr_l2_seq, llr_r3_seq, no

    def combine(self, llr_l2_seq, llr_r3_seq, delta_max: float):
        """R5 = clip(R3, L2 − Δ_max, L2 + Δ_max).

        Theorem 1 (exact, by construction):
          Δ_max = 0   →  R5 = L2 exactly.
          Δ_max = ∞   →  R5 = R3 exactly.
        """
        if math.isinf(delta_max):
            return llr_r3_seq
        if delta_max == 0.0:
            return llr_l2_seq
        return tf.clip_by_value(
            llr_r3_seq,
            llr_l2_seq - delta_max,
            llr_l2_seq + delta_max,
        )

    def _apply_l2_real(self, y, no):
        """Inline DMRS-estimated 2×2 mirror-pair inverse, same algorithm
        as r1_baseline.PdschSim.call() when real_iqi_comp=True."""
        h_hat_iqi, _ = self._inner.pdsch._ls_est_for_iqi(y, no)
        h_hat_iqi = tf.squeeze(h_hat_iqi, axis=[3, 4])
        num_taps_keep = 16
        F = self.cfg.fft_size
        h_cir = tf.signal.ifft(h_hat_iqi)
        mask_np = np.zeros(F, dtype=np.complex64)
        mask_np[:num_taps_keep] = 1.0
        mask_np[-num_taps_keep:] = 1.0
        cir_mask = tf.constant(mask_np, dtype=tf.complex64)
        cir_mask = tf.reshape(cir_mask, [1, 1, 1, 1, F])
        h_smooth = tf.signal.fft(h_cir * cir_mask)
        x_dmrs = tf.reshape(self._inner.pdsch._iqi_pilot_grid,
                            [1, 1, 1, self.cfg.n_symbols, F])
        mirror_mask = tf.reshape(self._inner.pdsch._iqi_mirror_mask,
                                 [1, 1, 1, self.cfg.n_symbols, F])
        residual = y - h_smooth * x_dmrs
        hx = h_smooth * x_dmrs
        hx_mirror = tf.roll(tf.reverse(hx, axis=[-1]), shift=1, axis=-1)
        mirror_product_conj = tf.math.conj(hx_mirror)
        mask_c = tf.cast(mirror_mask, tf.complex64)
        num_g = tf.reduce_sum(
            residual * tf.math.conj(mirror_product_conj) * mask_c,
            axis=[0, 1, 3, 4])
        den_g = tf.reduce_sum(
            tf.cast(tf.math.abs(mirror_product_conj) ** 2 * mirror_mask,
                    tf.complex64),
            axis=[0, 1, 3, 4])
        gamma = num_g / (den_g + tf.cast(1e-12, tf.complex64))
        y_mirror_conj = tf.math.conj(
            tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1))
        beta = tf.reshape(gamma, [1, 1, self.cfg.n_rx_ant, 1, 1])
        det = tf.cast(1.0, tf.complex64) - beta * tf.math.conj(beta)
        return (y - beta * y_mirror_conj) / det

    @tf.function(jit_compile=False)
    def decode(self, batch_size, ebno_db, delta_max):
        """End-to-end: forward + clip + LDPC → (b, b_hat)."""
        b, _c, llr_l2, llr_r3, _no = self.forward(batch_size, ebno_db)
        llr_r5 = self.combine(llr_l2, llr_r3, float(delta_max))
        b_hat, _crc = self._inner.pdsch.tb_dec(llr_r5)
        return b, b_hat


# ----------------------------------------------------------------------------
# Δ_max sweep — single trained R3, multiple Δ_max points, one SNR grid
# ----------------------------------------------------------------------------

def run_sweep(args):
    seed_all(args.seed)
    cfg = SimConfig(
        channel_model=args.channel, mcs_idx=args.mcs, dmrs_n_add=args.n_add,
        ue_speed_kmh=args.ue_speed_kmh, n_prb=args.n_prb,
        carrier_hz=args.carrier_hz, scs_hz=args.scs_hz,
        n_tx_ant=1, n_rx_ant=args.n_rx_ant, estimator="ls",
        pn_tx=args.pn_tx, pn_rx=args.pn_rx,
        pn_ue_offset_db=args.pn_ue_offset_db,
        iqi="spec", iqi_irr_db=args.eval_irr_db, real_iqi_comp=True,
        delay_spread_ns=args.delay_spread_ns,
    )
    q_m = MCS_TABLE[args.mcs][1]
    r3_net = NeuralReceiver(
        num_rx_ant=cfg.n_rx_ant,
        num_conv_channels=args.num_conv_channels,
        num_blocks=args.num_blocks,
        num_bits_per_symbol=q_m,
        # Run D uses the archived NO-MIRROR model (r3_full: 10 input channels
        # vs the mirror model's 14), so this can no longer be hardcoded. The
        # default is unchanged, so every prior invocation is byte-identical.
        use_mirror_feature=not args.no_mirror_feature,
    )
    sim = R5InferenceSim(cfg, r3_net,
                         num_conv_channels=args.num_conv_channels,
                         num_blocks=args.num_blocks)
    # Materialize weights with one forward, then load.
    _ = sim.forward(args.batch_size, tf.constant(20.0))
    r3_net.load_weights(os.path.join(args.r3_weights, "weights.h5"))
    logging.info(f"loaded R3 weights from {args.r3_weights}")

    delta_max_list = [parse_delta_max(s) for s in args.delta_max_list]
    delta_max_names = list(args.delta_max_list)

    # ---- Theorem 1 numerical check ----
    b, _c, llr_l2, llr_r3, _no = sim.forward(args.batch_size, tf.constant(20.0))
    llr_zero = sim.combine(llr_l2, llr_r3, 0.0)
    llr_inf = sim.combine(llr_l2, llr_r3, float("inf"))
    diff_zero = tf.reduce_max(tf.abs(llr_zero - llr_l2)).numpy()
    diff_inf = tf.reduce_max(tf.abs(llr_inf - llr_r3)).numpy()
    logging.info(f"Theorem 1 endpoints: "
                 f"max |R5(Δ=0) − L2| = {diff_zero:.2e}, "
                 f"max |R5(Δ=∞) − R3| = {diff_inf:.2e}")
    if diff_zero > 1e-5 or diff_inf > 1e-5:
        raise SystemExit("Theorem 1 numerical check FAILED")
    logging.info("Theorem 1 endpoints PASS")

    snr_range = np.arange(args.eval_snr_min, args.eval_snr_max + 1e-9,
                          args.eval_snr_step)
    n_snr = len(snr_range)
    n_d = len(delta_max_list)
    # bler[delta_idx, mc_idx, snr_idx]
    bler = np.zeros((n_d, args.eval_mc_runs, n_snr))

    # LLR artifact dump — opt-in. With --dump-llrs absent nothing below
    # changes.
    dump = None
    if args.dump_llrs:
        blocks_per_run = (max(1, args.eval_num_tti // args.batch_size)
                          * args.batch_size)
        dump = _LlrDump(
            args.dump_llrs, snr_range, blocks_per_run, args.eval_mc_runs,
            meta={
                "condition": _condition_token(args.channel, args.eval_irr_db),
                "channel": args.channel,
                "eval_irr_db": args.eval_irr_db,
                "seed": args.seed,
                "mcs": args.mcs,
                "n_prb": args.n_prb,
                "n_rx_ant": args.n_rx_ant,
                "carrier_hz": args.carrier_hz,
                "scs_hz": args.scs_hz,
                "eval_num_tti": args.eval_num_tti,
                "eval_mc_runs": args.eval_mc_runs,
                "batch_size": args.batch_size,
                "delta_max_list": list(args.delta_max_list),
                "r3_weights": args.r3_weights,
                "pn_tx": args.pn_tx,
                "pn_rx": args.pn_rx,
            })
        logging.info(f"LLR dump ENABLED -> {args.dump_llrs}")

    # IMPORTANT: identical seeded forward for every Δ_max at every (run, snr)
    # so the clip is the ONLY thing differing across bound values.
    # We achieve this by re-seeding before each (run, snr) iteration AND
    # by running all Δ_max values within the inner loop using the same
    # forward outputs.
    for run_idx in range(args.eval_mc_runs):
        seed = 1000 + run_idx   # kept for the log line only; see cell_seed
        for i, s in enumerate(snr_range):
            # Re-seed Sionna RNG so each (run, snr) cell starts the same
            # regardless of how many Δ_max points we evaluate. Within the
            # cell, the SAME stream of (b, llr_l2, llr_r3) is consumed by
            # all Δ_max bounds.
            # cell_seed must receive the index within the FULL grid,
            # not within this process's slice. A sweep sharded by SNR point
            # otherwise passes snr_idx=0 in every shard and silently draws
            # different realizations than the unsharded run. --snr-index-offset
            # lets a shard declare where its slice starts.
            seed_all(cell_seed(args.seed, run_idx, args.snr_index_offset + i))
            errs = [0] * n_d
            blocks = 0
            for _ in range(max(1, args.eval_num_tti // args.batch_size)):
                b, c, llr_l2, llr_r3, _no = sim.forward(
                    args.batch_size, tf.constant(float(s)))
                if dump is not None:
                    dump.write(run_idx, i, b, c, llr_l2, llr_r3)
                for di, d in enumerate(delta_max_list):
                    llr_r5 = sim.combine(llr_l2, llr_r3, float(d))
                    b_hat, _ = sim._inner.pdsch.tb_dec(llr_r5)
                    bf = tf.reshape(b, [args.batch_size, -1])
                    bhf = tf.reshape(b_hat, [args.batch_size, -1])
                    be = tf.reduce_any(tf.not_equal(bf, bhf), axis=-1)
                    errs[di] += int(tf.reduce_sum(tf.cast(be, tf.int32)))
                blocks += args.batch_size
            for di, name in enumerate(delta_max_names):
                bler[di, run_idx, i] = errs[di] / max(1, blocks)
            line = " ".join(
                f"{n}={bler[di, run_idx, i]:.3e}"
                for di, n in enumerate(delta_max_names))
            logging.info(f"run={run_idx} snr={s:+.1f} {line}")

    if dump is not None:
        dump.close()

    # ---- Persist + summarize ----
    os.makedirs(args.out_dir, exist_ok=True)
    mean = bler.mean(axis=1)               # [n_d, n_snr]
    sem = bler.std(axis=1, ddof=1) / max(1, math.sqrt(args.eval_mc_runs))

    # BLER vs SNR — one CSV per Δ_max
    for di, name in enumerate(delta_max_names):
        path = os.path.join(args.out_dir,
                            f"r5inf_bler_dmax{name}_irr{int(args.eval_irr_db)}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["snr_db", "bler_mean", "bler_sem", "n_runs", "n_tti",
                        "delta_max", "delta_max_value"])
            for i, s in enumerate(snr_range):
                w.writerow([s, mean[di, i], sem[di, i],
                            args.eval_mc_runs, args.eval_num_tti,
                            name, delta_max_list[di]])
        logging.info(f"wrote {path}")

    # 10%-BLER vs Δ_max — the gain-vs-safety curve
    summary_path = os.path.join(args.out_dir, "r5inf_dmax_sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_max_name", "delta_max_value", "snr10_db",
                    "floor_bler"])
        print()
        print(f"{'Δ_max':>12}  {'value':>10}  {'10%-BLER':>10}  {'floor':>10}")
        print(f"{'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
        for di, name in enumerate(delta_max_names):
            snr10 = None
            for i in range(1, n_snr):
                if mean[di, i-1] >= 0.1 > mean[di, i] \
                   and mean[di, i-1] > 0 and mean[di, i] > 0:
                    lb1 = math.log10(mean[di, i-1])
                    lb2 = math.log10(mean[di, i])
                    snr10 = snr_range[i-1] + (
                        (math.log10(0.1) - lb1)
                        * (snr_range[i] - snr_range[i-1])
                        / (lb2 - lb1))
                    break
            floor = float(np.min([x for x in mean[di] if x > 0])) \
                if (mean[di] > 0).any() else 0.0
            w.writerow([name, delta_max_list[di],
                        f"{snr10:.4f}" if snr10 is not None else "unreachable",
                        floor])
            v = (f"{delta_max_list[di]:.4f}"
                 if not math.isinf(delta_max_list[di]) else "inf")
            s10 = f"{snr10:.2f} dB" if snr10 is not None else "unreachable"
            print(f"{name:>12}  {v:>10}  {s10:>10}  {floor:>10.3e}")
    print()
    logging.info(f"wrote {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r3-weights", default="./results/r3_mirror/final")
    p.add_argument("--delta-max-list", nargs="+",
                   default=["0", "ln2", "ln4", "ln8", "ln16", "unbounded"])
    p.add_argument("--mcs", type=int, default=27)
    p.add_argument("--channel", default="C")
    p.add_argument("--n-add", type=int, default=1)
    p.add_argument("--n-prb", type=int, default=51)
    p.add_argument("--carrier-hz", type=float, default=7.125e9)
    p.add_argument("--scs-hz", type=float, default=30e3)
    p.add_argument("--ue-speed-kmh", type=float, default=3.0)
    p.add_argument("--n-rx-ant", type=int, default=2)
    p.add_argument("--pn-tx", default="on", choices=["on", "off"])
    p.add_argument("--pn-rx", default="on", choices=["on", "off"])
    p.add_argument("--pn-ue-offset-db", type=float, default=3.0)
    p.add_argument("--num-conv-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-irr-db", type=float, default=25.0)
    p.add_argument("--eval-snr-min", type=float, default=16.0)
    p.add_argument("--eval-snr-max", type=float, default=24.0)
    p.add_argument("--eval-snr-step", type=float, default=1.0)
    p.add_argument("--eval-num-tti", type=int, default=960)
    p.add_argument("--eval-mc-runs", type=int, default=2)
    p.add_argument("--delay-spread-ns", type=float, default=100.0,
                   help="CDL RMS delay spread in ns (default 100.0 = the "
                        "training point; 300 for run H's primary shift). "
                        "Default preserves prior behaviour exactly.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="./results/r5_inference")
    p.add_argument("--snr-index-offset", type=int, default=0,
                   help="Index of this run's first SNR point within the FULL "
                        "grid. Sharding by SNR without this makes every "
                        "shard use snr_idx=0 and draw different realizations "
                        "than the unsharded sweep. Default 0 = unsharded, so "
                        "the default path is unchanged.")
    p.add_argument("--no-mirror-feature", action="store_true",
                   help="Load a model trained WITHOUT the conjugate-mirror "
                        "input channels (run D's archived r3_full). Default "
                        "off, so the default path is unchanged.")
    p.add_argument("--dump-llrs", default=None, metavar="DIR",
                   help="Write per-bit artifacts (L_R3, L_L2-real, "
                        "coded bits, info bits, block boundaries) to DIR for "
                        "the offline runs (Δ_max recount, runs A/B). float32, ~0.53 MB per "
                        "slot. Omitted by default; when omitted the numerical "
                        "path is unchanged.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_sweep(args)


if __name__ == "__main__":
    main()
