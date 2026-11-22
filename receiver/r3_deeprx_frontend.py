"""r3_deeprx_frontend.py — R3 (DeepRx-style ResNet 2D conv) with a selectable
channel-estimate input: LS, or the practical-prior LMMSE estimate (the
LMMSE-fed network). See apply_front_end.

Can a fully-neural receiver cross 10%-BLER at MCS 27 / 7.125 GHz / CDL-C /
SIMO 1×2 with impairments ON, where the classical LS-R1 saturates at 80%?

Architecture (transcribed from Sionna 2.x Neural_Receiver tutorial, then
extended):
  - Input stack (channels-last):
      • Received y, real + imag, per RX antenna  → 2·num_rx_ant channels
      • LS channel estimate, real + imag, interpolated → 2·num_rx_ant channels
      • Binary pilot mask → 1 channel
      • log10(N0) broadcast → 1 channel
    Total: 2*(2·num_rx_ant) + 2 channels (= 10 for SIMO 1×2).
  - Input 3×3 conv → num_conv_channels feature maps.
  - N_blocks ResNet blocks: LN → ReLU → 3×3 conv → LN → ReLU → 3×3 conv → +skip.
  - Output 3×3 conv → num_bits_per_symbol per RE.

Training (the training loop is not part of this release; the trained
weights are):
  - Three-phase curriculum:
      A: imitation MSE vs APP-demapper oracle LLRs (~10% of iters)
      B: focal-BCE on coded bits, γ=2                  (the remaining iters)
      C: decoder-in-loop fine-tuning                    (conditional)
  - SNR ~ N(22, 3²) clipped to [16, 28] dB (sample_ebno_db below).
  - Adam, lr 1e-3, cosine decay over Phase B.
  - On-the-fly data generation via the existing PdschSim's TX chain +
    impairment pipeline (no stored dataset).

Entry point: --load-weights DIR loads a trained network (DIR/weights.h5) and
runs one BLER sweep at the configured channel and impairments (run_eval_only).
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import tensorflow as tf

# GPU memory growth + module load BEFORE any TF op
_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass

import sionna
from sionna.phy import config as sionna_config
from sionna.phy.ofdm import LSChannelEstimator, ResourceGridDemapper
from sionna.phy.mapping import Demapper

# Import the existing R1 baseline so we can reuse SimConfig, PdschSim,
# MCS_TABLE, the impairment classes, etc.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "r1_baseline",
    os.path.join(os.path.dirname(__file__), "r1_baseline.py"),
)
r1_baseline = importlib.util.module_from_spec(_spec)
sys.modules["r1_baseline"] = r1_baseline
_spec.loader.exec_module(r1_baseline)
SimConfig = r1_baseline.SimConfig
PdschSim = r1_baseline.PdschSim
MCS_TABLE = r1_baseline.MCS_TABLE

# ---- Channel-estimate front end. This file is a COPY of r3_deeprx.py plus
# apply_front_end. --front-end ls installs nothing (the stock LS estimate, byte for
# byte). --front-end lmmse-practical replaces R3's channel-estimate input
# (sim.ls_est, built once in R3TrainingSim.__init__ and read at first trace) by the
# practical-prior LMMSE estimator (lmmse_practical.make_estimator), loaded from the
# prior file only if its sha256 equals the recorded one. The received grid, pilot mask,
# noise level and conjugate-mirror feature are unchanged. Every call also writes the
# sidecar front_end.json (front end, prior sha, seed, argv) into --out-dir.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONT_END_ECHOED = []


def apply_front_end(sim, args):
    import hashlib
    sha = None
    if args.front_end == "lmmse-practical":
        import lmmse_practical as lp
        want = open(args.prior_sha).read().split()[0]
        sha = hashlib.sha256(open(args.prior, "rb").read()).hexdigest()
        if sha != want:
            raise SystemExit(f"REFUSED: prior {args.prior} sha256 {sha[:16]} != recorded {want[:16]}")
        R_freq, R_time, _meta = lp.load_prior(args.prior)
        _est = lp.make_estimator(sim.pdsch, R_freq, R_time, sim.cfg.lmmse_order)
        # Error-variance floor: the rank-deficient practical prior makes the interpolator's frequency step clamp
        # its error variance to 0 at isolated noise levels, the time step then solves a near-singular system, and
        # the estimate can overflow (one training run went NaN at 20.435810 dB). The floor
        # (lmmse_errvar_floor.py) changes only clamped entries; valid estimates are bit-identical.
        import lmmse_errvar_floor as lf
        lf.install(_est)
        # Sionna's LMMSEInterpolator builds its interpolation matrices with tf.linalg.lstsq(fast=False), which has
        # no gradient, so a training step under GradientTape failed. The estimate is an input feature with nothing
        # trainable upstream: stop_gradient on both outputs (identity in the forward pass) keeps the solve off the
        # backward graph.
        # Per-slot LS fallback at the output (a slot with any non-finite entry takes the stock LS estimate, the one
        # sim.ls_est held until here; counted, logged per call, totals at exit). Finite slots unchanged.
        _fb = lf.with_fallback(_est, sim.ls_est, log_path=os.path.join(args.out_dir, "fallback_log.txt"),
                               sidecar=os.path.join(args.out_dir, "front_end_fallback.json"))
        sim.ls_est = lambda y, no: tuple(tf.stop_gradient(t) for t in _fb(y, no))
    mode = "eval"   # this release runs evaluation only (the training loop recorded "train" here)
    side = {"front_end": args.front_end, "prior": args.prior if sha else None,
            "prior_sha256": sha, "seed": args.seed, "mode": mode,
            "load_weights": args.load_weights, "script": os.path.basename(__file__), "argv": sys.argv}
    if sha:
        side["errvar_floor_rel"] = lf.FLOOR_REL   # error-variance floor
        side["slot_fallback"] = "per-slot stock LS estimate where the LMMSE estimate is non-finite; totals in front_end_fallback.json"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "front_end.json"), "w") as f:
        json.dump(side, f, indent=1)
    if not _FRONT_END_ECHOED:
        _FRONT_END_ECHOED.append(1)
        logging.info(f"CONFIG front_end={args.front_end} prior_sha256={sha} seed={args.seed} mode={mode} "
                     f"lmmse_order={sim.cfg.lmmse_order if sha else '-'} out_dir={args.out_dir}")
    return sim


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class ResidualBlock(tf.keras.layers.Layer):
    """LN → ReLU → 3×3 conv → LN → ReLU → 3×3 conv → +skip.

    LayerNorm here normalises over the last (channels) axis only — TF/Keras
    LayerNormalization with default axis=-1.  This is different from the
    PyTorch reference which normalises over (C, H, W); we use the more
    standard Keras convention and the network adapts.
    """

    def __init__(self, num_conv_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_conv_channels = num_conv_channels
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.conv1 = tf.keras.layers.Conv2D(
            num_conv_channels, kernel_size=3, padding="same"
        )
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.conv2 = tf.keras.layers.Conv2D(
            num_conv_channels, kernel_size=3, padding="same"
        )

    def call(self, x, training=False):
        z = self.ln1(x)
        z = tf.nn.relu(z)
        z = self.conv1(z)
        z = self.ln2(z)
        z = tf.nn.relu(z)
        z = self.conv2(z)
        return z + x


class NeuralReceiver(tf.keras.Model):
    """R3 — DeepRx-style fully-convolutional neural OFDM receiver.

    Input
    -----
    y         : [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
    h_hat_ls  : same shape as y, complex (LS estimate interpolated to all REs)
    pilot_mask: [num_ofdm_symbols, fft_size], float (broadcast over batch)
    no        : [batch], float (noise variance)

    Output
    ------
    llr : [batch, num_ofdm_symbols, fft_size, num_bits_per_symbol], float
        Per-resource-element bit-LLRs.  Downstream code applies
        ResourceGridDemapper to align with the coded-bit sequence.
    """

    def __init__(self, num_rx_ant: int, num_conv_channels: int,
                 num_blocks: int, num_bits_per_symbol: int,
                 use_mirror_feature: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_rx_ant = num_rx_ant
        self.num_blocks = num_blocks
        self.num_bits_per_symbol = num_bits_per_symbol
        self.use_mirror_feature = use_mirror_feature
        # Channels-last input:
        #   I/Q of y per ant                              → 2·num_rx_ant
        #   I/Q of LS h_hat per ant                       → 2·num_rx_ant
        #   pilot mask + log10(N0)                        → 2
        #   (optional) I/Q of conjugate-mirror of y       → 2·num_rx_ant
        #
        # The mirror feature (`use_mirror_feature=True`, default) is
        # required to compensate IQ imbalance: Y(k) = α·S(k) + β·S*(−k)
        # couples bin k with bin (N−k) mod N, separated by up to fft_size
        # subcarriers — far outside the conv receptive field (~22 bins
        # for 11 blocks).  Including conj(y[mirror_of(k)]) at every
        # subcarrier position lets the local CNN see both members of
        # each mirror pair within its receptive field.
        base = 2 * (2 * num_rx_ant) + 2
        extra = (2 * num_rx_ant) if use_mirror_feature else 0
        self.num_input_channels = base + extra
        self.input_conv = tf.keras.layers.Conv2D(
            num_conv_channels, kernel_size=3, padding="same",
            name="input_conv",
        )
        self.blocks = [
            ResidualBlock(num_conv_channels, name=f"res_block_{i}")
            for i in range(num_blocks)
        ]
        self.output_conv = tf.keras.layers.Conv2D(
            num_bits_per_symbol, kernel_size=3, padding="same",
            name="output_conv",
        )

    def call(self, inputs, training=False):
        y, h_hat_ls, pilot_mask, no = inputs
        # y / h_hat_ls shape: [batch, num_rx=1, num_rx_ant, num_sym, fft_size]
        # Drop the singleton num_rx axis.
        y = tf.squeeze(y, axis=1)              # [B, num_rx_ant, T, F]
        h_hat_ls = tf.squeeze(h_hat_ls, axis=1)

        # Mirror-conjugate feature for IQI compensation.
        # Sionna's resource grid places DC at bin 0; bin k has frequency
        # +k·Δf for 0 ≤ k < N/2 and (k−N)·Δf for k ≥ N/2.  The IQI mirror
        # of bin k is bin (N−k) mod N.  tf.roll(tf.reverse(y, axis=F),
        # shift=1, axis=F) yields [y[0], y[N−1], y[N−2], …, y[1]] — exactly
        # y at the mirror of each bin.  Take its complex conjugate.
        if self.use_mirror_feature:
            y_mirror = tf.math.conj(
                tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1)
            )
        else:
            y_mirror = None

        # Channels-last: move antenna axis to the end and stack I/Q.
        # [B, num_rx_ant, T, F] → [B, T, F, num_rx_ant]
        y = tf.transpose(y, [0, 2, 3, 1])
        h_hat_ls = tf.transpose(h_hat_ls, [0, 2, 3, 1])
        y_re = tf.math.real(y)
        y_im = tf.math.imag(y)
        h_re = tf.math.real(h_hat_ls)
        h_im = tf.math.imag(h_hat_ls)
        # log10(no) broadcast.  no: [B] → [B, T, F, 1].
        T = tf.shape(y_re)[1]
        F = tf.shape(y_re)[2]
        no_log = tf.math.log(no) / tf.math.log(10.0)
        no_log = tf.reshape(no_log, [-1, 1, 1, 1])
        no_log = tf.broadcast_to(no_log, [tf.shape(y_re)[0], T, F, 1])
        # pilot_mask: [T, F] → [1, T, F, 1] broadcast over batch
        pm = tf.reshape(pilot_mask, [1, T, F, 1])
        pm = tf.broadcast_to(pm, [tf.shape(y_re)[0], T, F, 1])

        # Assemble input stack along channels axis.
        feature_list = [y_re, y_im, h_re, h_im]
        if y_mirror is not None:
            y_mirror = tf.transpose(y_mirror, [0, 2, 3, 1])
            feature_list.append(tf.math.real(y_mirror))
            feature_list.append(tf.math.imag(y_mirror))
        feature_list.extend([pm, no_log])
        z = tf.concat(feature_list, axis=-1)
        # Body
        z = self.input_conv(z)
        for block in self.blocks:
            z = block(z, training=training)
        z = self.output_conv(z)
        # Output: [B, T, F, Q_m] — LLR per coded bit per RE.
        return z


# ----------------------------------------------------------------------------
# Simulator: wraps PdschSim's TX chain + impairment pipeline. The training
# loop drove it through a training-step forward pass (not in this release);
# R3EvalSim below and r5_simo_inference.py build on it.
# ----------------------------------------------------------------------------

class R3TrainingSim(tf.keras.Model):
    """Wraps PdschSim: TX chain, impairments, channel, the LS estimator R3
    takes as input, and the per-RE-LLR to coded-bit-sequence mapping."""

    def __init__(self, cfg: SimConfig, r3: NeuralReceiver,
                 num_conv_channels: int, num_blocks: int):
        super().__init__()
        self.cfg = cfg
        # PdschSim provides everything we need: TX chain, impairments,
        # channel, LS estimator (we'll use it directly to produce h_hat).
        self.pdsch = PdschSim(cfg)
        self.r3 = r3
        # LS estimator — must be built outside the tf.function trace.
        self.ls_est = LSChannelEstimator(
            resource_grid=self.pdsch.rg, interpolation_type="lin"
        )
        # Demap helper: from per-RE LLRs to coded-bit-sequence LLRs.
        self.rg_demap = ResourceGridDemapper(
            self.pdsch.rg, self.pdsch.sm,
        )
        # Oracle APP demapper (the Phase A imitation target in training; unused
        # by evaluation).
        # Sionna's Demapper("app", ...) computes optimal LLRs given the
        # equalized signal x_hat and post-equalization noise variance.
        self.oracle_demap = Demapper(
            demapping_method="app",
            constellation_type="qam",
            num_bits_per_symbol=cfg.mcs[1],
        )

    def _build_pilot_mask(self):
        """Build a [num_ofdm_symbols, fft_size] binary mask for DMRS REs."""
        # Use the resource grid's pilot pattern directly.
        # rg.pilot_pattern.mask has shape [num_tx, num_streams, T, F].
        m = self.pdsch.rg.pilot_pattern.mask[0, 0]  # [T, F]
        return tf.cast(m, tf.float32)

    def _grid_to_seq(self, llr_grid):
        """Convert R3's per-RE LLRs [B, T, F, Q_m] to coded-bit sequence
        [B, n_coded_bits] matching the TBEncoder's output order.

        Uses ResourceGridDemapper's RE→sequence mapping.  Sionna expects
        the LLR tensor shaped [B, num_rx, num_streams_per_rx, T, F, Q_m]
        in our SIMO 1-stream config, num_rx=1 and num_streams_per_rx=1.
        """
        B = tf.shape(llr_grid)[0]
        T = self.cfg.n_symbols
        F = self.cfg.fft_size
        Q = self.r3.num_bits_per_symbol
        # Reshape to [B, num_rx=1, num_streams=1, T, F, Q]
        llr_grid_x = tf.reshape(llr_grid, [B, 1, 1, T, F, Q])
        seq = self.rg_demap(llr_grid_x)
        # Output: [B, num_rx=1, num_streams=1, n_coded_bits]
        # Reshape to [B, 1, 1, n_coded_bits] to match c.
        return tf.reshape(seq, [B, 1, 1, -1])


# ----------------------------------------------------------------------------
# Evaluation: full BLER sweep with LDPC decoder in the loop.
# ----------------------------------------------------------------------------

class R3EvalSim(R3TrainingSim):
    """Adds LDPC decoding on top of R3TrainingSim so we can measure BLER.

    Mirrors PdschSim.call() exactly except the LS+LMMSE+Gray receive chain
    is replaced by R3 → ResourceGridDemapper → TBDecoder.
    """

    @tf.function(jit_compile=False)
    def evaluate_step(self, batch_size, ebno_db):
        """Returns (b_true, b_hat) ready for BLER counting."""
        # Reuse the forward path to get c (= true coded bits) + R3 LLR seq.
        # We also need the original information bits b that fed TBEncoder
        # because the LDPC decoder reconstructs b (not c).
        b = self.pdsch.source([batch_size, 1, 1, self.pdsch.tb_size])
        c = self.pdsch.tb_enc(b)
        x = self.pdsch.mapper(c)
        x_rg = self.pdsch.rg_mapper(x)
        no = r1_baseline.ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=self.pdsch.q_m,
            coderate=self.pdsch.code_rate,
            resource_grid=self.pdsch.rg,
        )

        phi_tx_kept = None
        phi_rx_kept = None
        if self.pdsch._td_active:
            x_t_sym = tf.signal.ifft(x_rg)
            x_t = tf.reshape(
                x_t_sym, [batch_size, 1, 1, self.pdsch._slot_len]
            )
            if self.pdsch.pn_tx is not None:
                phi_tx_kept = self.pdsch.pn_tx((batch_size, 1, 1))
                x_t = x_t * tf.exp(tf.complex(0.0, phi_tx_kept))
            x_t_sym = tf.reshape(
                x_t, [batch_size, 1, 1,
                      self.cfg.n_symbols, self.cfg.fft_size]
            )
            x_rg = tf.signal.fft(x_t_sym)

        y, _h = self.pdsch.ofdm_channel(x_rg, no)

        if self.pdsch._td_active:
            y_t_sym = tf.signal.ifft(y)
            y_t = tf.reshape(
                y_t_sym,
                [batch_size, 1, self.cfg.n_rx_ant, self.pdsch._slot_len],
            )
            if self.pdsch.pn_rx is not None:
                phi_rx_kept = self.pdsch.pn_rx(
                    (batch_size, 1, self.cfg.n_rx_ant)
                )
                y_t = y_t * tf.exp(tf.complex(0.0, phi_rx_kept))
            if self.pdsch.iqi is not None:
                y_t = self.pdsch.iqi(y_t)
            if self.cfg.genie_phase_track:
                phi_total = tf.zeros(
                    [batch_size, 1, self.cfg.n_rx_ant, self.pdsch._slot_len],
                    dtype=tf.float32,
                )
                if phi_tx_kept is not None:
                    phi_total = phi_total + phi_tx_kept
                if phi_rx_kept is not None:
                    phi_total = phi_total + phi_rx_kept
                y_t = y_t * tf.exp(tf.complex(0.0, -phi_total))
            y_t_sym = tf.reshape(
                y_t,
                [batch_size, 1, self.cfg.n_rx_ant,
                 self.cfg.n_symbols, self.cfg.fft_size],
            )
            y = tf.signal.fft(y_t_sym)

        h_hat_ls, _err_var = self.ls_est(y, no)
        h_hat_ls = tf.squeeze(h_hat_ls, axis=[3, 4])
        pilot_mask = self._build_pilot_mask()
        no_b = tf.broadcast_to(tf.reshape(no, [-1]), [batch_size])
        r3_llr_grid = self.r3(
            (y, h_hat_ls, pilot_mask, no_b), training=False
        )
        r3_llr_seq = self._grid_to_seq(r3_llr_grid)
        b_hat, _crc = self.pdsch.tb_dec(r3_llr_seq)
        return b, b_hat


def evaluate_bler(eval_sim: R3EvalSim, snr_range, num_tti, batch_size,
                  num_mc_runs=2):
    """Sweep BLER across the SNR range using R3 + LDPC.  Returns
    `bler_all` of shape [num_mc_runs, len(snr_range)] for stats."""
    bler_all = np.zeros((num_mc_runs, len(snr_range)))
    for run_idx in range(num_mc_runs):
        seed = 1000 + run_idx
        tf.random.set_seed(seed)
        np.random.seed(seed)
        sionna_config.tf_rng.reset_from_seed(seed)
        for i, snr_db in enumerate(snr_range):
            total_blocks = 0
            total_errs = 0
            for _ in range(max(1, num_tti // batch_size)):
                b, b_hat = eval_sim.evaluate_step(
                    batch_size, tf.constant(float(snr_db))
                )
                block_err = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
                total_errs += int(tf.reduce_sum(tf.cast(block_err, tf.int32)))
                total_blocks += batch_size
            bler_all[run_idx, i] = total_errs / max(1, total_blocks)
    return bler_all


# ----------------------------------------------------------------------------
# SNR sampler — N(22, 3²) clipped to [16, 28] dB: the training distribution,
# also drawn by r1_practical_arm.py's practical-prior harvest.
# ----------------------------------------------------------------------------

def sample_ebno_db(batch_size, mu=22.0, sigma=3.0, lo=16.0, hi=28.0):
    """Truncated normal sampler.  Implemented via rejection: draws ~ N(μ, σ²),
    clamps out-of-range to the bounds (truncated-normal is unbiased to first
    order in σ for our range)."""
    x = tf.random.normal([batch_size], mean=mu, stddev=sigma)
    return tf.clip_by_value(x, lo, hi)


# ----------------------------------------------------------------------------
# Evaluation with trained weights
# ----------------------------------------------------------------------------

def run_eval_only(args):
    """Load pre-trained R3 weights, skip training, run a BLER sweep at the
    configured channel + impairment settings.  Single sweep, single CSV
    output.  Designed for OOD probes (channel shift, IRR sweeps, MCS
    sweeps) where retraining is not needed."""
    sionna_config.seed = args.seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    cfg = SimConfig(
        channel_model=args.channel, mcs_idx=args.mcs,
        dmrs_n_add=args.n_add, ue_speed_kmh=args.ue_speed_kmh,
        n_prb=args.n_prb, carrier_hz=args.carrier_hz, scs_hz=args.scs_hz,
        n_tx_ant=1, n_rx_ant=2, estimator="ls",
        pn_tx=args.pn_tx, pn_rx=args.pn_rx,
        pn_ue_offset_db=args.pn_ue_offset_db,
        iqi=args.iqi, iqi_irr_db=args.iqi_irr_db,
    )
    q_m = MCS_TABLE[args.mcs][1]
    r3 = NeuralReceiver(
        num_rx_ant=cfg.n_rx_ant,
        num_conv_channels=args.num_conv_channels,
        num_blocks=args.num_blocks,
        num_bits_per_symbol=q_m,
        use_mirror_feature=not args.no_mirror_feature,
    )
    # Materialize weights: build the model with a dummy forward pass at the
    # eval shape so Keras knows the variable shapes before load_weights.
    eval_sim = R3EvalSim(cfg, r3, args.num_conv_channels, args.num_blocks)
    apply_front_end(eval_sim, args)
    _ = eval_sim.evaluate_step(args.batch_size, tf.constant(23.0))
    weights_path = os.path.join(args.load_weights, "weights.h5")
    r3.load_weights(weights_path)
    logging.info(f"Loaded R3 weights from {weights_path}")

    # Read meta.json if present (just for the log).
    meta_path = os.path.join(args.load_weights, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        logging.info(
            f"Checkpoint meta: phase={meta.get('phase')}, "
            f"iteration={meta.get('iteration')}/{meta.get('total_iterations')}, "
            f"last_val_bler={meta.get('last_val_bler')}"
        )

    # BLER sweep at the eval config.
    snr_range = np.arange(
        args.eval_snr_min, args.eval_snr_max + 1e-9, args.eval_snr_step
    )
    logging.info(
        f"Eval-only sweep: channel={args.channel} MCS={args.mcs} "
        f"impairments=[pn_tx={args.pn_tx},pn_rx={args.pn_rx},"
        f"iqi={args.iqi}] SNR=[{args.eval_snr_min},{args.eval_snr_max}]"
    )
    bler_all = evaluate_bler(
        eval_sim, snr_range, args.eval_num_tti,
        args.batch_size, args.eval_mc_runs,
    )
    mean = bler_all.mean(axis=0)
    sem = bler_all.std(axis=0, ddof=1) / max(1, math.sqrt(args.eval_mc_runs))
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(
        args.out_dir,
        f"r3_bler_cdl{args.channel}_mcs{args.mcs}_eval.csv",
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "bler_mean", "bler_sem", "n_runs",
                    "n_blocks_per_run"])
        for i, s in enumerate(snr_range):
            w.writerow([s, mean[i], sem[i], args.eval_mc_runs,
                        args.eval_num_tti])
    logging.info(f"wrote {csv_path}")

    # 10%-BLER point + floor.
    snr_10 = None
    for i in range(1, len(snr_range)):
        if mean[i-1] >= 0.1 > mean[i] and mean[i-1] > 0 and mean[i] > 0:
            lb1 = math.log10(mean[i-1])
            lb2 = math.log10(mean[i])
            snr_10 = snr_range[i-1] + (
                (math.log10(0.1) - lb1) * (snr_range[i] - snr_range[i-1])
                / (lb2 - lb1)
            )
            break
    floor = float(np.min([b for b in mean if b > 0])) if (mean > 0).any() else 0.0
    op = f"{snr_10:.2f} dB" if snr_10 is not None else "unreachable"
    logging.info(
        f"=== Eval result: channel CDL-{args.channel} / MCS{args.mcs} === "
        f"10%-BLER = {op}, floor = {floor:.3e}"
    )
    return 0


def main():
    p = argparse.ArgumentParser()
    # Common
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-conv-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=11)
    p.add_argument("--carrier-hz", type=float, default=7.125e9)
    p.add_argument("--scs-hz", type=float, default=30e3)
    p.add_argument("--channel", choices=["A", "B", "C", "D", "E"], default="C")
    p.add_argument("--mcs", type=int, default=27, choices=[4, 13, 19, 27])
    p.add_argument("--n-add", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--ue-speed-kmh", type=float, default=3.0)
    p.add_argument("--n-prb", type=int, default=51)
    p.add_argument("--pn-tx", choices=["off", "on"], default="on")
    p.add_argument("--pn-rx", choices=["off", "on"], default="on")
    p.add_argument("--pn-ue-offset-db", type=float, default=3.0)
    p.add_argument("--iqi", choices=["off", "spec"], default="spec")
    p.add_argument("--iqi-irr-db", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=42)
    # Evaluation
    p.add_argument("--eval-snr-min", type=float, default=14.0)
    p.add_argument("--eval-snr-max", type=float, default=30.0)
    p.add_argument("--eval-snr-step", type=float, default=1.0)
    p.add_argument("--eval-num-tti", type=int, default=1920)
    p.add_argument("--eval-mc-runs", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="./results/r3_run")
    p.add_argument("--no-mirror-feature", action="store_true",
                   help="Disable the conjugate-mirror input channel "
                        "(default on — IQI compensation needs it).")
    # R3's channel-estimate input; see apply_front_end.
    p.add_argument("--front-end", choices=["ls", "lmmse-practical"], default="ls")
    p.add_argument("--prior", default=os.path.join(_PKG_DIR, "results/run_p/prior_practical.npz"))
    p.add_argument("--prior-sha", default=os.path.join(_PKG_DIR, "results/run_p/prior_practical.sha256"))
    # Evaluation with a trained network (load weights, sweep).
    p.add_argument("--load-weights", type=str, default=None,
                   help="Path to a trained network's folder containing "
                        "weights.h5. Loads these weights and runs a single "
                        "BLER sweep at the configured channel + impairments. "
                        "Used for channel-shift OOD probes and reload-only "
                        "evaluations.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.load_weights is not None:
        return run_eval_only(args)

    raise SystemExit(
        "Specify --load-weights DIR (a trained network's folder holding "
        "weights.h5). The training loop is not part of this release."
    )


if __name__ == "__main__":
    sys.exit(main())
