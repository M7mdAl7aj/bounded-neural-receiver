"""r3_mimo.py — Neural MIMO receiver (rank-2 256T→2R) + fair comparison.

Extends the DeepRx-style receiver (r3_deeprx) to 2-layer spatial
multiplexing. Operates on the post-precoder 2-RX-antenna received grid
and outputs LLRs for both layers. Trained on the genuine rank-2 MIMO
pipeline (r1_mimo: SVD wideband genie precoder, shared-LO TX PN,
per-chain RX PN + IQI), mixed-IRR distribution.

Fair comparison target: neural MIMO vs the classical L2-genie MIMO
baseline (per-chain genie IQI comp + LMMSE MIMO detection), which sits
at 0.69 dB ≈ the OFF floor (0.64 dB). The question: does the neural
receiver's end-to-end joint optimisation beat the classical
LS+LMMSE+Gray structure at MIMO, as it did by 1.56 dB in SIMO?

This release ships the network definition, the simulator and the evaluation
path (--eval --load-weights DIR); the training loop is not included. In
training, the front end (generate_rx) ran OUTSIDE the gradient tape (no
trainable params; the 256-antenna tensor is transient, collapsed to n_rx=2
by the channel sum).
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass

import sionna
from sionna.phy import config as sionna_config
from sionna.phy.ofdm import ResourceGridDemapper

import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(__file__)
r1_baseline = _load("r1_baseline", os.path.join(_here, "r1_baseline.py"))
r1_mimo = _load("r1_mimo", os.path.join(_here, "r1_mimo.py"))
r3_deeprx = _load("r3_deeprx", os.path.join(_here, "r3_deeprx.py"))

SimConfig = r1_baseline.SimConfig
MCS_TABLE = r1_baseline.MCS_TABLE
IQImbalance = r1_baseline.IQImbalance
MimoPdschSim = r1_mimo.MimoPdschSim
ResidualBlock = r3_deeprx.ResidualBlock


# ----------------------------------------------------------------------------
# Neural MIMO receiver (2-layer DeepRx with conjugate-mirror feature)
# ----------------------------------------------------------------------------

class NeuralMimoReceiver(tf.keras.Model):
    """2-layer neural MIMO receiver.

    Input (channels-last grid [B, T, F, C]):
      - y I/Q per RX antenna           : 2·n_rx_ant
      - conj-mirror of y, I/Q per ant  : 2·n_rx_ant  (IQI mirror-pair feature)
      - LS effective-channel estimate I/Q (n_rx_ant × n_layers)
      - pilot mask                     : 1
      - log10(N0)                      : 1
    For 2R/2-layer: 4 + 4 + 8 + 1 + 1 = 18 channels.

    Output: [B, T, F, n_layers, Q_m] per-RE per-layer LLRs.
    """

    def __init__(self, n_rx_ant, n_layers, num_conv_channels, num_blocks,
                 num_bits_per_symbol, transform="lmmse", **kwargs):
        super().__init__(**kwargs)
        self.n_rx_ant = n_rx_ant
        self.n_layers = n_layers
        self.num_bits_per_symbol = num_bits_per_symbol
        # DeepRx-MIMO transformation layer (Korpi et al.): a conv backbone
        # alone cannot perform the per-RE multiplicative spatial inversion
        # (x = H^{-1} y) — a bilinear op in (y, h) not expressible by local
        # additive convs → SNR-independent BER floor (incomplete spatial
        # separation). Transformation variants:
        #   "none"  : raw features only (pure conv; floors ~43%).
        #   "mrc"   : matched filter (h_l^H y)/(h_l^H h_l); does NOT cancel
        #             inter-layer interference (floors ~30%).
        #   "lmmse" : per-RE LMMSE solve x = (H^H H + no·I)^{-1} H^H y — the
        #             optimal LINEAR detector; cancels inter-layer
        #             interference. The conv backbone then only refines for
        #             IQI / impairments. CSI matched to classical: the
        #             transform uses the SAME LS-estimated effective channel
        #             that classical L2 detection uses, so the only thing
        #             tested is neural refinement, not CSI quality.
        assert transform in ("none", "mrc", "lmmse")
        self.transform = transform
        self.input_conv = tf.keras.layers.Conv2D(
            num_conv_channels, 3, padding="same", name="input_conv")
        self.blocks = [ResidualBlock(num_conv_channels, name=f"res_block_{i}")
                       for i in range(num_blocks)]
        self.output_conv = tf.keras.layers.Conv2D(
            n_layers * num_bits_per_symbol, 3, padding="same",
            name="output_conv")

    def call(self, inputs, training=False):
        y, h_hat_eff, pilot_mask, no = inputs
        # y: [B, num_rx=1, n_rx_ant, T, F]
        # h_hat_eff: [B, num_rx=1, n_rx_ant, num_tx=1, n_layers, T, F]
        y = tf.squeeze(y, axis=1)                    # [B, n_rx_ant, T, F]
        he = h_hat_eff[:, 0, :, 0, :, :, :]          # [B,n_rx,L,T,F] (LS estimate)

        # --- Transformation layer, computed on the RAW LS-estimated channel
        # --- and raw y (physical units → correct LMMSE regularization; CSI
        # --- matched to classical L2 which uses the same LS estimate). ---
        xfm_feats = []
        T = tf.shape(y)[2]; F = tf.shape(y)[3]; B = tf.shape(y)[0]
        L = self.n_layers; nrx = self.n_rx_ant
        if self.transform in ("mrc", "lmmse"):
            # Inter-layer cross-Gram (interference indicator, L=2): [B,T,F]
            cross = tf.reduce_sum(
                tf.math.conj(he[:, :, 0]) * he[:, :, 1], axis=1)
            gain = tf.reduce_sum(tf.math.abs(he) ** 2, axis=1)   # [B,L,T,F]
        if self.transform == "mrc":
            num = tf.einsum("brltf,brtf->bltf", tf.math.conj(he), y)
            x_t = num / tf.cast(gain + 1e-9, tf.complex64)        # [B,L,T,F]
        elif self.transform == "lmmse":
            # Per-RE LMMSE: x = (H^H H + no·I)^{-1} H^H y, optimal linear
            # detector. H = he per RE [n_rx, L]. Vectorize over (B,T,F).
            H = tf.transpose(he, [0, 3, 4, 1, 2])                 # [B,T,F,nrx,L]
            H = tf.reshape(H, [-1, nrx, L])                       # [M,nrx,L]
            yv = tf.transpose(y, [0, 2, 3, 1])                    # [B,T,F,nrx]
            yv = tf.reshape(yv, [-1, nrx, 1])                     # [M,nrx,1]
            HhH = tf.matmul(H, H, adjoint_a=True)                 # [M,L,L]
            # no is [B] but constant within a call (single SNR) → scalar reg.
            reg = (tf.cast(tf.reduce_mean(no), tf.complex64)
                   * tf.eye(L, dtype=tf.complex64))
            xl = tf.linalg.solve(HhH + reg,
                                 tf.matmul(H, yv, adjoint_a=True))  # [M,L,1]
            x_t = tf.reshape(xl[..., 0], [B, T, F, L])            # [B,T,F,L]
            x_t = tf.transpose(x_t, [0, 3, 1, 2])                 # [B,L,T,F]
        if self.transform in ("mrc", "lmmse"):
            x_t = tf.transpose(x_t, [0, 2, 3, 1])                 # [B,T,F,L]
            # gain/cross are raw (~O(channel_scale²)) — normalize by the
            # per-example mean gain so they don't swamp the conv (the same
            # scale failure that collapsed the pure-conv net). x_t is already
            # a unit-power symbol estimate and is fed as-is.
            gain = tf.transpose(gain, [0, 2, 3, 1])               # [B,T,F,L]
            g_scale = tf.reduce_mean(gain, axis=[1, 2, 3], keepdims=True) + 1e-9
            gain = gain / g_scale
            cross = cross / tf.cast(g_scale[:, :, :, 0], tf.complex64)
            xfm_feats = [tf.math.real(x_t), tf.math.imag(x_t), gain,
                         tf.math.real(cross)[..., tf.newaxis],
                         tf.math.imag(cross)[..., tf.newaxis]]

        # --- Input normalization for the RAW features (unit RMS per example).
        # The ~20 dB array gain makes raw y/he O(10); un-normalized O(10)
        # inputs swamp the conv (collapse to prior). Transform features (x_t)
        # are already ~unit-power symbol estimates, fed without renorm.
        y_rms = tf.sqrt(tf.reduce_mean(tf.math.abs(y) ** 2,
                                       axis=[1, 2, 3], keepdims=True) + 1e-9)
        y = y / tf.cast(y_rms, tf.complex64)
        he_rms = tf.sqrt(tf.reduce_mean(tf.math.abs(he) ** 2,
                                        axis=[1, 2, 3, 4], keepdims=True) + 1e-9)
        he = he / tf.cast(he_rms, tf.complex64)
        y_mirror = tf.math.conj(
            tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1))

        # Channels-last for raw features.
        y = tf.transpose(y, [0, 2, 3, 1])            # [B,T,F,n_rx_ant]
        y_mirror = tf.transpose(y_mirror, [0, 2, 3, 1])
        he_cl = tf.transpose(he, [0, 3, 4, 1, 2])    # [B,T,F,n_rx,L]
        he_cl = tf.reshape(he_cl, tf.concat(
            [tf.shape(he_cl)[:3], [self.n_rx_ant * self.n_layers]], axis=0))
        no_log = tf.math.log(no) / tf.math.log(10.0)
        no_log = tf.broadcast_to(tf.reshape(no_log, [-1, 1, 1, 1]), [B, T, F, 1])
        pm = tf.broadcast_to(tf.reshape(pilot_mask, [1, T, F, 1]), [B, T, F, 1])
        z = tf.concat(
            [tf.math.real(y), tf.math.imag(y),
             tf.math.real(y_mirror), tf.math.imag(y_mirror),
             tf.math.real(he_cl), tf.math.imag(he_cl)]
            + xfm_feats + [pm, no_log], axis=-1)
        z = self.input_conv(z)
        for blk in self.blocks:
            z = blk(z, training=training)
        z = self.output_conv(z)                      # [B,T,F,L*Qm]
        return tf.reshape(z, tf.concat(
            [tf.shape(z)[:3], [self.n_layers, self.num_bits_per_symbol]],
            axis=0))                                 # [B,T,F,L,Qm]


# ----------------------------------------------------------------------------
# Simulator (front end + neural receiver + LDPC)
# ----------------------------------------------------------------------------

class MimoR3Sim(tf.keras.Model):
    """Wraps MimoPdschSim's front end + the neural MIMO receiver.

    Mixed-IRR training swapped a set of IQImbalance blocks (one per IRR
    value plus an IQI-off case) into the underlying MimoPdschSim per batch;
    evaluation pins one of them (evaluate_neural). Shared-LO TX PN stays on
    for all draws.
    """

    def __init__(self, cfg, neural, num_conv_channels, num_blocks,
                 mixed_irr_db=(20, 25, 30, 35, 40, 45), include_iqi_off=True):
        super().__init__()
        self.cfg = cfg
        self.pdsch = MimoPdschSim(cfg)
        self.neural = neural
        self.rg_demap = ResourceGridDemapper(self.pdsch.rg, self.pdsch.sm)
        # Pre-build IQI blocks for the mixed-IRR distribution.
        self._iqi_bank = [
            IQImbalance(num_rx_ant=cfg.n_rx_ant, irr_db=float(irr))
            for irr in mixed_irr_db
        ]
        self._iqi_choices = list(range(len(self._iqi_bank)))
        if include_iqi_off:
            self._iqi_choices.append(-1)   # -1 → IQI off
        self.q_m = self.pdsch.q_m
        self.n_layers = self.pdsch.num_layers

    def set_iqi(self, choice):
        """Swap the underlying sim's IQI block (-1 → off)."""
        if choice == -1:
            self.pdsch.iqi = None
        else:
            self.pdsch.iqi = self._iqi_bank[choice]
        # _td_active recompute (PN may still be on).
        self.pdsch._td_active = (
            self.pdsch.pn_tx is not None or self.pdsch.pn_rx is not None
            or self.pdsch.iqi is not None)

    def _grid_to_seq(self, llr_grid):
        """[B,T,F,L,Qm] per-RE LLRs → coded-bit sequence [B,1,1,n].

        ResourceGridDemapper: [B,num_rx=1,num_streams=L,T,F,Qm] →
        [B,1,L,num_data*Qm]; then LayerDemapper → [B,1,1,n].
        """
        B = tf.shape(llr_grid)[0]
        T = self.cfg.n_symbols; F = self.cfg.fft_size
        Q = self.q_m; L = self.n_layers
        # [B,T,F,L,Qm] → [B, num_rx=1, L, T, F, Qm]
        g = tf.transpose(llr_grid, [0, 3, 1, 2, 4])   # [B,L,T,F,Qm]
        g = g[:, tf.newaxis]                          # [B,1,L,T,F,Qm]
        seq = self.rg_demap(g)                        # [B,1,L,num_data,Qm]
        # Flatten (num_data, Qm) → per-layer coded bits, Qm consecutive per
        # symbol (what LayerDemapper expects).
        seq = tf.reshape(seq, tf.concat(
            [tf.shape(seq)[:3], [-1]], axis=0))        # [B,1,L,num_data*Qm]
        seq = self.pdsch.layer_demapper(seq)          # [B,1,n] (rank 3!)
        # Force rank-4 [B,1,1,n] to MATCH the coded-bit tensor c. Without
        # this, the training loss on (2c-1)*llr broadcasts [B,1,1,n] against
        # [B,1,n] → [B,B,1,n], cross-comparing every example's LLRs with
        # every other example's bits → unlearnable, collapse to the prior.
        return tf.reshape(seq, [tf.shape(seq)[0], 1, 1, -1])

    @tf.function(jit_compile=False)
    def neural_forward(self, y, no):
        """Neural receiver forward → coded-bit-sequence LLRs [B,1,1,n].

        The LS effective-channel estimate is computed here (non-trainable)."""
        h_hat, _ = self.pdsch.chan_est(y, no)          # effective MIMO channel
        pm = tf.cast(self.pdsch.rg.pilot_pattern.mask[0, 0], tf.float32)
        # collapse the 2 streams' masks to a single grid mask (OR).
        if len(pm.shape) > 2:
            pm = tf.cast(tf.reduce_any(pm > 0, axis=0), tf.float32)
        no_b = tf.broadcast_to(tf.reshape(no, [-1]), [tf.shape(y)[0]])
        llr_grid = self.neural((y, h_hat, pm, no_b), training=False)
        return self._grid_to_seq(llr_grid)

    @tf.function(jit_compile=False)
    def neural_decode(self, batch_size, ebno_db):
        """Full neural RX: front end + neural receiver + LDPC → (b, b_hat)."""
        b, c, y, no = self.pdsch.generate_rx(batch_size, ebno_db)
        llr = self.neural_forward(y, no)
        b_hat, _ = self.pdsch.tb_dec(llr)
        return b, b_hat


def evaluate_neural(sim, snr_range, num_tti, batch_size, num_mc_runs, irr_db):
    """BLER sweep for the trained neural MIMO receiver at a fixed IRR."""
    # Pin the IQI block to the comparison IRR (default 25 dB).
    irr_idx = None
    for i, blk in enumerate(sim._iqi_bank):
        if abs(blk.irr_db - irr_db) < 1e-6:
            irr_idx = i
            break
    if irr_idx is None:   # build one if not in the bank
        sim._iqi_bank.append(IQImbalance(num_rx_ant=sim.cfg.n_rx_ant,
                                         irr_db=irr_db))
        irr_idx = len(sim._iqi_bank) - 1
    sim.set_iqi(irr_idx)
    bler_all = np.zeros((num_mc_runs, len(snr_range)))
    for run_idx in range(num_mc_runs):
        seed = 1000 + run_idx
        tf.random.set_seed(seed); np.random.seed(seed)
        sionna_config.tf_rng.reset_from_seed(seed)
        for i, s in enumerate(snr_range):
            errs = blocks = 0
            for _ in range(max(1, num_tti // batch_size)):
                b, b_hat = sim.neural_decode(batch_size, tf.constant(float(s)))
                bf = tf.reshape(b, [batch_size, -1])
                bhf = tf.reshape(b_hat, [batch_size, -1])
                be = tf.reduce_any(tf.not_equal(bf, bhf), axis=-1)
                errs += int(tf.reduce_sum(tf.cast(be, tf.int32)))
                blocks += batch_size
            bler_all[run_idx, i] = errs / max(1, blocks)
            logging.info(f"run={run_idx} snr={s:+.1f} bler={bler_all[run_idx,i]:.3e}")
    return bler_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true")
    p.add_argument("--load-weights", type=str, default=None)
    p.add_argument("--eval-irr-db", type=float, default=25.0)
    p.add_argument("--eval-snr-min", type=float, default=-4.0)
    p.add_argument("--eval-snr-max", type=float, default=12.0)
    p.add_argument("--eval-snr-step", type=float, default=1.0)
    p.add_argument("--eval-num-tti", type=int, default=960)
    p.add_argument("--eval-mc-runs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-conv-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=11)
    p.add_argument("--mcs", type=int, default=27)
    p.add_argument("--n-add", type=int, default=1)
    p.add_argument("--n-prb", type=int, default=51)
    p.add_argument("--carrier-hz", type=float, default=7.125e9)
    p.add_argument("--scs-hz", type=float, default=30e3)
    p.add_argument("--channel", default="C")
    p.add_argument("--ue-speed-kmh", type=float, default=3.0)
    p.add_argument("--pn-ue-offset-db", type=float, default=3.0)
    p.add_argument("--transform", choices=["none", "mrc", "lmmse"],
                   default="lmmse",
                   help="Spatial transformation layer: none (pure conv), "
                        "mrc (matched filter), or lmmse (optimal linear "
                        "detector, cancels inter-layer interference).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="./results/r3_mimo")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.eval:
        return run_eval(args)
    raise SystemExit("specify --eval --load-weights DIR (the training loop is "
                     "not part of this release)")


def run_eval(args):
    sionna_config.seed = args.seed
    cfg = SimConfig(
        channel_model=args.channel, mcs_idx=args.mcs, dmrs_n_add=args.n_add,
        ue_speed_kmh=args.ue_speed_kmh, n_prb=args.n_prb,
        carrier_hz=args.carrier_hz, scs_hz=args.scs_hz,
        n_tx_ant=256, n_rx_ant=2, estimator="ls",
        pn_tx="on", pn_rx="on", pn_ue_offset_db=args.pn_ue_offset_db,
        iqi="spec", iqi_irr_db=args.eval_irr_db)
    q_m = MCS_TABLE[args.mcs][1]
    neural = NeuralMimoReceiver(2, 2, args.num_conv_channels, args.num_blocks, q_m, transform=args.transform)
    sim = MimoR3Sim(cfg, neural, args.num_conv_channels, args.num_blocks)
    # Materialize weights with a dummy forward, then load.
    _ = sim.neural_decode(args.batch_size, tf.constant(2.0))
    neural.load_weights(os.path.join(args.load_weights, "weights.h5"))
    logging.info(f"loaded {args.load_weights}/weights.h5")
    snr_range = np.arange(args.eval_snr_min, args.eval_snr_max + 1e-9,
                          args.eval_snr_step)
    bler_all = evaluate_neural(sim, snr_range, args.eval_num_tti,
                               args.batch_size, args.eval_mc_runs,
                               args.eval_irr_db)
    mean = bler_all.mean(axis=0)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"neural_mimo_bler_irr{int(args.eval_irr_db)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["snr_db", "bler_mean", "n_runs", "n_tti"])
        for i, s in enumerate(snr_range):
            w.writerow([s, mean[i], args.eval_mc_runs, args.eval_num_tti])
    logging.info(f"wrote {path}")
    snr10 = None
    for i in range(1, len(snr_range)):
        if mean[i-1] >= 0.1 > mean[i] and mean[i-1] > 0 and mean[i] > 0:
            lb1, lb2 = math.log10(mean[i-1]), math.log10(mean[i])
            snr10 = snr_range[i-1] + (math.log10(0.1)-lb1)*(snr_range[i]-snr_range[i-1])/(lb2-lb1)
            break
    floor = float(np.min([x for x in mean if x > 0])) if (mean > 0).any() else 0.0
    op = f"{snr10:.2f} dB" if snr10 else "unreachable"
    logging.info(f"=== Neural MIMO (IRR {args.eval_irr_db}): "
                 f"10%-BLER = {op}, floor = {floor:.3e} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
