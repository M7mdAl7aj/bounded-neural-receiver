"""
r1_awgn_check.py — AWGN-only sanity check for the R1 baseline.

Runs the same TX/RX chain as r1_baseline.py but replaces the CDL fading
channel with an identity channel + AWGN (no fading, no multipath).  This
provides an independent, theory-backed anchor for validating the R1
receiver.

Supports:
  --n-rx-ant N    : SISO (1) or SIMO (2+) to match the CDL-C comparison
  --perfect-csi   : bypass LS estimation, feed h=1 + err_var=0 directly

The SNR axis is Eb/N0 (same as r1_baseline.py), with OFDM overhead
correction via sionna.phy.utils.ebnodb2no.

Usage:
    # SIMO 1x2 with LS estimation (matches CDL-C config)
    python r1_awgn_check.py --n-rx-ant 2 --snr-min 4 --snr-max 16 --snr-step 0.5

    # Perfect CSI SISO (isolates coding gap from estimation penalty)
    python r1_awgn_check.py --n-rx-ant 1 --perfect-csi --snr-min 4 --snr-max 16 --snr-step 0.5
"""

import argparse
import logging
import time

import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass

import sionna
from sionna.phy.channel import AWGN as AWGNChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid, ResourceGridMapper,
    LSChannelEstimator, LMMSEEqualizer,
)
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.nr import TBEncoder, TBDecoder, calculate_tb_size
from sionna.phy.utils import ebnodb2no

MCS_TABLE = {
    4:  ("QPSK",    2, 0.3008),
    13: ("16QAM",   4, 0.4785),
    19: ("64QAM",   6, 0.7539),
    27: ("256QAM",  8, 0.9258),
}


class AwgnPdschSim(tf.keras.Model):
    """R1 receiver chain with AWGN-only channel (identity + noise).

    For SIMO 1xN AWGN: each RX antenna sees the same signal plus
    independent noise.  The LS estimator estimates h≈1 per antenna,
    and the LMMSE equalizer performs MRC combining.

    With --perfect-csi, LS estimation is bypassed: h_hat=1, err_var=0.
    """

    def __init__(self, mcs_idx=19, n_prb=273, n_add=1, n_rx_ant=1,
                 perfect_csi=False):
        super().__init__()
        self.n_rx_ant = n_rx_ant
        self.perfect_csi = perfect_csi
        mod_name, q_m, code_rate = MCS_TABLE[mcs_idx]

        dmrs_symbols = {
            0: [2], 1: [2, 11], 2: [2, 7, 11], 3: [2, 5, 8, 11]
        }[n_add]

        self.n_symbols = 14
        self.fft_size = n_prb * 12

        self.rg = ResourceGrid(
            num_ofdm_symbols=self.n_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=dmrs_symbols,
        )

        self.sm = StreamManagement(
            rx_tx_association=np.array([[1]]),
            num_streams_per_tx=1,
        )

        num_dmrs_per_prb = len(dmrs_symbols) * 12
        tb_size, cb_size, num_cb, tb_crc_len, cb_crc_len, cw_length = (
            calculate_tb_size(
                modulation_order=q_m,
                target_coderate=code_rate,
                num_prbs=n_prb,
                num_ofdm_symbols=self.n_symbols,
                num_dmrs_per_prb=num_dmrs_per_prb,
                num_layers=1,
                return_cw_length=True,
                verbose=False,
            )
        )
        self.tb_size = int(tb_size)
        self.n = int(tf.reduce_sum(cw_length))
        self.q_m = q_m
        self.code_rate = self.tb_size / self.n

        self.source = BinarySource()
        self.tb_enc = TBEncoder(
            target_tb_size=self.tb_size,
            num_coded_bits=self.n,
            target_coderate=code_rate,
            num_bits_per_symbol=q_m,
            num_layers=1,
            channel_type="PDSCH",
        )
        self.mapper = Mapper("qam", q_m)
        self.rg_mapper = ResourceGridMapper(self.rg)

        self.awgn = AWGNChannel()

        if not perfect_csi:
            self.chan_est = LSChannelEstimator(
                resource_grid=self.rg,
                interpolation_type="lin",
            )
        self.equalizer = LMMSEEqualizer(
            resource_grid=self.rg,
            stream_management=self.sm,
        )
        self.demapper = Demapper(
            demapping_method="maxlog",
            constellation_type="qam",
            num_bits_per_symbol=q_m,
        )
        self.tb_dec = TBDecoder(
            encoder=self.tb_enc,
            num_bp_iter=25,
            cn_update="minsum",
        )

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        # TX
        b = self.source([batch_size, 1, 1, self.tb_size])
        c = self.tb_enc(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)
        # x_rg shape: [batch, 1(num_tx), 1(num_streams), n_ofdm, fft]

        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=self.q_m,
            coderate=self.code_rate,
            resource_grid=self.rg,
        )

        # AWGN SIMO channel: replicate signal across n_rx_ant antennas,
        # add independent noise to each.
        # Output shape: [batch, 1(num_rx), n_rx_ant, n_ofdm, fft]
        y = tf.tile(x_rg, [1, 1, self.n_rx_ant, 1, 1])
        y = self.awgn(y, no)

        if self.perfect_csi:
            # Identity channel: h=1 for each (rx_ant, tx=1, tx_ant=1) pair
            # h_hat shape: [batch, 1, n_rx_ant, 1, 1, n_ofdm, n_eff_sc]
            n_eff = self.fft_size
            h_hat = tf.ones(
                [batch_size, 1, self.n_rx_ant, 1, 1,
                 self.n_symbols, n_eff],
                dtype=tf.complex64,
            )
            err_var = tf.zeros(
                [batch_size, 1, self.n_rx_ant, 1, 1,
                 self.n_symbols, n_eff],
                dtype=tf.float32,
            )
        else:
            h_hat, err_var = self.chan_est(y, no)

        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        b_hat, _ = self.tb_dec(llr)
        return b, b_hat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mcs", type=int, default=19, choices=list(MCS_TABLE.keys()))
    p.add_argument("--n-prb", type=int, default=273)
    p.add_argument("--n-add", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--n-rx-ant", type=int, default=1,
                    help="Number of RX antennas (1=SISO, 2=SIMO 1x2)")
    p.add_argument("--perfect-csi", action="store_true",
                    help="Bypass LS estimation, use h=1 directly")
    p.add_argument("--snr-min", type=float, default=4.0)
    p.add_argument("--snr-max", type=float, default=16.0)
    p.add_argument("--snr-step", type=float, default=0.5)
    p.add_argument("--num-tti", type=int, default=500)
    p.add_argument("--num-mc-runs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mode = "perfect-CSI" if args.perfect_csi else "LS-estimated"
    ant_label = f"SISO" if args.n_rx_ant == 1 else f"SIMO 1x{args.n_rx_ant}"
    logging.info(f"AWGN check: {ant_label}, {mode}")

    sim = AwgnPdschSim(
        mcs_idx=args.mcs, n_prb=args.n_prb, n_add=args.n_add,
        n_rx_ant=args.n_rx_ant, perfect_csi=args.perfect_csi,
    )

    q_m = sim.q_m
    r = sim.code_rate
    offset_db = 10 * np.log10(r * q_m)
    logging.info(f"MCS {args.mcs}: Q_m={q_m}, effective coderate={r:.4f}")
    logging.info(f"Es/N0 = Eb/N0 + {offset_db:.2f} dB")

    snr_range = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    # Warmup
    _ = sim(args.batch_size, tf.constant(10.0))

    n_snr = len(snr_range)
    bler_all = np.zeros((args.num_mc_runs, n_snr))
    t0 = time.time()

    for run_idx in range(args.num_mc_runs):
        tf.random.set_seed(1000 + run_idx)
        np.random.seed(1000 + run_idx)

        for i, snr_db in enumerate(snr_range):
            total_blocks = 0
            total_errs = 0
            for _ in range(max(1, args.num_tti // args.batch_size)):
                b, b_hat = sim(args.batch_size, tf.constant(float(snr_db)))
                block_err = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
                total_errs += int(tf.reduce_sum(tf.cast(block_err, tf.int32)))
                total_blocks += args.batch_size
            bler = total_errs / max(1, total_blocks)
            bler_all[run_idx, i] = bler
            elapsed = time.time() - t0
            logging.info(
                f"run={run_idx} Eb/N0={snr_db:+.1f}dB bler={bler:.3e} "
                f"blocks={total_blocks} elapsed={elapsed/60:.1f}min"
            )

    mean = bler_all.mean(axis=0)

    print(f"\n=== AWGN {ant_label} {mode} ===")
    print(f"{'Eb/N0 (dB)':>12} {'Es/N0 (dB)':>12} {'BLER':>12}")
    print("-" * 40)
    for i, s in enumerate(snr_range):
        es_n0 = s + offset_db
        print(f"{s:12.1f} {es_n0:12.1f} {mean[i]:12.4e}")

    if np.any(mean < 0.1) and np.any(mean > 0.1):
        idx = np.argmax(mean < 0.1)
        if idx > 0:
            s1, s2 = snr_range[idx - 1], snr_range[idx]
            b1, b2 = mean[idx - 1], mean[idx]
            if b1 > 0 and b2 > 0:
                log_b1, log_b2 = np.log10(b1), np.log10(b2)
                target = np.log10(0.1)
                snr_10 = s1 + (target - log_b1) * (s2 - s1) / (log_b2 - log_b1)
                es_n0_10 = snr_10 + offset_db
                print(f"\n  10%-BLER operating Eb/N0 = {snr_10:.2f} dB")
                print(f"  10%-BLER operating Es/N0 = {es_n0_10:.2f} dB")
    elif np.all(mean < 0.1):
        print(f"\n  WARNING: BLER < 10% at all SNR points. Extend snr-min lower.")
    elif np.all(mean >= 0.1):
        print(f"\n  WARNING: BLER >= 10% at all SNR points. Extend snr-max higher.")


if __name__ == "__main__":
    main()
