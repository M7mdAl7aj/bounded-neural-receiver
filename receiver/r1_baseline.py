"""
r1_baseline.py — R1 classical baseline link-level simulator.

Produces BLER vs SNR curves for 3GPP-compliant PDSCH under CDL channels at
7.125 GHz cmWave, following the paper's §II-§III.

Receiver pipeline:
  1. LS channel estimation on DMRS with linear interpolation
  2. LMMSE equalization
  3. Gray LLR demapper (max-log)
  4. 3GPP LDPC decoder, 25 iterations

Methodology:
  * 12 Monte-Carlo runs per sweep point with common random numbers (CRN)
  * Mean ± SEM reporting
  * 3GPP TR 38.901 CDL channel, scaled for 7.125 GHz

Tested on NVIDIA RTX 6000 Ada, Python 3.10, TensorFlow 2.16.2 (tf-keras 2.16.0,
TF_USE_LEGACY_KERAS=1), Sionna 1.2.2, numpy 1.26.4.

Invocation:
  python r1_baseline.py --channel C --mcs 19 --n-add 1 \\
                         --snr-min 5 --snr-max 25 --snr-step 1 \\
                         --num-tti 4000 --num-mc-runs 12 \\
                         --out-dir ./results
"""

import argparse
import csv
import logging
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

# GPU memory growth — critical on RTX 6000 Ada / large GPUs to prevent TF
# from pre-allocating all 48 GB on startup. Must be called before any TF ops.
_gpus = tf.config.list_physical_devices('GPU')
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass  # Already initialized; ignore

# Deterministic ops are enabled at the START of main() (after Sionna RNG
# is seeded) — placed there rather than at module load because Sionna's
# config.seed setter lazily initializes its tf.random.Generator via
# tf.random.Generator.from_non_deterministic_state(), which raises once
# enable_op_determinism() has been called.  Order matters: seed Sionna
# first, then flip the determinism switch.

import sionna
from sionna.phy import config as sionna_config
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import OFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid, ResourceGridMapper,
    LSChannelEstimator, LMMSEEqualizer,
    LMMSEInterpolator,
    PilotPattern,
)
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.nr import TBEncoder, TBDecoder, calculate_tb_size
from sionna.phy.utils import ebnodb2no


# ----------------------------------------------------------------------------
# Deterministic seeding (shared by r1_mimo.py and r5_simo_inference.py)
# ----------------------------------------------------------------------------

def seed_all(seed):
    """Seed every generator this project draws from, from one integer.

    Measured: before this fix `--seed` pinned only the bit source. Two runs at the
    same seed produced byte-identical coded/info bits but LLRs differing in
    100% of entries. Two independent causes, both fixed here:

    1. The per-cell reseed blocks reset `sionna_config.tf_rng` and numpy's
       GLOBAL RNG. Sionna does not draw from numpy's global RNG -- it uses its
       own `config.np_rng` Generator, which nothing reset. Assigning
       `sionna_config.seed` resets tf_rng, py_rng and np_rng together, so that
       is what this uses.
    2. `PSDPhaseNoise.__call__` (and the MIMO AWGN in r1_mimo) drew from raw
       `tf.random.normal`. Inside a traced `tf.function` such an op has its
       seed pair fixed AT TRACE TIME from the graph-level seed; the graph seed
       was unset when tracing happened, so the op was seeded
       non-deterministically and no later `tf.random.set_seed` could reach it.
       Those draws now go through `sionna_config.tf_rng`, a stateful Generator
       that `reset_from_seed` genuinely rewinds.

    This buys FORWARD reproducibility only: realizations drawn before the fix
    were never seed-pinned and cannot be regenerated.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Last: also resets Sionna's tf_rng, py_rng and np_rng together.
    sionna_config.seed = seed


def cell_seed(base_seed, run_idx, snr_idx=0):
    """Per-cell CRN seed, derived from `--seed`.

    The sweeps used a hardcoded `1000 + run_idx`, which overwrote whatever
    `--seed` had set -- so `--seed` never reached the sweep body. Before the
    seeding fix that was invisible: runs at different seeds *did* differ, but
    from the non-deterministic phase-noise op, not from the seed. With the
    draws now deterministic the inertness is exposed, so the per-cell seed is
    derived from `--seed` here.

    Deterministic in (base_seed, run_idx, snr_idx), so two configurations run
    at the same `--seed` stay CRN-paired -- E2's L2-real vs L2-genie
    comparison depends on exactly that.
    """
    return (int(base_seed) * 1_000_003
            + int(run_idx) * 10_007
            + int(snr_idx) * 101) % (2 ** 31 - 1)

# 3GPP MCS table (TS 38.214 Table 5.1.3.1-2)
# Format: mcs_idx -> (name, Q_m, R_c)
# MCS 27's coderate is expressed as the exact fraction 948/1024 = 0.92578125
# because Sionna's TBEncoder asserts `target_coderate <= 948/1024` strictly;
# the rounded literal 0.9258 exceeds this in float and trips the assert.
# MCS 4/13/19 keep their original rounded literals so frozen baselines
# reproduce bit-for-bit.
MCS_TABLE = {
    4:  ("QPSK",    2, 0.3008),
    13: ("16QAM",   4, 0.4785),
    19: ("64QAM",   6, 0.7539),
    27: ("256QAM", 8, 948 / 1024),   # = 0.92578, exactly at Sionna's upper bound
}


def _array_geometry(n_ant: int) -> tuple[int, int, str]:
    """Map a total antenna count to a PanelArray (rows, cols, polarization).

    Specific cases preserve the paper's panel layouts:
        1   → (1, 1, "single")          1 antenna, no polarization
        2   → (1, 1, "dual")            SIMO RX default (cross-pol pair)
        8   → (2, 2, "dual")            paper RX preset (2×4 → 2×2 dual-pol)
        256 → (8, 16, "dual")           paper TX preset (8×16 dual-pol)

    Other counts: dual-pol with a square-ish (rows, cols) factorization of
    n_ant // 2. Odd counts > 1 are rejected.
    """
    import math

    if n_ant == 1:
        return (1, 1, "single")
    if n_ant <= 0 or n_ant % 2 != 0:
        raise ValueError(
            f"Antenna count {n_ant} not supported. Use 1, or any positive "
            "even integer (dual-pol gives 2 antennas per row/col cell)."
        )
    half = n_ant // 2
    # Pinned panel shapes matching the original paper config
    if half == 1:    return (1, 1,  "dual")   # n_ant = 2
    if half == 4:    return (2, 2,  "dual")   # n_ant = 8   (paper RX)
    if half == 128:  return (8, 16, "dual")   # n_ant = 256 (paper TX)
    # Square-ish factorization for the general case
    rows = int(math.isqrt(half))
    while rows > 1 and half % rows != 0:
        rows -= 1
    cols = half // rows
    return (rows, cols, "dual")


# --- RF impairment models (Section II-E) ---
#
# Time-domain pipeline: x_rg → iFFT → time samples → (TX PN) → FFT → freq grid
# → OFDMChannel (CDL + AWGN, freq domain) → iFFT → (RX PN) → (IQI) → FFT →
# receive chain.  Impairments off ⇒ the detour is mathematically equivalent
# to the original freq-domain shortcut up to FFT round-off (~1e-7 fp32),
# well below MC noise (1/480 ≈ 2e-3 BLER granularity).  Validation gate 4.


class PSDPhaseNoise:
    """Time-domain phase noise φ[n] from a rational-form PSD with
    fractional pole/zero exponents.

    Target PSD shape (TR 38.803 example-model style):

        PSD(f) [rad²/Hz] = psd0 · Π_k (1 + (f / fz_k)^α_z_k)
                                / Π_k (1 + (f / fp_k)^α_p_k)

    The fractional α's (e.g. 2.37, 2.7, 2.53, 3.3, 3.3, 1.0 in the 29.55
    GHz parameter set) mean this PSD is NOT a rational function of f, so
    no analytical spectral factorization into a finite-order IIR filter
    exists.  We synthesize φ[n] entirely in the frequency domain: build
    |H(f_k)| = sqrt(PSD(f_k)) on the one-sided FFT bin grid, multiply by
    a white-noise complex Gaussian draw, irfft to real time domain.

    Block-edge note: a length-N irfft is implicitly cyclic (samples 0 and
    N−1 are correlated).  An earlier version used `oversample=2` to crop
    the middle N from a 2N synthesis, but this rectangular crop window
    leaks the enormous near-DC PSD power (+7 to +19 dB bias across mid-
    band, confirmed by the gate-1 PSD diagnostics).  The default is
    now `oversample=1` — the irfft output is consumed directly.  The
    cyclic seam at sample 0/N−1 is harmless for the BLER sim: each slot
    gets an independent realization and no OFDM symbol straddles the
    block boundary.

    Parameters
    ----------
    psd0 : float
        PSD scale factor [rad²/Hz].  Already carrier-scaled and
        ue-offset-scaled by the caller.
    fz_list_hz, fp_list_hz : sequence of float
        Zero / pole corner frequencies [Hz].
    alpha_z, alpha_p : sequence of float
        Fractional exponents for each zero / pole (same length as
        fz_list_hz / fp_list_hz).
    fs_hz : float
        Time-domain sample rate [Hz] (slot sample rate = fft_size × scs).
    slot_length : int
        Number of time-domain samples the caller will consume per call.
    oversample : int, default 1
        Generate `oversample × slot_length` samples per FFT and crop the
        middle `slot_length`.  Default 1 (no crop) is correct; higher
        values corrupt the PSD via rectangular-window spectral leakage.
    """

    def __init__(self, psd0, fz_list_hz, alpha_z, fp_list_hz, alpha_p,
                 fs_hz, slot_length, oversample=1):
        self.psd0 = float(psd0)
        self.fz = np.asarray(fz_list_hz, dtype=np.float64)
        self.fp = np.asarray(fp_list_hz, dtype=np.float64)
        self.alpha_z = np.asarray(alpha_z, dtype=np.float64)
        self.alpha_p = np.asarray(alpha_p, dtype=np.float64)
        assert len(self.fz) == len(self.alpha_z), \
            "fz and alpha_z must match in length"
        assert len(self.fp) == len(self.alpha_p), \
            "fp and alpha_p must match in length"
        self.fs = float(fs_hz)
        self.N_out = int(slot_length)
        self.oversample = int(oversample)
        self.N = self.oversample * self.N_out  # generated length per FFT
        self.crop_start = (self.N - self.N_out) // 2

        # Build per-bin PSD on the one-sided FFT grid.  rfftfreq gives
        # bins at [0, fs/N, 2*fs/N, ..., fs/2] for an N-point real DFT.
        half = self.N // 2 + 1
        f = np.fft.rfftfreq(self.N, d=1.0 / self.fs)  # [half], in Hz
        psd = self._psd_curve(f)  # [half], one-sided, rad²/Hz

        # Per-bin variance budget: for a generic colored process the
        # bin-k variance is ∫_{f_k − df/2}^{f_k + df/2} PSD(f) df.  For
        # smoothly varying PSD this collapses to PSD(f_k) · df, but when
        # the PSD has a steep roll-off within a single bin (e.g. the DC
        # bin contains the entire flat-at-DC → f^−α transition that
        # happens around fp[0] ≪ df), naive sampling over-allocates by
        # orders of magnitude and the resulting DC leakage corrupts the
        # whole Welch estimate.  We integrate numerically for every bin.
        df = self.fs / self.N
        bin_var = self._integrate_psd_per_bin(f, df)  # [half], rad²

        # FFT amplitude shaping.  We generate complex Gaussian w[k] with
        # E|w[k]|² = 1, then multiply by amp[k].  Want time-domain x via
        # irfft(W) to have variance Σ_k bin_var[k] and one-sided PSD equal
        # to bin_var[k] / df ≈ PSD(f_k) for smooth bins.
        amp = float(self.N) * np.sqrt(np.maximum(bin_var, 0.0) / 2.0)
        self._amp = tf.constant(amp, dtype=tf.float32)
        self._bin_var = bin_var
        # Store the analytical PSD for gate-1 comparison.
        self._psd_analytical = psd
        self._freq_bins = f

    def _psd_curve(self, f):
        """Evaluate the rational PSD form at arbitrary frequencies [Hz]."""
        f = np.asarray(f, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (np.prod(1.0 + (f[:, None] / self.fz) ** self.alpha_z, axis=1)
                   if len(self.fz) else np.ones_like(f))
            den = (np.prod(1.0 + (f[:, None] / self.fp) ** self.alpha_p, axis=1)
                   if len(self.fp) else np.ones_like(f))
        return self.psd0 * num / den

    def _integrate_psd_per_bin(self, f_centers, df):
        """∫_{f − df/2}^{f + df/2} PSD(f) df per bin via Simpson's rule.

        Vectorized over bins for speed: 4M bins × 64 sub-samples = 256M
        PSD evaluations in numpy.  Negative-f portion of bin 0 is folded
        onto the positive half (one-sided PSD convention)."""
        n_sub = 65  # odd → composite Simpson works cleanly
        u = np.linspace(0.0, 1.0, n_sub)
        # Simpson weights (1, 4, 2, 4, 2, ..., 4, 1) / 3, scaled by step.
        w = np.ones(n_sub)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w /= 3.0 * (n_sub - 1)

        # Per-bin integration limits.
        lo = np.maximum(0.0, f_centers - df / 2.0)  # [B]
        hi = f_centers + df / 2.0                    # [B]
        # Sub-sample grid: [B, n_sub].
        fs_sub = lo[:, None] + (hi - lo)[:, None] * u[None, :]
        # Vectorized PSD evaluation on the flattened grid.
        psd_flat = self._psd_curve(fs_sub.reshape(-1))
        psd_grid = psd_flat.reshape(fs_sub.shape)            # [B, n_sub]
        return (hi - lo) * (psd_grid * w[None, :]).sum(axis=1)

    def __call__(self, leading_shape):
        """Generate φ[n] with shape `leading_shape + (slot_length,)`.

        `leading_shape` is a tuple (typically (batch_size, num_rx_ant) for
        RX PN or (batch_size, 1, 1) for TX PN).  Each leading-axis
        position gets an independent realization.
        """
        half = self.N // 2 + 1
        out_shape = list(leading_shape) + [half]
        # Via Sionna's seeded Generator, NOT raw tf.random.normal --
        # a raw op's seed is frozen at trace time and was non-deterministic.
        re = sionna_config.tf_rng.normal(out_shape, dtype=tf.float32)
        im = sionna_config.tf_rng.normal(out_shape, dtype=tf.float32)
        # |w[k]|² ~ ChiSq(2)/2, E|w[k]|² = 1.
        w = tf.complex(re, im) * tf.cast(1.0 / np.sqrt(2.0), tf.complex64)
        amp = tf.reshape(
            self._amp,
            [1] * len(leading_shape) + [half],
        )
        w = w * tf.cast(amp, tf.complex64)
        # irfft → real time samples (oversample × slot_length).
        phi_full = tf.signal.irfft(w, fft_length=[self.N])
        # Crop the central slot_length samples to isolate from cyclic seam.
        phi = phi_full[..., self.crop_start:self.crop_start + self.N_out]
        return phi  # [..., slot_length], real


class IQImbalance:
    """RX-side IQ imbalance: y_iqi = α·y + β·conj(y).

        α = (1 + g · exp(jφ)) / 2
        β = (1 − g · exp(−jφ)) / 2
        IRR = |α|² / |β|²

    For a target IRR (dB) we solve for a balanced (g, φ) pair: split the
    impairment evenly between gain and phase mismatch.  The override
    `g_overrides` / `phi_overrides` allows pinning specific values for
    sweeps.

    Per-RX-antenna independent (g_i, φ_i) drawn once at construction —
    physical analog mismatch is a property of the radio, not a per-symbol
    random variable.
    """

    def __init__(self, num_rx_ant, irr_db, seed=20260528,
                 g_overrides=None, phi_overrides=None):
        self.num_rx_ant = int(num_rx_ant)
        self.irr_db = float(irr_db)
        rng = np.random.default_rng(seed)

        # For target IRR, split evenly: pick g, φ such that the balanced
        # (g, φ) pair yields the exact target IRR with g = 1 − ε, φ = ε.
        # Exact solver (bisection) avoids the small-signal approximation
        # error that grows at lower IRR values.
        def _irr_at_eps(e):
            g_ = 1.0 - e
            a_ = 0.5 * (1.0 + g_ * np.exp(1j * e))
            b_ = 0.5 * (1.0 - g_ * np.exp(-1j * e))
            return 10.0 * np.log10(np.abs(a_) ** 2 / np.abs(b_) ** 2)

        def _bisect_irr(target):
            lo_, hi_ = 1e-12, 0.99
            for _ in range(100):
                mid_ = 0.5 * (lo_ + hi_)
                if _irr_at_eps(mid_) > target:
                    lo_ = mid_
                else:
                    hi_ = mid_
            return 0.5 * (lo_ + hi_)

        # Per-antenna jitter in dB space: draw per-antenna IRR targets
        # from N(irr_db, σ²) then solve for exact (g, φ) per antenna.
        # Jittering in dB space avoids the Jensen-inequality bias that
        # multiplicative jitter on (g, φ) would cause (IRR in dB is a
        # concave function of impairment magnitude).
        irr_spread_db = 2.0  # ±2 dB 1σ spread across antennas
        irr_per_ant = self.irr_db + irr_spread_db * rng.standard_normal(
            self.num_rx_ant)
        # Clamp to [5, 60] dB to avoid degenerate (g, phi).
        irr_per_ant = np.clip(irr_per_ant, 5.0, 60.0)
        eps_arr = np.array([_bisect_irr(t) for t in irr_per_ant])
        g_arr = 1.0 - eps_arr
        phi_arr = eps_arr
        if g_overrides is not None:
            g_arr = np.asarray(g_overrides, dtype=np.float64)
        if phi_overrides is not None:
            phi_arr = np.asarray(phi_overrides, dtype=np.float64)

        # α, β complex constants per antenna
        alpha = 0.5 * (1.0 + g_arr * np.exp(1j * phi_arr))
        beta = 0.5 * (1.0 - g_arr * np.exp(-1j * phi_arr))
        self.alpha = tf.constant(alpha.astype(np.complex64))   # [num_rx_ant]
        self.beta = tf.constant(beta.astype(np.complex64))     # [num_rx_ant]
        # Realized IRR per antenna for the validation log:
        self.realized_irr_db = 10.0 * np.log10(
            np.abs(alpha) ** 2 / np.abs(beta) ** 2
        )

    def __call__(self, y_t):
        """y_t shape [batch, num_rx, num_rx_ant, T]; broadcast α, β over
        leading axes."""
        # Reshape α, β to broadcast over (batch, num_rx, ..., T)
        # y_t rank is 4; α has shape [num_rx_ant]; broadcast to
        # [1, 1, num_rx_ant, 1].
        a = tf.reshape(self.alpha, [1, 1, self.num_rx_ant, 1])
        b = tf.reshape(self.beta, [1, 1, self.num_rx_ant, 1])
        return a * y_t + b * tf.math.conj(y_t)


# TR 38.803 v14.3.0 example PSD parameter set at the 29.55 GHz reference
# carrier.
#
# IMPORTANT: TR 38.803 does NOT label its example PSDs "PN-A / PN-B / PN-C".
# Those labels appear in the paper's §II-E but are paper
# terminology with no normative mapping into the TR.  The defensible
# substitute (this implementation): use the single 29.55 GHz example PSD
# for BOTH gNB (TX) and UE (RX), apply a UE oscillator-quality penalty of
# +3 dB at the UE to model the typical gNB-vs-UE LO gap.  The offset is
# exposed as --pn-ue-offset-db (default +3 dB) for sensitivity sweeps.
#
# Carrier scaling: 7.125 GHz has no native TR 38.803 PN spec.  Free-running
# PN PSD scales with carrier frequency squared:
#   L_target(f) [dBc/Hz] = L_ref(f) + 20·log10(f_target / f_ref)
# For 7.125 GHz from 29.55 GHz: scaling = 20·log10(7.125/29.55) = −12.4 dB.
# Applied by multiplying psd0 by 10^(-1.24) before instantiating
# PSDPhaseNoise.
_PN_29_55_GHZ = dict(
    psd0=32.0,
    fz_list_hz=[3e3, 550e3, 280e6],
    alpha_z=[2.37, 2.7, 2.53],
    fp_list_hz=[1.0, 1.6e6, 30e6],
    alpha_p=[3.3, 3.3, 1.0],
)
_PN_REFERENCE_CARRIER_HZ = 29.55e9   # TR 38.803 example PSD reference


def _pn_carrier_scale_db(carrier_hz: float) -> float:
    """20·log10(carrier / 29.55 GHz) — free-running PN PSD scales with f².

    Reference points:
        7.125 GHz → −12.36 dB  (the paper's cmWave config)
       24.25 GHz →  −1.71 dB  (top of FR3, preserves 6G branding)
       28.00 GHz →  −0.46 dB  (5G FR2 / mmWave standard)
       29.55 GHz →    0.00 dB  (TR 38.803 example PSD reference)
    """
    return 20.0 * float(np.log10(carrier_hz / _PN_REFERENCE_CARRIER_HZ))


# --- Config dataclasses ---

@dataclass
class SimConfig:
    # PHY
    scs_hz: float = 30e3             # subcarrier spacing; 30 kHz → μ=1
    n_prb: int = 273                 # primary config
    n_symbols: int = 14
    fft_size: int = field(init=False)

    # Antennas — parameterized. Default is SIMO 1 TX × 2 RX
    # (Theorem 1 is antenna-agnostic; SIMO is the primary track).
    # Paper's massive-MIMO config is reachable via n_tx_ant=256, n_rx_ant=8.
    n_tx_ant: int = 1
    n_rx_ant: int = 2
    carrier_hz: float = 7.125e9

    # DMRS
    dmrs_n_add: int = 1

    # MCS
    mcs_idx: int = 19

    # Channel
    channel_model: str = "C"         # "C" | "D" | "E"
    delay_spread_ns: float = 100.0
    ue_speed_kmh: float = 3.0

    # LDPC
    ldpc_iters: int = 25

    # Channel estimator — diagnostic options for the LS-vs-LMMSE comparison.
    # "ls" is the production default (LS at pilots + linear interpolation).
    #   ls              → LS at pilots + linear interp (paper code)
    #   lmmse-empirical → LS at pilots + LMMSEInterpolator with covariance
    #                     estimated from a CDL Monte-Carlo warmup pass.
    #                     Diagnostic upper bound only; CDL-C-tuned cov is
    #                     mismatched on shifted channels.
    #   perfect-csi     → bypass estimator entirely; feed true h, err_var=0.
    estimator: str = "ls"
    lmmse_warmup_slots: int = 200
    # Channel model the LMMSE warmup covariance is FITTED on. None = follow
    # channel_model, which preserves the behaviour before this option existed.
    # Set to a different model to fit the prior on one channel and EVALUATE on
    # another - the mismatch experiment that was previously inexpressible.
    lmmse_warmup_channel: "str | None" = None
    lmmse_order: str = "f-t"

    # Section II-E RF impairments. All default OFF — production
    # R1 retains the current AWGN-only channel until explicit ON-vs-OFF
    # measurement is requested.
    #
    # PN parameter sets are TR 38.803 v14.3.0 references whose numerical
    # values are pending external verification.
    # The synthesis machinery is parameter-agnostic; ON without verified
    # parameters will raise.
    #
    # debug_time_domain forces the time-domain detour even when all three
    # impairments are off — used to validate that the iFFT→ multiply →FFT
    # round-trip is gate-4 equivalent to the current freq-domain shortcut.
    # "off" | "on".  "on" uses the 29.55 GHz TR 38.803 example PSD scaled
    # to 7.125 GHz (−12.4 dB).  Same model at TX and RX; UE PN is +3 dB
    # above gNB PN by default (see pn_ue_offset_db).
    pn_tx: str = "off"
    pn_rx: str = "off"
    pn_ue_offset_db: float = 3.0  # UE oscillator quality gap above gNB.
    iqi: str = "off"             # "off" | "spec" (RX-side IQ imbalance)
    iqi_irr_db: float = 25.0     # IRR target when iqi == "spec"
    debug_time_domain: bool = False
    # Genie phase tracker — diagnostic upper bound on what any PTRS-like
    # classical CPE compensation could in principle achieve.  Subtracts
    # the exact realised phi_tx + phi_rx from the received time-domain
    # signal before the receive FFT.  Physically impossible at a real
    # receiver (no access to the synthesizer's noise process).  Used
    # ONLY for diagnostic bounding, NEVER for production receivers.
    genie_phase_track: bool = False
    # Genie IQI compensator — classical 2×2 mirror-pair inverse using the
    # TRUE (α, β) from the simulator.  Y[k] = (conj(α)·Y_iqi[k] −
    # β·conj(Y_iqi[N−k])) / (|α|² − |β|²).  This is the "L2-genie"
    # classical baseline that must be in the comparison matrix because
    # the mirror-conjugate input feature in R3 carries physically-necessary
    # IQI-pair information.  Applied AFTER the receive FFT and BEFORE the
    # channel estimator.  Physically impossible at a real receiver (no
    # access to true α, β); upper bound on classical IQI compensation.
    # L2-real (α, β estimated from DMRS) is a separate, deferred build.
    genie_iqi_comp: bool = False
    # L2-real IQI compensator — deployable classical baseline.  α, β are
    # estimated per RX antenna from DMRS-bin observations via the
    # small-impairment approximation:
    #   residual[k] = y[k] − h_hat_iqi[k]·x_dmrs[k]
    #               ≈ (β/α) · conj(h_hat_iqi[N−k]·x_dmrs[N−k])
    # γ = β/α is LS-fit over DMRS bins where both k and N−k are DMRS;
    # α is approximated as 1 (valid at IRR ≥ 25 dB where |β/α| ≤ 0.06).
    # Then the same 2×2 inverse as L2-genie is applied to y.  No genie
    # access; this is the receiver a real radio would use.
    real_iqi_comp: bool = False

    def __post_init__(self):
        self.fft_size = self.n_prb * 12

    @property
    def dmrs_symbols(self):
        # Type-1 DMRS, TS 38.211 Table 7.4.1.1.2-3
        return {0: [2], 1: [2, 11], 2: [2, 7, 11], 3: [2, 5, 8, 11]}[self.dmrs_n_add]

    @property
    def mcs(self):
        return MCS_TABLE[self.mcs_idx]


# --- Simulator model ---

class PdschSim(tf.keras.Model):
    """End-to-end PDSCH link-level sim with R1 classical receiver."""

    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.cfg = cfg
        mod_name, q_m, code_rate = cfg.mcs

        # Resource grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=cfg.n_symbols,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.scs_hz,
            num_tx=1,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=cfg.dmrs_symbols,
        )

        # Stream management
        self.sm = StreamManagement(
            rx_tx_association=np.array([[1]]),
            num_streams_per_tx=1,
        )

        # Transmit chain
        # Standards-compliant 3GPP TBS via TS 38.214 §5.1.3.2. Each DMRS OFDM
        # symbol in the Type-1 kronecker pattern occupies all 12 SC of every
        # PRB, so num_dmrs_per_prb = (# DMRS symbols) × 12.
        num_dmrs_per_prb = len(cfg.dmrs_symbols) * 12
        tb_size, cb_size, num_cb, tb_crc_len, cb_crc_len, cw_length = (
            calculate_tb_size(
                modulation_order=q_m,
                target_coderate=code_rate,
                num_prbs=cfg.n_prb,
                num_ofdm_symbols=cfg.n_symbols,
                num_dmrs_per_prb=num_dmrs_per_prb,
                num_layers=1,
                return_cw_length=True,
                verbose=False,
            )
        )
        self.tb_size = int(tb_size)
        self.n = int(tf.reduce_sum(cw_length))   # total coded bits = Σ cw_length
        self.num_cb = int(num_cb)
        self.q_m = q_m
        self.code_rate = self.tb_size / self.n   # effective coderate after TBS quantization

        self.source = BinarySource()
        # NR TBEncoder handles CRC attachment, code-block segmentation,
        # per-CB LDPC encoding, rate matching, and scrambling per 38.212/214.
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

        # Channel arrays. Antenna geometry is derived from total antenna
        # counts; see _array_geometry. SIMO 1×2 default; paper MIMO via
        # n_tx_ant=256, n_rx_ant=8.
        # polarization_type is a required positional in Sionna 1.x PanelArray:
        #   single → "V" (vertical only)
        #   dual   → "cross" (cross-polarized pair, matches paper)
        tx_rows, tx_cols, tx_pol = _array_geometry(cfg.n_tx_ant)
        rx_rows, rx_cols, rx_pol = _array_geometry(cfg.n_rx_ant)
        tx_pol_type = "cross" if tx_pol == "dual" else "V"
        rx_pol_type = "cross" if rx_pol == "dual" else "V"
        self.bs_array = PanelArray(
            num_rows_per_panel=tx_rows,
            num_cols_per_panel=tx_cols,
            polarization=tx_pol,
            polarization_type=tx_pol_type,
            antenna_pattern="38.901",
            carrier_frequency=cfg.carrier_hz,
        )
        self.ue_array = PanelArray(
            num_rows_per_panel=rx_rows,
            num_cols_per_panel=rx_cols,
            polarization=rx_pol,
            polarization_type=rx_pol_type,
            antenna_pattern="38.901",
            carrier_frequency=cfg.carrier_hz,
        )

        # CDL channel model
        ue_speed_mps = cfg.ue_speed_kmh / 3.6
        self.cdl = CDL(
            model=cfg.channel_model,
            delay_spread=cfg.delay_spread_ns * 1e-9,
            carrier_frequency=cfg.carrier_hz,
            ut_array=self.ue_array,
            bs_array=self.bs_array,
            direction="downlink",
            min_speed=ue_speed_mps,
            max_speed=ue_speed_mps,
        )
        # normalize_channel=True enforces unit average energy per RE on the
        # OFDM channel output. Without this, the 38.901 element pattern peak
        # gain (~8 dBi ≈ 6.3 linear) leaves trace(R_freq)/M ≈ 6 in the
        # empirical covariance — mathematically harmless for LS (antenna gain
        # cancels via equalization) but it shifts LMMSE's effective SNR
        # crossover and confuses sanity checks. Normalize for cleanliness.
        self.ofdm_channel = OFDMChannel(
            channel_model=self.cdl,
            resource_grid=self.rg,
            add_awgn=True,
            return_channel=True,
            normalize_channel=True,
        )

        # --- Section II-E impairment blocks (lazily built) ---
        # Slot sample rate at CP=0: fs = fft_size · subcarrier spacing.
        # Slot length in samples: n_symbols · fft_size.
        self._slot_fs_hz = cfg.fft_size * cfg.scs_hz
        self._slot_len = cfg.n_symbols * cfg.fft_size

        # 29.55 GHz example PSD, carrier-scaled to cfg.carrier_hz.  Same
        # model for gNB and UE; UE PSD is offset by cfg.pn_ue_offset_db
        # (default +3 dB) to model the gNB-vs-UE LO quality gap.
        carrier_scale_db = _pn_carrier_scale_db(cfg.carrier_hz)
        psd0_gnb = _PN_29_55_GHZ["psd0"] * 10.0 ** (carrier_scale_db / 10.0)
        psd0_ue = psd0_gnb * 10.0 ** (cfg.pn_ue_offset_db / 10.0)
        logging.info(
            f"PN carrier scaling (29.55 GHz → {cfg.carrier_hz / 1e9:.3f} GHz): "
            f"{carrier_scale_db:+.2f} dB"
        )

        def _make_pn(psd0):
            return PSDPhaseNoise(
                psd0=psd0,
                fz_list_hz=_PN_29_55_GHZ["fz_list_hz"],
                alpha_z=_PN_29_55_GHZ["alpha_z"],
                fp_list_hz=_PN_29_55_GHZ["fp_list_hz"],
                alpha_p=_PN_29_55_GHZ["alpha_p"],
                fs_hz=self._slot_fs_hz,
                slot_length=self._slot_len,
            )

        if cfg.pn_tx == "off":
            self.pn_tx = None
        elif cfg.pn_tx == "on":
            self.pn_tx = _make_pn(psd0_gnb)
            logging.info(
                f"PN-TX ON (gNB, 29.55→7.125 GHz scaled): psd0={psd0_gnb:.4f}"
            )
        else:
            raise ValueError(f"Unknown pn_tx mode {cfg.pn_tx!r}")

        if cfg.pn_rx == "off":
            self.pn_rx = None
        elif cfg.pn_rx == "on":
            self.pn_rx = _make_pn(psd0_ue)
            logging.info(
                f"PN-RX ON (UE, 29.55→7.125 GHz, +{cfg.pn_ue_offset_db:.1f} dB UE "
                f"offset): psd0={psd0_ue:.4f}"
            )
        else:
            raise ValueError(f"Unknown pn_rx mode {cfg.pn_rx!r}")

        # IQI (RX-side, IRR-spec).
        if cfg.iqi == "off":
            self.iqi = None
        elif cfg.iqi == "spec":
            self.iqi = IQImbalance(
                num_rx_ant=cfg.n_rx_ant,
                irr_db=cfg.iqi_irr_db,
            )
            logging.info(
                f"IQI ON (target IRR {cfg.iqi_irr_db:.1f} dB); per-antenna "
                f"realized IRR = {self.iqi.realized_irr_db.tolist()}"
            )
        else:
            raise ValueError(f"Unknown iqi mode {cfg.iqi!r}")

        # Are we using the time-domain detour?  True if any impairment is on,
        # OR if --debug-time-domain is requested (validates the detour itself),
        # OR if --genie-phase-track is requested (needs phi_tx/phi_rx access).
        self._td_active = (
            self.pn_tx is not None
            or self.pn_rx is not None
            or self.iqi is not None
            or cfg.debug_time_domain
            or cfg.genie_phase_track
        )
        if cfg.genie_phase_track:
            logging.info(
                "GENIE PHASE TRACKER ENABLED — diagnostic upper-bound only, "
                "not a deployment-realistic receiver. Removes exact realised "
                "PN before the receive FFT."
            )

        # Receive chain
        # Channel estimator depends on cfg.estimator:
        #   - "ls": built now (LS + linear interp; original paper code).
        #   - "lmmse-empirical": deferred — built by build_lmmse_estimator()
        #     after a Monte-Carlo warmup pass over the CDL channel.
        #   - "perfect-csi": never built; call() uses the true h directly.
        if cfg.estimator == "ls":
            self.chan_est = LSChannelEstimator(
                resource_grid=self.rg,
                interpolation_type="lin",
            )
        elif cfg.estimator in ("lmmse-empirical", "perfect-csi"):
            self.chan_est = None
        else:
            raise ValueError(
                f"Unknown estimator {cfg.estimator!r}. "
                "Expected one of: ls, lmmse-empirical, perfect-csi."
            )

        # L2-real first-pass LS estimator + DMRS pilot grid for IQI
        # parameter estimation.  Used only when cfg.real_iqi_comp is True.
        if cfg.real_iqi_comp:
            self._ls_est_for_iqi = LSChannelEstimator(
                resource_grid=self.rg,
                interpolation_type="lin",
            )
            # Build [T, F] pilot grid (zeros at non-DMRS REs) once at init.
            mask = self.rg.pilot_pattern.mask[0, 0]  # [T, F] bool
            pilots = self.rg.pilot_pattern.pilots[0, 0]  # [num_pilots] complex
            indices = tf.where(mask)
            pilot_grid = tf.scatter_nd(
                indices=indices,
                updates=tf.cast(pilots, tf.complex64),
                shape=tf.cast(tf.shape(mask), tf.int64),
            )
            self._iqi_pilot_grid = tf.constant(pilot_grid, dtype=tf.complex64)
            # Mask where both k and N−k are DMRS (mirror-pair valid).
            mask_f = tf.cast(mask, tf.float32)
            mask_mirror = tf.roll(
                tf.reverse(mask_f, axis=[-1]), shift=1, axis=-1
            )
            self._iqi_mirror_mask = tf.constant(
                mask_f * mask_mirror, dtype=tf.float32
            )
            logging.info(
                f"L2-real IQI compensator ON: "
                f"{int(tf.reduce_sum(self._iqi_mirror_mask))} mirror-valid "
                f"DMRS bins per slot (out of "
                f"{int(tf.reduce_sum(mask_f))} total DMRS bins)"
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
        # NR TBDecoder: per-CB LDPC decoding, rate dematching, descrambling,
        # CB+TB CRC checks, reassembly. Mirrors TBEncoder configuration.
        self.tb_dec = TBDecoder(
            encoder=self.tb_enc,
            num_bp_iter=cfg.ldpc_iters,
            cn_update="minsum",
        )

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        # Transmit. Source bits are the *transport block* (not raw LDPC k).
        # TBEncoder handles segmentation + per-CB LDPC + rate matching.
        b = self.source([batch_size, 1, 1, self.tb_size])
        c = self.tb_enc(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)

        # Noise variance
        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=self.q_m,
            coderate=self.code_rate,
            resource_grid=self.rg,
        )

        # --- TX side time-domain detour for PN-TX (gate-4 path when
        # --- self._td_active is on; otherwise we'd never leave freq domain).
        # phi_tx_kept and phi_rx_kept are saved across the channel for the
        # genie phase tracker (diagnostic upper-bound PTRS — see __init__).
        phi_tx_kept = None
        phi_rx_kept = None
        if self._td_active:
            # x_rg: [batch, 1, 1, n_sym, fft_size] freq grid.
            # OFDM CP=0 ⇒ per-symbol iFFT gives the time samples.  Stitch
            # the n_sym × fft_size samples into one contiguous time stream
            # per (batch, tx, stream) so the PN process is continuous across
            # symbol boundaries (matches a physical free-running oscillator).
            x_t_sym = tf.signal.ifft(x_rg)
            # Reshape (n_sym, fft_size) → (n_sym * fft_size) along the last
            # two axes.
            x_t = tf.reshape(
                x_t_sym,
                [batch_size, 1, 1, self._slot_len],
            )
            if self.pn_tx is not None:
                phi_tx_kept = self.pn_tx((batch_size, 1, 1))  # [B,1,1,slot_len]
                x_t = x_t * tf.exp(tf.complex(0.0, phi_tx_kept))
            # Back to per-symbol layout and FFT to freq grid.
            x_t_sym = tf.reshape(
                x_t,
                [batch_size, 1, 1, self.cfg.n_symbols, self.cfg.fft_size],
            )
            x_rg = tf.signal.fft(x_t_sym)

        # Channel — Sionna 1.x uses positional args, not list-packed inputs.
        # return_channel=True so we can use the true h for perfect-CSI mode.
        y, h = self.ofdm_channel(x_rg, no)

        # --- RX side time-domain detour for PN-RX + IQI.
        if self._td_active:
            # y: [batch, num_rx, num_rx_ant, n_sym, fft_size].  num_rx is 1
            # for this SIMO config; we hard-code it (matches StreamManagement
            # rx_tx_association above).
            y_t_sym = tf.signal.ifft(y)
            y_t = tf.reshape(
                y_t_sym,
                [batch_size, 1, self.cfg.n_rx_ant, self._slot_len],
            )
            if self.pn_rx is not None:
                phi_rx_kept = self.pn_rx((batch_size, 1, self.cfg.n_rx_ant))
                y_t = y_t * tf.exp(tf.complex(0.0, phi_rx_kept))
            if self.iqi is not None:
                y_t = self.iqi(y_t)

            # --- Genie phase tracker (diagnostic upper-bound for PTRS-like
            # compensation).  Placed AFTER IQI and BEFORE the receive FFT
            # by design.  This subtracts the exact realised
            # phase-noise process (TX PN carried through the channel +
            # per-antenna RX PN) from the time-domain signal — physically
            # impossible at a real receiver (genie has access to the
            # synthesizer's noise), but bounds what any PTRS-like
            # compensation could in principle achieve.  Residual floor
            # (if any) is from IQI-PN cross-terms + ICI-from-multipath-
            # interacting-with-TX-PN, neither of which a phase tracker
            # can undo.
            if self.cfg.genie_phase_track:
                phi_total = tf.zeros(
                    [batch_size, 1, self.cfg.n_rx_ant, self._slot_len],
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

        # --- L2-genie classical IQI compensator (diagnostic baseline). ---
        # Per-RX-antenna 2×2 inverse on the (k, N−k) mirror pair using the
        # synthesizer's TRUE (α, β).  Y_corrected[k] =
        #   (conj(α)·Y_iqi[k] − β·conj(Y_iqi[N−k])) / (|α|² − |β|²).
        # Applied AFTER the receive FFT and BEFORE the channel estimator.
        # Skipped if IQI is off (nothing to compensate).
        if self.cfg.genie_iqi_comp and self.iqi is not None:
            # y shape: [B, num_rx, num_rx_ant, T, F].  Mirror along F.
            y_mirror_conj = tf.math.conj(
                tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1)
            )
            # Broadcast α, β: [num_rx_ant] → [1, 1, num_rx_ant, 1, 1].
            alpha = tf.reshape(self.iqi.alpha,
                               [1, 1, self.cfg.n_rx_ant, 1, 1])
            beta = tf.reshape(self.iqi.beta,
                              [1, 1, self.cfg.n_rx_ant, 1, 1])
            det = tf.cast(
                tf.math.abs(alpha) ** 2 - tf.math.abs(beta) ** 2,
                tf.complex64,
            )
            y = (tf.math.conj(alpha) * y - beta * y_mirror_conj) / det

        # --- L2-real classical IQI compensator (deployable baseline). ---
        # Algorithm:
        #   1. First-pass LS estimate → h_hat_iqi (IQI-corrupted, ≈ α·H).
        #   2. residual[k] = y[k] − h_hat_iqi[k]·x_dmrs[k]
        #                  ≈ (β/α)·conj(h_hat_iqi[N−k]·x_dmrs[N−k])
        #   3. γ = β/α LS-fit per RX antenna over DMRS bins where both k
        #      and N−k are DMRS (the "mirror-valid" mask).
        #   4. Apply 2×2 inverse with α≈1, β=γ.  At IRR ≥ 25 dB, |β/α| ≤
        #      0.06 so α≈1 is a tight approximation.
        if self.cfg.real_iqi_comp and self.iqi is not None:
            # First-pass LS on the IQI'd y → h_hat_iqi at all REs (Sionna's
            # linear-interpolated estimate).  At DMRS bins this is exactly
            # y/X, so a residual computed against it would be identically
            # zero — useless for parameter estimation.
            h_hat_iqi, _ = self._ls_est_for_iqi(y, no)
            h_hat_iqi = tf.squeeze(h_hat_iqi, axis=[3, 4])  # [B, 1, n_rx_ant, T, F]

            # Smooth h_hat across the subcarrier axis by CIR truncation:
            # take IFFT along the F axis (giving the channel impulse response
            # in time-domain taps), keep only the low-delay taps that the
            # physical CDL channel can support, then FFT back.  This filters
            # OUT the IQI cross-term (which acts like wide-band noise across
            # subcarriers) and filters IN the true α·H[k] structure.  After
            # smoothing the residual is no longer identically zero at DMRS
            # bins; it carries the cross-term that γ-estimation needs.
            #
            # CDL-C at 100 ns delay spread × 18.36 MHz BW (51 PRB / 30 kHz):
            # CIR support ≈ 1.84 taps.  Keeping 16 taps + 16 wrap-around
            # taps is generous and well within bounds.
            num_taps_keep = 16
            F = self.cfg.fft_size
            h_cir = tf.signal.ifft(h_hat_iqi)   # [B,1,n_rx_ant,T,F] CIR
            mask_np = np.zeros(F, dtype=np.complex64)
            mask_np[:num_taps_keep] = 1.0
            mask_np[-num_taps_keep:] = 1.0
            cir_mask = tf.constant(mask_np, dtype=tf.complex64)
            cir_mask = tf.reshape(cir_mask, [1, 1, 1, 1, F])
            h_smooth = tf.signal.fft(h_cir * cir_mask)  # back to subcarriers

            # Pilot grid broadcast to [1, 1, 1, T, F].
            x_dmrs = tf.reshape(
                self._iqi_pilot_grid,
                [1, 1, 1, self.cfg.n_symbols, self.cfg.fft_size],
            )
            mirror_mask = tf.reshape(
                self._iqi_mirror_mask,
                [1, 1, 1, self.cfg.n_symbols, self.cfg.fft_size],
            )
            # residual = y − h_smooth · x_dmrs   (nonzero at DMRS bins now;
            # contains the IQI cross-term plus smoothing noise).
            residual = y - h_smooth * x_dmrs
            # Mirror of (h_smooth · x_dmrs):  conj(h_smooth[N-k]·x_dmrs[N-k]).
            hx = h_smooth * x_dmrs
            hx_mirror = tf.roll(tf.reverse(hx, axis=[-1]), shift=1, axis=-1)
            mirror_product_conj = tf.math.conj(hx_mirror)
            # LS fit γ per RX antenna over mirror-valid DMRS bins:
            #   γ_hat = Σ residual · conj(mirror_product_conj) · mask
            #         / Σ |mirror_product_conj|² · mask
            mask_c = tf.cast(mirror_mask, tf.complex64)
            num_g = tf.reduce_sum(
                residual * tf.math.conj(mirror_product_conj) * mask_c,
                axis=[0, 1, 3, 4],
            )  # [num_rx_ant]
            den_g = tf.reduce_sum(
                tf.cast(
                    tf.math.abs(mirror_product_conj) ** 2 * mirror_mask,
                    tf.complex64,
                ),
                axis=[0, 1, 3, 4],
            )
            gamma = num_g / (den_g + tf.cast(1e-12, tf.complex64))
            # Apply 2×2 inverse with α=1, β=γ (small-impairment approx).
            y_mirror_conj = tf.math.conj(
                tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1)
            )
            beta = tf.reshape(gamma, [1, 1, self.cfg.n_rx_ant, 1, 1])
            det = tf.cast(1.0, tf.complex64) - beta * tf.math.conj(beta)
            y = (y - beta * y_mirror_conj) / det

        # Receive
        if self.cfg.estimator == "perfect-csi":
            # True channel from OFDMChannel, zero estimation error. With
            # num_tx=1 and num_streams_per_tx=1, h's shape already matches
            # what LMMSEEqualizer expects from a channel estimator.
            h_hat = h
            err_var = tf.zeros(tf.shape(h), dtype=tf.float32)
        else:
            h_hat, err_var = self.chan_est(y, no)
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        # TBDecoder returns (b_hat, tb_crc_status); we only need the bits here.
        b_hat, _ = self.tb_dec(llr)
        return b, b_hat

    @tf.function(jit_compile=False)
    def _channel_sample(self, x_rg):
        """One batch through the OFDM channel; returns the true h tensor.

        Used by build_lmmse_estimator to harvest channel realizations for
        empirical covariance estimation. Noise is set negligibly small
        because y is discarded — only h is consumed.
        """
        no = tf.constant(1e-9, dtype=tf.float32)
        _, h = self.ofdm_channel(x_rg, no)
        return h

    def build_lmmse_estimator(self, batch_size: int, num_batches: int):
        """Estimate frequency/time covariance from a CDL Monte-Carlo warmup
        pass, then build LSChannelEstimator(interpolator=LMMSEInterpolator(...)).

        Only used when cfg.estimator == 'lmmse-empirical'.

        Diagnostic only: empirical R is fitted on the configured channel
        (e.g. CDL-C) and is mismatched if you then evaluate on CDL-D/E.
        """
        assert self.cfg.estimator == "lmmse-empirical", \
            "build_lmmse_estimator should only run for the lmmse-empirical estimator"

        M = self.cfg.fft_size
        N = self.cfg.n_symbols

        # Dummy TX signal — needed only so OFDMChannel has shapes to match;
        # we discard y inside _channel_sample.
        b = self.source([batch_size, 1, 1, self.tb_size])
        c = self.tb_enc(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)

        cdtype = tf.complex64
        R_freq_sum = tf.zeros([M, M], dtype=cdtype)
        R_time_sum = tf.zeros([N, N], dtype=cdtype)
        n_freq_samples = 0
        n_time_samples = 0

        # --- warmup channel selection ------------------------------------
        wch = self.cfg.lmmse_warmup_channel or self.cfg.channel_model
        saved = None
        if wch != self.cfg.channel_model:
            # Build a throwaway OFDM channel on the WARMUP model, harvest from
            # it, then restore. self.cdl / self.ofdm_channel are untouched for
            # the evaluation sweep.
            ue_speed_mps = self.cfg.ue_speed_kmh / 3.6
            warm_cdl = CDL(
                model=wch,
                delay_spread=self.cfg.delay_spread_ns * 1e-9,
                carrier_frequency=self.cfg.carrier_hz,
                ut_array=self.ue_array, bs_array=self.bs_array,
                direction="downlink",
                min_speed=ue_speed_mps, max_speed=ue_speed_mps,
            )
            saved = self.ofdm_channel
            self.ofdm_channel = OFDMChannel(
                channel_model=warm_cdl, resource_grid=self.rg, add_awgn=True,
                return_channel=True, normalize_channel=True,
            )
            logging.info(f"  LMMSE warmup fitted on CDL-{wch}, evaluation on "
                         f"CDL-{self.cfg.channel_model} (MISMATCHED BY DESIGN)")

        t0 = time.time()
        for b_i in range(num_batches):
            h = self._channel_sample(x_rg)
            # h shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, N, M]
            # Treat each (batch, rx-side, tx-side, ofdm-symbol) slice as one
            # sample of the M-dim frequency response.
            h_freq = tf.reshape(h, [-1, M])
            # Sample covariance R[i,j] = (1/K) Σ_k H[k,i] · conj(H[k,j]) requires
            # H^T · conj(H), NOT tf.matmul(H, H, adjoint_a=True) (which gives H^H·H
            # = K·conj(R)). The two differ only by conjugation, invisible on the
            # diagonal/trace but ruinous to LMMSE: feeding conj(R) inverts the
            # phase of off-diagonals → wrong Wiener filter → BLER pinned at 1.
            R_freq_sum += tf.matmul(h_freq, tf.math.conj(h_freq), transpose_a=True)
            n_freq_samples += int(tf.shape(h_freq)[0])

            # Time covariance: permute symbols to inner dim, treat each
            # (batch, rx-side, tx-side, subcarrier) slice as one N-dim sample.
            h_t = tf.transpose(h, [0, 1, 2, 3, 4, 6, 5])  # [..., M, N]
            h_time = tf.reshape(h_t, [-1, N])
            R_time_sum += tf.matmul(h_time, tf.math.conj(h_time), transpose_a=True)
            n_time_samples += int(tf.shape(h_time)[0])

            if (b_i + 1) % 20 == 0 or b_i == num_batches - 1:
                logging.info(
                    f"  lmmse-warmup batch {b_i+1}/{num_batches} "
                    f"(freq_samples={n_freq_samples}, time_samples={n_time_samples}, "
                    f"elapsed={time.time()-t0:.1f}s)"
                )

        R_freq = R_freq_sum / tf.cast(n_freq_samples, cdtype)
        R_time = R_time_sum / tf.cast(n_time_samples, cdtype)
        # Hermitianize defensively — finite-sample averages drift by O(1/sqrt(N)).
        R_freq = 0.5 * (R_freq + tf.linalg.adjoint(R_freq))
        R_time = 0.5 * (R_time + tf.linalg.adjoint(R_time))

        # Sanity checks: R must be Hermitian PSD with trace ≈ M (resp. N)
        # for a normalize_channel=True OFDM channel (unit avg |H|² per RE).
        tr_f = float(tf.math.real(tf.linalg.trace(R_freq)).numpy())
        tr_t = float(tf.math.real(tf.linalg.trace(R_time)).numpy())
        herm_err_f = float(tf.reduce_max(tf.abs(
            R_freq - tf.linalg.adjoint(R_freq))).numpy())
        herm_err_t = float(tf.reduce_max(tf.abs(
            R_time - tf.linalg.adjoint(R_time))).numpy())
        eig_f = tf.math.real(tf.linalg.eigvalsh(R_freq)).numpy()
        eig_t = tf.math.real(tf.linalg.eigvalsh(R_time)).numpy()
        logging.info(
            f"  empirical cov: trace(R_freq)={tr_f:.3f} (M={M}, ratio={tr_f/M:.3f}), "
            f"trace(R_time)={tr_t:.3f} (N={N}, ratio={tr_t/N:.3f})"
        )
        logging.info(
            f"  Hermitian residual: freq={herm_err_f:.2e}, time={herm_err_t:.2e}"
        )
        logging.info(
            f"  R_freq eig range: [{eig_f.min():.3e}, {eig_f.max():.3e}] "
            f"({int((eig_f > 1e-6 * eig_f.max()).sum())}/{M} above 1e-6 of max)"
        )
        logging.info(
            f"  R_time eig range: [{eig_t.min():.3e}, {eig_t.max():.3e}]"
        )
        # PSD: smallest eigenvalue must be >= 0 (tolerate ~1e-5 of the max
        # for finite-sample noise).
        if saved is not None:
            self.ofdm_channel = saved      # evaluation channel restored

        assert eig_f.min() >= -1e-5 * eig_f.max(), \
            f"R_freq not PSD: min eig = {eig_f.min()}"
        assert eig_t.min() >= -1e-5 * eig_t.max(), \
            f"R_time not PSD: min eig = {eig_t.min()}"

        interp = LMMSEInterpolator(
            pilot_pattern=self.rg.pilot_pattern,
            cov_mat_time=R_time,
            cov_mat_freq=R_freq,
            order=self.cfg.lmmse_order,
        )
        self.chan_est = LSChannelEstimator(
            resource_grid=self.rg,
            interpolator=interp,
        )
        logging.info(
            f"  lmmse-empirical estimator built (order={self.cfg.lmmse_order})"
        )


# --- Sweep driver ---

def run_sweep(cfg, snr_range, num_tti, batch_size, num_mc_runs, out_dir,
              base_seed=42):
    os.makedirs(out_dir, exist_ok=True)
    sim = PdschSim(cfg)

    # Build empirical-LMMSE estimator first if requested. Must happen before
    # the call-graph warmup so the chan_est attribute exists when call() is traced.
    if cfg.estimator == "lmmse-empirical":
        logging.info(
            f"Building empirical LMMSE estimator from "
            f"{cfg.lmmse_warmup_slots} warmup batches on CDL-{cfg.lmmse_warmup_channel or cfg.channel_model}..."
        )
        sim.build_lmmse_estimator(batch_size, cfg.lmmse_warmup_slots)

    # Warmup (compile XLA / prebuild graph)
    _ = sim(batch_size, tf.constant(10.0))

    n_snr = len(snr_range)
    bler_all = np.zeros((num_mc_runs, n_snr))
    t0 = time.time()

    for run_idx in range(num_mc_runs):
        # CRN: same seed per run, reset at the start.  Reset all three RNGs
        # the sim consumes: numpy (used by IQI init & similar), TF global
        # (some Sionna blocks fall back to it), and Sionna's own
        # tf.random.Generator (the one BinarySource etc. actually pull from).
        seed = cell_seed(base_seed, run_idx)
        seed_all(seed)

        for i, snr_db in enumerate(snr_range):
            total_blocks = 0
            total_errs = 0
            for _ in range(max(1, num_tti // batch_size)):
                b, b_hat = sim(batch_size, tf.constant(float(snr_db)))
                # BLER: any bit error in a transport block counts as a block error.
                block_err = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
                total_errs += int(tf.reduce_sum(tf.cast(block_err, tf.int32)))
                total_blocks += batch_size
            bler = total_errs / max(1, total_blocks)
            bler_all[run_idx, i] = bler
            elapsed = time.time() - t0
            logging.info(
                f"run={run_idx} snr={snr_db:+.1f}dB bler={bler:.3e} "
                f"blocks={total_blocks} elapsed={elapsed/60:.1f}min"
            )

    mean = bler_all.mean(axis=0)
    sem = bler_all.std(axis=0, ddof=1) / np.sqrt(num_mc_runs)

    # Write CSV. The default LS estimator keeps the original filename for
    # backwards compatibility with earlier CSVs; diagnostic modes append a tag.
    est_tag = "" if cfg.estimator == "ls" else f"_{cfg.estimator}"
    name = (
        f"bler_cdl{cfg.channel_model}_mcs{cfg.mcs_idx}_"
        f"nadd{cfg.dmrs_n_add}_v{int(cfg.ue_speed_kmh)}{est_tag}.csv"
    )
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "bler_mean", "bler_sem", "n_runs", "n_blocks_per_run"])
        for i, s in enumerate(snr_range):
            w.writerow([s, mean[i], sem[i], num_mc_runs, num_tti])
    logging.info(f"wrote {path}")

    # Quick console summary: 10% BLER operating point
    if np.any(mean < 0.1) and np.any(mean > 0.1):
        # Linear interp on log BLER
        idx = np.argmax(mean < 0.1)
        if idx > 0:
            s1, s2 = snr_range[idx-1], snr_range[idx]
            b1, b2 = mean[idx-1], mean[idx]
            if b1 > 0 and b2 > 0:
                log_b1, log_b2 = np.log10(b1), np.log10(b2)
                target = np.log10(0.1)
                snr_10 = s1 + (target - log_b1) * (s2 - s1) / (log_b2 - log_b1)
                print(f"\n  10%-BLER operating SNR ≈ {snr_10:.2f} dB "
                      f"(CDL-{cfg.channel_model}, MCS{cfg.mcs_idx}, "
                      f"N_add={cfg.dmrs_n_add}, {cfg.ue_speed_kmh} km/h)\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", choices=["A", "B", "C", "D", "E"], default="C")
    p.add_argument("--mcs", type=int, default=19, choices=list(MCS_TABLE.keys()))
    p.add_argument("--n-add", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--ue-speed-kmh", type=float, default=3.0)
    p.add_argument("--snr-min", type=float, default=5.0)
    p.add_argument("--snr-max", type=float, default=25.0)
    p.add_argument("--snr-step", type=float, default=1.0)
    p.add_argument("--num-tti", type=int, default=4000)
    p.add_argument("--num-mc-runs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--n-prb", type=int, default=273,
        help="Number of allocated PRBs (default 273). Use a small value "
             "(e.g. 4-16) for small-GPU smoke tests.",
    )
    p.add_argument(
        "--carrier-hz", type=float, default=7.125e9,
        help="Carrier frequency in Hz (default 7.125e9 = original cmWave "
             "FR3 config). 24.25e9 = top of FR3 (preserves 6G framing). "
             "28e9 = 5G FR2 mmWave standard. PN PSD is scaled accordingly "
             "from the 29.55 GHz TR 38.803 example PSD by 20·log10(f/29.55e9).",
    )
    p.add_argument(
        "--scs-hz", type=float, default=30e3,
        help="Subcarrier spacing (default 30 kHz, NR μ=1). Use 120e3 "
             "(μ=3) for mmWave NR numerology — pairs with smaller --n-prb "
             "for the same bandwidth and shrinks LMMSEInterpolator's M² "
             "covariance tensor.",
    )
    p.add_argument(
        "--n-tx-ant", type=int, default=1,
        help="Total number of TX antennas (default 1 = SIMO). "
             "Paper MIMO: --n-tx-ant 256 (mapped to 8×16 dual-pol).",
    )
    p.add_argument(
        "--n-rx-ant", type=int, default=2,
        help="Total number of RX antennas (default 2 = SIMO RX diversity). "
             "Paper MIMO: --n-rx-ant 8 (mapped to 2×2 dual-pol).",
    )
    p.add_argument(
        "--estimator",
        choices=["ls", "lmmse-empirical", "perfect-csi"],
        default="ls",
        help="Channel estimator. Default 'ls' (LS at pilots + linear "
             "interpolation). 'lmmse-empirical' is a diagnostic upper bound "
             "(LMMSEInterpolator with covariance from a CDL Monte-Carlo "
             "warmup — channel-tuned, not for production). 'perfect-csi' "
             "is the theoretical floor (true h fed to the equalizer).",
    )
    p.add_argument(
        "--delay-spread-ns", type=float, default=100.0,
        help="CDL RMS delay spread in ns (default 100.0 = the training point). "
             "Set to 300 for run H's primary shift condition. Plumbed into the "
             "CDL build; the default preserves prior behaviour exactly.",
    )
    p.add_argument(
        "--lmmse-warmup-channel", type=str, default=None, choices=["A","B","C","D","E"],
        help="Channel model the LMMSE warmup covariance is FITTED on. Default: "
             "follow --channel (previous behaviour). Set differently to fit the "
             "prior on one channel and evaluate on another.",
    )
    p.add_argument(
        "--lmmse-warmup-slots", type=int, default=200,
        help="Number of warmup batches for empirical covariance estimation "
             "(only used with --estimator lmmse-empirical).",
    )
    p.add_argument(
        "--lmmse-order", type=str, default="f-t",
        choices=["f-t", "t-f"],
        help="LMMSE interpolation order across the resource grid "
             "(only used with --estimator lmmse-empirical).",
    )
    # Section II-E impairment toggles. All off by default.
    p.add_argument(
        "--pn-tx", choices=["off", "on"], default="off",
        help="Phase noise at the gNB (TX side). 'on' uses the TR 38.803 "
             "29.55 GHz example PSD scaled to 7.125 GHz (−12.4 dB carrier "
             "scaling). NOTE: TR 38.803 has no native 7.125 GHz PN spec; "
             "PN-B/PN-C labels in paper §II-E are not TR labels.",
    )
    p.add_argument(
        "--pn-rx", choices=["off", "on"], default="off",
        help="Phase noise at the UE (RX side). Same 29.55 GHz model as "
             "--pn-tx with an additional +pn-ue-offset-db on the UE side.",
    )
    p.add_argument(
        "--pn-ue-offset-db", type=float, default=3.0,
        help="UE-vs-gNB phase-noise PSD offset in dB (default +3.0). "
             "Literature-anchored: UE oscillators are typically ~3 dB "
             "noisier than gNB references. Sweepable for sensitivity.",
    )
    p.add_argument(
        "--iqi", choices=["off", "spec"], default="off",
        help="RX-side IQ imbalance. 'spec' applies y_iqi = α·y + β·conj(y) "
             "with α, β derived from --iqi-irr-db (default 25 dB).",
    )
    p.add_argument(
        "--iqi-irr-db", type=float, default=25.0,
        help="Target image-rejection ratio in dB when --iqi=spec "
             "(default 25 dB; literature anchor for direct-conversion UE "
             "at cmWave).",
    )
    p.add_argument(
        "--debug-time-domain", action="store_true",
        help="Force the iFFT→FFT time-domain detour even when all "
             "impairments are off. Used for gate-4 validation that the "
             "detour is round-trip equivalent to the freq-domain "
             "OFDMChannel shortcut.",
    )
    p.add_argument(
        "--genie-phase-track", action="store_true",
        help="Diagnostic upper bound on what PTRS-like classical CPE "
             "compensation could in principle achieve. Subtracts the "
             "exact synthesizer-realised phi_tx + phi_rx_per_antenna "
             "from the time-domain RX signal before the receive FFT. "
             "Physically impossible at a real receiver. NOT for "
             "deployment-realistic measurements.",
    )
    p.add_argument(
        "--genie-iqi-comp", action="store_true",
        help="L2-genie classical IQI compensator using TRUE (α, β) from "
             "the simulator: 2×2 inverse on each (k, N−k) mirror pair "
             "after the receive FFT.  Upper bound on classical IQI "
             "compensation; physically impossible at a real receiver "
             "(no access to true α, β).",
    )
    p.add_argument(
        "--real-iqi-comp", action="store_true",
        help="L2-real deployable classical IQI compensator: α, β are "
             "estimated per RX antenna from DMRS-bin observations via "
             "LS-fit of the conjugate-mirror product against the "
             "first-pass-LS-channel-estimate residual.  Same 2×2 inverse "
             "as L2-genie applied with estimated parameters.  Realistic "
             "receiver baseline.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Sionna RNG seed (required for op-determinism mode).  Same "
             "seed across script invocations ⇒ bit-identical results.",
    )
    p.add_argument("--out-dir", type=str, default="./results")
    args = p.parse_args()

    # Seed Sionna's internal RNG.  Makes BinarySource (and other Sionna
    # blocks that consume randomness through sionna.phy.config.tf_rng)
    # reproducible across invocations modulo CUDA reduction nondeterminism.
    #
    # NOTE: tf.config.experimental.enable_op_determinism() would give true
    # bit-identical results across runs, but Sionna's LDPC decoder uses
    # DenseBincount which has no deterministic GPU kernel in this TF
    # version.  Living with the cuDNN/cuBLAS precision floor of ~0.05-0.1 dB
    # at the 10% BLER operating point for gate-4 comparisons.
    seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = SimConfig(
        channel_model=args.channel,
        mcs_idx=args.mcs,
        dmrs_n_add=args.n_add,
        ue_speed_kmh=args.ue_speed_kmh,
        n_prb=args.n_prb,
        carrier_hz=args.carrier_hz,
        scs_hz=args.scs_hz,
        n_tx_ant=args.n_tx_ant,
        n_rx_ant=args.n_rx_ant,
        estimator=args.estimator,
        lmmse_warmup_slots=args.lmmse_warmup_slots,
        lmmse_warmup_channel=args.lmmse_warmup_channel,
        delay_spread_ns=args.delay_spread_ns,
        lmmse_order=args.lmmse_order,
        pn_tx=args.pn_tx,
        pn_rx=args.pn_rx,
        pn_ue_offset_db=args.pn_ue_offset_db,
        iqi=args.iqi,
        iqi_irr_db=args.iqi_irr_db,
        debug_time_domain=args.debug_time_domain,
        genie_phase_track=args.genie_phase_track,
        genie_iqi_comp=args.genie_iqi_comp,
        real_iqi_comp=args.real_iqi_comp,
    )
    snr_range = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    run_sweep(cfg, snr_range, args.num_tti, args.batch_size,
              args.num_mc_runs, args.out_dir, base_seed=args.seed)


if __name__ == "__main__":
    main()
