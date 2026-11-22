"""r1_mimo.py — Genuine rank-2 spatial-multiplexing MIMO link.

256T (gNB) → 2R (UE) downlink, 2-layer spatial multiplexing (rank capped
at 2 by the UE antenna count). SVD/eigen-beamforming precoder with genie
CSI-at-TX, used as an EXPLICITLY-LABELLED SURROGATE for a DTR-AS-MBWC
beamformer (not implemented here).

This is the MIMO counterpart of r1_baseline.py's SIMO pipeline. It exists
so we can run the fair-baseline MIMO comparison (classical LMMSE MIMO
detection + L2 IQI compensation vs a neural MIMO receiver) and decide
whether spatial-IQI interaction makes the neural advantage larger than
the SIMO 1.58 dB.

MODELING CHOICES:
  - Precoder: per-RE SVD eigen-beamforming, genie CSI-at-TX. Top-2 right
    singular vectors of the 2×256 channel. Phase-aligned across REs to
    keep the effective channel smooth for LS interpolation. Surrogate
    for DTR-AS-MBWC.
  - TX phase noise: SHARED-LO model — one PN process applied to all 256
    antenna chains (co-located ELAA, shared local oscillator).
  - RX phase noise + IQ imbalance: per-RX-chain (2 chains), reusing the
    validated SIMO impairment blocks.
  - Single codeword over 2 layers (NR uses 1 codeword for ≤4 layers).

Reuses from r1_baseline.py: MCS_TABLE, _array_geometry, PSDPhaseNoise,
IQImbalance, _pn_carrier_scale_db, _PN_29_55_GHZ.
"""

import argparse
import csv
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
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer,
)
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.nr import (
    TBEncoder, TBDecoder, calculate_tb_size, LayerMapper, LayerDemapper,
)
from sionna.phy.utils import ebnodb2no

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "r1_baseline", os.path.join(os.path.dirname(__file__), "r1_baseline.py")
)
r1_baseline = importlib.util.module_from_spec(_spec)
sys.modules["r1_baseline"] = r1_baseline
_spec.loader.exec_module(r1_baseline)

MCS_TABLE = r1_baseline.MCS_TABLE
_array_geometry = r1_baseline._array_geometry
seed_all = r1_baseline.seed_all          # shared deterministic seeding
cell_seed = r1_baseline.cell_seed        # per-cell CRN seed from --seed
PSDPhaseNoise = r1_baseline.PSDPhaseNoise
IQImbalance = r1_baseline.IQImbalance
_pn_carrier_scale_db = r1_baseline._pn_carrier_scale_db
_PN_29_55_GHZ = r1_baseline._PN_29_55_GHZ

# ---------------------------------------------------------------------------
# Capability flag for the MIMO L2-real IQI compensator.
#
# An earlier `_l2_real_iqi_comp` was an uncalled NotImplementedError stub,
# while `--real-iqi-comp` was accepted on the CLI and threaded into the
# config -- so the flag was SILENTLY IGNORED. Running it produced a sweep with
# no IQI compensation applied (L1 behaviour) that would have been labelled
# L2-real. This flag makes that impossible: while it is False, requesting the
# real path aborts at construction time. It is True because the estimator is
# wired into the forward path.
# ---------------------------------------------------------------------------
_L2_REAL_MIMO_IMPLEMENTED = True    # wired into generate_rx below


def svd_precoder_genie(h, num_layers):
    """Per-slot WIDEBAND SVD eigen-beamforming precoder, genie CSI-at-TX.

    h : [B, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, T, F] complex
        True channel (genie CSI-at-TX).
    Returns
    -------
    W : [B, num_tx_ant, num_layers] complex
        ONE precoding matrix per slot (constant across T and F), mapping
        `num_layers` streams → num_tx_ant antennas.

    Why wideband (not per-RE): Sionna's kronecker DMRS frequency-multiplexes
    the layers, so the receiver must interpolate the effective channel across
    subcarriers. A per-RE SVD precoder has arbitrary per-subcarrier phase and
    can swap layer ordering between adjacent REs, making the effective channel
    non-smooth and destroying LS interpolation (observed: flat ~30-50% BLER
    floor, no waterfall). Wideband precoding keeps W constant across the band,
    so H_eff[f,t] = H[f,t] @ W is as smooth as the physical channel and the
    DMRS estimate is clean. This mirrors NR wideband-PMI precoding and is a
    standard, defensible surrogate.

    Wideband precoder = top-`num_layers` eigenvectors of the band-averaged
    transmit spatial covariance R = (1/TFRx) Σ_{t,f} Hᴴ H  (n_tx × n_tx).
    Covariance averaging adds powers (no phase cancellation), so it is robust
    on frequency-selective channels.
    """
    B = tf.shape(h)[0]
    n_rx = tf.shape(h)[2]
    n_tx = tf.shape(h)[4]
    # h_mat [B, n_rx, n_tx, T, F] → stack (t,f,rx) rows: [B, T*F*n_rx, n_tx]
    h_mat = h[:, 0, :, 0, :, :, :]                  # [B, n_rx, n_tx, T, F]
    h_stack = tf.transpose(h_mat, [0, 3, 4, 1, 2])  # [B, T, F, n_rx, n_tx]
    h_stack = tf.reshape(h_stack, [B, -1, n_tx])    # [B, K, n_tx], K=T*F*n_rx
    k = tf.cast(tf.shape(h_stack)[1], tf.complex64)
    # Transmit spatial covariance R = Hᴴ H / K  → [B, n_tx, n_tx]
    r = tf.matmul(h_stack, h_stack, adjoint_a=True) / k
    # Top-`num_layers` eigenvectors (eigh ascending → take last L, reverse).
    _, eigvec = tf.linalg.eigh(r)                   # [B, n_tx, n_tx]
    w = eigvec[..., ::-1][..., :num_layers]         # [B, n_tx, L] orthonormal
    # Power: orthonormal columns; split unit power across layers.
    w = w / tf.cast(math.sqrt(num_layers), tf.complex64)
    return w                                        # [B, n_tx, L]


def estimate_gamma_core(y, h_hat_layers, pilot_grid, mirror_mask,
                        num_taps_keep=16):
    """Pure form of the per-chain gamma estimator.

    Deliberately a free function taking explicit tensors, so a check can
    drive the shipped code with SIMO-shaped inputs (num_layers = 1) instead of
    a degenerate SIMO mode being built into MimoPdschSim.

    Args
      y             [B, 1, n_rx, T, F]      received grid
      h_hat_layers  [B, 1, n_rx, L, T, F]   first-pass LS, per (chain, layer)
      pilot_grid    [L, T, F]               per-layer pilot values, 0 where blanked
      mirror_mask   [T, F]                  1 where k and N-k are both occupied
    Returns
      gamma_hat     [n_rx] complex          beta/alpha per RX chain
    """
    F = int(pilot_grid.shape[-1])
    T = int(pilot_grid.shape[-2])
    L = int(pilot_grid.shape[0])

    h_cir = tf.signal.ifft(h_hat_layers)
    mask_np = np.zeros(F, dtype=np.complex64)
    mask_np[:num_taps_keep] = 1.0
    mask_np[-num_taps_keep:] = 1.0
    cir_mask = tf.reshape(tf.constant(mask_np, dtype=tf.complex64),
                          [1, 1, 1, 1, 1, F])
    h_smooth = tf.signal.fft(h_cir * cir_mask)

    x_l = tf.reshape(pilot_grid, [1, 1, 1, L, T, F])
    hx = tf.reduce_sum(h_smooth * x_l, axis=3)          # [B,1,n_rx,T,F]
    residual = y - hx
    hx_mirror = tf.roll(tf.reverse(hx, axis=[-1]), shift=1, axis=-1)
    m = tf.math.conj(hx_mirror)
    mask = tf.reshape(mirror_mask, [1, 1, 1, T, F])
    mask_c = tf.cast(mask, tf.complex64)
    num_g = tf.reduce_sum(residual * tf.math.conj(m) * mask_c,
                          axis=[0, 1, 3, 4])
    den_g = tf.reduce_sum(tf.cast(tf.math.abs(m) ** 2 * mask, tf.complex64),
                          axis=[0, 1, 3, 4])
    return num_g / (den_g + tf.cast(1e-12, tf.complex64))


class MimoPdschSim(tf.keras.Model):
    """Rank-2 256T→2R downlink with SVD genie precoding + classical RX."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_layers = 2
        mod_name, q_m, code_rate = cfg.mcs

        self.rg = ResourceGrid(
            num_ofdm_symbols=cfg.n_symbols,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.scs_hz,
            num_tx=1,
            num_streams_per_tx=self.num_layers,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=cfg.dmrs_symbols,
        )
        self.sm = StreamManagement(
            rx_tx_association=np.array([[1]]),
            num_streams_per_tx=self.num_layers,
        )

        num_dmrs_per_prb = len(cfg.dmrs_symbols) * 12
        tb_size, _, num_cb, _, _, cw_length = calculate_tb_size(
            modulation_order=q_m, target_coderate=code_rate,
            num_prbs=cfg.n_prb, num_ofdm_symbols=cfg.n_symbols,
            num_dmrs_per_prb=num_dmrs_per_prb, num_layers=self.num_layers,
            return_cw_length=True, verbose=False,
        )
        self.tb_size = int(tb_size)
        self.n = int(tf.reduce_sum(cw_length))
        self.q_m = q_m
        self.code_rate = self.tb_size / self.n

        self.source = BinarySource()
        self.tb_enc = TBEncoder(
            target_tb_size=self.tb_size, num_coded_bits=self.n,
            target_coderate=code_rate, num_bits_per_symbol=q_m,
            num_layers=self.num_layers, channel_type="PDSCH",
        )
        self.mapper = Mapper("qam", q_m)
        self.layer_mapper = LayerMapper(num_layers=self.num_layers)
        self.layer_demapper = LayerDemapper(
            self.layer_mapper, num_bits_per_symbol=q_m
        )
        self.rg_mapper = ResourceGridMapper(self.rg)

        # Arrays: 256 TX (8×16 dual-pol), 2 RX (1×1 dual-pol).
        tx_rows, tx_cols, tx_pol = _array_geometry(cfg.n_tx_ant)
        rx_rows, rx_cols, rx_pol = _array_geometry(cfg.n_rx_ant)
        self.bs_array = PanelArray(
            num_rows_per_panel=tx_rows, num_cols_per_panel=tx_cols,
            polarization=tx_pol,
            polarization_type="cross" if tx_pol == "dual" else "V",
            antenna_pattern="38.901", carrier_frequency=cfg.carrier_hz,
        )
        self.ue_array = PanelArray(
            num_rows_per_panel=rx_rows, num_cols_per_panel=rx_cols,
            polarization=rx_pol,
            polarization_type="cross" if rx_pol == "dual" else "V",
            antenna_pattern="38.901", carrier_frequency=cfg.carrier_hz,
        )
        ue_speed_mps = cfg.ue_speed_kmh / 3.6
        self.cdl = CDL(
            model=cfg.channel_model, delay_spread=cfg.delay_spread_ns * 1e-9,
            carrier_frequency=cfg.carrier_hz, ut_array=self.ue_array,
            bs_array=self.bs_array, direction="downlink",
            min_speed=ue_speed_mps, max_speed=ue_speed_mps,
        )
        self.frequencies = subcarrier_frequencies(cfg.fft_size, cfg.scs_hz)
        # OFDM symbol duration (CP=0): 1 / scs. Sampling freq for the CDL
        # time evolution = 1 / ofdm_symbol_duration (one channel tap per sym).
        self.ofdm_symbol_duration = 1.0 / cfg.scs_hz

        # Receive chain
        self.chan_est = LSChannelEstimator(
            resource_grid=self.rg, interpolation_type="lin"
        )

        # --- L2-real per-chain IQI estimator setup ---
        if cfg.real_iqi_comp:
            self._ls_est_for_iqi = LSChannelEstimator(
                resource_grid=self.rg, interpolation_type="lin"
            )
            # Per-LAYER pilot grids. Sionna's kronecker pattern frequency-
            # multiplexes the two layers: both masks mark every DMRS RE, but
            # the pilot VALUES are zero on subcarriers the layer does not own
            # (RESOLVE-E1: 612 nonzero each, overlap 0). Scattering the pilot
            # values therefore yields x_l that is already zero where blanked,
            # so the sum over layers degenerates to one nonzero term per bin.
            grids, masks = [], []
            for l in range(self.num_layers):
                mask_l = self.rg.pilot_pattern.mask[0, l]          # [T, F]
                pilots_l = self.rg.pilot_pattern.pilots[0, l]
                idx = tf.where(mask_l)
                grids.append(tf.scatter_nd(
                    indices=idx,
                    updates=tf.cast(pilots_l, tf.complex64),
                    shape=tf.cast(tf.shape(mask_l), tf.int64)))
                masks.append(tf.cast(mask_l, tf.float32))
            self._iqi_pilot_grid = tf.constant(
                tf.stack(grids, axis=0), dtype=tf.complex64)      # [L, T, F]
            # Mirror-valid mask on ANY-LAYER occupancy: RX IQI leaks the
            # conjugate of the COMPOSITE signal at the mirror bin, so a pair
            # (k, N-k) owned by different layers is exactly the case the model
            # requires -- not an approximation (RESOLVE-E2).
            occ = tf.cast(tf.reduce_max(tf.stack(masks, axis=0), axis=0) > 0,
                          tf.float32)                              # [T, F]
            occ_mirror = tf.roll(tf.reverse(occ, axis=[-1]), shift=1, axis=-1)
            self._iqi_mirror_mask = tf.constant(occ * occ_mirror,
                                                dtype=tf.float32)
            # Instrumentation: gamma_hat from the LIVE forward
            # path, readable after a call. A tf.Variable, so the assignment
            # works inside the traced generate_rx.
            self._last_gamma = tf.Variable(
                tf.zeros([cfg.n_rx_ant], dtype=tf.complex64),
                trainable=False, name="last_gamma_hat")
            self._mirror_valid_count = int(tf.reduce_sum(
                self._iqi_mirror_mask))
            logging.info(
                f"L2-real MIMO IQI compensator ON: "
                f"{int(tf.reduce_sum(self._iqi_mirror_mask))} mirror-valid "
                f"DMRS bins per slot (out of {int(tf.reduce_sum(occ))} "
                f"any-layer DMRS bins; SIMO reference 1224)")
        self.equalizer = LMMSEEqualizer(
            resource_grid=self.rg, stream_management=self.sm
        )
        self.demapper = Demapper(
            demapping_method="maxlog", constellation_type="qam",
            num_bits_per_symbol=q_m,
        )
        self.tb_dec = TBDecoder(
            encoder=self.tb_enc, num_bp_iter=cfg.ldpc_iters, cn_update="minsum",
        )

        # Impairment blocks (reused from r1_baseline). Built lazily.
        self._slot_fs_hz = cfg.fft_size * cfg.scs_hz
        self._slot_len = cfg.n_symbols * cfg.fft_size
        carrier_scale_db = _pn_carrier_scale_db(cfg.carrier_hz)
        psd0_gnb = _PN_29_55_GHZ["psd0"] * 10.0 ** (carrier_scale_db / 10.0)
        psd0_ue = psd0_gnb * 10.0 ** (cfg.pn_ue_offset_db / 10.0)

        def _make_pn(psd0):
            return PSDPhaseNoise(
                psd0=psd0, fz_list_hz=_PN_29_55_GHZ["fz_list_hz"],
                alpha_z=_PN_29_55_GHZ["alpha_z"],
                fp_list_hz=_PN_29_55_GHZ["fp_list_hz"],
                alpha_p=_PN_29_55_GHZ["alpha_p"],
                fs_hz=self._slot_fs_hz, slot_length=self._slot_len,
            )

        self.pn_tx = _make_pn(psd0_gnb) if cfg.pn_tx == "on" else None
        self.pn_rx = _make_pn(psd0_ue) if cfg.pn_rx == "on" else None
        self.iqi = (IQImbalance(num_rx_ant=cfg.n_rx_ant, irr_db=cfg.iqi_irr_db)
                    if cfg.iqi == "spec" else None)
        if self.iqi is not None:
            logging.info(
                f"IQI ON (IRR {cfg.iqi_irr_db} dB); realized per-chain IRR = "
                f"{self.iqi.realized_irr_db.tolist()}"
            )
        logging.info(
            f"PN carrier scaling (29.55 GHz → {cfg.carrier_hz/1e9:.3f} GHz): "
            f"{carrier_scale_db:+.2f} dB; pn_tx={cfg.pn_tx} (shared-LO), "
            f"pn_rx={cfg.pn_rx}, iqi={cfg.iqi}"
        )
        self._td_active = (self.pn_tx is not None or self.pn_rx is not None
                           or self.iqi is not None)

        # --- resolve, announce, and validate the active IQI-comp path ---
        if cfg.genie_iqi_comp and cfg.real_iqi_comp:
            raise ValueError(
                "IQI-comp path ambiguous: --genie-iqi-comp and --real-iqi-comp "
                "were both set. Pass exactly one (or neither for L1)."
            )
        if cfg.real_iqi_comp:
            self.iqi_comp_path = "real comp (L2-real, DMRS-estimated alpha,beta)"
        elif cfg.genie_iqi_comp:
            self.iqi_comp_path = "genie comp (L2-genie, true alpha,beta)"
        else:
            self.iqi_comp_path = "no comp (L1)"
        logging.info(f"IQI compensation path ACTIVE IN FORWARD PATH: "
                     f"{self.iqi_comp_path}")
        if cfg.real_iqi_comp and not _L2_REAL_MIMO_IMPLEMENTED:
            raise NotImplementedError(
                "--real-iqi-comp requested but the MIMO L2-real compensator is "
                "NOT implemented in this build (_L2_REAL_MIMO_IMPLEMENTED is "
                "False). Refusing to run: without the estimator the code path "
                "applies NO IQI compensation, so the resulting sweep would be L1 "
                "behaviour mislabelled as L2-real. Implement the per-RX-antenna "
                "DMRS estimator and flip the flag, or use --genie-iqi-comp."
            )
        if cfg.real_iqi_comp and self.iqi is None:
            raise ValueError(
                "--real-iqi-comp requested but IQI is off (--iqi off): there is "
                "nothing to compensate."
            )

    @tf.function(jit_compile=False)
    def generate_rx(self, batch_size, ebno_db):
        """TX → SVD precode → channel → impairments → (optional L2 IQI comp).

        Returns (b, c, y, no): info bits, coded bits, received grid
        [B,1,n_rx,T,F], noise variance. Shared front end for the classical
        tail (call) and the neural MIMO receiver (r3_mimo). No trainable
        params here — safe to run outside a gradient tape; the 256-antenna
        x_ant tensor is transient (collapsed by the channel sum to n_rx=2).
        """
        # --- TX: 2-layer codeword → QAM → resource grid ---
        b = self.source([batch_size, 1, 1, self.tb_size])
        c = self.tb_enc(b)
        x = self.mapper(c)              # [B, 1, 1, n/q_m] symbols
        x = self.layer_mapper(x)        # [B, 1(tx), 1(cw), L, sym_per_layer]
        x = tf.squeeze(x, axis=2)       # drop single-codeword axis → [B,1,L,sym]
        x_rg = self.rg_mapper(x)        # [B, num_tx=1, num_streams=2, T, F]
        no = ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=self.q_m,
                       coderate=self.code_rate, resource_grid=self.rg)

        # --- Genie channel realization (CSI-at-TX) ---
        a, tau = self.cdl(batch_size, self.cfg.n_symbols,
                          1.0 / self.ofdm_symbol_duration)
        h = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        # h: [B, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, T, F]

        # --- SVD genie precoder (per-slot wideband) ---
        w = svd_precoder_genie(h, self.num_layers)   # [B, n_tx_ant, L]

        # x_rg layers → antennas: x_ant[B,n_tx,T,F] = Σ_L w[B,n_tx,L]·x_layer[B,L,T,F]
        x_layers = x_rg[:, 0]            # [B, L, T, F]
        x_ant = tf.einsum("bnl,bltf->bntf", w, x_layers)  # [B, n_tx_ant, T, F]

        # --- Shared-LO TX phase noise (per-antenna time-domain, shared φ) ---
        if self.pn_tx is not None:
            x_ant = self._apply_shared_pn_tx(x_ant, batch_size)

        # --- Apply genie channel: Y[B,n_rx_ant,T,F] = Σ_txant H · x_ant ---
        h_apply = h[:, 0, :, 0, :, :, :]          # [B, n_rx_ant, n_tx_ant, T, F]
        y = tf.reduce_sum(h_apply * x_ant[:, tf.newaxis, :, :, :], axis=2)
        # AWGN
        # Seeded Generator, not raw tf.random.normal (trace-time seed).
        noise = tf.complex(
            sionna_config.tf_rng.normal(tf.shape(y), stddev=tf.sqrt(no/2.0)),
            sionna_config.tf_rng.normal(tf.shape(y), stddev=tf.sqrt(no/2.0)),
        )
        y = y + noise
        # Reshape to Sionna RX layout [B, num_rx=1, n_rx_ant, T, F]
        y = y[:, tf.newaxis, :, :, :]

        # --- RX impairments (per-chain PN + IQI) via time-domain detour ---
        if self.pn_rx is not None or self.iqi is not None:
            y = self._apply_rx_impairments(y, batch_size)

        # --- L2-genie IQI compensation (per RX chain, true α,β) ---
        # Conservative classical baseline: 2×2 mirror-pair inverse on each
        # RX chain using the synthesizer's true (α,β). Applied to the raw
        # received grid BEFORE channel estimation / MIMO detection — IQI is
        # an analog RX-chain impairment, independent of the spatial/MIMO
        # structure. Same operation as SIMO L2-genie, broadcast over the
        # 2 chains. (L2-genie ≈ L2-real in SIMO, and giving the classical
        # receiver genie IQI knowledge only raises the bar for the neural
        # receiver → conservative for the neural-advantage claim.)
        if self.cfg.genie_iqi_comp and self.iqi is not None:
            y_mirror_conj = tf.math.conj(
                tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1)
            )
            alpha = tf.reshape(self.iqi.alpha, [1, 1, self.cfg.n_rx_ant, 1, 1])
            beta = tf.reshape(self.iqi.beta, [1, 1, self.cfg.n_rx_ant, 1, 1])
            det = tf.cast(tf.math.abs(alpha) ** 2 - tf.math.abs(beta) ** 2,
                          tf.complex64)
            y = (tf.math.conj(alpha) * y - beta * y_mirror_conj) / det

        # --- L2-real IQI compensation (per RX chain, DMRS-estimated) ---
        # Same insertion point as genie and as SIMO: on the raw received
        # grid, BEFORE channel estimation, which then re-runs on compensated y.
        if self.cfg.real_iqi_comp and self.iqi is not None:
            y = self._l2_real_iqi_comp(y, no)

        return b, c, y, no

    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        """Classical receiver: shared front end + LMMSE MIMO detection."""
        b, c, y, no = self.generate_rx(batch_size, ebno_db)
        # --- Channel estimation of the EFFECTIVE channel + MIMO detection ---
        h_hat, err_var = self.chan_est(y, no)
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)     # per-layer LLRs
        llr = self.layer_demapper(llr)         # recombine layers → codeword
        b_hat, _ = self.tb_dec(llr)
        return b, b_hat

    def _apply_shared_pn_tx(self, x_ant, batch_size):
        """Shared-LO TX PN: one φ[n] applied to all n_tx_ant chains."""
        n_tx = self.cfg.n_tx_ant
        x_t = tf.signal.ifft(x_ant)               # [B, n_tx, T, F] → time per sym
        x_t = tf.reshape(x_t, [batch_size, n_tx, self._slot_len])
        phi = self.pn_tx((batch_size, 1))         # [B, 1, slot_len] shared
        x_t = x_t * tf.exp(tf.complex(0.0, phi))  # broadcast over n_tx
        x_t = tf.reshape(x_t, [batch_size, n_tx, self.cfg.n_symbols,
                               self.cfg.fft_size])
        return tf.signal.fft(x_t)

    def _apply_rx_impairments(self, y, batch_size):
        """Per-chain RX PN + IQI via the time-domain detour."""
        n_rx = self.cfg.n_rx_ant
        y_t = tf.signal.ifft(y)                    # [B,1,n_rx,T,F]
        y_t = tf.reshape(y_t, [batch_size, 1, n_rx, self._slot_len])
        if self.pn_rx is not None:
            phi_rx = self.pn_rx((batch_size, 1, n_rx))
            y_t = y_t * tf.exp(tf.complex(0.0, phi_rx))
        if self.iqi is not None:
            y_t = self.iqi(y_t)
        y_t = tf.reshape(y_t, [batch_size, 1, n_rx, self.cfg.n_symbols,
                               self.cfg.fft_size])
        return tf.signal.fft(y_t)

    def _estimate_gamma(self, y, no):
        """Per-RX-chain gamma_hat = beta/alpha from DMRS.

        Port of the SIMO estimator (r1_baseline), applied per RX chain with the
        per-layer reconstruction summed over layers. The math lives in
        `estimate_gamma_core` so it can be exercised directly.

        Conjugation: RX IQ imbalance leaks beta*conj(y[N-k]), so the regressor
        is conj(hx_mirror).
        """
        h_hat, _ = self._ls_est_for_iqi(y, no)      # [B,1,n_rx,1,L,T,F]
        h_hat = tf.squeeze(h_hat, axis=[3])          # [B,1,n_rx,L,T,F]
        return estimate_gamma_core(y, h_hat, self._iqi_pilot_grid,
                                   self._iqi_mirror_mask)

    def _l2_real_iqi_comp(self, y, no):
        """Per-chain 2x2 mirror-pair inverse with alpha=1, beta=gamma_hat
        -- the identical inverse to the genie path under
        (alpha, beta) -> (1, gamma_hat)."""
        gamma = self._estimate_gamma(y, no)
        self._last_gamma.assign(gamma)          # instrumentation
        beta = tf.reshape(gamma, [1, 1, self.cfg.n_rx_ant, 1, 1])
        y_mirror_conj = tf.math.conj(
            tf.roll(tf.reverse(y, axis=[-1]), shift=1, axis=-1))
        det = tf.cast(1.0, tf.complex64) - beta * tf.math.conj(beta)
        return (y - beta * y_mirror_conj) / det


def run_sweep(cfg, snr_range, num_tti, batch_size, num_mc_runs, out_dir, tag,
              base_seed=42):
    os.makedirs(out_dir, exist_ok=True)
    sim = MimoPdschSim(cfg)
    logging.info(f"MIMO sim: {cfg.n_tx_ant}T→{cfg.n_rx_ant}R, "
                 f"{sim.num_layers} layers, MCS {cfg.mcs_idx}, "
                 f"TBS={sim.tb_size}, eff.rate={sim.code_rate:.4f}")
    _ = sim(batch_size, tf.constant(float(snr_range[0])))  # warmup/trace
    bler_all = np.zeros((num_mc_runs, len(snr_range)))
    for run_idx in range(num_mc_runs):
        seed = cell_seed(base_seed, run_idx)
        seed_all(seed)
        for i, snr_db in enumerate(snr_range):
            errs = blocks = 0
            for _ in range(max(1, num_tti // batch_size)):
                b, b_hat = sim(batch_size, tf.constant(float(snr_db)))
                # b: [B,1,1,tb], b_hat: [B,1,tb] — flatten both to [B, tb]
                b_flat = tf.reshape(b, [batch_size, -1])
                b_hat_flat = tf.reshape(b_hat, [batch_size, -1])
                be = tf.reduce_any(tf.not_equal(b_flat, b_hat_flat), axis=-1)
                errs += int(tf.reduce_sum(tf.cast(be, tf.int32)))
                blocks += batch_size
            bler_all[run_idx, i] = errs / max(1, blocks)
            logging.info(f"run={run_idx} snr={snr_db:+.1f} "
                         f"bler={bler_all[run_idx,i]:.3e}")
    mean = bler_all.mean(axis=0)
    path = os.path.join(out_dir, f"mimo_bler_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "bler_mean", "n_runs", "n_tti"])
        for i, s in enumerate(snr_range):
            w.writerow([s, mean[i], num_mc_runs, num_tti])
    logging.info(f"wrote {path}")
    # 10%-BLER point
    snr10 = None
    for i in range(1, len(snr_range)):
        if mean[i-1] >= 0.1 > mean[i] and mean[i-1] > 0 and mean[i] > 0:
            lb1, lb2 = math.log10(mean[i-1]), math.log10(mean[i])
            snr10 = snr_range[i-1] + (math.log10(0.1)-lb1)*(snr_range[i]-snr_range[i-1])/(lb2-lb1)
            break
    floor = float(np.min([b for b in mean if b > 0])) if (mean > 0).any() else 0.0
    op = f"{snr10:.2f} dB" if snr10 else "unreachable"
    logging.info(f"=== MIMO {tag}: 10%-BLER = {op}, floor = {floor:.3e} ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carrier-hz", type=float, default=7.125e9)
    p.add_argument("--scs-hz", type=float, default=30e3)
    p.add_argument("--channel", choices=["A","B","C","D","E"], default="C")
    p.add_argument("--mcs", type=int, default=27, choices=[4,13,19,27])
    p.add_argument("--n-add", type=int, default=1, choices=[0,1,2,3])
    p.add_argument("--ue-speed-kmh", type=float, default=3.0)
    p.add_argument("--n-prb", type=int, default=51)
    p.add_argument("--n-tx-ant", type=int, default=256)
    p.add_argument("--n-rx-ant", type=int, default=2)
    p.add_argument("--pn-tx", choices=["off","on"], default="off")
    p.add_argument("--pn-rx", choices=["off","on"], default="off")
    p.add_argument("--pn-ue-offset-db", type=float, default=3.0)
    p.add_argument("--iqi", choices=["off","spec"], default="off")
    p.add_argument("--iqi-irr-db", type=float, default=25.0)
    p.add_argument("--genie-iqi-comp", action="store_true",
                   help="L2-genie classical IQI comp (true α,β per RX chain).")
    p.add_argument("--real-iqi-comp", action="store_true")
    p.add_argument("--snr-min", type=float, default=4.0)
    p.add_argument("--snr-max", type=float, default=24.0)
    p.add_argument("--snr-step", type=float, default=1.0)
    p.add_argument("--num-tti", type=int, default=480)
    p.add_argument("--num-mc-runs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--out-dir", type=str, default="./results/mimo")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    seed_all(args.seed)

    cfg = r1_baseline.SimConfig(
        channel_model=args.channel, mcs_idx=args.mcs, dmrs_n_add=args.n_add,
        ue_speed_kmh=args.ue_speed_kmh, n_prb=args.n_prb,
        carrier_hz=args.carrier_hz, scs_hz=args.scs_hz,
        n_tx_ant=args.n_tx_ant, n_rx_ant=args.n_rx_ant, estimator="ls",
        pn_tx=args.pn_tx, pn_rx=args.pn_rx, pn_ue_offset_db=args.pn_ue_offset_db,
        iqi=args.iqi, iqi_irr_db=args.iqi_irr_db, real_iqi_comp=args.real_iqi_comp,
        genie_iqi_comp=args.genie_iqi_comp,
    )
    snr_range = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    run_sweep(cfg, snr_range, args.num_tti, args.batch_size, args.num_mc_runs,
              args.out_dir, args.tag, base_seed=args.seed)


if __name__ == "__main__":
    main()
