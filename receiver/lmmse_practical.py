"""lmmse_practical.py - the practical-prior LMMSE estimator for the classical
chain. r1_baseline.py is imported unchanged: the estimator is injected into a
PdschSim after construction.

  oracle_prior(sim, ...)      the oracle diagnostic's prior, computed exactly as
                              PdschSim.build_lmmse_estimator (r1_baseline.py)
                              computes it (same ops, same RNG use), returned
                              instead of installed. Run through the injection
                              path it must reproduce run F (the oracle arm).
  make_estimator(sim, Rf, Rt) LSChannelEstimator(LMMSEInterpolator(...)) with the
                              arguments PdschSim.build_lmmse_estimator uses.
  harvest_practical(...)      the practical prior: LS estimates at the DMRS
                              positions of the full link (CDL-C, PN + IQI at the
                              training target, L2-real compensation on), one SNR
                              per step from the training script's own sampler
                              (r3_deeprx.sample_ebno_db, mean of 32 draws),
                              noise-debiased, projected to the PSD cone, with a
                              static (all-ones) time covariance.
"""
import json, logging, math, os, time
import numpy as np
import tensorflow as tf
from sionna.phy.ofdm import LSChannelEstimator, LMMSEInterpolator
from sionna.phy.channel.tr38901 import CDL          # the imports of r1_baseline.py
from sionna.phy.channel import OFDMChannel


def oracle_prior(sim, batch_size, num_batches):
    """Verbatim harvest of PdschSim.build_lmmse_estimator (r1_baseline.py),
    returning (R_freq, R_time) as complex64 tf tensors."""
    M = sim.cfg.fft_size
    N = sim.cfg.n_symbols
    b = sim.source([batch_size, 1, 1, sim.tb_size])
    c = sim.tb_enc(b)
    x = sim.mapper(c)
    x_rg = sim.rg_mapper(x)
    cdtype = tf.complex64
    R_freq_sum = tf.zeros([M, M], dtype=cdtype)
    R_time_sum = tf.zeros([N, N], dtype=cdtype)
    n_freq_samples = 0
    n_time_samples = 0
    wch = sim.cfg.lmmse_warmup_channel or sim.cfg.channel_model
    saved = None
    if wch != sim.cfg.channel_model:
        ue_speed_mps = sim.cfg.ue_speed_kmh / 3.6
        warm_cdl = CDL(model=wch, delay_spread=sim.cfg.delay_spread_ns * 1e-9,
                       carrier_frequency=sim.cfg.carrier_hz,
                       ut_array=sim.ue_array, bs_array=sim.bs_array,
                       direction="downlink", min_speed=ue_speed_mps, max_speed=ue_speed_mps)
        saved = sim.ofdm_channel
        sim.ofdm_channel = OFDMChannel(channel_model=warm_cdl, resource_grid=sim.rg, add_awgn=True,
                                       return_channel=True, normalize_channel=True)
    for b_i in range(num_batches):
        h = sim._channel_sample(x_rg)
        h_freq = tf.reshape(h, [-1, M])
        R_freq_sum += tf.matmul(h_freq, tf.math.conj(h_freq), transpose_a=True)
        n_freq_samples += int(tf.shape(h_freq)[0])
        h_t = tf.transpose(h, [0, 1, 2, 3, 4, 6, 5])
        h_time = tf.reshape(h_t, [-1, N])
        R_time_sum += tf.matmul(h_time, tf.math.conj(h_time), transpose_a=True)
        n_time_samples += int(tf.shape(h_time)[0])
    R_freq = R_freq_sum / tf.cast(n_freq_samples, cdtype)
    R_time = R_time_sum / tf.cast(n_time_samples, cdtype)
    R_freq = 0.5 * (R_freq + tf.linalg.adjoint(R_freq))
    R_time = 0.5 * (R_time + tf.linalg.adjoint(R_time))
    if saved is not None:
        sim.ofdm_channel = saved
    logging.info(f"oracle_prior: {num_batches} batches, freq_samples={n_freq_samples}, "
                 f"time_samples={n_time_samples}, trace(R_freq)={float(tf.math.real(tf.linalg.trace(R_freq))):.3f}")
    return R_freq, R_time


def make_estimator(sim, R_freq, R_time, order):
    """The oracle path's estimator construction (end of PdschSim.build_lmmse_estimator)."""
    interp = LMMSEInterpolator(pilot_pattern=sim.rg.pilot_pattern, cov_mat_time=R_time,
                               cov_mat_freq=R_freq, order=order)
    return LSChannelEstimator(resource_grid=sim.rg, interpolator=interp)


def save_prior(path, R_freq, R_time, meta):
    np.savez(path, R_freq=np.asarray(R_freq), R_time=np.asarray(R_time),
             meta=np.array(json.dumps(meta, sort_keys=True)))


def load_prior(path):
    z = np.load(path, allow_pickle=False)
    return (tf.constant(z["R_freq"], dtype=tf.complex64), tf.constant(z["R_time"], dtype=tf.complex64),
            json.loads(str(z["meta"])))


class _RecordingLS:
    """Wraps the sim's LS estimator inside its traced call: passes the estimate
    through unchanged and stores the estimate at the DMRS symbols and the noise
    level in tf.Variables, read back after every call."""

    def __init__(self, inner, batch_size, n_rx_ant, dmrs_symbols, fft_size):
        self.inner = inner
        self.sym = list(dmrs_symbols)
        self.h_var = tf.Variable(tf.zeros([batch_size, n_rx_ant, len(self.sym), fft_size], tf.complex64),
                                 trainable=False)
        self.no_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    def __call__(self, y, no):
        h_hat, err_var = self.inner(y, no)
        # h_hat: [B, num_rx=1, n_rx_ant, num_tx=1, num_streams=1, T, F]
        h_p = tf.gather(h_hat[:, 0, :, 0, 0, :, :], self.sym, axis=2)
        self.h_var.assign(h_p)
        self.no_var.assign(tf.cast(tf.reshape(no, []), tf.float32))
        return h_hat, err_var


def harvest_practical(sim, batch_size, num_steps, sampler, log_every=50):
    """Practical prior from IQ-imbalance-compensated LS pilot estimates of the
    full link. `sim` must be a PdschSim built with estimator='ls' and
    real_iqi_comp=True; `sampler(batch_size)` is r3_deeprx.sample_ebno_db, and
    the step SNR is its mean, as in the training loop."""
    M = sim.cfg.fft_size
    rec = _RecordingLS(sim.chan_est, batch_size, sim.cfg.n_rx_ant, sim.cfg.dmrs_symbols, M)
    sim.chan_est = rec                       # injected after construction
    R_sum = np.zeros((M, M), np.complex128)
    K = 0
    no_list, snr_list = [], []
    t0 = time.time()
    for step in range(num_steps):
        ebno = tf.reduce_mean(sampler(batch_size))
        _ = sim(batch_size, ebno)
        hp = rec.h_var.numpy().reshape(-1, M).astype(np.complex128)   # [B * n_rx * n_dmrs, M]
        R_sum += hp.T @ hp.conj()            # sum_k h_k h_k^H with h as a row: H^T conj(H)
        K += hp.shape[0]
        no_list.append(float(rec.no_var.numpy()))
        snr_list.append(float(ebno.numpy()))
        if (step + 1) % log_every == 0 or step == num_steps - 1:
            logging.info(f"harvest step {step + 1}/{num_steps}, vectors={K}, elapsed={time.time() - t0:.0f}s")
    R_raw = R_sum / K
    sigma2 = float(np.mean(no_list))
    R_db = R_raw - sigma2 * np.eye(M)
    R_db = 0.5 * (R_db + R_db.conj().T)
    w, V = np.linalg.eigh(R_db)
    neg_mass = float(np.abs(w[w < 0]).sum()) / float(np.abs(w).sum())
    R_psd = (V * np.clip(w, 0.0, None)) @ V.conj().T
    R_psd = 0.5 * (R_psd + R_psd.conj().T)
    N = sim.cfg.n_symbols
    R_time = np.ones((N, N), np.complex128)
    stats = dict(vectors=K, steps=num_steps, batch_size=batch_size, sigma2_mean=sigma2,
                 trace_raw=float(np.real(np.trace(R_raw))), trace_debiased=float(np.real(np.trace(R_db))),
                 trace_psd=float(np.real(np.trace(R_psd))), clipped_eigen_mass=neg_mass,
                 n_negative_eigs=int((w < 0).sum()), min_eig=float(w.min()), max_eig=float(w.max()),
                 snr_step_mean=float(np.mean(snr_list)), snr_step_sd=float(np.std(snr_list, ddof=1)),
                 snr_step_min=float(np.min(snr_list)), snr_step_max=float(np.max(snr_list)),
                 harvest_seconds=round(time.time() - t0, 1))
    return R_psd.astype(np.complex64), R_time.astype(np.complex64), stats, np.array(snr_list)
