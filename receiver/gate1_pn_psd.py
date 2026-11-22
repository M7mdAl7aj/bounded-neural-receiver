"""gate1_pn_psd.py — Validate that PSDPhaseNoise's realized PSD matches the
analytical spec mask within ±1 dB across [1 kHz, 10 MHz].

Uses a **multi-realization averaged periodogram**: generate K independent
phase-noise realizations of length N with `oversample=1`, compute the
periodogram of each (which is exact for the cyclic irfft output — zero
spectral leakage), and average.  This avoids the Welch-window mainlobe
leakage that corrupts the lowest in-band bins when the PSD has a steep
low-frequency slope (~100 dB from DC to 1 kHz).

Usage:
    python gate1_pn_psd.py
    python gate1_pn_psd.py --n-real 500 --seg-len 131072

Seeded (42): every run gives the same result.

Exit code 0 ⇒ gate 1 PASS (max |error| ≤ 1 dB over [1 kHz, 10 MHz]).
Exit code 1 ⇒ FAIL (max error reported).
"""
import argparse
import logging
import sys

import numpy as np
import tensorflow as tf

import importlib.util
_spec = importlib.util.spec_from_file_location("r1", "r1_baseline.py")
_r1 = importlib.util.module_from_spec(_spec)
sys.modules["r1"] = _r1
_spec.loader.exec_module(_r1)
PSDPhaseNoise = _r1.PSDPhaseNoise
_PN_29_55_GHZ = _r1._PN_29_55_GHZ
_PN_CARRIER_SCALE_DB = _r1._pn_carrier_scale_db(7.125e9)   # 20·log10(7.125/29.55) dB, the simulator's own scaling


def analytical_psd(f_hz, psd0):
    """Return PSD(f) in rad²/Hz on the 29.55 GHz model, scaled by psd0."""
    f = np.asarray(f_hz, dtype=np.float64)
    fz = np.asarray(_PN_29_55_GHZ["fz_list_hz"], dtype=np.float64)
    fp = np.asarray(_PN_29_55_GHZ["fp_list_hz"], dtype=np.float64)
    az = np.asarray(_PN_29_55_GHZ["alpha_z"], dtype=np.float64)
    ap = np.asarray(_PN_29_55_GHZ["alpha_p"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.prod(1.0 + (f[:, None] / fz) ** az, axis=1)
        den = np.prod(1.0 + (f[:, None] / fp) ** ap, axis=1)
    return psd0 * num / den


def validate_one(label, psd0, fs_hz, seg_len, n_real, batch_size, tol_db,
                 f_lo=1e3, f_hi=1e7):
    """Multi-realization averaged periodogram.

    Each realization is generated with `oversample=1` — the irfft output
    is a cyclic signal whose DFT recovers the exact Fourier coefficients
    with zero spectral leakage.  Averaging K independent realizations
    reduces the chi²(2) per-bin variance to chi²(2K), giving
    1σ ≈ 4.343/sqrt(K) dB.

    The previous Welch approach (single long signal, windowed segments)
    suffered from two issues:
    1. `oversample=2` cropping introduced +7–19 dB bias via rectangular-
       window spectral leakage from the enormous near-DC PSD.
    2. Even with `oversample=1`, the Blackman-Harris mainlobe (−3.34 dB
       at 1-bin offset) smeared the steep low-frequency PSD into the
       first few in-band bins (+4–5 dB at 1526 Hz with 16M samples).
    Both issues vanish with the multi-realization approach.
    """
    N = seg_len
    half = N // 2 + 1
    df = fs_hz / N

    # No explicit oversample — relies on the production default (1).
    # This ensures gate 1 validates the same code path as the BLER sim.
    pn = PSDPhaseNoise(
        psd0=psd0,
        fz_list_hz=_PN_29_55_GHZ["fz_list_hz"],
        alpha_z=_PN_29_55_GHZ["alpha_z"],
        fp_list_hz=_PN_29_55_GHZ["fp_list_hz"],
        alpha_p=_PN_29_55_GHZ["alpha_p"],
        fs_hz=fs_hz,
        slot_length=N,
    )

    psd_accum = np.zeros(half, dtype=np.float64)
    n_done = 0
    while n_done < n_real:
        bs = min(batch_size, n_real - n_done)
        phi = pn((bs,))  # [bs, N], float32
        X = tf.signal.rfft(phi)  # [bs, half], complex64
        X_np = X.numpy().astype(np.complex128)
        psd_accum += np.sum(np.abs(X_np) ** 2, axis=0)
        n_done += bs

    logging.info(
        f"[{label}] averaged {n_done} realizations, seg_len={N}, df={df:.1f} Hz"
    )

    # One-sided PSD: PSD(f_k) = 2 |rfft(x)[k]|² / (N · fs) averaged over K.
    psd_meas = 2.0 * psd_accum / (n_done * N * fs_hz)
    psd_meas[0] /= 2.0  # DC: factor 1, not 2
    psd_meas[-1] /= 2.0  # Nyquist: factor 1
    f_bins = np.fft.rfftfreq(N, d=1.0 / fs_hz)

    sigma_db = 10.0 / (np.log(10.0) * np.sqrt(n_done))
    logging.info(f"[{label}] 1σ ≈ {sigma_db:.3f} dB")

    band_mask = (f_bins >= f_lo) & (f_bins <= f_hi) & (f_bins > 0)
    f_band = f_bins[band_mask]
    psd_band_meas = psd_meas[band_mask]
    psd_band_true = analytical_psd(f_band, psd0)
    err_db = 10.0 * np.log10(psd_band_meas / psd_band_true)
    max_err = float(np.max(np.abs(err_db)))
    mean_err = float(np.mean(err_db))

    order = np.argsort(np.abs(err_db))[::-1]
    worst_idx = order[:5]
    logging.info(
        f"[{label}] over [{f_lo:.0e}, {f_hi:.0e}] Hz: "
        f"mean error = {mean_err:+.3f} dB, max |error| = {max_err:.3f} dB "
        f"(tolerance {tol_db:.2f} dB)"
    )
    logging.info(f"[{label}] worst-error bins:")
    for idx in worst_idx:
        logging.info(
            f"    f={f_band[idx]:.3e} Hz: err={err_db[idx]:+.3f} dB "
            f"(meas={10 * np.log10(psd_band_meas[idx]):+.2f}, "
            f"true={10 * np.log10(psd_band_true[idx]):+.2f})"
        )
    for fp_check in [1e3, 1e4, 1e5, 1e6, 5e6]:
        if fp_check < f_band[0] or fp_check > f_band[-1]:
            continue
        idx = int(np.argmin(np.abs(f_band - fp_check)))
        logging.info(
            f"    f={f_band[idx]:.2e} Hz: meas={10 * np.log10(psd_band_meas[idx]):+7.2f} "
            f"dB, true={10 * np.log10(psd_band_true[idx]):+7.2f} dB, "
            f"diff={err_db[idx]:+.3f} dB"
        )
    return max_err, mean_err, f_band, psd_band_meas, psd_band_true


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg-len", type=int, default=2 ** 17,
                   help="FFT length per realization (default 131072).")
    p.add_argument("--n-real", type=int, default=500,
                   help="Number of independent realizations to average.")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Realizations per GPU batch.")
    p.add_argument("--fs-mhz", type=float, default=100.0,
                   help="Sample rate in MHz.")
    p.add_argument("--ue-offset-db", type=float, default=3.0)
    p.add_argument("--tol-db", type=float, default=1.0,
                   help="Pass threshold (max |error| in dB).")
    p.add_argument("--save-plot", type=str, default=None,
                   help="Path to save PSD-vs-mask PNG.")
    args = p.parse_args()
    # One seed, set once: every draw (PSDPhaseNoise samples sionna_config.tf_rng) derives from it, so every
    # run gives the same result.
    _r1.seed_all(42)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    fs_hz = args.fs_mhz * 1e6
    psd0_gnb = _PN_29_55_GHZ["psd0"] * 10.0 ** (_PN_CARRIER_SCALE_DB / 10.0)
    psd0_ue = psd0_gnb * 10.0 ** (args.ue_offset_db / 10.0)

    logging.info(
        f"29.55 GHz model carrier-scaled to 7.125 GHz: "
        f"psd0_gnb={psd0_gnb:.4f}, psd0_ue={psd0_ue:.4f} "
        f"(+{args.ue_offset_db:.1f} dB UE)"
    )

    results = []
    for label, psd0 in [("gNB", psd0_gnb),
                         (f"UE (+{args.ue_offset_db:.1f} dB)", psd0_ue)]:
        max_err, mean_err, f, p_meas, p_true = validate_one(
            label, psd0, fs_hz, args.seg_len, args.n_real,
            args.batch_size, tol_db=args.tol_db,
        )
        results.append((label, max_err, mean_err, f, p_meas, p_true))

    worst = max(r[1] for r in results)
    print()
    print("=" * 64)
    if worst <= args.tol_db:
        print(f"GATE 1 PASS: max |error| = {worst:.3f} dB ≤ {args.tol_db:.2f} dB")
        verdict_code = 0
    else:
        print(f"GATE 1 FAIL: max |error| = {worst:.3f} dB > {args.tol_db:.2f} dB")
        verdict_code = 1
    print("=" * 64)

    if args.save_plot is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            for label, _, _, f, p_meas, p_true in results:
                axes[0].loglog(f, p_meas, label=f"{label} measured", alpha=0.7)
                axes[0].loglog(f, p_true, "--", label=f"{label} analytical")
                err = 10 * np.log10(p_meas / p_true)
                axes[1].semilogx(f, err, label=label, alpha=0.7)
            axes[0].set_ylabel("PSD [rad²/Hz]")
            axes[0].set_title(
                "Gate 1: PN PSD realized vs analytical "
                "(29.55 GHz → 7.125 GHz)"
            )
            axes[0].grid(True, which="both", alpha=0.3)
            axes[0].legend()
            axes[1].axhline(args.tol_db, color="r", ls="--", alpha=0.5)
            axes[1].axhline(-args.tol_db, color="r", ls="--", alpha=0.5)
            axes[1].set_xlabel("Offset frequency [Hz]")
            axes[1].set_ylabel("Error [dB]")
            axes[1].set_title(f"Error (tolerance ±{args.tol_db} dB)")
            axes[1].grid(True, which="both", alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(args.save_plot, dpi=150)
            logging.info(f"saved {args.save_plot}")
        except ImportError:
            logging.warning("matplotlib not available; skipping plot")

    sys.exit(verdict_code)


if __name__ == "__main__":
    main()
