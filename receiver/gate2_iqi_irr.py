"""gate2_iqi_irr.py -- Validate IQImbalance realized IRR within +/-0.5 dB.

Four tests:
  A. Bisection solver: verify the exact (g_nom, phi_nom) from the solver
     yields IRR == target to machine precision (< 0.01 dB).
  B. Empirical tone (no jitter): inject exp(j*2*pi*f0*t) per-antenna with
     overrides pinned to nominal, measure |image|^2/|signal|^2 via FFT.
  C. Jittered analytical (100 antennas): mean realized IRR within +/-0.5 dB.
  D. Jittered empirical (100 antennas): mean IRR from tone test within
     +/-0.5 dB of the per-antenna analytical values (validates __call__).

Exit code 0 => PASS; 1 => FAIL.

Usage:
    python gate2_iqi_irr.py
    python gate2_iqi_irr.py --irr-db 30.0
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
IQImbalance = _r1.IQImbalance


def _bisect_eps(irr_db):
    """Replicate the class's bisection to find nominal (g, phi)."""
    def _irr_at_eps(e):
        g_ = 1.0 - e
        a_ = 0.5 * (1.0 + g_ * np.exp(1j * e))
        b_ = 0.5 * (1.0 - g_ * np.exp(-1j * e))
        return 10.0 * np.log10(np.abs(a_) ** 2 / np.abs(b_) ** 2)
    lo, hi = 1e-12, 0.99
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _irr_at_eps(mid) > irr_db:
            lo = mid
        else:
            hi = mid
    eps = 0.5 * (lo + hi)
    return eps


def test_bisection_exact(irr_db, tol_db=0.01):
    """Test A: verify the bisection solver's nominal IRR == target."""
    eps = _bisect_eps(irr_db)
    g_nom = 1.0 - eps
    phi_nom = eps
    # Exact computation
    alpha = 0.5 * (1.0 + g_nom * np.exp(1j * phi_nom))
    beta = 0.5 * (1.0 - g_nom * np.exp(-1j * phi_nom))
    realized = 10.0 * np.log10(np.abs(alpha) ** 2 / np.abs(beta) ** 2)
    err = abs(realized - irr_db)
    return realized, err, err <= tol_db


def test_class_no_jitter(irr_db, tol_db=0.01):
    """Test B: create IQImbalance with overrides at nominal, check IRR."""
    eps = _bisect_eps(irr_db)
    g_nom = 1.0 - eps
    phi_nom = eps
    num_ant = 2
    iqi = IQImbalance(
        num_rx_ant=num_ant,
        irr_db=irr_db,
        g_overrides=[g_nom] * num_ant,
        phi_overrides=[phi_nom] * num_ant,
    )
    realized = iqi.realized_irr_db
    max_err = float(np.max(np.abs(realized - irr_db)))
    return realized, max_err, max_err <= tol_db


def test_empirical_tone_no_jitter(irr_db, tol_db=0.05):
    """Test C: inject tone at nominal (g, phi), measure IRR from FFT."""
    eps = _bisect_eps(irr_db)
    g_nom = 1.0 - eps
    phi_nom = eps
    num_ant = 2
    iqi = IQImbalance(
        num_rx_ant=num_ant,
        irr_db=irr_db,
        g_overrides=[g_nom] * num_ant,
        phi_overrides=[phi_nom] * num_ant,
    )

    N = 8192
    f0_bin = 100
    t = np.arange(N, dtype=np.float64)
    tone = np.exp(2j * np.pi * f0_bin * t / N).astype(np.complex64)
    y = tf.constant(tone.reshape(1, 1, 1, N))
    y = tf.broadcast_to(y, [1, 1, num_ant, N])

    y_iqi = iqi(y)
    Y = tf.signal.fft(tf.cast(y_iqi, tf.complex64))
    power = tf.abs(Y) ** 2
    p_sig = power[0, 0, :, f0_bin].numpy()
    p_img = power[0, 0, :, N - f0_bin].numpy()
    irr_meas = 10.0 * np.log10(p_sig / p_img)
    max_err = float(np.max(np.abs(irr_meas - irr_db)))
    return irr_meas, max_err, max_err <= tol_db


def test_jittered_analytical(irr_db, num_ant=100, tol_db=0.5, seed=20260528):
    """Test D: jittered (g, phi), mean realized IRR within tolerance."""
    iqi = IQImbalance(
        num_rx_ant=num_ant,
        irr_db=irr_db,
        seed=seed,
    )
    realized = iqi.realized_irr_db
    mean_irr = float(np.mean(realized))
    std_irr = float(np.std(realized))
    err = abs(mean_irr - irr_db)
    return mean_irr, std_irr, realized, err, err <= tol_db


def test_jittered_empirical_consistency(irr_db, num_ant=100, tol_db=0.1,
                                        seed=20260528):
    """Test E: jittered, verify tone-measured IRR matches analytical per-ant."""
    iqi = IQImbalance(
        num_rx_ant=num_ant,
        irr_db=irr_db,
        seed=seed,
    )

    N = 8192
    f0_bin = 100
    t = np.arange(N, dtype=np.float64)
    tone = np.exp(2j * np.pi * f0_bin * t / N).astype(np.complex64)
    y = tf.constant(tone.reshape(1, 1, 1, N))
    y = tf.broadcast_to(y, [1, 1, num_ant, N])

    y_iqi = iqi(y)
    Y = tf.signal.fft(tf.cast(y_iqi, tf.complex64))
    power = tf.abs(Y) ** 2
    p_sig = power[0, 0, :, f0_bin].numpy()
    p_img = power[0, 0, :, N - f0_bin].numpy()
    irr_meas = 10.0 * np.log10(p_sig / p_img)

    # Compare per-antenna: empirical vs analytical
    diff = np.abs(irr_meas - iqi.realized_irr_db)
    max_diff = float(np.max(diff))
    return irr_meas, iqi.realized_irr_db, max_diff, max_diff <= tol_db


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--irr-db", type=float, default=25.0)
    p.add_argument("--tol-db", type=float, default=0.5,
                   help="Mean IRR tolerance for jittered test (default 0.5 dB)")
    p.add_argument("--num-ant-jitter", type=int, default=100)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    irr = args.irr_db
    all_pass = True

    # --- Test A: bisection solver exact ---
    realized_a, err_a, pass_a = test_bisection_exact(irr)
    logging.info(
        f"[A] Bisection solver: target={irr:.2f} dB, "
        f"realized={realized_a:.6f} dB, |err|={err_a:.6f} dB "
        f"{'PASS' if pass_a else 'FAIL'}"
    )
    all_pass &= pass_a

    # --- Test B: class no-jitter analytical ---
    realized_b, err_b, pass_b = test_class_no_jitter(irr)
    logging.info(
        f"[B] Class no-jitter: per-antenna IRR = "
        f"{realized_b.tolist()}, max |err|={err_b:.6f} dB "
        f"{'PASS' if pass_b else 'FAIL'}"
    )
    all_pass &= pass_b

    # --- Test C: empirical tone no-jitter ---
    irr_meas_c, err_c, pass_c = test_empirical_tone_no_jitter(irr)
    logging.info(
        f"[C] Empirical tone (no jitter): per-antenna IRR = "
        f"{irr_meas_c.tolist()}, max |err|={err_c:.4f} dB "
        f"{'PASS' if pass_c else 'FAIL'}"
    )
    all_pass &= pass_c

    # --- Test D: jittered analytical mean ---
    mean_d, std_d, per_ant_d, err_d, pass_d = test_jittered_analytical(
        irr, num_ant=args.num_ant_jitter, tol_db=args.tol_db
    )
    logging.info(
        f"[D] Jittered analytical ({args.num_ant_jitter} antennas): "
        f"mean={mean_d:.3f} dB, std={std_d:.3f} dB, |err|={err_d:.3f} dB "
        f"(tol {args.tol_db:.2f} dB) {'PASS' if pass_d else 'FAIL'}"
    )
    logging.info(
        f"    per-antenna range: [{float(np.min(per_ant_d)):.2f}, "
        f"{float(np.max(per_ant_d)):.2f}] dB"
    )
    all_pass &= pass_d

    # --- Test E: jittered empirical consistency ---
    irr_emp, irr_ana, max_diff, pass_e = test_jittered_empirical_consistency(
        irr, num_ant=args.num_ant_jitter
    )
    logging.info(
        f"[E] Jittered empirical consistency: max |tone - analytical| = "
        f"{max_diff:.4f} dB {'PASS' if pass_e else 'FAIL'}"
    )
    all_pass &= pass_e

    # --- Production config report ---
    iqi_prod = IQImbalance(num_rx_ant=2, irr_db=irr)
    logging.info(
        f"[Production] 2 antennas, target={irr:.1f} dB: "
        f"realized_irr = {iqi_prod.realized_irr_db.tolist()}"
    )

    print()
    print("=" * 64)
    if all_pass:
        print(f"GATE 2 PASS: all IRR tests within tolerance")
        code = 0
    else:
        print(f"GATE 2 FAIL: one or more IRR tests exceeded tolerance")
        code = 1
    print("=" * 64)

    sys.exit(code)


if __name__ == "__main__":
    main()
