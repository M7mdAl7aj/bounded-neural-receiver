"""Floor the LMMSE interpolator's intermediate error variance (the lmmse-practical front end of the LMMSE-fed
network only; imported by r3_deeprx_frontend.py and r5_frontend_eval.py inside their lmmse-practical branch).

The practical prior (results/run_p/prior_practical.npz) is numerically rank-deficient: R_freq has 608 of 612
eigenvalues below float32's rank tolerance, R_time 13 of 14. Sionna's LMMSEInterpolator (order "f-t") first
interpolates in frequency and clamps that step's posterior error variance at 0 (`tf.maximum(err_var, 0)`); the
clamped variance is the diagonal loading of the time step's least-squares solve. At isolated noise levels the
frequency step's variance goes negative and is clamped to 0 (probed: 20.3 dB, and 20.4355-20.4366 dB around one
training run's first NaN batch at 20.435810 dB), the time step then solves a near-singular system, and on some
slots the channel estimate overflows to inf / NaN, which enters R3's input features unchanged.

The fix floors the frequency step's output error variance at FLOOR_REL times the largest pilot error variance of
the same slot (the LS no / |p|^2), before it reaches the time step. Where the variance was valid it is far above the
floor (probed: at least 8.5e-3 times no over 16-28 dB, against FLOOR_REL = 1e-3), so tf.maximum returns it
unchanged and the estimate is bit-identical; only clamped entries change. lmmse_practical.py (run P's classical arm)
is not touched.
"""
import tensorflow as tf
from sionna.phy.ofdm import LMMSEInterpolator

FLOOR_REL = 1e-3


class _FloorFreqStep:
    """Wraps the frequency step of an LMMSEInterpolator: same call, output error variance floored."""

    def __init__(self, inner, floor_rel):
        self.inner, self.floor_rel = inner, floor_rel

    def __call__(self, h_hat, err_var):
        # err_var here is the LS pilot error variance scattered on the grid (0 off the pilots):
        # [batch, num_rx, num_rx_ant, num_tx, num_streams, outer, inner]
        ref = tf.reduce_max(err_var, axis=list(range(1, len(err_var.shape))), keepdims=True)
        h_hat, err_var = self.inner(h_hat, err_var)
        return h_hat, tf.maximum(err_var, self.floor_rel * ref)


def install(estimator, floor_rel=FLOOR_REL):
    """Install the floor on an LSChannelEstimator built by lmmse_practical.make_estimator (order f-t)."""
    interp = estimator._interpol
    if not isinstance(interp, LMMSEInterpolator):
        raise SystemExit(f"lmmse_errvar_floor: expected an LMMSEInterpolator, got {type(interp).__name__}")
    order = list(interp._order)
    if order != ["f", "t"] or len(interp._interpolators) != 2:
        raise SystemExit(f"lmmse_errvar_floor: expected order f-t with two steps, got {order}")
    if not isinstance(interp._interpolators[0], _FloorFreqStep):
        interp._interpolators[0] = _FloorFreqStep(interp._interpolators[0], floor_rel)
    return estimator


# ---------------------------------------------------------------- per-slot fallback at the output
# The floor above does not stop every overflow: a second training run went NaN at 20.3121 dB, and a CPU probe showed
# the non-finite values arising inside the frequency step's own solve, upstream of the floor. So: if any entry of a
# slot's LMMSE channel estimate (both receive antennas) is non-finite, that slot takes
# the stock LS estimate (LS at the pilots, the interpolator's own input, linearly interpolated: the ls front end's
# estimator), and the slot is counted. The LS estimate is computed only when a slot fails (tf.cond), so every call
# without a failing slot returns the LMMSE output itself, and every finite slot of a failing call is selected from the
# LMMSE output by tf.where: finite slots are bit-identical. Counters: calls, slots, fallback slots; one log line per call
# (fallback_log.txt) and the totals at exit (front_end_fallback.json).
import atexit as _atexit
import json as _json

_REGISTRY = []


class SlotFallback:
    def __init__(self, lmmse_est, ls_est, log_path=None):
        self.lmmse, self.ls, self.log_path = lmmse_est, ls_est, log_path
        self.calls = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.slots = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.fallback = tf.Variable(0, dtype=tf.int64, trainable=False)

    def __call__(self, y, no):
        h, e = self.lmmse(y, no)
        ax = list(range(1, len(h.shape)))
        ok = tf.reduce_all(tf.logical_and(tf.math.is_finite(tf.math.real(h)), tf.math.is_finite(tf.math.imag(h))), axis=ax)
        bad = tf.logical_not(ok)                                            # [batch]
        n = tf.reduce_sum(tf.cast(bad, tf.int64))

        def take_ls():
            hl, el = self.ls(y, no)
            m = tf.reshape(bad, [-1] + [1] * (len(h.shape) - 1))
            return tf.where(m, hl, h), tf.where(m, tf.cast(el, e.dtype), e)

        h2, e2 = tf.cond(n > 0, take_ls, lambda: (h, e))
        self.calls.assign_add(1)
        self.slots.assign_add(tf.cast(tf.shape(h)[0], tf.int64))
        self.fallback.assign_add(n)
        if self.log_path:
            tf.print("call", self.calls, "fallback_slots", n, "of", tf.shape(h)[0], output_stream="file://" + self.log_path)
        return h2, e2

    def totals(self):
        return dict(calls=int(self.calls.numpy()), slots=int(self.slots.numpy()), fallback_slots=int(self.fallback.numpy()))


def with_fallback(lmmse_est, ls_est, log_path=None, sidecar=None):
    """Wrap an LMMSE estimator (floor installed) with the per-slot LS fallback; the totals are written to
    `sidecar` (JSON) at interpreter exit."""
    fb = SlotFallback(lmmse_est, ls_est, log_path)
    _REGISTRY.append((fb, sidecar))
    return fb


@_atexit.register
def _write_sidecars():
    """One sidecar per path: the totals summed over every estimator instance that writes there (a training run builds
    one for training and one for validation), and each instance's own totals."""
    by_path = {}
    for fb, path in _REGISTRY:
        if path: by_path.setdefault(path, []).append(fb)
    for path, fbs in by_path.items():
        try:
            per = [fb.totals() for fb in fbs]
            tot = {k: sum(t[k] for t in per) for k in ("calls", "slots", "fallback_slots")}
            with open(path, "w") as f:
                _json.dump(dict(tot, instances=per, rule="per-slot LS fallback", log=fbs[0].log_path), f, indent=1)
        except Exception:
            pass
