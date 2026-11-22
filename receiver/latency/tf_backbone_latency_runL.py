"""Run L: the TensorFlow row of Table II, measured. The trained R3 conv backbone (input_conv ->
11 ResBlocks -> output_conv, the functional model trt_export_onnx.py exports) in
TensorFlow FP32, batch 1, 51 PRB, input [1, 14, 612, 14] resident on the GPU, tf.function.

Two passes of this script, never both in one process (profiling inflates wall time):
  --pass kernel  (a) the Table II value: the GPU kernel window per slot, first kernel start
                 to last kernel end of the backbone call, from the TensorFlow profiler's
                 CUPTI timestamps (trace_viewer@ export, microseconds; memcpy/memset
                 excluded). TensorFlow exposes no CUDA stream to Python, so CUDA events
                 cannot bracket its kernels as the TensorRT timer does.
  --pass wall    (b) recorded beside it: host perf_counter around the same call followed by
                 tf.test.experimental.sync_devices(), profiler off.
Each pass: 50 warm-up slots, then the median (p10, p90) of 300 slots. L-G3 is checked in the
pass: no other compute process before start, exactly this process on the card once the GPU is
initialized. Across passes, the driver that runs both checks that two kernel passes agree within
2% and that the kernel median is <= the wall median (STOP otherwise).

  python latency/tf_backbone_latency_runL.py --pass kernel OUT_DIR    (from receiver/, paper3trt env)
Writes OUT_DIR/runL_<pass>.json and prints it.
"""
import argparse, glob, json, os, statistics, subprocess, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
HP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HP); os.chdir(HP)
ap = argparse.ArgumentParser()
ap.add_argument("--pass", dest="mode", choices=["kernel", "wall"], required=True)
ap.add_argument("out")
A = ap.parse_args()
OUT = os.path.abspath(A.out); os.makedirs(OUT, exist_ok=True)
N_WARM, N_MEAS = 50, 300


def compute_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()


if compute_apps():
    sys.exit(f"L-G3 STOP: compute processes on the GPU before start: {compute_apps()}")

import numpy as np, tensorflow as tf
from r1_baseline import SimConfig, MCS_TABLE
from r3_deeprx import NeuralReceiver

if not tf.config.list_physical_devices("GPU"):
    sys.exit("no GPU visible")
cfg = SimConfig(n_prb=51, mcs_idx=27, n_tx_ant=1, n_rx_ant=2)
F, T, qm, nrx = cfg.fft_size, cfg.n_symbols, MCS_TABLE[27][1], 2
C = 2 * (2 * nrx) + 2 + 2 * nrx                      # 14 input channels (SIMO 1x2 + mirror)
net = NeuralReceiver(num_rx_ant=nrx, num_conv_channels=128, num_blocks=11,
                     num_bits_per_symbol=qm, use_mirror_feature=True)
yb = tf.complex(tf.zeros([1, 1, nrx, T, F]), tf.zeros([1, 1, nrx, T, F]))
_ = net((yb, yb, tf.zeros([T, F]), tf.ones([1])))
net.load_weights("results/r3_mirror/final/weights.h5")
inp = tf.keras.Input(shape=(T, F, C), batch_size=1, dtype=tf.float32)
x = net.input_conv(inp)
for blk in net.blocks:
    x = blk(x, training=False)
backbone = tf.keras.Model(inp, net.output_conv(x))


@tf.function
def step(feat):
    return backbone(feat, training=False)


with tf.device("/GPU:0"):
    feat = tf.constant(np.random.default_rng(0).standard_normal((1, T, F, C)).astype(np.float32))
for _ in range(N_WARM):
    step(feat); tf.test.experimental.sync_devices()
apps = compute_apps()
if apps != [str(os.getpid())]:
    sys.exit(f"L-G3 STOP: the card holds {apps}, expected only this process {os.getpid()}")


def summ(v):
    v = sorted(v)
    return dict(median=statistics.median(v), p10=v[int(0.1 * len(v))], p90=v[int(0.9 * len(v))], n=len(v))


res = dict(what="R3 conv backbone, TensorFlow FP32, batch 1, 51 PRB, tf.function, input resident on GPU", pass_mode=A.mode,
           warmup=N_WARM, measured=N_MEAS, budget_ms=0.5, LG3_apps=apps,
           gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
           tf=tf.__version__, python=sys.executable, weights="results/r3_mirror/final/weights.h5",
           env={k: os.environ.get(k) for k in ("TF_USE_LEGACY_KERAS", "TF_CUDNN_DETERMINISTIC", "CUDA_VISIBLE_DEVICES")})

if A.mode == "wall":
    wall = []
    for _ in range(N_MEAS):
        t0 = time.perf_counter()
        step(feat); tf.test.experimental.sync_devices()
        wall.append((time.perf_counter() - t0) * 1e3)
    res["wall_synced_ms"] = summ(wall)
    res["value_ms"] = res["wall_synced_ms"]["median"]
else:
    from tensorflow.python.profiler.internal import _pywrap_profiler as pw
    logdir = os.path.join(OUT, "profile")
    tf.profiler.experimental.start(logdir)
    for i in range(N_MEAS):
        with tf.profiler.experimental.Trace("runL_step", step_num=i, _r=1):
            step(feat); tf.test.experimental.sync_devices()
    tf.profiler.experimental.stop()
    xp = sorted(glob.glob(os.path.join(logdir, "plugins", "profile", "*", "*.xplane.pb")))
    if not xp:
        sys.exit("no xplane.pb written")
    # "trace_viewer@" returns Chrome-trace JSON (ts/dur in microseconds, checked against
    # perf_counter); plain "trace_viewer" returns a protobuf.
    data, ok = pw.xspace_to_tools_data([xp[-1]], "trace_viewer@", {})
    if not ok:
        sys.exit("xspace_to_tools_data failed")
    ev = json.loads(data)["traceEvents"]
    pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    tname = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    gpu_pids = {p for p, n in pname.items() if "/device:GPU:0" in n}
    kern = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                  if e.get("ph") == "X" and e.get("pid") in gpu_pids
                  and tname.get((e["pid"], e["tid"]), "").startswith("Stream")
                  and not any(k in e.get("name", "") for k in ("Memcpy", "Memset", "memcpy", "memset")))
    steps = sorted(e["ts"] for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("runL_step") and e.get("pid") not in gpu_pids)
    if not kern or len(steps) != N_MEAS:
        sys.exit(f"trace parse failed: {len(kern)} kernels, {len(steps)} step markers (expected {N_MEAS})")
    bounds = steps + [float("inf")]
    dev, ksum, nk = [], [], []
    for i in range(N_MEAS):
        ks = [k for k in kern if bounds[i] <= k[0] < bounds[i + 1]]
        if ks:
            dev.append((max(b for _, b in ks) - min(a for a, _ in ks)) / 1e3)      # us -> ms
            ksum.append(sum(b - a for a, b in ks) / 1e3)
            nk.append(len(ks))
    if len(dev) < 200:
        sys.exit(f"only {len(dev)} slots carry GPU kernels (need >= 200)")
    res.update(kernel_window_ms=summ(dev), kernel_sum_ms=summ(ksum), kernels_per_slot=summ(nk),
               value_ms=statistics.median(dev), xplane=xp[-1])
res["x_budget"] = res["value_ms"] / 0.5
json.dump(res, open(os.path.join(OUT, f"runL_{A.mode}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
