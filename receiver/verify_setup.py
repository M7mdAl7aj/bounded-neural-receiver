"""
verify_setup.py — run this first on your GPU to confirm the environment
is good before running the simulations.

Expected output if all OK:
  Python: 3.10.x
  TensorFlow: 2.16.x
  GPUs: 1
  10x 4096x4096 matmul: 150-250 ms
  Sionna: 1.x
  All Sionna imports OK
  === Setup verified ===
"""
import sys
print(f"Python: {sys.version.split()[0]}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
assert tf.__version__.startswith("2.16"), "Need TF 2.16.x (you have {})".format(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")
assert len(gpus) >= 1, "No GPU visible to TensorFlow"

# Enable memory growth
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass

# Quick compute test
import time
with tf.device('/GPU:0'):
    x = tf.random.normal([4096, 4096])
    y = tf.random.normal([4096, 4096])
    # Warmup
    _ = tf.matmul(x, y).numpy()
    t0 = time.time()
    for _ in range(10):
        z = tf.matmul(x, y)
    _ = z.numpy()
    dt = time.time() - t0
print(f"10x 4096x4096 matmul: {dt*1000:.1f} ms (expected 150-250 ms on RTX 6000 Ada)")

import sionna
print(f"Sionna: {sionna.__version__}")
assert sionna.__version__.startswith("1."), "Need Sionna 1.x (you have {})".format(sionna.__version__)

# Import all Sionna modules we'll use
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import OFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
print("All Sionna imports OK")

print("\n=== Setup verified ===")
