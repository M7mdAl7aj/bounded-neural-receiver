"""Export R3's real-valued conv backbone (input [B,14,612,14] -> LLR [B,14,612,8])
to ONNX. The complex mirror/IQ feature-assembly is elementwise (us-class) and is
kept outside the engine; the conv backbone is the ~5.6e10-FLOP compute path."""
import os, sys, tempfile
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
HP=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX=os.path.join(tempfile.gettempdir(), "trt_work", "r3_backbone.onnx")
sys.path.insert(0, HP); os.chdir(HP)
import numpy as np, tensorflow as tf
from r1_baseline import SimConfig, MCS_TABLE
from r3_deeprx import NeuralReceiver
import tf2onnx, onnx

cfg = SimConfig(n_prb=51, mcs_idx=27, n_tx_ant=1, n_rx_ant=2)
F=cfg.fft_size; T=cfg.n_symbols; qm=MCS_TABLE[27][1]
nrx=2; C = 2*(2*nrx)+2 + 2*nrx   # 14 input channels (SIMO 1x2 + mirror)
net = NeuralReceiver(num_rx_ant=nrx, num_conv_channels=128, num_blocks=11,
                     num_bits_per_symbol=qm, use_mirror_feature=True)
# materialize with a dummy complex forward, then load trained weights
yb=tf.complex(tf.zeros([1,1,nrx,T,F]),tf.zeros([1,1,nrx,T,F]))
pm=tf.zeros([T,F]); no=tf.ones([1])
_=net((yb,yb,pm,no))
net.load_weights("results/r3_mirror/final/weights.h5")
print("loaded R3 weights; params", sum(int(np.prod(v.shape)) for v in net.trainable_variables))

# functional backbone: real feature stack -> LLRs (reuses trained layers)
inp = tf.keras.Input(shape=(T, F, C), batch_size=1, name="feat", dtype=tf.float32)
x = net.input_conv(inp)
for blk in net.blocks: x = blk(x, training=False)
out = net.output_conv(x)
backbone = tf.keras.Model(inp, out, name="r3_backbone")
print("backbone in", backbone.input_shape, "out", backbone.output_shape)

os.makedirs(os.path.dirname(ONNX), exist_ok=True)
spec = (tf.TensorSpec((1,T,F,C), tf.float32, name="feat"),)
mp, _ = tf2onnx.convert.from_keras(backbone, input_signature=spec, opset=17,
                                   output_path=ONNX)
m = onnx.load(ONNX); onnx.checker.check_model(m)
print("ONNX OK ->", ONNX,
      "| inputs", [i.name for i in m.graph.input], "| opset", m.opset_import[0].version)
