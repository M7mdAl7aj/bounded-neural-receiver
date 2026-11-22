import os, sys, time, statistics, math, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import r1_baseline
from r1_baseline import SimConfig, MCS_TABLE, ebnodb2no
from r3_deeprx import NeuralReceiver
from r5_simo_inference import R5InferenceSim

cfg = SimConfig(channel_model="C", mcs_idx=27, dmrs_n_add=1, ue_speed_kmh=3.0,
                n_prb=51, carrier_hz=7.125e9, scs_hz=30e3, n_tx_ant=1, n_rx_ant=2,
                estimator="ls", pn_tx="on", pn_rx="on", pn_ue_offset_db=3.0,
                iqi="spec", iqi_irr_db=25.0, real_iqi_comp=True)
qm = MCS_TABLE[27][1]
r3 = NeuralReceiver(num_rx_ant=2, num_conv_channels=128, num_blocks=11,
                    num_bits_per_symbol=qm, use_mirror_feature=True)
sim = R5InferenceSim(cfg, r3, num_conv_channels=128, num_blocks=11)
_ = sim.forward(1, tf.constant(20.0))          # materialize
r3.load_weights("results/r3_mirror/final/weights.h5")
inner = sim._inner; pd = inner.pdsch
B=1; F=cfg.fft_size; T=cfg.n_symbols
print(f"grid F={F} T={T} Qm={qm}  (51 PRB, SIMO 1x2, MCS27, PN+IQI IRR25)")

# ---- generate ONE fixed received grid y (TX+impaired channel), batch=1 ----
def gen_y(ebno=20.0):
    b = pd.source([B,1,1,pd.tb_size]); c = pd.tb_enc(b); x = pd.mapper(c); x_rg = pd.rg_mapper(x)
    no = ebnodb2no(ebno_db=ebno, num_bits_per_symbol=pd.q_m, coderate=pd.code_rate, resource_grid=pd.rg)
    if pd._td_active:
        x_t = tf.reshape(tf.signal.ifft(x_rg),[B,1,1,pd._slot_len])
        if pd.pn_tx is not None: x_t = x_t*tf.exp(tf.complex(0.0, pd.pn_tx((B,1,1))))
        x_rg = tf.signal.fft(tf.reshape(x_t,[B,1,1,T,F]))
    y,_ = pd.ofdm_channel(x_rg, no)
    if pd._td_active:
        y_t = tf.reshape(tf.signal.ifft(y),[B,1,cfg.n_rx_ant,pd._slot_len])
        if pd.pn_rx is not None: y_t = y_t*tf.exp(tf.complex(0.0, pd.pn_rx((B,1,cfg.n_rx_ant))))
        if pd.iqi is not None: y_t = pd.iqi(y_t)
        y = tf.signal.fft(tf.reshape(y_t,[B,1,cfg.n_rx_ant,T,F]))
    return y, no
y, no = gen_y()
pm = inner._build_pilot_mask(); no_b = tf.broadcast_to(tf.reshape(no,[-1]),[B]); DMAX=math.log(4.0)

@tf.function(reduce_retracing=True)
def R1_L1(y,no):
    h,ev = pd.chan_est(y,no); xh,ne = pd.equalizer(y,h,ev,no); return pd.demapper(xh,ne)
@tf.function(reduce_retracing=True)
def R1_L2(y,no):
    yl = sim._apply_l2_real(y,no); h,ev = pd.chan_est(yl,no); xh,ne = pd.equalizer(yl,h,ev,no); return pd.demapper(xh,ne)
@tf.function(reduce_retracing=True)
def R3_(y,no,pm,no_b):
    h,_ = inner.ls_est(y,no); h=tf.squeeze(h,[3,4]); return inner._grid_to_seq(sim.r3_net((y,h,pm,no_b),training=False))
@tf.function(reduce_retracing=True)
def R5_(y,no,pm,no_b):
    yl = sim._apply_l2_real(y,no); h2,ev2 = pd.chan_est(yl,no); xh2,ne2 = pd.equalizer(yl,h2,ev2,no)
    l2 = tf.reshape(pd.demapper(xh2,ne2),[B,1,1,-1])
    h3,_ = inner.ls_est(y,no); h3=tf.squeeze(h3,[3,4]); r3s = inner._grid_to_seq(sim.r3_net((y,h3,pm,no_b),training=False))
    return tf.clip_by_value(r3s, l2-DMAX, l2+DMAX)

def bench(fn, args, n=200, w=30):
    for _ in range(w): _=fn(*args).numpy()
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); _=fn(*args).numpy(); ts.append((time.perf_counter()-t0)*1e3)
    ts.sort(); return statistics.median(ts), ts[int(0.9*len(ts))], min(ts)
rows=[("R1-L1 (classical, no IQI comp)", R1_L1,(y,no)),
      ("R1-L2-real (classical + IQI inv.)", R1_L2,(y,no)),
      ("R3 (neural, LS+net)", R3_,(y,no,pm,no_b)),
      ("R5 (clip(R3, L2±ln4))", R5_,(y,no,pm,no_b))]
print(f"\n{'receiver':36s} {'median(ms)':>11s} {'p90':>7s} {'min':>7s} {'x budget':>9s}")
for name,fn,a in rows:
    p50,p90,mn = bench(fn,a)
    print(f"{name:36s} {p50:11.3f} {p90:7.3f} {mn:7.3f} {p50/0.5:8.1f}x")
