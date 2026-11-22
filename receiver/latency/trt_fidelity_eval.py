"""Paired fidelity: fp32-TF R3 vs a TensorRT engine (mode = fp16|fp32|mixed).
mixed = FP16 + last-K layers (output conv + final block) forced FP32 (OBEY)."""
import os, sys, math, tempfile, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
HP=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX=os.path.join(tempfile.gettempdir(), "trt_work", "r3_backbone.onnx")
sys.path.insert(0, HP); os.chdir(HP)
import tensorflow as tf, statistics, time
from r1_baseline import SimConfig, MCS_TABLE, ebnodb2no
from r3_deeprx import NeuralReceiver
from r5_simo_inference import R5InferenceSim
import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit
N=int(sys.argv[1]) if len(sys.argv)>1 else 300
MODE=sys.argv[2] if len(sys.argv)>2 else "fp16"
KLAST=int(sys.argv[3]) if len(sys.argv)>3 else 8

cfg = SimConfig(channel_model="C", mcs_idx=27, dmrs_n_add=1, ue_speed_kmh=3.0, n_prb=51,
                carrier_hz=7.125e9, scs_hz=30e3, n_tx_ant=1, n_rx_ant=2, estimator="ls",
                pn_tx="on", pn_rx="on", pn_ue_offset_db=3.0, iqi="spec", iqi_irr_db=25.0, real_iqi_comp=True)
qm=MCS_TABLE[27][1]; F=cfg.fft_size; T=cfg.n_symbols; ANT=2
net=NeuralReceiver(num_rx_ant=ANT,num_conv_channels=128,num_blocks=11,num_bits_per_symbol=qm,use_mirror_feature=True)
sim=R5InferenceSim(cfg,net,num_conv_channels=128,num_blocks=11); _=sim.forward(1,tf.constant(20.0))
net.load_weights("results/r3_mirror/final/weights.h5"); inner=sim._inner; pd=inner.pdsch

LOG=trt.Logger(trt.Logger.ERROR); B=1
bld=trt.Builder(LOG); netw=bld.create_network(0); pr=trt.OnnxParser(netw,LOG)
pr.parse(open(ONNX,"rb").read())
c=bld.create_builder_config(); c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<31)
if MODE in ("fp16","mixed"): c.set_flag(trt.BuilderFlag.FP16)
if MODE=="mixed":
    c.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    nl=netw.num_layers; forced=0
    for i in range(nl):
        L=netw.get_layer(i)
        nm=L.name.lower()
        is_norm = ("norm" in nm) or ("reduce" in nm) or (L.type==trt.LayerType.NORMALIZATION)
        if i>=nl-KLAST or is_norm:
            try:
                L.precision=trt.float32
                for o in range(L.num_outputs): L.set_output_type(o, trt.float32)
                forced+=1
            except Exception: pass
    print(f"mixed: forced {forced}/{nl} layers to FP32")
eng=trt.Runtime(LOG).deserialize_cuda_engine(bld.build_serialized_network(netw,c)); ctx=eng.create_execution_context()
ins=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
inn=[n for n in ins if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT][0]
outn=[n for n in ins if eng.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT][0]
d_in=cuda.mem_alloc(int(np.prod((1,T,F,14)))*4); d_out=cuda.mem_alloc(int(np.prod((1,T,F,qm)))*4)
ctx.set_tensor_address(inn,int(d_in)); ctx.set_tensor_address(outn,int(d_out)); stream=cuda.Stream()
def trt_run(z):
    z=np.ascontiguousarray(z,dtype=np.float32); out=np.empty((1,T,F,qm),np.float32)
    cuda.memcpy_htod_async(d_in,z,stream); ctx.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out,d_out,stream); stream.synchronize(); return out
# latency
e0=cuda.Event(); e1=cuda.Event(); zr=np.random.randn(1,T,F,14).astype(np.float32); cuda.memcpy_htod(d_in,zr)
for _ in range(50): ctx.execute_async_v3(stream.handle)
stream.synchronize(); lt=[]
for _ in range(500): e0.record(stream); ctx.execute_async_v3(stream.handle); e1.record(stream); e1.synchronize(); lt.append(e0.time_till(e1))
print(f"[{MODE}] latency median={statistics.median(lt):.4f} ms  ({statistics.median(lt)/0.5:.2f}x budget)")

def assemble(y,h,no_b,pm):
    y=tf.squeeze(y,1); h=tf.squeeze(h,1)
    ym=tf.math.conj(tf.roll(tf.reverse(y,axis=[-1]),shift=1,axis=-1))
    y=tf.transpose(y,[0,2,3,1]); h=tf.transpose(h,[0,2,3,1]); ym=tf.transpose(ym,[0,2,3,1])
    nol=tf.broadcast_to(tf.reshape(tf.math.log(no_b)/tf.math.log(10.0),[-1,1,1,1]),[1,T,F,1])
    pmb=tf.broadcast_to(tf.reshape(pm,[1,T,F,1]),[1,T,F,1])
    return tf.concat([tf.math.real(y),tf.math.imag(y),tf.math.real(h),tf.math.imag(h),
                      tf.math.real(ym),tf.math.imag(ym),pmb,nol],axis=-1)
def gen(ebno):
    b=pd.source([B,1,1,pd.tb_size]); cc=pd.tb_enc(b); x=pd.mapper(cc); xr=pd.rg_mapper(x)
    no=ebnodb2no(ebno_db=ebno,num_bits_per_symbol=pd.q_m,coderate=pd.code_rate,resource_grid=pd.rg)
    if pd._td_active:
        xt=tf.reshape(tf.signal.ifft(xr),[B,1,1,pd._slot_len])
        if pd.pn_tx is not None: xt=xt*tf.exp(tf.complex(0.0,pd.pn_tx((B,1,1))))
        xr=tf.signal.fft(tf.reshape(xt,[B,1,1,T,F]))
    y,_=pd.ofdm_channel(xr,no)
    if pd._td_active:
        yt=tf.reshape(tf.signal.ifft(y),[B,1,ANT,pd._slot_len])
        if pd.pn_rx is not None: yt=yt*tf.exp(tf.complex(0.0,pd.pn_rx((B,1,ANT))))
        if pd.iqi is not None: yt=pd.iqi(yt)
        y=tf.signal.fft(tf.reshape(yt,[B,1,ANT,T,F]))
    return b,y,no
pm=inner._build_pilot_mask()
def decode(grid): return pd.tb_dec(inner._grid_to_seq(tf.convert_to_tensor(grid)))[0]
SNRS=[17.,18.,19.,20.,21.]; res={}; md=0.0
print(f"N={N}/SNR  mode={MODE}\n{'SNR':>5}{'fp32':>10}{'TRT':>10}")
for s in SNRS:
    tf.random.set_seed(int(s*1000)); e32=etr=0
    for _ in range(N):
        b,y,no=gen(s); h,_=inner.ls_est(y,no); h=tf.squeeze(h,[3,4]); nob=tf.broadcast_to(tf.reshape(no,[-1]),[B])
        g32=net((y,h,pm,nob),training=False); gtr=trt_run(assemble(y,h,nob,pm).numpy())
        md=max(md,np.abs(g32.numpy()-gtr).max())
        bf=tf.reshape(b,[B,-1])
        e32+=int(tf.reduce_any(tf.not_equal(bf,tf.reshape(decode(g32),[B,-1]))))
        etr+=int(tf.reduce_any(tf.not_equal(bf,tf.reshape(decode(gtr),[B,-1]))))
    res[s]=(e32/N,etr/N); print(f"{s:5.0f}{e32/N:10.4f}{etr/N:10.4f}")
def cross(d):
    xs=sorted(d)
    for i in range(len(xs)-1):
        a,b=xs[i],xs[i+1]
        if d[a]>=0.1>=d[b]: return a+(d[a]-0.1)/(d[a]-d[b])*(b-a)
    return None
print(f"\n10%-BLER fp32={cross({s:res[s][0] for s in SNRS})}  {MODE}-TRT={cross({s:res[s][1] for s in SNRS})}  LLRmaxdiff={md:.3f}")
