"""Latency vs batch (throughput view) for the FP16 engine — supports the
AI-RAN batched-inference framing. CUDA-event timing, per-slot = total/batch."""
import os, tempfile
import numpy as np, tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit, statistics
ONNX=os.path.join(tempfile.gettempdir(), "trt_work", "r3_backbone.onnx"); LOG=trt.Logger(trt.Logger.ERROR); T,F,C,Q=14,612,14,8
def build_fp16_dyn():
    b=trt.Builder(LOG); n=b.create_network(0); p=trt.OnnxParser(n,LOG); p.parse(open(ONNX,"rb").read())
    # rebuild with dynamic batch by overriding input dim0
    inp=n.get_input(0); inp.shape=[-1,T,F,C]
    cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<31); cfg.set_flag(trt.BuilderFlag.FP16)
    prof=b.create_optimization_profile(); prof.set_shape(inp.name,(1,T,F,C),(16,T,F,C),(64,T,F,C)); cfg.add_optimization_profile(prof)
    return trt.Runtime(LOG).deserialize_cuda_engine(b.build_serialized_network(n,cfg))
eng=build_fp16_dyn()
print(f"{'batch':>5} {'total(ms)':>10} {'/slot(ms)':>10} {'slots/s':>10}")
for B in (1,4,16,64):
    ctx=eng.create_execution_context(); ctx.set_input_shape("feat",(B,T,F,C))
    din=cuda.mem_alloc(B*T*F*C*4); dout=cuda.mem_alloc(B*T*F*Q*4)
    ctx.set_tensor_address("feat",int(din))
    on=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT][0]
    ctx.set_tensor_address(on,int(dout))
    cuda.memcpy_htod(din,np.random.randn(B,T,F,C).astype(np.float32))
    st=cuda.Stream(); e0=cuda.Event(); e1=cuda.Event()
    for _ in range(50): ctx.execute_async_v3(st.handle)
    st.synchronize(); ts=[]
    for _ in range(300):
        e0.record(st); ctx.execute_async_v3(st.handle); e1.record(st); e1.synchronize(); ts.append(e0.time_till(e1))
    m=statistics.median(ts); print(f"{B:5d} {m:10.3f} {m/B:10.4f} {B/(m/1e3):10.0f}")
