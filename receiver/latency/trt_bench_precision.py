"""Build FP16 + INT8 TensorRT engines from r3_backbone.onnx and benchmark with
CUDA events (batch=1). INT8 uses random calibration: latency is calibration-
independent (only accuracy would need real calibration data)."""
import numpy as np, tensorrt as trt
import pycuda.driver as cuda, pycuda.autoinit
import os, tempfile
ONNX=os.path.join(tempfile.gettempdir(), "trt_work", "r3_backbone.onnx"); SLOT_MS=0.5
LOG=trt.Logger(trt.Logger.WARNING)
print("TensorRT", trt.__version__)

def build(mode):
    builder=trt.Builder(LOG)
    net=builder.create_network(0)
    parser=trt.OnnxParser(net, LOG)
    with open(ONNX,"rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors): print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")
    cfg=builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<31)
    inp=net.get_input(0); shp=tuple(inp.shape)
    if mode=="fp16": cfg.set_flag(trt.BuilderFlag.FP16)
    if mode=="bf16": cfg.set_flag(trt.BuilderFlag.BF16)
    if mode=="int8":
        cfg.set_flag(trt.BuilderFlag.INT8); cfg.set_flag(trt.BuilderFlag.FP16)
        class Calib(trt.IInt8EntropyCalibrator2):
            def __init__(s):
                trt.IInt8EntropyCalibrator2.__init__(s); s.i=0; s.n=30
                s.dev=cuda.mem_alloc(int(np.prod(shp))*4)
            def get_batch_size(s): return shp[0]
            def get_batch(s,names):
                if s.i>=s.n: return None
                cuda.memcpy_htod(s.dev, np.random.randn(*shp).astype(np.float32)); s.i+=1
                return [int(s.dev)]
            def read_calibration_cache(s): return None
            def write_calibration_cache(s,c): pass
        cfg.int8_calibrator=Calib()
    ser=builder.build_serialized_network(net, cfg)
    if ser is None: raise SystemExit(f"{mode} build failed")
    eng=trt.Runtime(LOG).deserialize_cuda_engine(ser)
    return eng, shp

def bench(eng, shp, n=1000, w=100):
    ctx=eng.create_execution_context()
    names=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    bufs={}; out_name=None; in_name=None
    for nm in names:
        s=tuple(ctx.get_tensor_shape(nm)); nbytes=int(np.prod(s))*4
        d=cuda.mem_alloc(nbytes); bufs[nm]=d; ctx.set_tensor_address(nm, int(d))
        if eng.get_tensor_mode(nm)==trt.TensorIOMode.INPUT: in_name=nm
        else: out_name=nm
    cuda.memcpy_htod(bufs[in_name], np.random.randn(*shp).astype(np.float32))
    stream=cuda.Stream(); e0=cuda.Event(); e1=cuda.Event()
    for _ in range(w): ctx.execute_async_v3(stream.handle)
    stream.synchronize()
    ts=[]
    for _ in range(n):
        e0.record(stream); ctx.execute_async_v3(stream.handle); e1.record(stream)
        e1.synchronize(); ts.append(e1.time_till(e0)*-1 if False else e0.time_till(e1))
    ts.sort(); import statistics
    return statistics.median(ts), ts[int(0.9*len(ts))], min(ts)

print(f"\n{'precision':10s} {'median(ms)':>11s} {'p90':>7s} {'min':>7s} {'xbudget':>8s}")
for mode in ("fp32","fp16","bf16","int8"):
    eng,shp=build(mode); p50,p90,mn=bench(eng,shp)
    print(f"{mode:10s} {p50:11.4f} {p90:7.4f} {mn:7.4f} {p50/SLOT_MS:7.2f}x")
