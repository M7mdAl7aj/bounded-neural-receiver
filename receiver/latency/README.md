# Deployment-latency benchmark (R3/R5 neural detector)

Measures the **GPU compute latency** of the R3 conv backbone (the dominant
neural path; R5 inherits it) with deployment-grade optimization, and verifies
the optimized engine **reproduces the fp32 BLER** (the neural gain is preserved).

## Why a separate environment
TensorRT is **not** in the locked `paper3` stack (which is pinned for the BLER
campaign). All of this runs in a **clone**, `paper3trt`, so the locked env is
untouched:

```
conda create -y -n paper3trt --clone paper3
conda activate paper3trt
pip install "tensorrt-cu12<11" tf2onnx onnx pycuda
```

(`tensorrt-cu12<11` = TRT 10.x, which keeps the FP16/INT8 builder-flag +
calibrator API; TRT 11 is strongly-typed-only. cu12 matches the CUDA-12 / TF-2.16
ecosystem.)

## Pipeline
1. `trt_export_onnx.py` — load trained R3 (`results/r3_mirror/final`), wrap the
   real-valued conv backbone (input_conv → 11 ResBlocks → output_conv;
   input `[1,14,612,14]` → LLR `[1,14,612,8]`), export → `r3_backbone.onnx`.
   The complex mirror/IQ feature-assembly is µs-class elementwise and is kept
   outside the engine.
2. `trt_bench_precision.py` — build FP32/FP16/BF16/INT8 engines, CUDA-event
   latency at batch=1.
3. `trt_fidelity_eval.py N MODE KLAST` — paired fp32-TF vs TRT(MODE) BLER around
   the waterfall + LLR diff. `MODE=mixed` forces the last `KLAST` layers and all
   norm/reduce ops to FP32 (convs stay FP16).
4. `trt_bench_batchsweep.py` — latency vs batch (throughput view).
5. `tf_latency_overhead_probe.py` — shows the naive TF batch=1 numbers are
   framework-overhead-bound (classical receivers especially), i.e. why a TF
   wall-clock cross-receiver table is invalid and the FLOP comparison is the
   honest one.
6. `tf_backbone_latency_runL.py --pass {kernel,wall} OUT_DIR` — the TensorFlow
   FP32 row of Table II (profiler kernel window, and synchronized wall clock).

## Run
From `receiver/`, in the activated `paper3trt` env (the ONNX export is written to
`trt_work/r3_backbone.onnx` under the system temporary directory):
```
python latency/trt_export_onnx.py
python latency/trt_bench_precision.py
python latency/trt_fidelity_eval.py 500 mixed 8
python latency/trt_bench_batchsweep.py
```

## Scope / caveats
- GPU **compute** (kernel) latency via CUDA events — excludes host↔device I/O,
  LDPC decode, and the µs-class complex feature-prep.
- **RTX 6000 Ada is a 300 W datacenter GPU, not UE silicon** — this establishes
  feasibility on AI-RAN-class hardware, not on a handset. The UE-silicon gap
  stays analytic.
- The measured values are reported in Table II of the paper.
