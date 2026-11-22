# bounded-neural-receiver

This repository is being prepared. It will hold the code, configurations, trained
weights, result files and an interactive page for a bounded neural receiver with an
exact classical fallback. Until the first tagged release, nothing here is final.

So far:

- `receiver/` - the link-level simulator, the classical receivers, the neural
  receivers (network definitions and evaluation; the training code is not
  published), the offline tools that work on stored LLRs, and the latency
  benchmarks (`receiver/latency/`).
- `scripts/` - figure generators.

Environments: everything runs in the locked `paper3` environment (its environment
file comes in a later step), except the four TensorRT scripts in `receiver/latency/`
(`trt_*.py`), which need TensorRT: they run in `paper3trt`, a clone of `paper3` with
TensorRT added (see `receiver/latency/README.md`).

Licence: MIT, see `LICENSE`.
