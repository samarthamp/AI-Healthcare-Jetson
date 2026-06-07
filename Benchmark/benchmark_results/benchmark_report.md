# TinyResECG Inference Benchmark

## Configuration

- Checkpoint: `tinyresecg_v3.pth`
- Device: `cuda`
- Input shape: `(1, 2, 1280)`
- dtype: `float32`
- Warmup iterations: `100`
- Timed iterations per repeat: `1000`
- Repeats: `3`
- cuDNN benchmark: `False`

## Platform

- Jetson model: `NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super`
- JetPack/L4T: `# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia`
- Python: `3.10.20 (main, Mar 11 2026, 17:41:27) [GCC 14.3.0]`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA available: `True`
- CUDA device: `Orin`
- CUDA runtime: `12.6`
- cuDNN: `90300`

## Model Size

- Total parameters: **49,762**
- Trainable parameters: **49,762**
- Parameter memory fp32: **0.1898 MiB**
- Checkpoint format: `raw_state_dict`
- Missing keys: `[]`
- Unexpected keys: `[]`
- Ignored profiling keys: `26`

## Operation Estimate

- MACs per forward pass: **19,663,120**
- Non-MAC ops per forward pass: **729,762**
- Estimated FLOPs per forward pass: **40,056,002**
- Layer breakdown: `layer_breakdown.csv`

Operation estimates come from PyTorch forward hooks. Conv1d and Linear are
reported as MACs and converted to FLOPs as `2 * MACs`; BatchNorm, activation,
pooling, SE averaging/scaling, and residual adds are counted as non-MAC ops.

## Latency

- Mean: **6.988 ms**
- Median: **6.254 ms**
- Min / Max: **0.855 / 25.238 ms**
- Std dev: **2.580 ms**
- p90 / p95 / p99: **10.366 / 10.835 / 12.543 ms**
- Throughput: **143.10 samples/s**

## Memory

- CUDA memory allocated after benchmark: `8.353 MiB`
- CUDA max memory allocated: `9.3535 MiB`
- CUDA max memory reserved: `22.0 MiB`

## Jetson Power/Clock Notes

For stable Jetson numbers, record the active `nvpmodel` mode and whether
`jetson_clocks` was enabled before running the benchmark.

- nvpmodel output captured: `True`
- jetson_clocks output captured: `True`
- tegrastats log: `tegrastats.log`
