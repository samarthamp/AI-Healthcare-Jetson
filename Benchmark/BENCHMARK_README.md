# TinyResECG Jetson Inference Benchmark

This benchmark is for `model.py` + `tinyresecg_v3.pth`.

## Run

On the Jetson Orin Nano with JetPack 6.2 and PyTorch installed:

```bash
python3 benchmark_tinyresecg.py \
  --checkpoint tinyresecg_v3.pth \
  --device cuda \
  --batch-size 1 \
  --channels 2 \
  --samples 1280 \
  --warmup 100 \
  --iterations 1000 \
  --repeats 3 \
  --with-tegrastats
```

Outputs are written to `benchmark_results/`:

- `benchmark_report.md`: human-readable benchmark report
- `benchmark_report.json`: complete machine-readable report
- `layer_breakdown.csv`: per-layer hook results with params, shapes, MACs, and ops
- `tegrastats.log`: Jetson telemetry, only when `--with-tegrastats` is used

## Recommended Stable Jetson Setup

For reproducible numbers, set the power/performance mode before benchmarking:

```bash
sudo nvpmodel -q
sudo jetson_clocks
```

Record the active `nvpmodel` output in the report. The script also attempts to
capture `nvpmodel -q`, `jetson_clocks --show`, `/proc/device-tree/model`, and
`/etc/nv_tegra_release`.

## Input

The default benchmark input is a synthetic tensor shaped `(1, 2, 1280)`, matching
the model training sanity check and ECG dataset format. To benchmark a real ECG
window, pass:

```bash
python3 benchmark_tinyresecg.py --input-npy path/to/window.npy
```

Accepted `.npy` shapes are `(T, C)`, `(C, T)`, `(B, T, C)`, or `(B, C, T)`.

## Notes

- Latency uses CUDA events on GPU and `time.perf_counter()` on CPU.
- Operation counts are estimates from PyTorch forward hooks.
- Conv1d and Linear layers are reported as MACs and converted to FLOPs as
  `2 * MACs`; BatchNorm, activations, pooling, SE mean/scaling, and residual
  additions are reported as non-MAC ops.
- Use `--dtype float16` only if you intentionally want FP16 inference numbers.
- Use `--allow-missing-keys` only when loading a non-identical checkpoint is
  intentional.
