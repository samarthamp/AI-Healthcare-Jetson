#!/usr/bin/env python3
"""
Inference benchmark for TinyResECG on Jetson Orin Nano / JetPack.

The script loads model.py + a checkpoint, runs warmup and timed forward passes,
uses hooks to collect per-layer tensor shapes and operation estimates, and writes
JSON, CSV, and Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model import ResBlock, SEGate, TinyResECG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TinyResECG inference and generate a report."
    )
    parser.add_argument("--checkpoint", default="tinyresecg_v3.pth")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1280)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--input-npy", default=None, help="Optional ECG sample .npy")
    parser.add_argument("--no-load-checkpoint", action="store_true")
    parser.add_argument("--allow-missing-keys", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument(
        "--with-tegrastats",
        action="store_true",
        help="Run tegrastats while timing if available.",
    )
    return parser.parse_args()


def run_text_command(command: List[str], timeout: float = 2.0) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip()
    return text or None


def read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(errors="replace").strip("\x00\n ")
    except OSError:
        return None


def collect_system_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": None,
        "device_requested": str(device),
        "jetson_model": read_text("/proc/device-tree/model"),
        "nv_tegra_release": read_text("/etc/nv_tegra_release"),
        "nvpmodel": run_text_command(["nvpmodel", "-q"]),
        "jetson_clocks": run_text_command(["jetson_clocks", "--show"]),
    }
    if device.type == "cuda":
        info["cuda_available"] = torch.cuda.is_available()
        info.update(
            {
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_capability": torch.cuda.get_device_capability(device),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return info


def load_checkpoint(model: nn.Module, checkpoint_path: Path, strict: bool) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata: Dict[str, Any] = {"checkpoint_path": str(checkpoint_path)}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata["checkpoint_format"] = "dict[state_dict]"
        metadata["checkpoint_keys"] = sorted(str(k) for k in checkpoint.keys())
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata["checkpoint_format"] = "dict[model_state_dict]"
        metadata["checkpoint_keys"] = sorted(str(k) for k in checkpoint.keys())
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata["checkpoint_format"] = "raw_state_dict"
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    cleaned = {}
    ignored_profile_keys = []
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        if clean_key in {"total_ops", "total_params"} or clean_key.endswith(
            (".total_ops", ".total_params")
        ):
            ignored_profile_keys.append(clean_key)
            continue
        cleaned[clean_key] = value
    result = model.load_state_dict(cleaned, strict=strict)
    metadata["missing_keys"] = list(result.missing_keys)
    metadata["unexpected_keys"] = list(result.unexpected_keys)
    metadata["ignored_profile_keys"] = ignored_profile_keys
    return metadata


def make_input(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(args.seed)
    if args.input_npy:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("--input-npy requires numpy") from exc
        array = np.load(args.input_npy)
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 2 and tensor.shape == (args.samples, args.channels):
            tensor = tensor.transpose(0, 1).unsqueeze(0)
        elif tensor.ndim == 2 and tensor.shape == (args.channels, args.samples):
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-2:] == (args.samples, args.channels):
            tensor = tensor.transpose(1, 2)
        elif tensor.ndim != 3:
            raise ValueError(
                "Input .npy must be shaped (T,C), (C,T), (B,T,C), or (B,C,T)."
            )
        if tensor.shape[0] != args.batch_size:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(args.batch_size, 1, 1)
            else:
                raise ValueError(
                    f"Input batch {tensor.shape[0]} does not match --batch-size {args.batch_size}."
                )
    else:
        tensor = torch.randn(args.batch_size, args.channels, args.samples)
    return tensor.to(device=device, dtype=dtype)


def tensor_shape(value: Any) -> Optional[List[int]]:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = tensor_shape(item)
            if shape is not None:
                return shape
    return None


def count_params(module: nn.Module, recurse: bool = False) -> int:
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def conv1d_ops(module: nn.Conv1d, output: torch.Tensor) -> Tuple[int, int]:
    batch, out_channels, out_len = output.shape
    kernel_mul = module.in_channels // module.groups * module.kernel_size[0]
    macs = batch * out_channels * out_len * kernel_mul
    non_mac_ops = 0
    if module.bias is not None:
        non_mac_ops += batch * out_channels * out_len
    return macs, non_mac_ops


def linear_ops(module: nn.Linear, output: torch.Tensor) -> Tuple[int, int]:
    out_elements = output.numel()
    macs = out_elements * module.in_features
    non_mac_ops = 0
    if module.bias is not None:
        non_mac_ops += out_elements
    return macs, non_mac_ops


def batchnorm_ops(output: torch.Tensor) -> Tuple[int, int]:
    elements = output.numel()
    return 0, 2 * elements


def activation_ops(output: torch.Tensor) -> Tuple[int, int]:
    return 0, output.numel()


def pool_ops(module: nn.Module, input_shape: List[int], output: torch.Tensor) -> Tuple[int, int]:
    if isinstance(module, nn.AdaptiveAvgPool1d):
        in_len = input_shape[-1]
        out_len = output.shape[-1]
        per_output = max(math.ceil(in_len / out_len), 1)
        return 0, output.numel() * per_output
    kernel = module.kernel_size
    if isinstance(kernel, tuple):
        kernel = kernel[0]
    return 0, output.numel() * int(kernel)


def collect_layer_stats(model: nn.Module, sample: torch.Tensor) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    handles = []
    module_names = {module: name for name, module in model.named_modules()}

    def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        out_shape = tensor_shape(output)
        in_shape = tensor_shape(inputs)
        if out_shape is None:
            return
        output_tensor = output if torch.is_tensor(output) else output[0]
        macs = 0
        ops = 0
        note = ""
        if isinstance(module, nn.Conv1d):
            macs, ops = conv1d_ops(module, output_tensor)
        elif isinstance(module, nn.Linear):
            macs, ops = linear_ops(module, output_tensor)
        elif isinstance(module, nn.BatchNorm1d):
            macs, ops = batchnorm_ops(output_tensor)
        elif isinstance(module, (nn.ReLU, nn.Sigmoid)):
            macs, ops = activation_ops(output_tensor)
        elif isinstance(module, (nn.MaxPool1d, nn.AvgPool1d, nn.AdaptiveAvgPool1d)):
            macs, ops = pool_ops(module, in_shape or out_shape, output_tensor)
        elif isinstance(module, SEGate):
            batch = output_tensor.shape[0]
            channels = output_tensor.shape[1]
            timesteps = output_tensor.shape[2]
            hidden = module.fc1.out_features
            ops = batch * channels * timesteps
            ops += batch * hidden
            ops += batch * channels
            ops += output_tensor.numel()
            note = "SE mean, activation, sigmoid, and scaling only; Linear ops are counted separately."
        elif isinstance(module, ResBlock):
            ops = output_tensor.numel()
            note = "Residual add only; child module ops are counted separately."
        else:
            return
        rows.append(
            {
                "name": module_names.get(module, ""),
                "type": module.__class__.__name__,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "params": count_params(module, recurse=False),
                "macs": int(macs),
                "ops": int(ops),
                "note": note,
            }
        )

    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Conv1d,
                nn.Linear,
                nn.BatchNorm1d,
                nn.ReLU,
                nn.Sigmoid,
                nn.MaxPool1d,
                nn.AvgPool1d,
                nn.AdaptiveAvgPool1d,
                SEGate,
                ResBlock,
            ),
        ):
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    totals = {
        "macs": sum(row["macs"] for row in rows),
        "ops": sum(row["ops"] for row in rows),
        "estimated_flops": sum(row["macs"] * 2 + row["ops"] for row in rows),
    }
    return rows, totals


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def start_tegrastats(output_dir: Path) -> Tuple[Optional[subprocess.Popen], Optional[Path]]:
    path = output_dir / "tegrastats.log"
    try:
        handle = path.open("w")
        process = subprocess.Popen(
            ["tegrastats", "--interval", "100"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None, None
    process._codex_log_handle = handle  # type: ignore[attr-defined]
    return process, path


def stop_tegrastats(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
    log_handle = getattr(process, "_codex_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def benchmark_latency(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    all_latencies: List[float] = []
    repeat_summaries: List[Dict[str, float]] = []

    for _ in range(repeats):
        latencies: List[float] = []
        with torch.inference_mode():
            for _ in range(iterations):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(sample)
                    end.record()
                    torch.cuda.synchronize(device)
                    latency_ms = float(start.elapsed_time(end))
                else:
                    start_time = time.perf_counter()
                    model(sample)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                latencies.append(latency_ms)
        all_latencies.extend(latencies)
        repeat_summaries.append(summarize_latencies(latencies, sample.shape[0]))

    summary = summarize_latencies(all_latencies, sample.shape[0])
    summary["repeats"] = repeat_summaries
    summary["iterations"] = iterations
    summary["warmup"] = warmup
    return summary


def summarize_latencies(latencies: List[float], batch_size: int) -> Dict[str, float]:
    mean_ms = statistics.fmean(latencies)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        "p90_ms": percentile(latencies, 90),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "throughput_samples_per_s": batch_size * 1000.0 / mean_ms,
    }


def bytes_to_mib(value: int) -> float:
    return value / (1024.0**2)


def write_layer_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "type",
                "input_shape",
                "output_shape",
                "params",
                "macs",
                "ops",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_int(value: float) -> str:
    return f"{int(value):,}"


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    system: Dict[str, Any],
    checkpoint: Dict[str, Any],
    model_stats: Dict[str, Any],
    op_totals: Dict[str, int],
    latency: Dict[str, Any],
    memory: Dict[str, Any],
    layer_csv_name: str,
    tegrastats_path: Optional[Path],
) -> None:
    lines = [
        "# TinyResECG Inference Benchmark",
        "",
        "## Configuration",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Device: `{system.get('device_requested')}`",
        f"- Input shape: `({args.batch_size}, {args.channels}, {args.samples})`",
        f"- dtype: `{args.dtype}`",
        f"- Warmup iterations: `{args.warmup}`",
        f"- Timed iterations per repeat: `{args.iterations}`",
        f"- Repeats: `{args.repeats}`",
        f"- cuDNN benchmark: `{args.cudnn_benchmark}`",
        "",
        "## Platform",
        "",
        f"- Jetson model: `{system.get('jetson_model')}`",
        f"- JetPack/L4T: `{system.get('nv_tegra_release')}`",
        f"- Python: `{system.get('python')}`",
        f"- PyTorch: `{system.get('torch')}`",
        f"- CUDA available: `{system.get('cuda_available')}`",
        f"- CUDA device: `{system.get('cuda_device')}`",
        f"- CUDA runtime: `{system.get('cuda_runtime')}`",
        f"- cuDNN: `{system.get('cudnn')}`",
        "",
        "## Model Size",
        "",
        f"- Total parameters: **{fmt_int(model_stats['total_params'])}**",
        f"- Trainable parameters: **{fmt_int(model_stats['trainable_params'])}**",
        f"- Parameter memory fp32: **{fmt_float(model_stats['param_memory_mib'], 4)} MiB**",
        f"- Checkpoint format: `{checkpoint.get('checkpoint_format')}`",
        f"- Missing keys: `{checkpoint.get('missing_keys')}`",
        f"- Unexpected keys: `{checkpoint.get('unexpected_keys')}`",
        f"- Ignored profiling keys: `{len(checkpoint.get('ignored_profile_keys', []))}`",
        "",
        "## Operation Estimate",
        "",
        f"- MACs per forward pass: **{fmt_int(op_totals['macs'])}**",
        f"- Non-MAC ops per forward pass: **{fmt_int(op_totals['ops'])}**",
        f"- Estimated FLOPs per forward pass: **{fmt_int(op_totals['estimated_flops'])}**",
        f"- Layer breakdown: `{layer_csv_name}`",
        "",
        "Operation estimates come from PyTorch forward hooks. Conv1d and Linear are",
        "reported as MACs and converted to FLOPs as `2 * MACs`; BatchNorm, activation,",
        "pooling, SE averaging/scaling, and residual adds are counted as non-MAC ops.",
        "",
        "## Latency",
        "",
        f"- Mean: **{fmt_float(latency['mean_ms'])} ms**",
        f"- Median: **{fmt_float(latency['median_ms'])} ms**",
        f"- Min / Max: **{fmt_float(latency['min_ms'])} / {fmt_float(latency['max_ms'])} ms**",
        f"- Std dev: **{fmt_float(latency['std_ms'])} ms**",
        f"- p90 / p95 / p99: **{fmt_float(latency['p90_ms'])} / {fmt_float(latency['p95_ms'])} / {fmt_float(latency['p99_ms'])} ms**",
        f"- Throughput: **{fmt_float(latency['throughput_samples_per_s'], 2)} samples/s**",
        "",
        "## Memory",
        "",
        f"- CUDA memory allocated after benchmark: `{memory.get('allocated_mib')} MiB`",
        f"- CUDA max memory allocated: `{memory.get('max_allocated_mib')} MiB`",
        f"- CUDA max memory reserved: `{memory.get('max_reserved_mib')} MiB`",
        "",
        "## Jetson Power/Clock Notes",
        "",
        "For stable Jetson numbers, record the active `nvpmodel` mode and whether",
        "`jetson_clocks` was enabled before running the benchmark.",
        "",
        f"- nvpmodel output captured: `{system.get('nvpmodel') is not None}`",
        f"- jetson_clocks output captured: `{system.get('jetson_clocks') is not None}`",
        f"- tegrastats log: `{tegrastats_path.name if tegrastats_path else None}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model = TinyResECG(num_classes=args.num_classes, dropout=args.dropout)
    checkpoint_metadata: Dict[str, Any] = {"checkpoint_format": "not_loaded"}
    if not args.no_load_checkpoint:
        checkpoint_metadata = load_checkpoint(
            model, Path(args.checkpoint), strict=not args.allow_missing_keys
        )
    model.eval().to(device=device, dtype=dtype)

    sample = make_input(args, device, dtype)
    system = collect_system_info(device)
    model_stats = {
        "total_params": count_params(model, recurse=True),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    model_stats["param_memory_mib"] = model_stats["total_params"] * 4 / (1024.0**2)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    layer_rows, op_totals = collect_layer_stats(model, sample)

    tegrastats_process = None
    tegrastats_path = None
    if args.with_tegrastats:
        tegrastats_process, tegrastats_path = start_tegrastats(output_dir)
    try:
        latency = benchmark_latency(
            model=model,
            sample=sample,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            device=device,
        )
    finally:
        stop_tegrastats(tegrastats_process)

    with torch.inference_mode():
        output = model(sample)

    memory: Dict[str, Any] = {
        "allocated_mib": None,
        "max_allocated_mib": None,
        "max_reserved_mib": None,
    }
    if device.type == "cuda":
        memory = {
            "allocated_mib": round(bytes_to_mib(torch.cuda.memory_allocated(device)), 4),
            "max_allocated_mib": round(bytes_to_mib(torch.cuda.max_memory_allocated(device)), 4),
            "max_reserved_mib": round(bytes_to_mib(torch.cuda.max_memory_reserved(device)), 4),
        }

    if args.save_output:
        torch.save(output.detach().cpu(), output_dir / "forward_output.pt")

    layer_csv = output_dir / "layer_breakdown.csv"
    report_md = output_dir / "benchmark_report.md"
    report_json = output_dir / "benchmark_report.json"
    write_layer_csv(layer_csv, layer_rows)

    full_report = {
        "args": vars(args),
        "system": system,
        "checkpoint": checkpoint_metadata,
        "model": model_stats,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "operations": op_totals,
        "latency": latency,
        "memory": memory,
        "layer_breakdown": layer_rows,
        "tegrastats_log": str(tegrastats_path) if tegrastats_path else None,
    }
    report_json.write_text(json.dumps(full_report, indent=2))
    write_markdown_report(
        report_md,
        args,
        system,
        checkpoint_metadata,
        model_stats,
        op_totals,
        latency,
        memory,
        layer_csv.name,
        tegrastats_path,
    )

    print(f"Report written to: {report_md}")
    print(f"JSON written to  : {report_json}")
    print(f"Layers written to: {layer_csv}")
    print(
        "Summary: "
        f"params={model_stats['total_params']:,}, "
        f"MACs={op_totals['macs']:,}, "
        f"FLOPs~={op_totals['estimated_flops']:,}, "
        f"mean_latency={latency['mean_ms']:.3f} ms, "
        f"throughput={latency['throughput_samples_per_s']:.2f} samples/s"
    )


if __name__ == "__main__":
    main()
