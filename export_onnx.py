#!/usr/bin/env python3
"""
export_onnx.py - export MvANet to ONNX
======================================
Export trained model to ONNX format for edge deployment
(e.g., on embedded devices, Raspberry Pi, NVIDIA Jetson, etc.)

Usage:
    # Basic export
    python export_onnx.py --checkpoint runs/mvanet_3/run_01/best.pth

    # Custom output and input size
    python export_onnx.py --checkpoint best.pth --output mvanet.onnx --input-size 96

    # With simplification and dynamic batch
    python export_onnx.py --checkpoint best.pth --simplify --dynamic-batch

    # Verify exported model
    python export_onnx.py --checkpoint best.pth --verify
"""

import argparse
import os
import sys
import time
import warnings

import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    # Avoid Windows cp1252 crashes when torch.onnx prints Unicode status text.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export MvANet to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model (.pth)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override architecture (default: from checkpoint)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX file path")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Input resolution (default: from checkpoint)")
    parser.add_argument("--opset-version", type=int, default=18,
                        help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true",
                        dest="dynamic_batch", default=True,
                        help="Enable dynamic batch size (default)")
    parser.add_argument("--static-batch", action="store_false",
                        dest="dynamic_batch",
                        help="Export a fixed batch-size graph")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify ONNX model (requires onnx-simplifier)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX model output matches PyTorch")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark ONNX runtime inference speed")
    parser.add_argument("--fp16", action="store_true",
                        help="Convert to FP16 (half precision)")
    return parser.parse_args()


def export_onnx(model: nn.Module, output_path: str, input_size: int = 32,
                opset_version: int = 18, dynamic_batch: bool = True,
                in_channels: int = 3):
    """Export PyTorch model to ONNX format."""
    model.eval()
    device = next(model.parameters()).device

    # Dummy input (batch=5 as in paper: 5 images per jujube)
    dummy_input = torch.randn(5, in_channels, input_size, input_size).to(device)

    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
    except ModuleNotFoundError as exc:
        missing = exc.name or "an ONNX dependency"
        raise RuntimeError(
            f"ONNX export requires optional package '{missing}'. "
            "Install optional dependencies with: "
            "pip install -r requirements-optional.txt"
        ) from exc

    file_size = os.path.getsize(output_path) / 1024
    print(f"  Exported ONNX model: {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    return output_path


def simplify_onnx(onnx_path: str) -> str:
    """Simplify ONNX model (remove redundant ops)."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  Warning: onnx-simplifier not installed. "
              "Install with: pip install onnxsim")
        return onnx_path

    model = onnx.load(onnx_path)
    model_simplified, check = simplify(model)

    if check:
        simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
        onnx.save(model_simplified, simplified_path)
        original_size = os.path.getsize(onnx_path) / 1024
        new_size = os.path.getsize(simplified_path) / 1024
        print(f"  Simplified: {original_size:.1f} KB -> {new_size:.1f} KB")
        return simplified_path
    else:
        print("  Warning: Simplification check failed, using original")
        return onnx_path


def verify_onnx(pytorch_model: nn.Module, onnx_path: str,
                input_size: int = 32, in_channels: int = 3):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed. "
              "Install with: pip install onnxruntime")
        return False

    import numpy as np

    device = next(pytorch_model.parameters()).device
    dummy = torch.randn(5, in_channels, input_size, input_size).to(device)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy).cpu().numpy()

    # ONNX Runtime output
    session = ort.InferenceSession(onnx_path)
    ort_input = {session.get_inputs()[0].name: dummy.cpu().numpy()}
    ort_output = session.run(None, ort_input)[0]

    # Compare
    max_diff = np.abs(pt_output - ort_output).max()
    mean_diff = np.abs(pt_output - ort_output).mean()

    print(f"\n  Verification:")
    print(f"    Max absolute difference: {max_diff:.6e}")
    print(f"    Mean absolute difference: {mean_diff:.6e}")

    passed = max_diff < 1e-4
    print(f"    Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def benchmark_onnx(onnx_path: str, input_size: int = 32,
                   num_runs: int = 100, in_channels: int = 3):
    """Benchmark ONNX Runtime inference speed."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  Warning: onnxruntime not installed.")
        return

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(5, in_channels, input_size, input_size).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    times = sorted(times)
    avg = sum(times) / len(times)
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]

    print(f"\n  ONNX Runtime Benchmark ({num_runs} runs, batch=5):")
    print(f"    Avg:    {avg:.2f} ms/batch ({avg/5:.2f} ms/image)")
    print(f"    Median: {median:.2f} ms/batch")
    print(f"    P95:    {p95:.2f} ms/batch")
    print(f"    Meets real-time (<4ms/image): "
          f"{'YES' if avg/5 < 4 else 'NO'}")


def convert_fp16(onnx_path: str) -> str:
    """Convert ONNX model to FP16 (half precision)."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("  Warning: onnxconverter-common not installed. "
              "Install with: pip install onnxconverter-common")
        return onnx_path

    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)

    fp16_path = onnx_path.replace('.onnx', '_fp16.onnx')
    onnx.save(model_fp16, fp16_path)

    original_size = os.path.getsize(onnx_path) / 1024
    new_size = os.path.getsize(fp16_path) / 1024
    print(f"  FP16 conversion: {original_size:.1f} KB -> {new_size:.1f} KB")
    return fp16_path


def main():
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cpu")  # export on CPU for compatibility
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model_name = cfg.get("model", {}).get("name", "mvanet_3")
    num_classes = cfg.get("model", {}).get("num_classes", 4)
    in_channels = cfg.get("model", {}).get("in_channels", 3)
    input_size = args.input_size or cfg.get("data", {}).get("input_size", 32)
    arch = args.arch or cfg.get("model", {}).get("arch", "MvANet")

    # Build and load model
    model = build_model(
        arch=arch,
        name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {arch}/{model_name} | Input: {input_size}x{input_size} | "
          f"Params: {model.count_parameters():,}")

    # Output path
    output_path = args.output or f"exports/{model_name}_{input_size}x{input_size}.onnx"

    # Export
    print(f"\nExporting to ONNX (opset {args.opset_version})...")
    onnx_path = export_onnx(model, output_path, input_size,
                            args.opset_version, args.dynamic_batch,
                            in_channels=in_channels)

    # Simplify
    if args.simplify:
        print("\nSimplifying ONNX model...")
        onnx_path = simplify_onnx(onnx_path)

    # Verify
    if args.verify:
        print("\nVerifying ONNX output...")
        if not verify_onnx(model, onnx_path, input_size, in_channels=in_channels):
            raise RuntimeError("ONNX verification failed: max diff >= 1e-4")

    # FP16 conversion intentionally runs after verification because a converted
    # FP16 graph is not expected to be bit-close to FP32 PyTorch outputs.
    if args.fp16:
        print("\nConverting to FP16...")
        onnx_path = convert_fp16(onnx_path)

    # Benchmark
    if args.benchmark:
        benchmark_onnx(onnx_path, input_size, in_channels=in_channels)

    print(f"\nExport complete: {onnx_path}")


if __name__ == "__main__":
    main()
