#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import create_dataloaders
from loss import build_loss
from models import build_model, list_models, model_params_M
from utils import (
    MetricTracker, MultiSeedLogger, TrainingLogger,
    collect_environment, get_rng_state, seed_everything, set_rng_state,
    plot_confusion_matrix, plot_per_class_metrics, plot_training_curves,
)

DATASET_CONFIGS = {
    "jujube": "configs/jujube.yaml",
    "jujube_ours": "configs/jujube_ours.yaml",
}
SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train lightweight models on jujube datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --dataset jujube --model mobilenet_v3_small --loss cross_entropy --seed 1\n"
            "  python train.py --dataset jujube_ours --model efficientnet_b0 --loss focal --seed 1 2 3 4 5\n"
            "  python train.py --dataset jujube --model resnet18 --loss label_smoothing --seed 1 2 3\n"
        ),
    )
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model", required=True,
                   help=f"Model name. Available: {list_models()}")
    p.add_argument("--loss", default="cross_entropy",
                   choices=["cross_entropy", "focal", "label_smoothing"])
    p.add_argument("--seed", type=int, nargs="+", required=True,
                   help="One or more training seeds (e.g. --seed 1 2 3 4 5)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no-benchmark", action="store_true",
                   help="Skip inference benchmark (faster CI runs)")
    p.add_argument("--benchmark-warmup", type=int, default=50,
                   help="GPU warmup iterations before latency measurement")
    p.add_argument("--benchmark-runs", type=int, default=200,
                   help="Measurement iterations for latency benchmark")
    return p.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def get_run_dir(dataset: str, model: str, loss: str, seed: int) -> str:
    return os.path.join("runs", dataset, model, loss, f"seed_{seed}")


def write_json(path: str, data: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_optimizer(model: nn.Module, cfg: Dict):
    opt_name = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg: Dict, epochs: int):
    sched = cfg.get("lr_scheduler", "cosine").lower()
    warmup = cfg.get("warmup_epochs", 5)
    if sched == "cosine":
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1), eta_min=1e-6
        )
        if warmup > 0:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup), cos],
                milestones=[warmup],
            )
        return cos
    if sched == "step":
        milestones = cfg.get("step_milestones",
                             [int(epochs * 0.6), int(epochs * 0.85)])
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)
    if sched == "none":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: 1.0)
    raise ValueError(f"Unknown scheduler: {sched}")


def train_one_epoch(model: nn.Module, loader, criterion, optimizer,
                    device: torch.device, num_classes: int,
                    class_names: List[str]) -> Dict:
    model.train()
    tracker = MetricTracker(num_classes, class_names)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        tracker.update(preds, labels, float(loss.item()), len(labels))
    return tracker.summary()


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device,
             num_classes: int, class_names: List[str]) -> Dict:
    model.eval()
    tracker = MetricTracker(num_classes, class_names)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        tracker.update(preds, labels, float(loss.item()), len(labels))
    return tracker.summary(), tracker


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Edge AI efficiency metrics (required for A*/Q1 paper comparison tables)
# ---------------------------------------------------------------------------

def get_model_size_mb(model: nn.Module) -> float:
    """Disk size of model weights in MB (float32 baseline)."""
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return (params + buffers) * 4 / (1024 ** 2)


def get_flops_gmacs(model: nn.Module, input_size: int,
                    in_channels: int = 3) -> Optional[float]:
    """Try thop → ptflops → return None if neither installed.

    Returns GMACs (= GFLOPs / 2 in multiply-accumulate convention).
    Papers on jujube edge AI (JujubeNet, Ju-LiteMobileAtt, RJ-TinyViT) all
    report FLOPs in this unit.
    """
    dummy = torch.zeros(1, in_channels, input_size, input_size,
                        device=next(model.parameters()).device)
    # --- thop (preferred, pip install thop) ---
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return round(macs / 1e9, 4)
    except ImportError:
        pass
    # --- ptflops fallback ---
    try:
        import ptflops
        macs_str, _ = ptflops.get_model_complexity_info(
            model,
            input_res=(in_channels, input_size, input_size),
            verbose=False,
            print_per_layer_stat=False,
        )
        parts = macs_str.split()
        value = float(parts[0])
        unit = parts[1] if len(parts) > 1 else ""
        if "G" in unit:
            return round(value, 4)
        if "M" in unit:
            return round(value / 1_000, 4)
        if "K" in unit:
            return round(value / 1_000_000, 4)
        return round(value / 1e9, 4)
    except ImportError:
        pass
    return None


def benchmark_inference(
    model: nn.Module,
    input_size: int,
    in_channels: int = 3,
    device: Optional[torch.device] = None,
    warmup: int = 50,
    runs: int = 200,
) -> Dict:
    """GPU-T4 inference benchmark following Edge AI paper protocol.

    Reports (all latency values in ms):
      - batch1_mean_ms / std_ms / p95_ms — per-image latency (edge-realistic)
      - batch1_fps                        — single-image throughput
      - batch32_fps                       — sustained throughput (industrial line)
      - device_name                       — e.g. "Tesla T4"

    Protocol matches JujubeNet / Ju-LiteMobileAtt / RJ-TinyViT benchmarks:
    GPU sync before each timing, CUDA events for sub-ms precision on GPU.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    use_cuda = device.type == "cuda"

    x1 = torch.randn(1, in_channels, input_size, input_size, device=device)
    x32 = torch.randn(32, in_channels, input_size, input_size, device=device)

    def _time_batch(x: torch.Tensor, n: int) -> List[float]:
        times = []
        if use_cuda:
            for _ in range(n):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                with torch.no_grad():
                    model(x)
                end_evt.record()
                torch.cuda.synchronize()
                times.append(start_evt.elapsed_time(end_evt))
        else:
            for _ in range(n):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(x)
                times.append((time.perf_counter() - t0) * 1000)
        return times

    # Warmup
    _time_batch(x1, warmup)

    # Batch-1 latency
    times1 = sorted(_time_batch(x1, runs))
    mean1 = float(np.mean(times1))
    std1 = float(np.std(times1, ddof=1))
    p95_1 = float(np.percentile(times1, 95))
    fps1 = round(1000.0 / mean1, 1) if mean1 > 0 else 0.0

    # Batch-32 throughput (warmup already done)
    times32 = _time_batch(x32, max(runs // 4, 50))
    mean32 = float(np.mean(times32))
    fps32 = round(32 * 1000.0 / mean32, 1) if mean32 > 0 else 0.0

    if use_cuda:
        device_name = torch.cuda.get_device_name(device)
    else:
        import platform
        device_name = f"CPU ({platform.processor() or platform.machine()})"

    # ------------------------------------------------------------------
    # Peak GPU memory during inference (MB)
    # Required by Edge AI papers to prove model fits on constrained devices.
    # Measured separately with memory tracking enabled.
    # ------------------------------------------------------------------
    peak_mem_mb: Optional[float] = None
    if use_cuda:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                model(x1)
            torch.cuda.synchronize(device)
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem_mb = round(peak_bytes / (1024 ** 2), 2)
        except Exception:
            peak_mem_mb = None

    return {
        "device": device_name,
        "input_size": input_size,
        "warmup_iters": warmup,
        "measurement_iters": runs,
        "batch1_latency_mean_ms": round(mean1, 3),
        "batch1_latency_std_ms": round(std1, 3),
        "batch1_latency_p95_ms": round(p95_1, 3),
        "batch1_fps": fps1,
        "batch32_fps": fps32,
        # Peak GPU memory for a single-image forward pass
        "peak_gpu_mem_mb": peak_mem_mb,
        # Real-time threshold: 30 FPS is standard for industrial sorting lines
        "meets_realtime_30fps": fps1 >= 30.0,
    }


def print_efficiency_report(eff: Dict) -> None:
    """Pretty-print the Edge AI efficiency table after test evaluation."""
    print(f"\n{'='*65}")
    print("  Edge AI Efficiency Metrics")
    print(f"{'='*65}")
    print(f"  Params          : {eff['params']:,} ({eff['params_M']:.2f} M)")
    print(f"  Model size      : {eff['model_size_mb']:.2f} MB")
    if eff.get("flops_gmacs") is not None:
        print(f"  FLOPs           : {eff['flops_gmacs']:.4f} G MACs")
    else:
        print("  FLOPs           : N/A (install thop or ptflops)")
    if "benchmark" in eff:
        b = eff["benchmark"]
        print(f"  Device          : {b['device']}")
        print(f"  Latency (bs=1)  : {b['batch1_latency_mean_ms']:.2f} ± "
              f"{b['batch1_latency_std_ms']:.2f} ms  "
              f"(p95={b['batch1_latency_p95_ms']:.2f} ms)")
        print(f"  Throughput      : {b['batch1_fps']:.1f} FPS (bs=1)  |  "
              f"{b['batch32_fps']:.1f} FPS (bs=32)")
        if b.get("peak_gpu_mem_mb") is not None:
            print(f"  Peak GPU Mem    : {b['peak_gpu_mem_mb']:.1f} MB")
        rt = "✓ YES" if b["meets_realtime_30fps"] else "✗ NO"
        print(f"  Real-time ≥30fps: {rt}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Model info / training helpers
# ---------------------------------------------------------------------------

def print_model_info(model_name: str, model: nn.Module, cfg: Dict):
    params = count_parameters(model)
    print(f"\n  Model     : {model_name}")
    print(f"  Params    : {params:,} ({params / 1e6:.2f} M)")
    print(f"  Input     : {cfg['input_size']}x{cfg['input_size']}")
    print(f"  Classes   : {cfg['num_classes']} ({cfg['class_names']})")


def run_seed(seed: int, cfg: Dict, loaders: Dict, args,
             device: torch.device) -> Dict:
    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]
    epochs = cfg["epochs"]
    input_size = cfg["input_size"]
    in_channels = cfg.get("in_channels", 3)

    run_dir = get_run_dir(args.dataset, args.model, args.loss, seed)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Seed {seed} | dataset={args.dataset} "
          f"model={args.model} loss={args.loss}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*65}")

    seed_everything(seed)

    model = build_model(args.model, num_classes).to(device)

    if seed == args.seed[0]:
        print_model_info(args.model, model, cfg)

    loss_cfg = cfg.get("loss", {})
    loss_cfg_clean = {k: v for k, v in loss_cfg.items() if k != "name"}
    criterion = build_loss(args.loss, num_classes=num_classes,
                           loss_cfg=loss_cfg_clean)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    logger = TrainingLogger(run_dir, seed)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    print(f"\n  Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer,
            device, num_classes, class_names)
        val_metrics, _ = evaluate(
            model, loaders["val"], criterion, device, num_classes, class_names)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
        if is_best:
            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_acc": best_val_acc,
                "seed": seed,
                "config": {
                    "model": args.model,
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                },
            }, os.path.join(run_dir, "best.pth"))

    logger.save_history()
    history = logger.get_history()
    plot_training_curves(
        history, save_path=os.path.join(run_dir, "training_curves.png"))

    # Expose training time stats before loading best weights
    train_summary = logger.training_summary()
    print(f"\n  Best epoch       : {train_summary['best_epoch']}  "
          f"(val acc = {train_summary['best_val_acc']:.2f}%)")
    print(f"  Total train time : {train_summary['total_train_time_min']:.1f} min  "
          f"({train_summary['mean_epoch_time_s']:.1f} s/epoch)")

    print(f"\n  Best val accuracy: {best_val_acc:.2f}% (loaded for test)")
    model.load_state_dict(best_state)

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------
    test_metrics, test_tracker = evaluate(
        model, loaders["test"], criterion, device, num_classes, class_names)
    test_tracker.print_report(split=f"Test (seed={seed})")

    cm = test_tracker.confusion_matrix()
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(run_dir, "confusion_matrix.png"))
    plot_per_class_metrics(
        test_tracker.per_class_metrics(),
        save_path=os.path.join(run_dir, "per_class_metrics.png"))

    # -----------------------------------------------------------------------
    # Edge AI efficiency metrics (A*/Q1 paper requirement)
    # Collected once per seed (model weights fixed after best checkpoint load)
    # -----------------------------------------------------------------------
    params = count_parameters(model)
    efficiency: Dict = {
        "params": params,
        "params_M": round(params / 1e6, 4),
        "model_size_mb": round(get_model_size_mb(model), 3),
        "flops_gmacs": get_flops_gmacs(model, input_size, in_channels),
    }

    if not args.no_benchmark:
        efficiency["benchmark"] = benchmark_inference(
            model,
            input_size=input_size,
            in_channels=in_channels,
            device=device,
            warmup=args.benchmark_warmup,
            runs=args.benchmark_runs,
        )

    print_efficiency_report(efficiency)

    # -----------------------------------------------------------------------
    # Save everything to test_results.json
    # -----------------------------------------------------------------------
    result = {
        "seed": seed,
        "best_val_acc": best_val_acc,
        "training": train_summary,         # best_epoch, total_time, s/epoch
        "test": test_metrics,
        "per_class": test_tracker.per_class_metrics(),
        "efficiency": efficiency,
    }
    write_json(os.path.join(run_dir, "test_results.json"), result)

    return test_metrics


def main():
    args = parse_args()

    config_path = DATASET_CONFIGS[args.dataset]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.epochs:
        cfg["epochs"] = args.epochs

    device = get_device(args.device)

    print(f"\n{'='*65}")
    print(f"  Jujube Classification")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Loss    : {args.loss}")
    print(f"  Seeds   : {args.seed}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Split   : 70/15/15 (seed={SPLIT_SEED}, fixed)")
    print(f"{'='*65}")

    print(f"\n  Creating 70/15/15 stratified split (seed={SPLIT_SEED})...")
    loaders = create_dataloaders(cfg, split_seed=SPLIT_SEED)
    sizes = loaders["split_sizes"]
    print(f"  Train: {sizes['train']} | Val: {sizes['val']} | Test: {sizes['test']}")

    write_json(
        os.path.join("runs", args.dataset, args.model, args.loss,
                     "split_info.json"),
        {"split_seed": SPLIT_SEED, "sizes": sizes,
         "class_names": cfg["class_names"]},
    )

    multi_logger = MultiSeedLogger()

    for seed in args.seed:
        test_metrics = run_seed(seed, cfg, loaders, args, device)
        multi_logger.add(seed, test_metrics)

    if len(args.seed) > 1:
        multi_logger.print_summary(args.model, args.loss, args.dataset)

    summary_path = os.path.join(
        "runs", args.dataset, args.model, args.loss, "summary.json")
    multi_logger.save(summary_path, args.model, args.loss, args.dataset)
    print(f"  Summary saved → {summary_path}")

    env = collect_environment()
    write_json(
        os.path.join("runs", args.dataset, args.model, args.loss,
                     "environment.json"),
        env,
    )


if __name__ == "__main__":
    main()
