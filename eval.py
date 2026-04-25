#!/usr/bin/env python3
"""
eval.py — MvANet Evaluation Script (standalone)
================================================
Evaluate a trained model on the test set.

Usage:
    python eval.py --checkpoint runs/mvanet_2/run_01/best.pth --data-root ./dataset
    python eval.py --checkpoint best.pth --data-root ./dataset --visualize

Hoặc dùng qua train.py:
    python train.py eval --checkpoint best.pth --data-root ./dataset
"""

import argparse
import os
import sys
import time
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model
from data import get_eval_transforms
from utils import (
    MetricTracker,
    plot_confusion_matrix, plot_per_class_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MvANet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--arch", type=str, default=None,
                        help="Override architecture (default: from checkpoint)")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    try:
        import torch.serialization, numpy as _np
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
    except Exception:
        pass

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    arch = args.arch or cfg.get("model", {}).get("arch", "MvANet")
    model_name = cfg.get("model", {}).get("name", "mvanet_3")
    num_classes = cfg.get("model", {}).get("num_classes", 4)
    input_size = args.input_size or cfg.get("data", {}).get("input_size", 32)
    class_names = cfg.get("data", {}).get("classes", [])

    print(f"Model: {arch}/{model_name} | Input: {input_size}x{input_size} | Classes: {num_classes}")

    model = build_model(arch=arch, name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Parameters: {model.count_parameters():,}")

    # Data
    from torchvision import datasets
    from torch.utils.data import DataLoader

    eval_tf = get_eval_transforms(input_size=input_size)
    test_dir = os.path.join(args.data_root, "test")
    assert os.path.isdir(test_dir), f"Test dir not found: {test_dir}"
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    if not class_names:
        class_names = test_ds.classes
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nEvaluating on {len(test_ds)} test images...")
    criterion = nn.CrossEntropyLoss()
    tracker = MetricTracker(num_classes, class_names)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            tracker.update(preds, labels, loss.item(), images.size(0))

    tracker.print_report()

    # Inference time
    print("Measuring inference time...")
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    avg_time = sum(times) / len(times)
    print(f"  Avg inference: {avg_time:.2f} ms/image")

    # Visualize & save
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    if args.visualize:
        plot_confusion_matrix(tracker.confusion_matrix(), class_names,
                              os.path.join(output_dir, "eval_confusion_matrix.png"))
        plot_confusion_matrix(tracker.confusion_matrix(), class_names,
                              os.path.join(output_dir, "eval_confusion_matrix_norm.png"),
                              normalize=True)
        plot_per_class_metrics(tracker.per_class_metrics(),
                               os.path.join(output_dir, "eval_per_class_metrics.png"))

    results = {
        "checkpoint": args.checkpoint,
        "arch": arch, "model": model_name,
        "test_accuracy": tracker.accuracy,
        "test_error": tracker.error_rate,
        "avg_time_ms": avg_time,
        "per_class": tracker.per_class_metrics(),
        "summary": tracker.summary(),
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
