#!/usr/bin/env python3
"""Standalone evaluation wrapper for MvANet checkpoints.

This script reuses the same deterministic DataLoader and metric code as
``train.py eval`` so standalone evaluation cannot drift from the main CLI.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from train import (
    build_eval_dataloaders_from_config,
    evaluate_and_save,
    get_device,
    load_checkpoint,
    load_config,
    seed_everything,
    write_json,
)
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MvANet checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["root"] = args.data_root
    if args.batch_size is not None:
        cfg["data"]["batch_size_eval"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    cfg.setdefault("eval", {})
    cfg["eval"]["visualize"] = bool(args.visualize)

    device = get_device(args.device)
    seed = int(cfg.get("train", {}).get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_cfg = ckpt.get("config", {})
    ckpt_model = ckpt_cfg.get("model", {})
    ckpt_data = ckpt_cfg.get("data", {})

    arch = args.arch or ckpt_model.get("arch", cfg["model"]["arch"])
    model_name = args.model or ckpt_model.get("name", cfg["model"]["name"])
    num_classes = int(ckpt_model.get("num_classes", cfg["model"]["num_classes"]))
    in_channels = int(ckpt_model.get("in_channels", cfg["model"].get("in_channels", 3)))
    input_size = args.input_size or ckpt_data.get("input_size", cfg["data"]["input_size"])
    class_names = ckpt_data.get("classes", [])

    cfg["model"].update(
        {
            "arch": arch,
            "name": model_name,
            "num_classes": num_classes,
            "in_channels": in_channels,
        }
    )
    cfg["data"]["input_size"] = int(input_size)
    cfg["data"]["normalize_mean"] = ckpt_data.get(
        "normalize_mean", cfg["data"].get("normalize_mean")
    )
    cfg["data"]["normalize_std"] = ckpt_data.get(
        "normalize_std", cfg["data"].get("normalize_std")
    )

    model = build_model(
        arch=arch,
        name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = build_eval_dataloaders_from_config(
        cfg, seed, device, class_names or None
    )
    if not class_names:
        class_names = loaders["class_names"]
    elif list(class_names) != list(loaders["class_names"]):
        raise ValueError(
            "Checkpoint class order does not match dataset class order: "
            f"checkpoint={class_names} dataset={loaders['class_names']}"
        )

    criterion = torch.nn.CrossEntropyLoss()
    output_dir = args.output_dir or os.path.dirname(args.checkpoint) or "."

    results = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "model": model_name,
        "classes": class_names,
    }
    if "test" in loaders:
        print("\nEvaluating test split...")
        results["test"] = evaluate_and_save(
            model,
            loaders["test"],
            criterion,
            device,
            class_names,
            num_classes,
            output_dir,
            "eval_test",
            bool(args.visualize),
        )
    if "val" in loaders:
        print("\nEvaluating val split...")
        results["val"] = evaluate_and_save(
            model,
            loaders["val"],
            criterion,
            device,
            class_names,
            num_classes,
            output_dir,
            "eval_val",
            bool(args.visualize),
        )

    results_path = os.path.join(output_dir, "eval_results.json")
    write_json(results_path, results)
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
