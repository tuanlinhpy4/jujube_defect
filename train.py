#!/usr/bin/env python3
"""
train.py — MvANet CLI
======================
Thay thế notebook bằng giao diện CLI duy nhất.

Usage (Colab hoặc Local):
    # Xem danh sách model
    python train.py info

    # Train với default config
    python train.py train

    # Train với config tùy chỉnh
    python train.py train --arch MvANet_opt --model mvanet_1_tiny --lr 0.01 --epochs 100

    # Train từ file config khác
    python train.py train --config configs/my_config.yaml --epochs 50

    # Đánh giá trên test set
    python train.py eval --checkpoint runs/mvanet_2/run_01/best.pth

    # Export ONNX
    python train.py export --checkpoint runs/mvanet_2/run_01/best.pth

    # Resume training
    python train.py train --resume runs/mvanet_2/run_01/last.pth

Nguyên tắc:
    - Mọi flag CLI đều KHÔNG bắt buộc → dùng giá trị trong YAML
    - Flag CLI luôn ghi đè YAML
    - Thêm model mới: tạo folder models/TênModel/ với __init__.py
      export build_model + MODEL_CONFIGS → xong.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Đảm bảo project root nằm trong sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model, registry
from data import get_train_transforms, get_eval_transforms
from utils import MetricTracker, TrainingLogger, MultiRunLogger
from utils import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Config loading: YAML → defaults, CLI → overrides
# ═══════════════════════════════════════════════════════════════════════════

def _flat(d, prefix="", sep="."):
    """Flatten nested dict: {'a': {'b': 1}} → {'a.b': 1}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key, sep))
        else:
            out[key] = v
    return out


def _unflatten(d, sep="."):
    """Unflatten: {'a.b': 1} → {'a': {'b': 1}}"""
    out = {}
    for k, v in d.items():
        parts = k.split(sep)
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return out


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def merge_cli_into_config(cfg: dict, cli_args: argparse.Namespace) -> dict:
    """Ghi đè config bằng CLI args (chỉ các arg được truyền thực tế)."""
    flat_cfg = _flat(cfg)

    # Mapping: CLI flag name → flat config key
    CLI_MAP = {
        # model
        "arch":             "model.arch",
        "model":            "model.name",
        "num_classes":      "model.num_classes",
        "in_channels":      "model.in_channels",
        # data
        "data_root":        "data.root",
        "input_size":       "data.input_size",
        "batch_size":       "data.batch_size_train",
        "batch_size_eval":  "data.batch_size_eval",
        "num_workers":      "data.num_workers",
        # train
        "epochs":           "train.epochs",
        "optimizer":        "train.optimizer",
        "lr":               "train.lr",
        "momentum":         "train.momentum",
        "weight_decay":     "train.weight_decay",
        "lr_scheduler":     "train.lr_scheduler",
        "plateau_factor":   "train.plateau_factor",
        "plateau_patience": "train.plateau_patience",
        "plateau_min_lr":   "train.plateau_min_lr",
        "cosine_T_max":     "train.cosine_T_max",
        "cosine_eta_min":   "train.cosine_eta_min",
        "step_milestones":  "train.step_milestones",
        "step_gamma":       "train.step_gamma",
        "use_class_weights":"train.use_class_weights",
        "num_runs":         "train.num_runs",
        "seed":             "train.seed",
        "save_dir":         "train.save_dir",
        "save_every":       "train.save_every",
        "resume":           "train.resume",
        # eval
        "checkpoint":       "eval.checkpoint",
        "visualize":        "eval.visualize",
        # export
        "output":           "export.output",
        "opset_version":    "export.opset_version",
        "dynamic_batch":    "export.dynamic_batch",
        "simplify":         "export.simplify",
        "verify":           "export.verify",
    }

    cli_dict = vars(cli_args)
    for cli_key, cfg_key in CLI_MAP.items():
        val = cli_dict.get(cli_key)
        if val is not None:
            flat_cfg[cfg_key] = val

    return _unflatten(flat_cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(hint: str = "auto") -> torch.device:
    if hint == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(hint)


def train_one_epoch(model, loader, criterion, optimizer, device, tracker):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = outputs.max(1)
        tracker.update(preds, labels, loss.item(), images.size(0))


@torch.no_grad()
def evaluate(model, loader, criterion, device, tracker):
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, preds = outputs.max(1)
        tracker.update(preds, labels, loss.item(), images.size(0))


def save_checkpoint(model, optimizer, scheduler, epoch, metrics,
                    save_path, cfg):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": {
            "model": {
                "arch": cfg["model"]["arch"],
                "name": cfg["model"]["name"],
                "num_classes": cfg["model"]["num_classes"],
            },
            "data": {
                "input_size": cfg["data"]["input_size"],
                "classes": cfg.get("_class_names", []),
            },
        },
    }, save_path)


def build_dataloaders(cfg: dict):
    """Tạo DataLoader cho train/val/test từ config."""
    from torchvision import datasets
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    root = data_cfg["root"]
    input_size = data_cfg["input_size"]
    mean = tuple(data_cfg.get("normalize_mean", [0.5, 0.5, 0.5]))
    std = tuple(data_cfg.get("normalize_std", [0.5, 0.5, 0.5]))
    num_workers = data_cfg.get("num_workers", 2)

    eval_tf = get_eval_transforms(input_size=input_size, mean=mean, std=std)

    loaders = {}
    class_names = None
    num_classes = None

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        ds = datasets.ImageFolder(split_dir, transform=eval_tf)
        if class_names is None:
            class_names = ds.classes
            num_classes = len(class_names)
        bs = data_cfg.get("batch_size_train", 64) if split == "train" else data_cfg.get("batch_size_eval", 64)
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"  {split}: {len(ds)} images, {len(loaders[split])} batches")

    loaders["class_names"] = class_names or []
    loaders["num_classes"] = num_classes or cfg["model"]["num_classes"]
    return loaders


def compute_class_weights(loader, num_classes, device):
    """Tính class weights dựa trên inverse-frequency."""
    counts = np.zeros(num_classes)
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * num_classes
    return torch.FloatTensor(w).to(device)


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]
    opt_name = train_cfg.get("optimizer", "sgd").lower()
    lr = train_cfg.get("lr", 0.001)
    wd = train_cfg.get("weight_decay", 1e-4)

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=wd,
        )
    elif opt_name in ("adam", "amsgrad"):
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd,
            amsgrad=(opt_name == "amsgrad"),
        )
    else:
        raise ValueError(f"Optimizer '{opt_name}' không hỗ trợ. Dùng: sgd, adam, amsgrad")


def build_scheduler(optimizer, cfg):
    train_cfg = cfg["train"]
    sched_name = train_cfg.get("lr_scheduler", "plateau").lower()

    if sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=train_cfg.get("plateau_mode", "min"),
            factor=train_cfg.get("plateau_factor", 0.5),
            patience=train_cfg.get("plateau_patience", 15),
            min_lr=train_cfg.get("plateau_min_lr", 1e-5),
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("cosine_T_max", train_cfg.get("epochs", 300)),
            eta_min=train_cfg.get("cosine_eta_min", 1e-5),
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_cfg.get("step_milestones", [120, 210, 270]),
            gamma=train_cfg.get("step_gamma", 0.1),
        )
    else:
        raise ValueError(f"Scheduler '{sched_name}' không hỗ trợ. Dùng: plateau, cosine, step")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: info
# ═══════════════════════════════════════════════════════════════════════════

def cmd_info(cfg, args):
    """In danh sách models/architectures có sẵn."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║           MvANet — Available Models              ║")
    print("╚══════════════════════════════════════════════════╝\n")

    all_models = registry.list_all()
    for arch, variants in all_models.items():
        print(f"  📦 {arch}/")
        for v in variants:
            model = registry.build(arch, v, num_classes=4)
            params = model.count_parameters()
            print(f"     ├─ {v:<16} {params:>8,} params ({params*4/1024:.1f} KB)")
        print()

    print("Sử dụng:")
    print("  python train.py train --arch MvANet --model mvanet_2")
    print("  python train.py train --arch MvANet_opt --model mvanet_1_tiny")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: train
# ═══════════════════════════════════════════════════════════════════════════

def cmd_train(cfg, args):
    """Train loop — tái hiện logic từ notebook."""
    # Fix PyTorch ≥2.6 safe globals
    try:
        import torch.serialization
        import numpy as _np
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
    except Exception:
        pass

    device = get_device(getattr(args, "device", "auto"))
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    arch = model_cfg["arch"]
    model_name = model_cfg["name"]
    num_classes = model_cfg["num_classes"]
    in_channels = model_cfg.get("in_channels", 3)
    epochs = train_cfg["epochs"]
    num_runs = train_cfg.get("num_runs", 1)
    base_seed = train_cfg.get("seed", 42)
    save_dir_root = train_cfg.get("save_dir", "./runs")
    save_every = train_cfg.get("save_every", 50)

    print(f"\n{'='*60}")
    print(f"  MvANet Training")
    print(f"  Arch: {arch} | Model: {model_name} | Classes: {num_classes}")
    print(f"  Input: {data_cfg['input_size']}x{data_cfg['input_size']}")
    print(f"  Epochs: {epochs} | Runs: {num_runs} | Device: {device}")
    print(f"  Optimizer: {train_cfg.get('optimizer', 'sgd')} | LR: {train_cfg.get('lr', 0.001)}")
    print(f"  Scheduler: {train_cfg.get('lr_scheduler', 'plateau')}")
    print(f"{'='*60}\n")

    # DataLoader
    print("Loading dataset...")
    loaders = build_dataloaders(cfg)
    class_names = loaders["class_names"]
    actual_num_classes = loaders["num_classes"]

    assert actual_num_classes == num_classes, (
        f"Config num_classes={num_classes} nhưng dataset có "
        f"{actual_num_classes} class: {class_names}"
    )
    cfg["_class_names"] = list(class_names)

    print(f"\nClasses: {class_names}")

    # Class weights
    class_weights = None
    if train_cfg.get("use_class_weights", False) and "train" in loaders:
        class_weights = compute_class_weights(
            loaders["train"], num_classes, device
        )
        print(f"Class weights: {class_weights.tolist()}")

    # Multi-run
    multi_logger = MultiRunLogger()

    for run_id in range(1, num_runs + 1):
        seed = base_seed + run_id
        set_seed(seed)

        run_name = f"run_{run_id:02d}"
        save_dir = os.path.join(save_dir_root, model_name, run_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Run {run_id}/{num_runs} | {arch}/{model_name} | Seed: {seed}")
        print(f"  Save: {save_dir}")
        print(f"{'='*60}")

        # Build model
        model = build_model(
            arch=arch, name=model_name,
            num_classes=num_classes, in_channels=in_channels,
        ).to(device)
        print(f"  Params: {model.count_parameters():,}")

        # Optimizer & scheduler
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        # Resume
        start_epoch = 1
        resume_path = train_cfg.get("resume")
        if resume_path and os.path.isfile(resume_path):
            print(f"  Resuming from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"  Resumed at epoch {start_epoch}")

        # Loss
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        logger = TrainingLogger(save_dir, run_name)

        # ---------------------------------------------------------------
        # Epoch loop
        # ---------------------------------------------------------------
        for epoch in range(start_epoch, epochs + 1):
            # Train
            train_tracker = MetricTracker(num_classes, class_names)
            train_one_epoch(model, loaders["train"], criterion,
                            optimizer, device, train_tracker)

            # Validate
            val_loader = loaders.get("val", loaders.get("test"))
            val_tracker = MetricTracker(num_classes, class_names)
            evaluate(model, val_loader, criterion, device, val_tracker)

            lr = optimizer.param_groups[0]["lr"]
            is_best = logger.log_epoch(
                epoch, train_tracker.summary(), val_tracker.summary(), lr
            )

            # Scheduler step
            sched_name = train_cfg.get("lr_scheduler", "plateau").lower()
            if sched_name == "plateau":
                scheduler.step(val_tracker.summary()["loss"])
            else:
                scheduler.step()

            # Save best
            if is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_tracker.summary(),
                    os.path.join(save_dir, "best.pth"), cfg,
                )

            # Save periodic
            if epoch % save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_tracker.summary(),
                    os.path.join(save_dir, f"epoch_{epoch}.pth"), cfg,
                )

        # Save last
        save_checkpoint(
            model, optimizer, scheduler, epochs,
            val_tracker.summary(),
            os.path.join(save_dir, "last.pth"), cfg,
        )

        logger.save_history()
        logger.print_summary()

        # Lưu config vào run folder
        config_save_path = os.path.join(save_dir, "config.yaml")
        cfg_to_save = {k: v for k, v in cfg.items() if not k.startswith("_")}
        with open(config_save_path, "w") as f:
            yaml.dump(cfg_to_save, f, default_flow_style=False, allow_unicode=True)

    # ===================================================================
    # Đánh giá trên test set (sau run cuối)
    # ===================================================================
    if "test" in loaders:
        print("\n" + "=" * 60)
        print("  ĐÁNH GIÁ TRÊN TẬP TEST")
        print("=" * 60)

        best_path = os.path.join(save_dir, "best.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        test_tracker = MetricTracker(num_classes, class_names)
        evaluate(model, loaders["test"], criterion, device, test_tracker)
        test_tracker.print_report()

        # Vẽ biểu đồ
        history = logger.get_history()
        plot_training_curves(history, os.path.join(save_dir, "training_curves.png"))
        plot_confusion_matrix(
            test_tracker.confusion_matrix(), class_names,
            os.path.join(save_dir, "confusion_matrix.png"),
        )
        plot_confusion_matrix(
            test_tracker.confusion_matrix(), class_names,
            os.path.join(save_dir, "confusion_matrix_norm.png"),
            normalize=True,
        )
        plot_per_class_metrics(
            test_tracker.per_class_metrics(),
            os.path.join(save_dir, "per_class_metrics.png"),
        )

        # Lưu kết quả JSON
        results = {
            "arch": arch,
            "model": model_name,
            "input_size": data_cfg["input_size"],
            "epochs": epochs,
            "num_classes": num_classes,
            "classes": list(class_names),
            "test_accuracy": test_tracker.accuracy,
            "test_error": test_tracker.error_rate,
            "best_val_error": logger.best_error,
            "per_class": test_tracker.per_class_metrics(),
        }
        results_path = os.path.join(save_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nKết quả đã lưu: {results_path}")

    print("\n✅ Training hoàn tất!")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: eval
# ═══════════════════════════════════════════════════════════════════════════

def cmd_eval(cfg, args):
    """Đánh giá model trên test set."""
    try:
        import torch.serialization
        import numpy as _np
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
    except Exception:
        pass

    device = get_device(getattr(args, "device", "auto"))

    ckpt_path = cfg["eval"].get("checkpoint") or getattr(args, "checkpoint", None)
    if not ckpt_path:
        print("❌ Cần truyền --checkpoint <path>")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})

    arch = ckpt_cfg.get("model", {}).get("arch", cfg["model"]["arch"])
    model_name = ckpt_cfg.get("model", {}).get("name", cfg["model"]["name"])
    num_classes = ckpt_cfg.get("model", {}).get("num_classes", cfg["model"]["num_classes"])
    input_size = ckpt_cfg.get("data", {}).get("input_size", cfg["data"]["input_size"])
    class_names = ckpt_cfg.get("data", {}).get("classes", [])

    # Override data config from CLI / yaml
    cfg["data"]["input_size"] = input_size

    model = build_model(arch=arch, name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {arch}/{model_name} | Params: {model.count_parameters():,}")

    # Data
    loaders = build_dataloaders(cfg)
    if class_names:
        pass  # dùng class_names từ checkpoint
    else:
        class_names = loaders["class_names"]

    criterion = nn.CrossEntropyLoss()

    # Test
    if "test" in loaders:
        print("\nĐánh giá trên test set...")
        tracker = MetricTracker(num_classes, class_names)
        evaluate(model, loaders["test"], criterion, device, tracker)
        tracker.print_report()

        # Visualize
        output_dir = os.path.dirname(ckpt_path)
        if cfg["eval"].get("visualize", True):
            plot_confusion_matrix(
                tracker.confusion_matrix(), class_names,
                os.path.join(output_dir, "eval_confusion_matrix.png"),
            )
            plot_confusion_matrix(
                tracker.confusion_matrix(), class_names,
                os.path.join(output_dir, "eval_confusion_matrix_norm.png"),
                normalize=True,
            )
            plot_per_class_metrics(
                tracker.per_class_metrics(),
                os.path.join(output_dir, "eval_per_class_metrics.png"),
            )

        # JSON
        results = {
            "checkpoint": ckpt_path,
            "arch": arch,
            "model": model_name,
            "test_accuracy": tracker.accuracy,
            "test_error": tracker.error_rate,
            "per_class": tracker.per_class_metrics(),
            "summary": tracker.summary(),
        }
        results_path = os.path.join(output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nKết quả: {results_path}")

    # Val
    if "val" in loaders:
        print("\nĐánh giá trên val set...")
        val_tracker = MetricTracker(num_classes, class_names)
        evaluate(model, loaders["val"], criterion, device, val_tracker)
        val_tracker.print_report()


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: export
# ═══════════════════════════════════════════════════════════════════════════

def cmd_export(cfg, args):
    """Export model sang ONNX."""
    try:
        import torch.serialization
        import numpy as _np
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
    except Exception:
        pass

    ckpt_path = cfg["eval"].get("checkpoint") or getattr(args, "checkpoint", None)
    if not ckpt_path:
        print("❌ Cần truyền --checkpoint <path>")
        sys.exit(1)

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})

    arch = ckpt_cfg.get("model", {}).get("arch", cfg["model"]["arch"])
    model_name = ckpt_cfg.get("model", {}).get("name", cfg["model"]["name"])
    num_classes = ckpt_cfg.get("model", {}).get("num_classes", cfg["model"]["num_classes"])
    input_size = ckpt_cfg.get("data", {}).get("input_size", cfg["data"]["input_size"])

    model = build_model(arch=arch, name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {arch}/{model_name} | Params: {model.count_parameters():,}")

    export_cfg = cfg.get("export", {})
    output_path = export_cfg.get("output") or (
        f"exports/{model_name}_{input_size}x{input_size}.onnx"
    )
    opset = export_cfg.get("opset_version", 17)
    dynamic = export_cfg.get("dynamic_batch", True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy = torch.randn(1, 3, input_size, input_size)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    print(f"Exporting ONNX (opset {opset})...")
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        external_data=False,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ ONNX: {output_path} ({size_kb:.1f} KB)")

    # Verify
    if export_cfg.get("verify", False):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)
            input_name = session.get_inputs()[0].name
            pt_out = model(dummy).detach().numpy()
            ort_out = session.run(None, {input_name: dummy.numpy()})[0]
            max_diff = np.abs(pt_out - ort_out).max()
            print(f"  Verify: max diff = {max_diff:.2e} "
                  f"{'✅ PASS' if max_diff < 1e-4 else '⚠️ DIFF'}")
        except ImportError:
            print("  ⚠️ onnxruntime chưa cài. pip install onnxruntime")


# ═══════════════════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="MvANet CLI — Train, Evaluate, Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py info
  python train.py train --arch MvANet --model mvanet_2 --epochs 100
  python train.py train --config configs/default.yaml --lr 0.01
  python train.py eval  --checkpoint runs/mvanet_2/run_01/best.pth
  python train.py export --checkpoint runs/mvanet_2/run_01/best.pth
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Lệnh cần chạy")

    # --- info ---
    sub.add_parser("info", help="Liệt kê tất cả model có sẵn")

    # --- train ---
    p_train = sub.add_parser("train", help="Train model")
    _add_common_args(p_train)
    _add_train_args(p_train)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Đánh giá model trên test/val set")
    _add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, help="Đường dẫn .pth")
    p_eval.add_argument("--visualize", action="store_true", default=None,
                        help="Lưu biểu đồ")

    # --- export ---
    p_export = sub.add_parser("export", help="Export model sang ONNX")
    _add_common_args(p_export)
    p_export.add_argument("--checkpoint", type=str, help="Đường dẫn .pth")
    p_export.add_argument("--output", type=str, help="Đường dẫn file .onnx output")
    p_export.add_argument("--opset-version", type=int, dest="opset_version",
                          help="ONNX opset (default: 17)")
    p_export.add_argument("--dynamic-batch", action="store_true",
                          dest="dynamic_batch", default=None)
    p_export.add_argument("--simplify", action="store_true", default=None)
    p_export.add_argument("--verify", action="store_true", default=None)

    return parser


def _add_common_args(p):
    """Thêm các flag dùng chung cho mọi subcommand."""
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="Đường dẫn file config YAML (default: configs/default.yaml)")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto | cpu | cuda")
    # Model
    p.add_argument("--arch", type=str, help="Architecture folder (MvANet, MvANet_opt, ...)")
    p.add_argument("--model", type=str, help="Variant name (mvanet_1_tiny, mvanet_2, ...)")
    p.add_argument("--num-classes", type=int, dest="num_classes", help="Số lớp phân loại")
    p.add_argument("--in-channels", type=int, dest="in_channels", help="Kênh đầu vào")
    # Data
    p.add_argument("--data-root", type=str, dest="data_root", help="Thư mục dataset")
    p.add_argument("--input-size", type=int, dest="input_size", help="Input resolution")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size (train)")
    p.add_argument("--batch-size-eval", type=int, dest="batch_size_eval")
    p.add_argument("--num-workers", type=int, dest="num_workers")


def _add_train_args(p):
    """Thêm các flag riêng cho train."""
    p.add_argument("--epochs", type=int)
    p.add_argument("--optimizer", type=str, help="sgd | adam | amsgrad")
    p.add_argument("--lr", type=float, help="Learning rate")
    p.add_argument("--momentum", type=float)
    p.add_argument("--weight-decay", type=float, dest="weight_decay")
    p.add_argument("--lr-scheduler", type=str, dest="lr_scheduler",
                   help="plateau | cosine | step")
    p.add_argument("--plateau-factor", type=float, dest="plateau_factor")
    p.add_argument("--plateau-patience", type=int, dest="plateau_patience")
    p.add_argument("--plateau-min-lr", type=float, dest="plateau_min_lr")
    p.add_argument("--cosine-T-max", type=int, dest="cosine_T_max")
    p.add_argument("--cosine-eta-min", type=float, dest="cosine_eta_min")
    p.add_argument("--step-milestones", type=int, nargs="+", dest="step_milestones")
    p.add_argument("--step-gamma", type=float, dest="step_gamma")
    p.add_argument("--use-class-weights", action="store_true",
                   dest="use_class_weights", default=None)
    p.add_argument("--no-class-weights", action="store_false",
                   dest="use_class_weights")
    p.add_argument("--num-runs", type=int, dest="num_runs")
    p.add_argument("--seed", type=int)
    p.add_argument("--save-dir", type=str, dest="save_dir")
    p.add_argument("--save-every", type=int, dest="save_every")
    p.add_argument("--resume", type=str, help="Đường dẫn checkpoint để resume")
    # eval flag cũng cần cho train (eval sau train)
    p.add_argument("--checkpoint", type=str,
                   help="Alias cho --resume (backward compat)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load config
    config_path = getattr(args, "config", "configs/default.yaml")
    if os.path.isfile(config_path):
        cfg = load_config(config_path)
        print(f"Config: {config_path}")
    else:
        print(f"⚠️ Config file '{config_path}' không tìm thấy, dùng default.")
        cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))

    # Merge CLI overrides
    cfg = merge_cli_into_config(cfg, args)

    # Handle --checkpoint alias for train resume
    if args.command == "train" and getattr(args, "checkpoint", None):
        if not cfg["train"].get("resume"):
            cfg["train"]["resume"] = args.checkpoint

    # Dispatch
    commands = {
        "info": cmd_info,
        "train": cmd_train,
        "eval": cmd_eval,
        "export": cmd_export,
    }
    commands[args.command](cfg, args)


if __name__ == "__main__":
    main()
