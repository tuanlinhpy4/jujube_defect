#!/usr/bin/env python3
"""Unified CLI for MvANet training, evaluation, and ONNX export.

The CLI is intentionally config-first:

    python train.py train --config configs/default.yaml --data-root ./dataset
    python train.py eval --checkpoint runs/mvanet_3/run_01/best.pth
    python train.py export --checkpoint runs/mvanet_3/run_01/best.pth --verify

Every CLI flag is optional unless the subcommand needs an external artifact
such as a checkpoint path. CLI values override YAML values.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles may default to cp1252. PyTorch's ONNX exporter can print
    # Unicode status symbols, so replace unsupported characters instead of
    # crashing the export command.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import (
    create_dataloaders,
    create_eval_dataloaders,
    compute_class_weights,
)
from models import build_model, registry
from utils import (
    MetricTracker,
    MultiRunLogger,
    TrainingLogger,
    collect_environment,
    get_rng_state,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_training_curves,
    seed_everything,
    set_rng_state,
)

FIVE_SEED_PROTOCOL = [42, 123, 456, 789, 2024]


# ---------------------------------------------------------------------------
# Config loading and CLI overrides
# ---------------------------------------------------------------------------


def _flat(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested config dicts for simple CLI override mapping."""

    out: Dict[str, Any] = {}
    for key, value in d.items():
        flat_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flat(value, flat_key, sep))
        else:
            out[flat_key] = value
    return out


def _unflatten(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflatten ``{"a.b": 1}`` back to ``{"a": {"b": 1}}``."""

    out: Dict[str, Any] = {}
    for key, value in d.items():
        node = out
        parts = key.split(sep)
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config with UTF-8 encoding."""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_cli_into_config(cfg: Dict[str, Any],
                          cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Override YAML config with CLI args that were explicitly provided."""

    flat_cfg = _flat(cfg)
    cli_map = {
        # model
        "arch": "model.arch",
        "model": "model.name",
        "num_classes": "model.num_classes",
        "in_channels": "model.in_channels",
        # data
        "data_root": "data.root",
        "input_size": "data.input_size",
        "batch_size": "data.batch_size_train",
        "batch_size_eval": "data.batch_size_eval",
        "num_workers": "data.num_workers",
        "val_size": "data.val_size",
        "pin_memory": "data.pin_memory",
        "drop_last_train": "data.drop_last_train",
        "rotation": "data.augmentation.rotation",
        "color_jitter": "data.augmentation.color_jitter",
        "horizontal_flip": "data.augmentation.horizontal_flip",
        "vertical_flip": "data.augmentation.vertical_flip",
        # train
        "epochs": "train.epochs",
        "optimizer": "train.optimizer",
        "lr": "train.lr",
        "momentum": "train.momentum",
        "weight_decay": "train.weight_decay",
        "lr_scheduler": "train.lr_scheduler",
        "plateau_factor": "train.plateau_factor",
        "plateau_patience": "train.plateau_patience",
        "plateau_min_lr": "train.plateau_min_lr",
        "cosine_T_max": "train.cosine_T_max",
        "cosine_eta_min": "train.cosine_eta_min",
        "step_milestones": "train.step_milestones",
        "step_gamma": "train.step_gamma",
        "use_class_weights": "train.use_class_weights",
        "num_runs": "train.num_runs",
        "seed": "train.seed",
        "seed_mode": "train.seed_mode",
        "seeds": "train.seeds",
        "deterministic": "train.deterministic",
        "save_dir": "train.save_dir",
        "save_every": "train.save_every",
        "resume": "train.resume",
        # eval/export checkpoints are command-specific and handled below.
        "visualize": "eval.visualize",
        "output_dir": "eval.output_dir",
        # export
        "output": "export.output",
        "opset_version": "export.opset_version",
        "dynamic_batch": "export.dynamic_batch",
        "simplify": "export.simplify",
        "verify": "export.verify",
    }

    cli_dict = vars(cli_args)
    for cli_key, cfg_key in cli_map.items():
        value = cli_dict.get(cli_key)
        if value is not None:
            flat_cfg[cfg_key] = value

    checkpoint = cli_dict.get("checkpoint")
    if checkpoint is not None:
        command = getattr(cli_args, "command", None)
        if command == "export":
            flat_cfg["export.checkpoint"] = checkpoint
        elif command == "eval":
            flat_cfg["eval.checkpoint"] = checkpoint

    return _unflatten(flat_cfg)


def _strip_private(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove runtime-only keys before persisting config."""

    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


def resolve_run_seeds(train_cfg: Dict[str, Any]) -> list[int]:
    """Resolve the exact seed protocol for a train command.

    Supported modes:
    - single: one run with seed 42 by default
    - five: fixed 5-seed protocol [42, 123, 456, 789, 2024]
    - range: backward-compatible base_seed + run index
    - custom: explicit train.seeds / --seeds list
    """

    explicit = train_cfg.get("seeds")
    if explicit:
        return [int(seed) for seed in explicit]

    mode = str(train_cfg.get("seed_mode", "single")).lower().replace("-", "_")
    num_runs = int(train_cfg.get("num_runs", 1))
    if mode in ("five", "5", "5_seed", "5_seeds", "five_seed", "five_seeds"):
        return list(FIVE_SEED_PROTOCOL)
    if mode == "single":
        if num_runs == 5:
            return list(FIVE_SEED_PROTOCOL)
        if num_runs != 1:
            raise ValueError(
                "seed_mode=single supports only num_runs=1, except num_runs=5 "
                "which selects the fixed five-seed protocol. Use "
                "--seed-mode range or explicit --seeds for other repeat counts."
            )
        return [int(train_cfg.get("seed", 42))]
    if mode == "range":
        base_seed = int(train_cfg.get("seed", 42))
        return [base_seed + idx for idx in range(num_runs)]
    if mode == "custom":
        raise ValueError("seed_mode=custom requires train.seeds or --seeds.")
    raise ValueError("seed_mode must be one of: single, five, range, custom")


def _jsonable(value: Any) -> Any:
    """Convert tensors/NumPy values to JSON-serializable Python objects."""

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def write_yaml(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def get_device(hint: str = "auto") -> torch.device:
    if hint == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(hint)


def build_dataloaders_from_config(cfg: Dict[str, Any], seed: int,
                                  device: torch.device) -> Dict[str, Any]:
    """Create DataLoaders from config and print a compact split summary."""

    data_cfg = cfg["data"]
    augmentation = data_cfg.get("augmentation", {}) or {}
    pin_cfg = data_cfg.get("pin_memory", None)
    pin_memory = (device.type == "cuda") if pin_cfg is None else bool(pin_cfg)

    loaders = create_dataloaders(
        root=data_cfg["root"],
        input_size=int(data_cfg.get("input_size", 32)),
        batch_size_train=int(data_cfg.get("batch_size_train", 64)),
        batch_size_eval=int(data_cfg.get("batch_size_eval", 5)),
        val_size=data_cfg.get("val_size", 1280),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=pin_memory,
        mean=data_cfg.get("normalize_mean"),
        std=data_cfg.get("normalize_std"),
        seed=int(seed),
        drop_last_train=bool(data_cfg.get("drop_last_train", False)),
        rotation=int(augmentation.get("rotation", 15)),
        color_jitter=float(augmentation.get("color_jitter", 0.0)),
        horizontal_flip=bool(augmentation.get("horizontal_flip", True)),
        vertical_flip=bool(augmentation.get("vertical_flip", False)),
    )

    print(f"Dataset: {data_cfg['root']} | split={loaders['split_info']}")
    for split in ("train", "val", "test"):
        info = loaders["summary"]["splits"].get(split)
        if info:
            batch_count = len(loaders[split])
            print(f"  {split}: {info['num_images']} images, {batch_count} batches")
    print(f"  classes: {loaders['class_names']}")
    return loaders


def build_eval_dataloaders_from_config(
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
    class_order=None,
) -> Dict[str, Any]:
    """Create eval-only DataLoaders from config and checkpoint class order."""

    data_cfg = cfg["data"]
    pin_cfg = data_cfg.get("pin_memory", None)
    pin_memory = (device.type == "cuda") if pin_cfg is None else bool(pin_cfg)

    loaders = create_eval_dataloaders(
        root=data_cfg["root"],
        input_size=int(data_cfg.get("input_size", 32)),
        batch_size_eval=int(data_cfg.get("batch_size_eval", 5)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=pin_memory,
        mean=data_cfg.get("normalize_mean"),
        std=data_cfg.get("normalize_std"),
        seed=int(seed),
        class_order=class_order,
    )

    print(f"Dataset: {data_cfg['root']} | split={loaders['split_info']}")
    for split in ("test", "val"):
        info = loaders["summary"]["splits"].get(split)
        if info:
            batch_count = len(loaders[split])
            print(f"  {split}: {info['num_images']} images, {batch_count} batches")
    print(f"  classes: {loaders['class_names']}")
    return loaders


def train_one_epoch(model: nn.Module, loader, criterion, optimizer,
                    device: torch.device, tracker: MetricTracker) -> None:
    model.train()
    non_blocking = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        tracker.update(preds, labels, loss.item(), images.size(0))


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device,
             tracker: MetricTracker) -> None:
    model.eval()
    non_blocking = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        tracker.update(preds, labels, loss.item(), images.size(0))


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    train_cfg = cfg["train"]
    opt_name = str(train_cfg.get("optimizer", "sgd")).lower()
    lr = float(train_cfg.get("lr", 0.1))
    wd = float(train_cfg.get("weight_decay", 0.0))

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if opt_name in ("adam", "amsgrad"):
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            amsgrad=(opt_name == "amsgrad"),
        )
    raise ValueError("optimizer must be one of: sgd, adam, amsgrad")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    train_cfg = cfg["train"]
    sched_name = str(train_cfg.get("lr_scheduler", "step")).lower()

    if sched_name in ("none", "constant", "off"):
        return None
    if sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(train_cfg.get("plateau_mode", "min")),
            factor=float(train_cfg.get("plateau_factor", 0.5)),
            patience=int(train_cfg.get("plateau_patience", 15)),
            min_lr=float(train_cfg.get("plateau_min_lr", 1e-5)),
        )
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("cosine_T_max", train_cfg.get("epochs", 300))),
            eta_min=float(train_cfg.get("cosine_eta_min", 1e-5)),
        )
    if sched_name == "step":
        milestones = train_cfg.get("step_milestones")
        if not milestones:
            epochs = int(train_cfg.get("epochs", 300))
            milestones = [round(epochs * 0.4), round(epochs * 0.7),
                          round(epochs * 0.9)]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(x) for x in milestones],
            gamma=float(train_cfg.get("step_gamma", 0.1)),
        )
    raise ValueError("lr_scheduler must be one of: step, plateau, cosine, none")


def scheduler_step(scheduler, cfg: Dict[str, Any], val_loss: float) -> None:
    if scheduler is None:
        return
    sched_name = str(cfg["train"].get("lr_scheduler", "step")).lower()
    if sched_name == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, Any],
    save_path: str,
    cfg: Dict[str, Any],
    logger: TrainingLogger,
    train_generator: Optional[torch.Generator],
) -> None:
    """Save model state plus config and RNG state for deterministic resume."""

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": _jsonable(metrics),
        "best_val_error": float(logger.best_error),
        "best_epoch": int(logger.best_epoch),
        "config": _strip_private(cfg),
        "rng_state": get_rng_state(train_generator),
    }
    torch.save(payload, save_path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load a trusted training checkpoint.

    ``weights_only=False`` is intentional because our checkpoints contain
    optimizer, scheduler, config, and RNG metadata in addition to tensors.
    """

    return torch.load(path, map_location=device, weights_only=False)


def _checkpoint_model_fields(ckpt_cfg: Dict[str, Any],
                             fallback_cfg: Dict[str, Any],
                             args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve model settings from CLI, checkpoint, then YAML fallback."""

    ckpt_model = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    ckpt_data = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else {}
    return {
        "arch": getattr(args, "arch", None)
        or ckpt_model.get("arch")
        or fallback_cfg["model"]["arch"],
        "name": getattr(args, "model", None)
        or ckpt_model.get("name")
        or fallback_cfg["model"]["name"],
        "num_classes": getattr(args, "num_classes", None)
        or ckpt_model.get("num_classes")
        or fallback_cfg["model"]["num_classes"],
        "in_channels": getattr(args, "in_channels", None)
        or ckpt_model.get("in_channels")
        or fallback_cfg["model"].get("in_channels", 3),
        "input_size": getattr(args, "input_size", None)
        or ckpt_data.get("input_size")
        or fallback_cfg["data"]["input_size"],
        "classes": ckpt_data.get("classes", []),
        "normalize_mean": ckpt_data.get(
            "normalize_mean", fallback_cfg["data"].get("normalize_mean")
        ),
        "normalize_std": ckpt_data.get(
            "normalize_std", fallback_cfg["data"].get("normalize_std")
        ),
    }


def evaluate_and_save(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    class_names,
    num_classes: int,
    output_dir: str,
    prefix: str,
    visualize: bool,
) -> Dict[str, Any]:
    """Evaluate one split and persist metrics/plots."""

    tracker = MetricTracker(num_classes, class_names)
    evaluate(model, loader, criterion, device, tracker)
    tracker.print_report()

    if visualize:
        plot_confusion_matrix(
            tracker.confusion_matrix(),
            class_names,
            os.path.join(output_dir, f"{prefix}_confusion_matrix.png"),
        )
        plot_confusion_matrix(
            tracker.confusion_matrix(),
            class_names,
            os.path.join(output_dir, f"{prefix}_confusion_matrix_norm.png"),
            normalize=True,
        )
        plot_per_class_metrics(
            tracker.per_class_metrics(),
            os.path.join(output_dir, f"{prefix}_per_class_metrics.png"),
        )

    return {
        "summary": tracker.summary(),
        "per_class": tracker.per_class_metrics(),
        "confusion_matrix": tracker.confusion_matrix(),
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_info(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print available model architectures and variants."""

    print("\nMvANet available models")
    print("=" * 60)
    for arch, variants in registry.list_all().items():
        print(f"{arch}/")
        for variant in variants:
            model = registry.build(arch, variant, num_classes=4)
            params = model.count_parameters()
            print(f"  {variant:<16} {params:>10,} params ({params * 4 / 1024:.1f} KB)")
        print()


def cmd_train(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Train one or more seeded runs and evaluate each best checkpoint."""

    device = get_device(getattr(args, "device", "auto"))
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    arch = registry.canonical_name(model_cfg["arch"])
    model_cfg["arch"] = arch
    model_name = model_cfg["name"]
    num_classes = int(model_cfg["num_classes"])
    in_channels = int(model_cfg.get("in_channels", 3))
    epochs = int(train_cfg["epochs"])
    run_seeds = resolve_run_seeds(train_cfg)
    num_runs = len(run_seeds)
    deterministic = bool(train_cfg.get("deterministic", True))
    save_dir_root = train_cfg.get("save_dir", "./runs")
    save_every = int(train_cfg.get("save_every", 50))
    resume_path = train_cfg.get("resume")

    # A resumed checkpoint is a continuation of one stochastic trajectory.
    # Applying it to multiple seeds would corrupt the independent-run protocol.
    if resume_path and num_runs != 1:
        raise ValueError(
            "Resume is only supported for one seed/run at a time. "
            "Do not combine --resume with --seed-mode five, multiple --seeds, "
            "or --num-runs > 1."
        )

    print("\n" + "=" * 60)
    print("MvANet Training")
    print(f"  Arch/Model: {arch}/{model_name}")
    print(f"  Epochs/Runs: {epochs}/{num_runs}")
    print(f"  Device: {device}")
    print(f"  Seeds: {run_seeds} | deterministic={deterministic}")
    print(f"  Optimizer: {train_cfg.get('optimizer')} | LR: {train_cfg.get('lr')}")
    print(f"  Scheduler: {train_cfg.get('lr_scheduler')}")
    print("=" * 60)

    multi_logger = MultiRunLogger()
    summary_dir = os.path.join(save_dir_root, model_name)
    os.makedirs(summary_dir, exist_ok=True)

    for run_idx, seed in enumerate(run_seeds):
        run_id = run_idx + 1
        seed_everything(seed, deterministic=deterministic)
        resume_ckpt: Optional[Dict[str, Any]] = None

        if resume_path:
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            resume_ckpt = load_checkpoint(resume_path, device)
            # The DataLoader split, shuffling order, model initialization, and
            # augmentation stream are all tied to the saved seed.
            ckpt_seed = resume_ckpt.get("config", {}).get("train", {}).get("seed")
            if ckpt_seed is not None and int(ckpt_seed) != int(seed):
                raise ValueError(
                    f"Resume seed mismatch: checkpoint seed={ckpt_seed}, "
                    f"current seed={seed}. Pass --seed {ckpt_seed} or use the "
                    "original run config."
                )

        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("train", {})
        run_cfg["train"]["seed"] = seed
        run_cfg["train"]["run_id"] = run_id
        run_cfg["train"]["resolved_seeds"] = list(run_seeds)
        run_cfg["train"]["num_runs"] = num_runs

        run_name = f"run_{run_id:02d}"
        save_dir = os.path.join(save_dir_root, model_name, run_name)
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"Run {run_id}/{num_runs} | seed={seed} | save={save_dir}")
        print("=" * 60)

        loaders = build_dataloaders_from_config(run_cfg, seed, device)
        class_names = loaders["class_names"]
        actual_num_classes = int(loaders["num_classes"])
        if actual_num_classes != num_classes:
            raise ValueError(
                f"Config num_classes={num_classes}, but dataset has "
                f"{actual_num_classes} classes: {class_names}"
            )
        if "val" not in loaders:
            raise ValueError(
                "Training requires a validation split. Add dataset/val or set "
                "data.val_size > 0 to hold out validation data from train/."
            )
        if "test" not in loaders:
            raise ValueError(
                "Training requires dataset/test because train mode automatically "
                "evaluates the best checkpoint on the test split."
            )

        run_cfg["_class_names"] = list(class_names)
        run_cfg["model"]["in_channels"] = in_channels
        run_cfg["data"]["classes"] = list(class_names)

        write_yaml(os.path.join(save_dir, "config.yaml"), _strip_private(run_cfg))
        write_json(os.path.join(save_dir, "environment.json"), collect_environment())
        write_json(os.path.join(save_dir, "dataset_summary.json"), loaders["summary"])

        model = build_model(
            arch=arch,
            name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
        ).to(device)
        print(f"Model parameters: {model.count_parameters():,}")

        optimizer = build_optimizer(model, run_cfg)
        scheduler = build_scheduler(optimizer, run_cfg)

        class_weights = None
        if train_cfg.get("use_class_weights", False):
            class_weights = compute_class_weights(
                loaders["train_dataset"], num_classes, device
            )
            print(f"Class weights: {[round(x, 6) for x in class_weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        logger = TrainingLogger(save_dir, run_name)
        start_epoch = 1
        last_val_summary: Dict[str, Any] = {}

        if resume_path:
            print(f"Resuming from: {resume_path}")
            if resume_ckpt is None:
                raise RuntimeError("Internal error: resume checkpoint was not loaded.")
            ckpt = resume_ckpt
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            set_rng_state(ckpt.get("rng_state"), loaders.get("train_generator"))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            logger.best_error = float(ckpt.get("best_val_error", logger.best_error))
            logger.best_epoch = int(ckpt.get("best_epoch", 0))
            last_val_summary = ckpt.get("metrics", {})
            print(f"Resumed at epoch {start_epoch}")

        val_loader = loaders.get("val", loaders.get("test"))
        train_generator = loaders.get("train_generator")

        for epoch in range(start_epoch, epochs + 1):
            train_tracker = MetricTracker(num_classes, class_names)
            train_one_epoch(
                model, loaders["train"], criterion, optimizer, device, train_tracker
            )

            val_tracker = MetricTracker(num_classes, class_names)
            evaluate(model, val_loader, criterion, device, val_tracker)
            last_val_summary = val_tracker.summary()

            lr = float(optimizer.param_groups[0]["lr"])
            is_best = logger.log_epoch(
                epoch, train_tracker.summary(), last_val_summary, lr
            )

            scheduler_step(scheduler, run_cfg, float(last_val_summary["loss"]))

            if is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    last_val_summary,
                    os.path.join(save_dir, "best.pth"),
                    run_cfg,
                    logger,
                    train_generator,
                )

            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    last_val_summary,
                    os.path.join(save_dir, f"epoch_{epoch}.pth"),
                    run_cfg,
                    logger,
                    train_generator,
                )

        final_epoch = max(start_epoch - 1, epochs)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            final_epoch,
            last_val_summary,
            os.path.join(save_dir, "last.pth"),
            run_cfg,
            logger,
            train_generator,
        )

        logger.save_history()
        logger.print_summary()
        plot_training_curves(
            logger.get_history(), os.path.join(save_dir, "training_curves.png")
        )

        test_error: Optional[float] = None
        if "test" in loaders:
            best_path = os.path.join(save_dir, "best.pth")
            if os.path.exists(best_path):
                best_ckpt = load_checkpoint(best_path, device)
                model.load_state_dict(best_ckpt["model_state_dict"])

            print("\nEvaluating best checkpoint on test split...")
            test_result = evaluate_and_save(
                model,
                loaders["test"],
                criterion,
                device,
                class_names,
                num_classes,
                save_dir,
                "test",
                visualize=bool(run_cfg.get("eval", {}).get("visualize", True)),
            )
            test_error = float(test_result["summary"]["error_rate"])

            results = {
                "arch": arch,
                "model": model_name,
                "input_size": run_cfg["data"]["input_size"],
                "epochs": epochs,
                "seed": seed,
                "num_classes": num_classes,
                "classes": list(class_names),
                "best_val_error": logger.best_error,
                "best_epoch": logger.best_epoch,
                "test": test_result,
            }
            write_json(os.path.join(save_dir, "results.json"), results)

        final_val_error = float(
            last_val_summary.get("error_rate", logger.best_error)
        )
        multi_logger.add_run(
            run_id=run_id,
            seed=seed,
            best_val_error=logger.best_error,
            final_val_error=final_val_error,
            test_error=test_error,
        )

    multi_path = os.path.join(summary_dir, "multirun_summary.json")
    multi_logger.save(multi_path)
    multi_logger.print_summary()
    print(f"\nTraining complete. Multi-run summary: {multi_path}")


def cmd_eval(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on test and/or val splits."""

    device = get_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("train", {}).get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    ckpt_path = cfg.get("eval", {}).get("checkpoint") or getattr(args, "checkpoint", None)
    if not ckpt_path:
        raise ValueError("eval requires --checkpoint <path>")

    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_cfg = ckpt.get("config", {})
    fields = _checkpoint_model_fields(ckpt_cfg, cfg, args)

    cfg["model"]["arch"] = fields["arch"]
    cfg["model"]["name"] = fields["name"]
    cfg["model"]["num_classes"] = int(fields["num_classes"])
    cfg["model"]["in_channels"] = int(fields["in_channels"])
    cfg["data"]["input_size"] = int(fields["input_size"])
    cfg["data"]["normalize_mean"] = fields["normalize_mean"]
    cfg["data"]["normalize_std"] = fields["normalize_std"]

    model = build_model(
        arch=fields["arch"],
        name=fields["name"],
        num_classes=int(fields["num_classes"]),
        in_channels=int(fields["in_channels"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = build_eval_dataloaders_from_config(
        cfg, seed, device, fields["classes"] or None
    )
    class_names = fields["classes"] or loaders["class_names"]
    if fields["classes"] and list(fields["classes"]) != list(loaders["class_names"]):
        raise ValueError(
            "Checkpoint class order does not match dataset class order: "
            f"checkpoint={fields['classes']} dataset={loaders['class_names']}"
        )

    criterion = nn.CrossEntropyLoss()
    output_dir = cfg.get("eval", {}).get("output_dir") or os.path.dirname(ckpt_path) or "."
    visualize = bool(cfg.get("eval", {}).get("visualize", True))

    print(f"Checkpoint: {ckpt_path}")
    print(f"Model: {fields['arch']}/{fields['name']} | Params: {model.count_parameters():,}")

    results: Dict[str, Any] = {
        "checkpoint": ckpt_path,
        "arch": fields["arch"],
        "model": fields["name"],
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
            int(fields["num_classes"]),
            output_dir,
            "eval_test",
            visualize,
        )

    if "val" in loaders:
        print("\nEvaluating val split...")
        results["val"] = evaluate_and_save(
            model,
            loaders["val"],
            criterion,
            device,
            class_names,
            int(fields["num_classes"]),
            output_dir,
            "eval_val",
            visualize,
        )

    results_path = os.path.join(output_dir, "eval_results.json")
    write_json(results_path, results)
    print(f"\nEvaluation results: {results_path}")


def _simplify_onnx(path: str) -> str:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("onnxsim is not installed; keeping original ONNX file.")
        return path

    model = onnx.load(path)
    simplified, ok = simplify(model)
    if not ok:
        print("ONNX simplification check failed; keeping original ONNX file.")
        return path
    simplified_path = path.replace(".onnx", "_simplified.onnx")
    onnx.save(simplified, simplified_path)
    return simplified_path


def cmd_export(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Export a checkpoint to ONNX and optionally verify with onnxruntime."""

    ckpt_path = (
        cfg.get("export", {}).get("checkpoint")
        or cfg.get("eval", {}).get("checkpoint")
        or getattr(args, "checkpoint", None)
    )
    if not ckpt_path:
        raise ValueError("export requires --checkpoint <path>")

    device = torch.device("cpu")
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_cfg = ckpt.get("config", {})
    fields = _checkpoint_model_fields(ckpt_cfg, cfg, args)

    model = build_model(
        arch=fields["arch"],
        name=fields["name"],
        num_classes=int(fields["num_classes"]),
        in_channels=int(fields["in_channels"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    export_cfg = cfg.get("export", {})
    input_size = int(fields["input_size"])
    output_path = export_cfg.get("output") or (
        f"exports/{fields['name']}_{input_size}x{input_size}.onnx"
    )
    opset = int(export_cfg.get("opset_version", 18))
    dynamic = bool(export_cfg.get("dynamic_batch", True))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy = torch.randn(1, int(fields["in_channels"]), input_size, input_size)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    print(f"Exporting {fields['arch']}/{fields['name']} to {output_path}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                model,
                dummy,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
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

    final_path = output_path
    if export_cfg.get("simplify", False):
        final_path = _simplify_onnx(output_path)

    size_kb = os.path.getsize(final_path) / 1024
    print(f"ONNX saved: {final_path} ({size_kb:.1f} KB)")

    if export_cfg.get("verify", False):
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime is not installed; skipping verification.")
            return

        session = ort.InferenceSession(final_path)
        input_name = session.get_inputs()[0].name
        with torch.no_grad():
            pt_out = model(dummy).detach().numpy()
        ort_out = session.run(None, {input_name: dummy.numpy()})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        print(f"ONNX verify max abs diff: {max_diff:.3e}")
        if max_diff >= 1e-4:
            raise RuntimeError("ONNX verification failed: max diff >= 1e-4")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="MvANet CLI: train, evaluate, and export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py info\n"
            "  python train.py train --data-root ./dataset\n"
            "  python train.py eval --checkpoint runs/mvanet_3/run_01/best.pth\n"
            "  python train.py export --checkpoint runs/mvanet_3/run_01/best.pth --verify\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", help="subcommand")
    sub.add_parser("info", help="list registered architectures and variants")

    p_train = sub.add_parser("train", help="train a model")
    _add_common_args(p_train)
    _add_train_args(p_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint")
    _add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, help="checkpoint .pth path")
    p_eval.add_argument("--visualize", action="store_true", default=None)
    p_eval.add_argument("--no-visualize", action="store_false",
                        dest="visualize")
    p_eval.add_argument("--output-dir", type=str, dest="output_dir")

    p_export = sub.add_parser("export", help="export a checkpoint to ONNX")
    _add_common_args(p_export)
    p_export.add_argument("--checkpoint", type=str, help="checkpoint .pth path")
    p_export.add_argument("--output", type=str, help="output .onnx path")
    p_export.add_argument("--opset-version", type=int, dest="opset_version")
    p_export.add_argument("--dynamic-batch", action="store_true",
                          dest="dynamic_batch", default=None)
    p_export.add_argument("--static-batch", action="store_false",
                          dest="dynamic_batch")
    p_export.add_argument("--simplify", action="store_true", default=None)
    p_export.add_argument("--verify", action="store_true", default=None)

    return parser


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="YAML config path")
    p.add_argument("--device", type=str, default="auto",
                   help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--arch", type=str, help="architecture folder name")
    p.add_argument("--model", type=str, help="model variant name")
    p.add_argument("--num-classes", type=int, dest="num_classes")
    p.add_argument("--in-channels", type=int, dest="in_channels")

    p.add_argument("--data-root", type=str, dest="data_root")
    p.add_argument("--input-size", type=int, dest="input_size")
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--batch-size-eval", type=int, dest="batch_size_eval")
    p.add_argument("--num-workers", type=int, dest="num_workers")
    p.add_argument("--val-size", type=float, dest="val_size",
                   help="validation holdout count or fraction when no val/ dir exists")
    p.add_argument("--pin-memory", action="store_true", dest="pin_memory",
                   default=None)
    p.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    p.add_argument("--drop-last-train", action="store_true",
                   dest="drop_last_train", default=None)
    p.add_argument("--no-drop-last-train", action="store_false",
                   dest="drop_last_train")
    p.add_argument("--rotation", type=int)
    p.add_argument("--color-jitter", type=float, dest="color_jitter")
    p.add_argument("--horizontal-flip", action="store_true",
                   dest="horizontal_flip", default=None)
    p.add_argument("--no-horizontal-flip", action="store_false",
                   dest="horizontal_flip")
    p.add_argument("--vertical-flip", action="store_true",
                   dest="vertical_flip", default=None)
    p.add_argument("--no-vertical-flip", action="store_false",
                   dest="vertical_flip")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int)
    p.add_argument("--optimizer", type=str, help="sgd, adam, amsgrad")
    p.add_argument("--lr", type=float)
    p.add_argument("--momentum", type=float)
    p.add_argument("--weight-decay", type=float, dest="weight_decay")
    p.add_argument("--lr-scheduler", type=str, dest="lr_scheduler",
                   help="step, plateau, cosine, none")
    p.add_argument("--plateau-factor", type=float, dest="plateau_factor")
    p.add_argument("--plateau-patience", type=int, dest="plateau_patience")
    p.add_argument("--plateau-min-lr", type=float, dest="plateau_min_lr")
    p.add_argument("--cosine-T-max", type=int, dest="cosine_T_max")
    p.add_argument("--cosine-eta-min", type=float, dest="cosine_eta_min")
    p.add_argument("--step-milestones", type=int, nargs="+",
                   dest="step_milestones")
    p.add_argument("--step-gamma", type=float, dest="step_gamma")
    p.add_argument("--use-class-weights", action="store_true",
                   dest="use_class_weights", default=None)
    p.add_argument("--no-class-weights", action="store_false",
                   dest="use_class_weights")
    p.add_argument("--num-runs", type=int, dest="num_runs")
    p.add_argument("--seed", type=int)
    p.add_argument("--seed-mode", type=str, dest="seed_mode",
                   choices=["single", "five", "range", "custom"],
                   help="single=[42], five=[42,123,456,789,2024], range=seed+i")
    p.add_argument("--seeds", type=int, nargs="+",
                   help="explicit seed list; implies custom seed protocol")
    p.add_argument("--deterministic", action="store_true", default=None)
    p.add_argument("--non-deterministic", action="store_false",
                   dest="deterministic")
    p.add_argument("--save-dir", type=str, dest="save_dir")
    p.add_argument("--save-every", type=int, dest="save_every")
    p.add_argument("--resume", type=str)
    p.add_argument("--checkpoint", type=str,
                   help="alias for --resume in train mode")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    config_path = getattr(args, "config", "configs/default.yaml")
    if os.path.isfile(config_path):
        cfg = load_config(config_path)
    else:
        fallback = PROJECT_ROOT / "configs" / "default.yaml"
        print(f"Config not found: {config_path}; using {fallback}")
        cfg = load_config(str(fallback))

    cfg = merge_cli_into_config(cfg, args)
    if args.command == "train" and getattr(args, "checkpoint", None):
        cfg.setdefault("train", {})
        if not cfg["train"].get("resume"):
            cfg["train"]["resume"] = args.checkpoint

    commands = {
        "info": cmd_info,
        "train": cmd_train,
        "eval": cmd_eval,
        "export": cmd_export,
    }
    commands[args.command](cfg, args)


if __name__ == "__main__":
    main()
