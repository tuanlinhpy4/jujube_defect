from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.reproducibility import make_generator, seed_worker

SPLIT_SEED = 42


def _stratified_split(targets: List[int], train_ratio: float = 0.70,
                      val_ratio: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    classes = sorted(set(targets))
    train_idx, val_idx, test_idx = [], [], []
    for cls in classes:
        cls_indices = [i for i, t in enumerate(targets) if t == cls]
        cls_indices = list(rng.permutation(cls_indices))
        n = len(cls_indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:])
    return train_idx, val_idx, test_idx


def get_train_transforms(input_size: int, mean: Sequence[float],
                          std: Sequence[float], aug: dict) -> transforms.Compose:
    steps = [
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip() if aug.get("horizontal_flip") else transforms.Lambda(lambda x: x),
    ]
    if aug.get("vertical_flip"):
        steps.append(transforms.RandomVerticalFlip())
    if aug.get("rotation", 0) > 0:
        steps.append(transforms.RandomRotation(aug["rotation"]))
    if aug.get("color_jitter", 0.0) > 0:
        cj = aug["color_jitter"]
        steps.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(cj * 0.5, 0.5)))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=mean, std=std))
    if aug.get("random_erasing", 0.0) > 0:
        steps.append(transforms.RandomErasing(p=aug["random_erasing"]))
    return transforms.Compose(steps)


def get_eval_transforms(input_size: int, mean: Sequence[float],
                         std: Sequence[float]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def create_dataloaders(cfg: dict, split_seed: int = SPLIT_SEED) -> Dict:
    root = Path(cfg["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    input_size = cfg["input_size"]
    mean = cfg["normalize_mean"]
    std = cfg["normalize_std"]
    aug = cfg.get("augmentation", {})
    class_names = cfg["class_names"]
    batch_train = cfg["batch_size_train"]
    batch_eval = cfg["batch_size_eval"]
    num_workers = cfg.get("num_workers", 4)

    train_tf = get_train_transforms(input_size, mean, std, aug)
    eval_tf = get_eval_transforms(input_size, mean, std)

    full_aug = datasets.ImageFolder(str(root))
    full_aug_tf = datasets.ImageFolder(str(root), transform=train_tf)
    full_eval_tf = datasets.ImageFolder(str(root), transform=eval_tf)

    # actual_classes is sorted alphabetically by ImageFolder — this IS the ground-truth
    # label order (ImageFolder assigns label 0 to the first alphabetical folder, etc.).
    # class_names from yaml is used only as a sanity-check, NOT as the label mapping.
    actual_classes = full_aug.classes
    for c in class_names:
        if c not in actual_classes:
            raise ValueError(f"Class '{c}' not found in {root}. Found: {actual_classes}")
    if set(class_names) != set(actual_classes):
        raise ValueError(
            f"yaml class_names {sorted(class_names)} != folders {actual_classes}"
        )

    targets = full_aug.targets
    train_idx, val_idx, test_idx = _stratified_split(targets, seed=split_seed)

    train_ds = Subset(full_aug_tf, train_idx)
    val_ds = Subset(full_eval_tf, val_idx)
    test_ds = Subset(full_eval_tf, test_idx)

    gen = make_generator(split_seed)
    loader_kw = {"num_workers": num_workers, "pin_memory": True,
                 "worker_init_fn": seed_worker, "generator": gen}

    return {
        "train": DataLoader(train_ds, batch_size=batch_train, shuffle=True, drop_last=False, **loader_kw),
        "val": DataLoader(val_ds, batch_size=batch_eval, shuffle=False, **loader_kw),
        "test": DataLoader(test_ds, batch_size=batch_eval, shuffle=False, **loader_kw),
        "class_names": actual_classes,   # alphabetical = ImageFolder label order; yaml order is WRONG
        "num_classes": cfg["num_classes"],
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    }
