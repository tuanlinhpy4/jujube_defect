"""
Jujube Dataset Loader
=====================
Expects the following directory structure:

    dataset/
    ├── train/
    │   ├── invalid/
    │   ├── rotten/
    │   ├── wizened/
    │   └── normal/
    └── test/
        ├── invalid/
        ├── rotten/
        ├── wizened/
        └── normal/

Each subfolder contains .jpg/.png images of that class.
Supports train/val split, augmentation, and the paper's normalization.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms, datasets


# Default class names (matching paper)
DEFAULT_CLASSES = ["invalid", "rotten", "wizened", "normal"]


def get_train_transforms(input_size: int = 32,
                         mean: Tuple = (0.5, 0.5, 0.5),
                         std: Tuple = (0.5, 0.5, 0.5),
                         rotation: int = 15,
                         color_jitter: float = 0.1,
                         horizontal_flip: bool = True,
                         vertical_flip: bool = False) -> transforms.Compose:
    """Training transforms with augmentation (paper: mirroring + rotating)."""
    t = [transforms.Resize((input_size, input_size))]

    if horizontal_flip:
        t.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        t.append(transforms.RandomVerticalFlip())
    if rotation > 0:
        t.append(transforms.RandomRotation(rotation))
    if color_jitter > 0:
        t.append(transforms.ColorJitter(
            brightness=color_jitter, contrast=color_jitter,
            saturation=color_jitter, hue=color_jitter * 0.5
        ))

    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(t)


def get_eval_transforms(input_size: int = 32,
                        mean: Tuple = (0.5, 0.5, 0.5),
                        std: Tuple = (0.5, 0.5, 0.5)) -> transforms.Compose:
    """Evaluation transforms (resize + normalize only)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def create_dataloaders(
    root: str,
    input_size: int = 32,
    batch_size_train: int = 64,
    batch_size_eval: int = 5,
    val_size: int = 1280,
    num_workers: int = 4,
    pin_memory: bool = True,
    mean: Tuple = (0.5, 0.5, 0.5),
    std: Tuple = (0.5, 0.5, 0.5),
    seed: int = 42,
    **aug_kwargs,
) -> Dict[str, DataLoader]:
    """Create train, val, and test dataloaders.

    Args:
        root:             dataset root with train/ and test/ subdirs
        input_size:       image resize resolution (32 or 96)
        batch_size_train: training batch size (paper: 64)
        batch_size_eval:  eval batch size (paper: 5 — one jujube = 5 images)
        val_size:         number of training images held out for validation
                          (paper: 1280)
        num_workers:      dataloader workers
        pin_memory:       pin memory for CUDA
        mean, std:        normalization parameters
        seed:             random seed for val split
        **aug_kwargs:     passed to get_train_transforms

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'class_names' list
    """
    root = Path(root)
    train_dir = root / "train"
    test_dir = root / "test"

    assert train_dir.exists(), f"Training directory not found: {train_dir}"
    assert test_dir.exists(), f"Test directory not found: {test_dir}"

    # Transforms
    train_tf = get_train_transforms(input_size, mean, std, **aug_kwargs)
    eval_tf = get_eval_transforms(input_size, mean, std)

    # Datasets
    full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    class_names = full_train.classes

    # Train/Val split
    n_train = len(full_train) - val_size
    assert n_train > 0, f"val_size ({val_size}) >= dataset size ({len(full_train)})"

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset_raw = random_split(
        full_train, [n_train, val_size], generator=generator
    )

    # Val dataset uses eval transforms (no augmentation)
    val_dataset = ValSubset(full_train, val_dataset_raw.indices, eval_tf)

    # Dataloaders
    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size_eval, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset, batch_size=batch_size_eval, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "class_names": class_names,
        "num_classes": len(class_names),
    }

    print(f"Dataset loaded from: {root}")
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)}")

    return loaders


class ValSubset(Dataset):
    """Subset with different transform (for validation without augmentation)."""

    def __init__(self, dataset: datasets.ImageFolder,
                 indices: List[int], transform: transforms.Compose):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.dataset.samples[real_idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(dataloader: DataLoader,
                          num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = torch.zeros(num_classes)
    for _, labels in dataloader:
        for label in labels:
            counts[label] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights
