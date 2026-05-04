"""Dataset and DataLoader utilities for MvANet experiments.

Expected dataset layout for the original paper setting::

    dataset/
      train/
        invalid/
        rotten/
        wizened/
        normal/
      test/
        invalid/
        rotten/
        wizened/
        normal/

If ``dataset/val`` is present it is used directly. Otherwise ``val_size`` images
are held out from ``dataset/train`` with a seeded split, matching the paper's
reported 1,280-image validation holdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from utils.reproducibility import make_generator, seed_worker


DEFAULT_CLASSES = ["invalid", "rotten", "wizened", "normal"]

MeanStd = Optional[Sequence[float]]


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder with a stable paper-defined class order.

    Torchvision sorts class folders alphabetically by default. For this dataset
    that would be ``invalid, normal, rotten, wizened``, which does not match the
    class order used throughout the paper and README. This subclass keeps known
    jujube classes in ``DEFAULT_CLASSES`` order and appends any extra folders
    alphabetically after them.
    """

    def __init__(self, root: str, transform=None,
                 class_order: Optional[Sequence[str]] = None):
        self.class_order = list(class_order or DEFAULT_CLASSES)
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        root = Path(directory)
        present = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
        ordered = [name for name in self.class_order if name in present]
        extras = [name for name in present if name not in ordered]
        classes = ordered + extras
        if not classes:
            raise FileNotFoundError(f"No class folders found in {directory}")
        return classes, {cls_name: idx for idx, cls_name in enumerate(classes)}


def _normalization_steps(mean: MeanStd, std: MeanStd) -> List[Any]:
    """Return tensor conversion plus optional channel normalization.

    The paper states that images are normalized but does not publish dataset
    mean/std values. Therefore the config can set ``normalize_mean`` and
    ``normalize_std`` to null to use only ToTensor(), which scales pixels to
    [0, 1], or provide explicit values for standardization.
    """

    steps: List[Any] = [transforms.ToTensor()]
    if mean is not None and std is not None:
        steps.append(transforms.Normalize(mean=tuple(mean), std=tuple(std)))
    return steps


def get_train_transforms(
    input_size: int = 32,
    mean: MeanStd = None,
    std: MeanStd = None,
    rotation: int = 15,
    color_jitter: float = 0.0,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
) -> transforms.Compose:
    """Build train-time transforms.

    The paper explicitly reports mirroring and rotation augmentation. Color
    jitter is disabled by default because it is not part of the reported setup,
    but remains configurable for ablations.
    """

    steps: List[Any] = [transforms.Resize((input_size, input_size))]

    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        steps.append(transforms.RandomVerticalFlip())
    if rotation > 0:
        steps.append(transforms.RandomRotation(rotation))
    if color_jitter > 0:
        steps.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(color_jitter * 0.5, 0.5),
            )
        )

    steps.extend(_normalization_steps(mean, std))
    return transforms.Compose(steps)


def get_eval_transforms(
    input_size: int = 32,
    mean: MeanStd = None,
    std: MeanStd = None,
) -> transforms.Compose:
    """Build deterministic validation/test transforms."""

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            *_normalization_steps(mean, std),
        ]
    )


def _resolve_val_size(val_size: Union[int, float], train_len: int) -> int:
    """Convert absolute/fractional validation size to an integer count."""

    if isinstance(val_size, float) and 0.0 < val_size < 1.0:
        resolved = int(round(train_len * val_size))
    else:
        resolved = int(val_size)
    if resolved < 0:
        raise ValueError(f"val_size must be non-negative, got {val_size}")
    if resolved >= train_len and train_len > 0:
        raise ValueError(
            f"val_size ({resolved}) must be smaller than train size ({train_len})"
        )
    return resolved


def _subset(dataset: Dataset, indices: Iterable[int]) -> Subset:
    """Create a Subset with indices materialized for reproducible metadata."""

    return Subset(dataset, list(indices))


def _check_class_map(reference: datasets.ImageFolder,
                     candidate: datasets.ImageFolder,
                     split_name: str) -> None:
    """Fail early when split folders produce different class-to-index maps."""

    if candidate.class_to_idx != reference.class_to_idx:
        raise ValueError(
            f"Class folders in split '{split_name}' do not match train/. "
            f"train={reference.class_to_idx}, {split_name}={candidate.class_to_idx}"
        )


def _check_expected_classes(candidate: datasets.ImageFolder,
                            expected_classes: Sequence[str],
                            split_name: str) -> None:
    """Ensure an eval split preserves the checkpoint class-to-index mapping."""

    expected = list(expected_classes)
    if list(candidate.classes) != expected:
        raise ValueError(
            f"Class folders in split '{split_name}' do not match checkpoint "
            f"class order. expected={expected}, found={candidate.classes}. "
            "Evaluate with the same class folders used by the checkpoint."
        )


def _class_counts_from_targets(targets: Sequence[int],
                               num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for target in targets:
        counts[int(target)] += 1
    return counts


def _targets_from_dataset(dataset: Dataset) -> List[int]:
    """Extract integer labels from ImageFolder or Subset(ImageFolder)."""

    if isinstance(dataset, Subset):
        base_targets = _targets_from_dataset(dataset.dataset)
        return [base_targets[i] for i in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(x) for x in getattr(dataset, "targets")]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in getattr(dataset, "samples")]
    raise TypeError(f"Cannot extract class targets from {type(dataset)!r}")


def summarize_dataset(loaders: Dict[str, Any]) -> Dict[str, Any]:
    """Return split sizes and class counts for run metadata."""

    class_names = loaders.get("class_names", [])
    num_classes = len(class_names)
    summary: Dict[str, Any] = {"class_names": class_names, "splits": {}}
    for split in ("train", "val", "test"):
        dataset = loaders.get(f"{split}_dataset")
        if dataset is None:
            continue
        targets = _targets_from_dataset(dataset)
        summary["splits"][split] = {
            "num_images": len(dataset),
            "class_counts": {
                class_names[i]: count
                for i, count in enumerate(
                    _class_counts_from_targets(targets, num_classes)
                )
            },
        }
    return summary


def compute_class_weights(
    dataset: Dataset,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from the full train dataset.

    The weights are computed from dataset labels, not from DataLoader batches,
    so they do not depend on shuffle order or ``drop_last``.
    """

    targets = _targets_from_dataset(dataset)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for target in targets:
        counts[target] += 1.0

    weights = 1.0 / counts.clamp_min(1.0)
    weights = weights / weights.sum() * num_classes
    return weights.to(device) if device is not None else weights


def _loader_kwargs(num_workers: int, pin_memory: bool,
                   seed: int) -> Tuple[Dict[str, Any], torch.Generator]:
    """Build shared deterministic DataLoader kwargs."""

    loader_generator = make_generator(seed)
    return {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
    }, loader_generator


def create_dataloaders(
    root: str,
    input_size: int = 32,
    batch_size_train: int = 64,
    batch_size_eval: int = 5,
    val_size: Union[int, float] = 1280,
    num_workers: int = 2,
    pin_memory: bool = True,
    mean: MeanStd = None,
    std: MeanStd = None,
    seed: int = 42,
    drop_last_train: bool = False,
    **aug_kwargs: Any,
) -> Dict[str, Any]:
    """Create reproducible train/val/test DataLoaders.

    Returns a dict containing DataLoaders, raw datasets, class names, the
    DataLoader generator, and a dataset summary.
    """

    root_path = Path(root)
    train_dir = root_path / "train"
    val_dir = root_path / "val"
    test_dir = root_path / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    train_tf = get_train_transforms(input_size, mean, std, **aug_kwargs)
    eval_tf = get_eval_transforms(input_size, mean, std)

    full_train_aug = OrderedImageFolder(str(train_dir), transform=train_tf)
    full_train_eval = OrderedImageFolder(str(train_dir), transform=eval_tf)
    class_names = full_train_aug.classes
    num_classes = len(class_names)

    if val_dir.is_dir():
        train_dataset: Dataset = full_train_aug
        val_dataset: Optional[Dataset] = OrderedImageFolder(
            str(val_dir), transform=eval_tf
        )
        _check_class_map(full_train_aug, val_dataset, "val")
        split_info = "val_dir"
    else:
        resolved_val_size = _resolve_val_size(val_size, len(full_train_aug))
        if resolved_val_size > 0:
            indices = torch.randperm(
                len(full_train_aug), generator=make_generator(seed)
            ).tolist()
            train_indices = indices[:-resolved_val_size]
            val_indices = indices[-resolved_val_size:]
            train_dataset = _subset(full_train_aug, train_indices)
            val_dataset = _subset(full_train_eval, val_indices)
            split_info = f"seeded_holdout_{resolved_val_size}"
        else:
            train_dataset = full_train_aug
            val_dataset = None
            split_info = "none"

    test_dataset: Optional[Dataset] = None
    if test_dir.is_dir():
        candidate = OrderedImageFolder(str(test_dir), transform=eval_tf)
        _check_class_map(full_train_aug, candidate, "test")
        test_dataset = candidate

    loader_kwargs, loader_generator = _loader_kwargs(num_workers, pin_memory, seed)

    loaders: Dict[str, Any] = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            drop_last=drop_last_train,
            **loader_kwargs,
        ),
        "train_dataset": train_dataset,
        "class_names": class_names,
        "num_classes": num_classes,
        "train_generator": loader_generator,
        "split_info": split_info,
    }

    if val_dataset is not None:
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        loaders["val_dataset"] = val_dataset

    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        loaders["test_dataset"] = test_dataset

    loaders["summary"] = summarize_dataset(loaders)
    return loaders


def create_eval_dataloaders(
    root: str,
    input_size: int = 32,
    batch_size_eval: int = 5,
    num_workers: int = 2,
    pin_memory: bool = True,
    mean: MeanStd = None,
    std: MeanStd = None,
    seed: int = 42,
    class_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create deterministic validation/test loaders without requiring train/.

    This is used by checkpoint evaluation. When a checkpoint stores the class
    order, every eval split must expose the same folders in the same mapping so
    labels cannot silently shift.
    """

    root_path = Path(root)
    eval_tf = get_eval_transforms(input_size, mean, std)
    loader_kwargs, loader_generator = _loader_kwargs(num_workers, pin_memory, seed)

    loaders: Dict[str, Any] = {
        "class_names": list(class_order or []),
        "num_classes": len(class_order or []),
        "train_generator": loader_generator,
        "split_info": "eval_only",
    }
    reference: Optional[datasets.ImageFolder] = None

    for split in ("test", "val"):
        split_dir = root_path / split
        if not split_dir.is_dir():
            continue

        dataset = OrderedImageFolder(
            str(split_dir), transform=eval_tf, class_order=class_order
        )
        if class_order:
            _check_expected_classes(dataset, class_order, split)
        elif reference is not None:
            _check_class_map(reference, dataset, split)

        if reference is None:
            reference = dataset
            loaders["class_names"] = dataset.classes
            loaders["num_classes"] = len(dataset.classes)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        loaders[f"{split}_dataset"] = dataset

    if "test" not in loaders and "val" not in loaders:
        raise FileNotFoundError(
            f"No eval split found in {root_path}. Expected test/ or val/."
        )

    loaders["summary"] = summarize_dataset(loaders)
    return loaders
