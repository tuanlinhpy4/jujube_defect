#!/usr/bin/env python3
"""Auto-download jujube and fruitvision datasets.

Usage (standalone):
    python download_data.py                        # download all datasets
    python download_data.py --dataset jujube
    python download_data.py --dataset jujube_ours
    python download_data.py --dataset fruitvision
    python download_data.py --dataset sapota
    python download_data.py --dataset jujube_healthy
    python download_data.py --dataset mango_dhds

Usage (from code):
    from download_data import ensure_dataset
    ensure_dataset("jujube",         data_root="./data/jujube")
    ensure_dataset("jujube_ours",    data_root="./data/jujube_ours")
    ensure_dataset("fruitvision",    data_root="./data/fruitvision")
    ensure_dataset("sapota",         data_root="./data/sapota")
    ensure_dataset("jujube_healthy", data_root="./data/jujube_healthy")
    ensure_dataset("mango_dhds",     data_root="./data/mango_dhds")
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

_FRUITVISION_FRUITS = ["Apple", "Banana", "Grape", "Mango", "Orange"]
_FRUITVISION_CONDITIONS = ["Fresh", "Rotten", "Formalin-mixed"]
_FRUITVISION_CLASSES = [
    f"{fruit}_{condition}"
    for fruit in _FRUITVISION_FRUITS
    for condition in _FRUITVISION_CONDITIONS
]

DATASETS: Dict[str, Dict] = {
    "jujube": {
        "gdrive_id": "1HIel83uhuvH4MxYrSQLZvZl5cTWqolNJ",
        "default_root": "./data/jujube",
        "expected_classes": [
            "niaozhuo", "meibian", "liekou",
            "haozao", "gantiao", "bianxing",
        ],
    },
    "jujube_ours": {
        "gdrive_id": "19Q8LhNdSsdWjAy0HkaeMMutN2KBgxoH5",
        "default_root": "./data/jujube_ours",
        "expected_classes": [
            "dry", "insect_damaged", "cracked", "mold", "invalid",
        ],
    },
    "fruitvision": {
        "hf_repo_id": "doantrongthai/FruitVision",
        "hf_filename": "FruitVision.zip",
        "default_root": "./data/fruitvision",
        "expected_classes": _FRUITVISION_CLASSES,
        "fruit_dirs": _FRUITVISION_FRUITS,
        "condition_dirs": _FRUITVISION_CONDITIONS,
    },
    # Same HF repo (doantrongthai/FruitVision) also ships these three
    # standalone zips. Each one already extracts into flat class folders,
    # so no fruit/condition reorganizing is needed — just extract + flatten.
    "sapota": {
        "hf_repo_id": "doantrongthai/FruitVision",
        "hf_filename": "sapota.zip",
        "default_root": "./data/sapota",
        "expected_classes": ["Large", "Medium", "Small", "Spoiled"],
    },
    "jujube_healthy": {
        "hf_repo_id": "doantrongthai/FruitVision",
        "hf_filename": "jujube_healthy.zip",
        "default_root": "./data/jujube_healthy",
        "expected_classes": ["bruised", "healthy"],
    },
    "mango_dhds": {
        "hf_repo_id": "doantrongthai/FruitVision",
        "hf_filename": "mangoDHDS.zip",
        "default_root": "./data/mango_dhds",
        "expected_classes": [
            "Anthracnose", "Bacterial Canker", "Healthy", "Scab", "Stem End Rot",
        ],
    },
}


def _require_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "\n  [download_data] gdown is required to download datasets.\n"
            "  Install it with:  pip install gdown\n"
        ) from exc


def _require_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "\n  [download_data] huggingface_hub is required to download datasets.\n"
            "  Install it with:  pip install huggingface_hub\n"
        ) from exc


def _is_complete(data_root: str, expected_classes: List[str]) -> bool:
    root = Path(data_root)
    if not root.is_dir():
        return False
    for cls in expected_classes:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            return False
        if not any(cls_dir.iterdir()):
            return False
    return True


def _download_gdrive(file_id: str, dest: str) -> None:
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {url}")
    gdown.download(url, dest, quiet=False, fuzzy=True)


def _download_huggingface(repo_id: str, filename: str, dest: str) -> None:
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {filename} from Hugging Face ({repo_id})")
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    shutil.copy(downloaded, dest)


def _extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"  Extracting {zip_path} → {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _flatten_if_nested(data_root: str, expected_classes: List[str]) -> None:
    root = Path(data_root)
    if any((root / cls).is_dir() for cls in expected_classes):
        return

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        nested = subdirs[0]
        if any((nested / cls).is_dir() for cls in expected_classes):
            print(f"  Moving contents from {nested.name}/ up one level...")
            for item in nested.iterdir():
                item.rename(root / item.name)
            nested.rmdir()


def _reorganize_fruitvision(extract_root: Path, fruit_dirs: List[str],
                            condition_dirs: List[str], target_root: Path) -> None:
    search_root = extract_root
    if not any((search_root / fruit).is_dir() for fruit in fruit_dirs):
        subdirs = [d for d in search_root.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and any((subdirs[0] / fruit).is_dir() for fruit in fruit_dirs):
            search_root = subdirs[0]

    target_root.mkdir(parents=True, exist_ok=True)
    for fruit in fruit_dirs:
        fruit_dir = search_root / fruit
        if not fruit_dir.is_dir():
            continue
        for condition in condition_dirs:
            cond_dir = fruit_dir / condition
            if not cond_dir.is_dir():
                continue
            dst = target_root / f"{fruit}_{condition}"
            dst.mkdir(parents=True, exist_ok=True)
            for img in cond_dir.iterdir():
                if img.is_file():
                    shutil.move(str(img), str(dst / img.name))


def ensure_dataset(dataset: str, data_root: str | None = None) -> str:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}"
        )

    info = DATASETS[dataset]
    root = data_root or info["default_root"]
    expected = info["expected_classes"]

    if _is_complete(root, expected):
        print(f"  [{dataset}] Already present at '{root}' — skipping download.")
        return root

    print(f"\n  [{dataset}] Dataset not found or incomplete at '{root}'.")

    if "hf_repo_id" in info:
        _require_huggingface_hub()
        zip_path = f"/tmp/{dataset}.zip"
        _download_huggingface(info["hf_repo_id"], info["hf_filename"], zip_path)

        if "fruit_dirs" in info:
            # fruitvision.zip ships a nested Fruit/Condition layout that
            # needs merging into combined "Fruit_Condition" class folders.
            extract_dir = f"/tmp/{dataset}_extract"
            if os.path.isdir(extract_dir):
                shutil.rmtree(extract_dir)
            _extract_zip(zip_path, extract_dir)

            _reorganize_fruitvision(
                Path(extract_dir), info["fruit_dirs"], info["condition_dirs"], Path(root)
            )
            shutil.rmtree(extract_dir, ignore_errors=True)
        else:
            # sapota.zip / jujube_healthy.zip / mangoDHDS.zip already
            # extract straight into flat class folders — just unzip and
            # flatten one wrapper level if the zip has one.
            _extract_zip(zip_path, root)
            _flatten_if_nested(root, expected)
    else:
        _require_gdown()
        zip_path = f"/tmp/{dataset}.zip"
        _download_gdrive(info["gdrive_id"], zip_path)
        _extract_zip(zip_path, root)
        _flatten_if_nested(root, expected)

    try:
        os.remove(zip_path)
    except OSError:
        pass

    if not _is_complete(root, expected):
        missing = [
            cls for cls in expected
            if not (Path(root) / cls).is_dir()
        ]
        source = info.get("hf_repo_id") or info.get("gdrive_id")
        raise RuntimeError(
            f"[{dataset}] Download/extract finished but these class folders "
            f"are still missing: {missing}\n"
            f"Please check the source: {source}"
        )

    print(f"  [{dataset}] Ready at '{root}'.\n")
    return root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download jujube / fruitvision datasets")
    p.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default=None,
        help="Which dataset to download (default: all)",
    )
    p.add_argument("--data-root", type=str, default=None,
                   help="Override destination directory")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    targets = [args.dataset] if args.dataset else list(DATASETS)
    for ds in targets:
        ensure_dataset(ds, args.data_root)


if __name__ == "__main__":
    main()