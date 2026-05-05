#!/usr/bin/env python3
"""Auto-download jujube datasets from Google Drive.

Usage (standalone):
    python download_data.py                  # download both datasets
    python download_data.py --dataset jujube
    python download_data.py --dataset jujube_ours

Usage (from code):
    from download_data import ensure_dataset
    ensure_dataset("jujube",      data_root="./data/jujube")
    ensure_dataset("jujube_ours", data_root="./data/jujube_ours")
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
        "gdrive_id": "1G1t1NRWBLPjxmjlEaRiJFueEeHS3GI9D",
        "default_root": "./data/jujube_ours",
        "expected_classes": [
            "dry", "insect_damaged", "cracked", "mold", "invalid",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "\n  [download_data] gdown is required to download datasets.\n"
            "  Install it with:  pip install gdown\n"
        ) from exc


def _is_complete(data_root: str, expected_classes: List[str]) -> bool:
    """Return True only when every expected class folder exists and is non-empty."""
    root = Path(data_root)
    if not root.is_dir():
        return False
    for cls in expected_classes:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            return False
        if not any(cls_dir.iterdir()):     # empty folder = incomplete
            return False
    return True


def _download_gdrive(file_id: str, dest: str) -> None:
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {url}")
    gdown.download(url, dest, quiet=False, fuzzy=True)


def _extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"  Extracting {zip_path} → {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _flatten_if_nested(data_root: str, expected_classes: List[str]) -> None:
    """If zip extracted into a single sub-folder, move contents up one level.

    Handles zips that contain a top-level folder (e.g. jujube/niaozhuo/...)
    instead of placing class folders directly at the root.
    """
    root = Path(data_root)
    # Already flat?
    if any((root / cls).is_dir() for cls in expected_classes):
        return

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        nested = subdirs[0]
        # Check if class folders are inside the nested dir
        if any((nested / cls).is_dir() for cls in expected_classes):
            print(f"  Moving contents from {nested.name}/ up one level...")
            for item in nested.iterdir():
                item.rename(root / item.name)
            nested.rmdir()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_dataset(dataset: str, data_root: str | None = None) -> str:
    """Download and extract *dataset* if not already present.

    Args:
        dataset:   one of ``"jujube"`` or ``"jujube_ours"``.
        data_root: path where class folders should live.
                   Defaults to the value in ``DATASETS`` registry.

    Returns:
        Resolved ``data_root`` path (useful when caller used the default).
    """
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
    _require_gdown()

    zip_path = f"/tmp/{dataset}.zip"
    _download_gdrive(info["gdrive_id"], zip_path)
    _extract_zip(zip_path, root)
    _flatten_if_nested(root, expected)

    # Clean up zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    if not _is_complete(root, expected):
        missing = [
            cls for cls in expected
            if not (Path(root) / cls).is_dir()
        ]
        raise RuntimeError(
            f"[{dataset}] Download/extract finished but these class folders "
            f"are still missing: {missing}\n"
            f"Please download manually from:\n"
            f"  https://drive.google.com/uc?id={info['gdrive_id']}"
        )

    print(f"  [{dataset}] Ready at '{root}'.\n")
    return root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download jujube datasets")
    p.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default=None,
        help="Which dataset to download (default: both)",
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
