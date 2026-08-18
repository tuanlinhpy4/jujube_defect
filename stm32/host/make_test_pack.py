#!/usr/bin/env python3
"""Bake the jujube test split into `test_data.pack` (Phase 3, weight-independent).

Mỗi ảnh test được preprocess y hệt eval pipeline rồi ghi nhị phân float32 NCHW + label.
KHỚP BIT-FOR-BIT: dùng lại `get_eval_transforms` từ data/dataset.py thay vì chép tay
(nếu transform trong repo đổi, pack đổi theo). Xem stm32/host/PACK_FORMAT.md.

Chạy (WSL/conda có torch+torchvision, KHÔNG cần weight/TF):
  python stm32/host/make_test_pack.py \
      --test-dir data_jujube/data_resize/64_split_70_15_15_seed42/test \
      --out stm32/out/test_data.pack
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from torchvision import datasets

# --- import transform thật từ repo để bit-exact ---------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # stm32/host/ -> stm32/ -> repo
sys.path.insert(0, str(REPO_ROOT))
from data.dataset import get_eval_transforms  # noqa: E402

MAGIC = b"JPK1"
VERSION = 1
DTYPE_F32 = 0
LAYOUT_NCHW = 0
LABEL_BYTES = 4

# ImageNet normalize — phải trùng configs/jujube*.yaml và §6 của PLAN.md
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Thứ tự lớp = ImageFolder sort alphabet: cracked=0, dry=1, insect_damaged=2, invalid=3
DEFAULT_EXPECTED_CLASSES = ["cracked", "dry", "insect_damaged", "invalid"]


def parse_class_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True,
                    help="thư mục test/ chứa <class>/*.png")
    ap.add_argument("--out", required=True, help="đường dẫn test_data.pack đầu ra")
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument(
        "--class-names",
        default=",".join(DEFAULT_EXPECTED_CLASSES),
        help="Comma-separated class folders in ImageFolder label order.",
    )
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        ap.error(f"--test-dir không tồn tại: {test_dir}")

    eval_tf = get_eval_transforms(args.input_size, MEAN, STD)
    ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)
    expected_classes = parse_class_names(args.class_names)

    if expected_classes and ds.classes != expected_classes:
        ap.error(f"thứ tự lớp lệch: {ds.classes} != {expected_classes} "
                 "(label index PHẢI khớp eval/MCU)")

    n = len(ds)
    c, h, w = 3, args.input_size, args.input_size
    img_bytes = c * h * w * 4
    print(f"[pack] {n} samples, {c}x{h}x{w} float32, classes={ds.classes}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # đếm phân bố lớp để in kiểm tra
    per_class = [0 for _ in ds.classes]

    header = MAGIC + struct.pack(
        "<9I", VERSION, n, c, h, w, DTYPE_F32, LAYOUT_NCHW, LABEL_BYTES, 0)
    assert len(header) == 40, len(header)

    with out_path.open("wb") as f:
        f.write(header)
        for i in range(n):
            tensor, label = ds[i]                      # tensor: float32 CHW, đã normalize
            arr = tensor.numpy()
            assert arr.shape == (c, h, w), arr.shape
            assert arr.dtype == np.float32, arr.dtype
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
            f.write(struct.pack("<I", int(label)))
            per_class[int(label)] += 1
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"\r[pack] {i+1}/{n}", end="", flush=True)
    print()

    total = 40 + n * (img_bytes + LABEL_BYTES)
    print(f"[pack] per-class {dict(zip(ds.classes, per_class))}")
    print(f"[pack] wrote {out_path}  ({total/1e6:.2f} MB, stride={img_bytes+LABEL_BYTES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
