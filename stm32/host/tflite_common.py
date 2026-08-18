#!/usr/bin/env python3
"""Tiện ích dùng chung cho M4.0 (khâu venv, KHÔNG torch): tiền xử lý, đọc pack, macro-F1.

Tiền xử lý KHỚP pth_to_int8_onnx.preprocess_image: Resize(giữ tỉ lệ, cạnh ngắn =
int(size*256/224)) bilinear -> CenterCrop(size) -> /255 -> (x-mean)/std, RGB, NCHW.
Dùng cho ẢNH CALIB (train/val). Với GOLDEN nên đọc thẳng test_data.pack để trùng MCU.
"""
from __future__ import annotations

import random
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ("cracked", "dry", "insect_damaged", "invalid")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# pack format (khớp make_test_pack.py / test_pack.c)
_MAGIC = b"JPK1"


def preprocess_nchw(path: Path, input_size: int = 64,
                    mean: np.ndarray = MEAN, std: np.ndarray = STD) -> np.ndarray:
    """Trả về float32 NCHW (3, H, W) đã chuẩn hoá — khớp preprocess_image."""
    resize_size = int(input_size * 256 / 224)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w <= h:
        nw = resize_size
        nh = int(round(h * resize_size / w))
    else:
        nh = resize_size
        nw = int(round(w * resize_size / h))
    img = img.resize((nw, nh), Image.BILINEAR)
    left = max((nw - input_size) // 2, 0)
    top = max((nh - input_size) // 2, 0)
    img = img.crop((left, top, left + input_size, top + input_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = (arr - mean) / std
    return arr.astype(np.float32, copy=False)


def collect_calib_images(data_root: Path, splits: Sequence[str],
                         class_names: Sequence[str], seed: int,
                         max_images: int) -> list[Path]:
    """Thu ảnh calib từ train/val (TỪ CHỐI test). Khớp collect_calibration_images."""
    names = [s.strip() for s in splits if s.strip()]
    if not names:
        raise ValueError("cần ít nhất 1 split calib")
    if "test" in {s.lower() for s in names}:
        raise ValueError("Từ chối dùng split test để calib.")
    images: list[Path] = []
    for split in names:
        sd = data_root / split
        if not sd.is_dir():
            raise FileNotFoundError(f"thiếu split calib: {sd}")
        for cls in class_names:
            cd = sd / cls
            if not cd.is_dir():
                raise FileNotFoundError(f"thiếu thư mục lớp: {cd}")
            images.extend(p for p in cd.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    images = sorted(images)
    if not images:
        raise RuntimeError(f"không có ảnh calib trong {data_root}")
    random.Random(seed).shuffle(images)
    if max_images > 0:
        images = images[:max_images]
    return images


def read_pack(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Đọc test_data.pack -> (imgs float32 NCHW [N,3,H,W], labels int [N]).

    Trùng BIT-FOR-BIT với input MCU đọc trên thẻ.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != _MAGIC:
        raise ValueError(f"magic sai: {raw[:4]!r}")
    ver, n, c, h, w, dtype, layout, label_bytes, _ = struct.unpack_from("<9I", raw, 4)
    if dtype != 0 or layout != 0:
        raise ValueError(f"chỉ hỗ trợ float32 NCHW (dtype={dtype}, layout={layout})")
    if label_bytes != 4:
        raise ValueError(f"label_bytes={label_bytes} != 4")
    img_floats = c * h * w
    stride = img_floats * 4 + label_bytes
    off = 40
    imgs = np.empty((n, c, h, w), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)
    for i in range(n):
        base = off + i * stride
        imgs[i] = np.frombuffer(raw, dtype="<f4", count=img_floats,
                                offset=base).reshape(c, h, w)
        (labels[i],) = struct.unpack_from("<I", raw, base + img_floats * 4)
    return imgs, labels


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float, list]:
    """Trả về (accuracy, macro_f1, [f1 từng lớp])."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    f1s = []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(float(f1))
    return acc, float(np.mean(f1s)), f1s
