#!/usr/bin/env python3
"""M4.3 — Dựng nội dung thẻ SD: copy 60 .tflite -> SD/models + manifest.csv 60 dòng.

Chạy ENV CHÍNH (pure python). CRC32 = zlib.crc32 (khớp crc32.c firmware).
Giữ nguyên SD/test/test_data.pack. Xoá model dummy cũ.

  python stm32/host/build_sd.py
  python stm32/host/build_sd.py --sd F:/   # ghi thẳng ra thẻ
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
INPUT_W = INPUT_H = 64
INPUT_C = 3
NUM_CLASSES = 4
HEADER = "model_id,model_bin,input_w,input_h,input_c,num_classes,crc32"


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite-dir", type=Path, default=HERE / "out" / "tflite")
    ap.add_argument("--sd", type=Path, default=REPO_ROOT / "stm32" / "out" / "SD",
                    help="thư mục SD đích (mặc định staging stm32/out/SD)")
    ap.add_argument("--sd-subdir", default="models")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = ap.parse_args()

    files = sorted(args.tflite_dir.glob("*.tflite"))
    if not files:
        ap.error(f"không có .tflite trong {args.tflite_dir}")

    models_dir = args.sd / args.sd_subdir
    if models_dir.exists():
        for old in models_dir.glob("*"):
            if old.is_file():
                old.unlink()
    models_dir.mkdir(parents=True, exist_ok=True)

    lines = [HEADER]
    total = 0
    for fp in files:
        dst = models_dir / fp.name
        shutil.copy2(fp, dst)
        crc = crc32_file(dst)
        model_id = fp.stem
        model_bin = f"{args.sd_subdir}/{fp.name}"
        lines.append(f"{model_id},{model_bin},{INPUT_W},{INPUT_H},{INPUT_C},"
                     f"{args.num_classes},{crc:08x}")
        total += dst.stat().st_size
        print(f"[sd] {model_id:32s} {dst.stat().st_size/1e6:6.2f} MB  crc={crc:08x}")

    manifest = args.sd / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    (args.sd / "results").mkdir(exist_ok=True)
    print(f"\n[sd] {len(files)} model, tổng {total/1e6:.1f} MB -> {models_dir}")
    print(f"[sd] manifest -> {manifest}")
    print(f"[sd] nhớ giữ {args.sd / 'test' / 'test_data.pack'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
