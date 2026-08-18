#!/usr/bin/env python3
"""Quét thư mục model → sinh manifest.csv (model_id, path, dims, num_classes, crc32).

Phase 3: test harness với blob dummy `.bin`. Phase 4: model thật `.tflite`.
CRC32 = zlib.crc32 (CRC-32/ISO-HDLC) — khớp crc32.c trên firmware. Xem PACK_FORMAT.md.

  python stm32/host/make_manifest.py \
      --models-dir SD/models --ext .tflite \
      --sd-subdir models --out SD/manifest.csv
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

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
    ap.add_argument("--models-dir", required=True, help="thư mục chứa file model")
    ap.add_argument("--ext", default=".tflite", help="đuôi file model (.tflite hoặc .bin)")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--sd-subdir", default="models",
                    help="thư mục model trên SD (đưa vào cột model_bin, vd 'models')")
    ap.add_argument("--out", required=True, help="đường dẫn manifest.csv đầu ra")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        ap.error(f"--models-dir không tồn tại: {models_dir}")

    files = sorted(models_dir.glob(f"*{args.ext}"))
    if not files:
        ap.error(f"không thấy file *{args.ext} trong {models_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [HEADER]
    for fp in files:
        crc = crc32_file(fp)
        model_id = fp.stem
        model_bin = f"{args.sd_subdir}/{fp.name}" if args.sd_subdir else fp.name
        lines.append(f"{model_id},{model_bin},{INPUT_W},{INPUT_H},{INPUT_C},"
                     f"{args.num_classes},{crc:08x}")
        print(f"[manifest] {model_id:32s} {fp.stat().st_size/1e6:6.2f} MB  crc={crc:08x}")

    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"[manifest] wrote {out_path}  ({len(files)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
