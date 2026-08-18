#!/usr/bin/env python3
"""M4.0 — Khâu 2/3: FP32 ONNX -> INT8 .tflite (IO float, int8 bên trong).

CHẠY TRONG VENV: stm32/host/.venv_tflite (onnx2tf + tensorflow). KHÔNG cần torch.

Quy trình mỗi model:
  ONNX --onnx2tf--> SavedModel (TF, NHWC) --tf.lite (PTQ INT8, rep-dataset)--> .tflite
- Lượng tử hoá lại bằng calib train+val (tiền xử lý 64x64 RGB chuẩn) -> KHÔNG dùng _int8.onnx.
- inference_input_type/output_type = float32  => IO float, Quantize/Dequantize nằm trong graph,
  trọng số + activation int8 => khớp CMSIS-NN. NHWC do onnx2tf sinh ra (lý tưởng cho TFLM).

  ./.venv_tflite/Scripts/python onnx_to_tflite.py
  ./.venv_tflite/Scripts/python onnx_to_tflite.py --only mnasnet0_5_s1
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tflite_common import (  # noqa: E402
    DEFAULT_CLASS_NAMES,
    collect_calib_images,
    preprocess_nchw,
)

REPO_ROOT = HERE.parents[1]
DEFAULT_DATA = REPO_ROOT / "data_jujube" / "data_resize" / "64_split_70_15_15_seed42"


def build_calib_nhwc(data_root: Path, input_size: int, seed: int, max_images: int,
                     class_names: tuple[str, ...]) -> np.ndarray:
    """Mảng calib NHWC float32 [N,H,W,3] (dùng chung cho mọi model -> tất định)."""
    paths = collect_calib_images(data_root, ["train", "val"], class_names,
                                 seed=seed, max_images=max_images)
    arrs = [preprocess_nchw(p, input_size).transpose(1, 2, 0) for p in paths]  # CHW->HWC
    return np.ascontiguousarray(np.stack(arrs, 0), dtype=np.float32)


def convert_one(onnx_path: Path, sm_dir: Path, out_tflite: Path,
                calib_nhwc: np.ndarray, disable_strict: bool = False) -> dict:
    import onnx2tf
    import onnx2tf.onnx2tf as _o2t
    import tensorflow as tf

    # onnx2tf tải 1 ảnh mẫu (.npy pickled) để check accuracy -> numpy 2.x từ chối load.
    # Ta KHÔNG bật check closeness và KHÔNG dùng int8 của onnx2tf, nên test_data chỉ để in thử
    # -> thay bằng chính calib của mình (NHWC float32) để bỏ qua việc tải mạng.
    _o2t.download_test_image_data = lambda: calib_nhwc.astype(np.float32, copy=False)

    rec = {"onnx": str(onnx_path), "tflite": str(out_tflite), "status": "", "note": ""}
    if sm_dir.exists():
        shutil.rmtree(sm_dir, ignore_errors=True)
    sm_dir.mkdir(parents=True, exist_ok=True)

    # ONNX -> SavedModel (NHWC). non_verbose để bớt log; giữ tên IO.
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(sm_dir),
        output_signaturedefs=True,   # -osd: sửa lỗi sinh SavedModel khi tên op ONNX có ký tự lạ
        disable_strict_mode=disable_strict,  # bỏ qua sửa-layout-strict (cứu ghostnet/shuffle) — PHẢI verify accuracy
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    def rep_dataset():
        for i in range(calib_nhwc.shape[0]):
            yield [calib_nhwc[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32   # IO float
    conv.inference_output_type = tf.float32
    tfl_bytes = conv.convert()

    out_tflite.parent.mkdir(parents=True, exist_ok=True)
    out_tflite.write_bytes(tfl_bytes)

    # kiểm tra IO float + đếm op int8
    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]
    rec.update(status="OK", bytes=len(tfl_bytes),
               in_dtype=str(np.dtype(inp["dtype"])),
               in_shape="x".join(map(str, inp["shape"])),
               out_dtype=str(np.dtype(outp["dtype"])))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", type=Path, default=HERE / "out" / "onnx")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out" / "tflite")
    ap.add_argument("--sm-root", type=Path, default=HERE / "out" / "_savedmodel")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="thứ tự/tên thư mục lớp, phân tách bằng dấu phẩy",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-calib", type=int, default=0,
                    help="0 = dùng TOÀN BỘ train+val (chất lượng INT8 cao nhất, khớp ONNX baseline)")
    ap.add_argument("--calib-npy", type=Path, default=None,
                    help="cache calib NHWC: nếu có thì nạp; nếu chưa thì build rồi lưu (tăng tốc chạy mỗi-model-1-tiến-trình)")
    ap.add_argument("--only", nargs="*", default=None, help="lọc theo model_id (stem onnx)")
    ap.add_argument("--disable-strict", action="store_true",
                    help="onnx2tf disable_strict_mode (cứu arch lỗi layout; PHẢI verify golden)")
    ap.add_argument("--keep-savedmodel", action="store_true")
    args = ap.parse_args()
    class_names = tuple(x.strip() for x in args.class_names.split(",") if x.strip())
    if not class_names:
        ap.error("--class-names phải chứa ít nhất một lớp")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    onnx_files = sorted(args.onnx_dir.glob("*.onnx"))
    if args.only:
        onnx_files = [p for p in onnx_files if p.stem in set(args.only)]
    if not onnx_files:
        print(f"[err] không có .onnx trong {args.onnx_dir}")
        return 1

    if args.calib_npy and args.calib_npy.is_file():
        calib = np.load(args.calib_npy)
        if args.max_calib > 0:
            calib = calib[:args.max_calib]
        print(f"[calib] nạp cache {args.calib_npy} -> {calib.shape[0]} ảnh NHWC {calib.shape[1:]}\n")
    else:
        print(f"[calib] thu ảnh train+val (max {args.max_calib})...")
        calib = build_calib_nhwc(
            args.data_root, args.input_size, args.seed, args.max_calib, class_names
        )
        print(f"[calib] {calib.shape[0]} ảnh, shape NHWC {calib.shape[1:]}")
        if args.calib_npy:
            args.calib_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.calib_npy, calib)
            print(f"[calib] lưu cache -> {args.calib_npy}")
        print()

    print(f"# onnx_to_tflite: {len(onnx_files)} model -> {args.out_dir}\n")
    rows, ok = [], 0
    for i, onnx_path in enumerate(onnx_files, 1):
        mid = onnx_path.stem
        out_tflite = args.out_dir / f"{mid}.tflite"
        sm_dir = args.sm_root / mid
        print(f"[{i:>2}/{len(onnx_files)}] {mid:<32} ", end="", flush=True)
        rec = {"model_id": mid}
        try:
            rec.update(convert_one(onnx_path, sm_dir, out_tflite, calib,
                                   disable_strict=args.disable_strict))
            ok += 1
            print(f"OK  {rec['bytes']/1e6:6.2f} MB  IO={rec['in_dtype']}->{rec['out_dtype']}  "
                  f"in[{rec['in_shape']}]")
        except Exception as e:
            rec["status"] = "FAIL"
            rec["note"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL  {rec['note']}")
            traceback.print_exc()
        finally:
            if not args.keep_savedmodel:
                shutil.rmtree(sm_dir, ignore_errors=True)
        rows.append(rec)

    csv_path = args.out_dir.parent / "onnx_to_tflite.csv"
    cols = ["model_id", "status", "bytes", "in_dtype", "in_shape", "out_dtype", "note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {ok}/{len(onnx_files)} OK  -> log: {csv_path}")
    return 0 if ok == len(onnx_files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
