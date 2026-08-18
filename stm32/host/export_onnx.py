#!/usr/bin/env python3
"""M4.0 — Khâu 1/3: export 60 checkpoint .pth -> FP32 ONNX (batch CỐ ĐỊNH = 1).

Chạy trong ENV CHÍNH của repo (có torch + timm + models). KHÔNG cần TensorFlow.
Tái dùng đúng cách load/infer/build từ pth_to_int8_onnx.py để khớp tiền xử lý/cấu hình.

Batch cố định 1 (không dynamic axes) để onnx2tf chuyển sạch sang TFLite NHWC.

  python stm32/host/export_onnx.py
  python stm32/host/export_onnx.py --only mnasnet0_5 --seeds 1

Đầu ra: stm32/host/out/onnx/<arch>_s<seed>.onnx  + export_onnx.csv (log).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Tái dùng nguyên các hàm đã kiểm chứng trong pipeline ONNX cũ.
from pth_to_int8_onnx import (  # noqa: E402
    build_checkpoint_model,
    infer_model_name,
    infer_num_classes,
    load_checkpoint,
)

# 12 kiến trúc GO (đúng danh sách plan). Loại junet_edge (legacy, 0 seed) + resnet18.
GO_ARCHS = [
    "efficientnet_lite0",
    "fbnetc_100",
    "ghostnetv2_100",
    "junet_edge_l",
    "mnasnet0_5",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenetv4_conv_small",
    "regnetx_002",
    "shufflenet_v2_x0_5",
    "squeezenet1_1",
    "tinynet_e",
]

SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)


def discover_checkpoints(weight_root: Path, only: list[str], seeds: list[str]):
    """Trả về list (arch, seed_num, model_id, pth_path) cho các .pth GO."""
    found = []
    for arch in GO_ARCHS:
        if only and arch not in only:
            continue
        arch_dir = weight_root / arch
        if not arch_dir.is_dir():
            print(f"[warn] thiếu thư mục arch: {arch_dir}")
            continue
        for seed_dir in sorted(arch_dir.glob("seed_*")):
            m = SEED_RE.search(seed_dir.name)
            seed_num = m.group(1) if m else seed_dir.name.replace("seed_", "")
            if seeds and seed_num not in seeds:
                continue
            pths = sorted(seed_dir.glob("*.pth"))
            if not pths:
                print(f"[warn] không có .pth trong {seed_dir}")
                continue
            pth = pths[0]
            model_id = f"{arch}_s{seed_num}"
            found.append((arch, seed_num, model_id, pth))
    return found


def export_one(pth: Path, input_size: int, opset: int, out_path: Path) -> dict:
    rec = {"pth": str(pth), "onnx": str(out_path), "status": "", "note": ""}
    state, meta = load_checkpoint(pth)
    # model name: ưu tiên config['model'] trong ckpt; fallback theo tên thư mục arch.
    arch_from_dir = pth.parent.parent.name
    try:
        model_name = infer_model_name("auto", meta, pth)
    except Exception:
        model_name = arch_from_dir
    num_classes = infer_num_classes(None, meta, state)

    model = build_checkpoint_model(model_name, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    if hasattr(model, "convert_to_deploy"):
        model.convert_to_deploy()
        model.eval()
    if hasattr(model, "fuse_for_deploy"):
        model.fuse_for_deploy()
        model.eval()

    dummy = torch.zeros(1, 3, input_size, input_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # batch CỐ ĐỊNH = 1, KHÔNG dynamic_axes -> onnx2tf chuyển tĩnh, NHWC sạch.
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )
    rec["status"] = "OK"
    rec["model_name"] = model_name
    rec["num_classes"] = num_classes
    rec["bytes"] = out_path.stat().st_size
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight-root", type=Path, default=REPO_ROOT / "weight")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "stm32" / "host" / "out" / "onnx")
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--only", nargs="*", default=None, help="lọc theo tên arch")
    ap.add_argument("--seeds", nargs="*", default=None, help="lọc theo số seed")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ckpts = discover_checkpoints(args.weight_root, args.only or [], args.seeds or [])
    if not ckpts:
        print("[err] không tìm thấy checkpoint nào.")
        return 1

    print(f"# export_onnx: {len(ckpts)} checkpoint -> {args.out_dir}\n")
    rows = []
    ok = 0
    for i, (arch, seed, model_id, pth) in enumerate(ckpts, 1):
        out_path = args.out_dir / f"{model_id}.onnx"
        print(f"[{i:>2}/{len(ckpts)}] {model_id:<32} ", end="", flush=True)
        rec = {"arch": arch, "seed": seed, "model_id": model_id}
        try:
            rec.update(export_one(pth, args.input_size, args.opset, out_path))
            ok += 1
            print(f"OK  {rec['model_name']:<24} {rec['bytes']/1e6:6.2f} MB")
        except Exception as e:
            rec["status"] = "FAIL"
            rec["note"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL  {rec['note']}")
            traceback.print_exc()
        rows.append(rec)

    csv_path = args.out_dir.parent / "export_onnx.csv"
    cols = ["arch", "seed", "model_id", "model_name", "num_classes", "status", "bytes", "note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {ok}/{len(ckpts)} OK  -> log: {csv_path}")
    return 0 if ok == len(ckpts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
