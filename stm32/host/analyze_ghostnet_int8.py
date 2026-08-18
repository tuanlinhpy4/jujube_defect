#!/usr/bin/env python3
"""Đối chiếu GhostNetV2: báo cáo PTH -> ONNX raw -> ONNX MCU -> INT8 TFLite.

Chạy trong .venv_tflite để có cả onnxruntime và TensorFlow:
  stm32/host/.venv_tflite/Scripts/python stm32/host/analyze_ghostnet_int8.py

Mỗi runtime dùng đúng test_data.pack của MCU. Báo cáo giúp tách lỗi export/rewrite
khỏi sai số PTQ INT8 và được lưu dưới out/supplement/ghostnet_int8_analysis.*.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tflite_common import macro_f1, read_pack  # noqa: E402

OUT_ROOT = HERE / "out" / "supplement"
SEED_RE = re.compile(r"_s(\d+)$")


def run_onnx(path: Path, images: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    outputs = []
    for image in images:
        outputs.append(session.run(None, {name: image[None]})[0][0])
    return np.asarray(outputs, dtype=np.float32)


def run_tflite(path: Path, images: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    outputs = []
    for image in images:
        nhwc = np.transpose(image, (1, 2, 0))[None]
        interp.set_tensor(inp["index"], nhwc.astype(inp["dtype"], copy=False))
        interp.invoke()
        outputs.append(interp.get_tensor(out["index"])[0])
    return np.asarray(outputs, dtype=np.float32)


def reported_pth_f1(source: Path, dataset: str, seed: int) -> float:
    group = "ghostnetv2_100_jujube" if dataset == "jujube" else "ghostnetv2_100_ours"
    matches = sorted((source / group).rglob(f"seed_{seed}/test_results.json"))
    if len(matches) != 1:
        return float("nan")
    report = json.loads(matches[0].read_text(encoding="utf-8"))
    return float(report["test"]["f1_macro"]) / 100.0


def analyze_dataset(dataset: str, source: Path) -> list[dict]:
    base = OUT_ROOT / dataset
    images, labels = read_pack(base / "SD" / "test" / "test_data.pack")
    classes = 6 if dataset == "jujube" else 4
    rows = []
    for tflite in sorted((base / "tflite").glob("ghostnetv2_100_*.tflite")):
        model_id = tflite.stem
        match = SEED_RE.search(model_id)
        seed = int(match.group(1)) if match else 0
        raw_path = base / "onnx_raw" / f"{model_id}.onnx"
        mcu_path = base / "onnx" / f"{model_id}.onnx"
        print(f"[{dataset}] seed {seed}: raw ONNX", flush=True)
        raw = run_onnx(raw_path, images)
        print(f"[{dataset}] seed {seed}: rewritten ONNX", flush=True)
        mcu = run_onnx(mcu_path, images)
        print(f"[{dataset}] seed {seed}: INT8 TFLite", flush=True)
        q = run_tflite(tflite, images)

        raw_pred = raw.argmax(1)
        mcu_pred = mcu.argmax(1)
        q_pred = q.argmax(1)
        _, raw_f1, _ = macro_f1(labels, raw_pred, classes)
        _, mcu_f1, _ = macro_f1(labels, mcu_pred, classes)
        _, q_f1, _ = macro_f1(labels, q_pred, classes)
        pth_f1 = reported_pth_f1(source, dataset, seed)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "model_id": model_id,
            "pth_report_f1": round(pth_f1, 6),
            "onnx_raw_f1": round(raw_f1, 6),
            "onnx_mcu_f1": round(mcu_f1, 6),
            "tflite_int8_f1": round(q_f1, 6),
            "export_delta_pp": round((raw_f1 - pth_f1) * 100.0, 4),
            "rewrite_delta_pp": round((mcu_f1 - raw_f1) * 100.0, 4),
            "int8_delta_pp": round((q_f1 - mcu_f1) * 100.0, 4),
            "rewrite_logits_max_abs": float(np.max(np.abs(mcu - raw))),
            "rewrite_pred_agreement": round(float(np.mean(mcu_pred == raw_pred)), 6),
            "int8_pred_agreement": round(float(np.mean(q_pred == mcu_pred)), 6),
            "int8_logits_mae": round(float(np.mean(np.abs(q - mcu))), 6),
        })
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def write_reports(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GhostNetV2 INT8 conversion analysis", "",
        "| dataset | seed | PTH F1 | ONNX raw | ONNX MCU | INT8 F1 | INT8 delta (pp) | pred agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['pth_report_f1']:.4f} "
            f"| {row['onnx_raw_f1']:.4f} | {row['onnx_mcu_f1']:.4f} "
            f"| {row['tflite_int8_f1']:.4f} | {row['int8_delta_pp']:.2f} "
            f"| {row['int8_pred_agreement']:.3f} |"
        )
    lines += ["", "## Mean ± population std over five seeds", ""]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    for dataset, items in grouped.items():
        pth_m, pth_s = mean_std([x["pth_report_f1"] for x in items])
        q_m, q_s = mean_std([x["tflite_int8_f1"] for x in items])
        drop_m, drop_s = mean_std([x["int8_delta_pp"] for x in items])
        agree_m, agree_s = mean_std([x["int8_pred_agreement"] for x in items])
        lines.append(
            f"- **{dataset}**: PTH {pth_m:.4f}±{pth_s:.4f}; INT8 {q_m:.4f}±{q_s:.4f}; "
            f"delta {drop_m:.2f}±{drop_s:.2f} pp; prediction agreement {agree_m:.3f}±{agree_s:.3f}."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path.home() / "Downloads" / "bổ sung"
    )
    parser.add_argument("--datasets", nargs="+", choices=("jujube", "jujube_ours"),
                        default=["jujube", "jujube_ours"])
    parser.add_argument("--out-csv", type=Path,
                        default=OUT_ROOT / "ghostnet_int8_analysis.csv")
    parser.add_argument("--out-md", type=Path,
                        default=OUT_ROOT / "ghostnet_int8_analysis.md")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rows = []
    for dataset in args.datasets:
        rows.extend(analyze_dataset(dataset, args.source))
    if not rows:
        print("Không tìm thấy GhostNet TFLite.")
        return 1
    write_reports(rows, args.out_csv, args.out_md)
    print(f"[done] {len(rows)} model -> {args.out_csv} | {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
