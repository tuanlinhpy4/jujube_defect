#!/usr/bin/env python3
"""Prepare the 20 supplemental checkpoints for the existing STM32 pipeline.

Stages: audit, export, pack, convert, golden, sd, or all.
The two datasets are always staged separately because their labels/test packs differ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pth_to_int8_onnx import (  # noqa: E402
    build_checkpoint_model,
    infer_model_name,
    infer_num_classes,
    load_checkpoint,
)
@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    classes: tuple[str, ...]


@dataclass(frozen=True)
class GroupSpec:
    folder: str
    dataset: str
    arch: str


DATASETS = {
    "jujube": DatasetSpec(
        REPO_ROOT / "data_jujube" / "jujube_64_colab_split_70_15_15_seed42",
        ("bianxing", "gantiao", "haozao", "liekou", "meibian", "niaozhuo"),
    ),
    "jujube_ours": DatasetSpec(
        REPO_ROOT / "data_jujube" / "data_resize" / "64_split_70_15_15_seed42",
        ("cracked", "dry", "insect_damaged", "invalid"),
    ),
}
GROUPS = (
    GroupSpec("ghostnetv2_100_jujube", "jujube", "ghostnetv2_100"),
    GroupSpec("ghostnetv2_100_ours", "jujube_ours", "ghostnetv2_100"),
    GroupSpec("ju_lite_mobile_att_jujube", "jujube", "ju_lite_mobile_att"),
    GroupSpec("ju_lite_mobile_att_ours", "jujube_ours", "ju_lite_mobile_att"),
    GroupSpec("repghostnet/repghostnet_050 (1)", "jujube", "repghostnet_050"),
    GroupSpec("repghostnet/repghostnet_050", "jujube_ours", "repghostnet_050"),
    GroupSpec("ghostnetv1_jujube", "jujube", "ghostnetv1"),
    GroupSpec("ghostnetv1_ours", "jujube_ours", "ghostnetv1"),
    GroupSpec("ghostnetv1_100__jujube", "jujube", "ghostnetv1_100"),
    GroupSpec("ghostnetv1_100_ours", "jujube_ours", "ghostnetv1_100"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_classes(state: dict[str, Any]) -> int | None:
    for key, value in state.items():
        if key.endswith("classifier.weight") and torch.is_tensor(value):
            return int(value.shape[0])
    return None


def audit(
    source: Path,
    selected: set[str],
    selected_arches: set[str],
    output: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        if group.dataset not in selected:
            continue
        if selected_arches and group.arch not in selected_arches:
            continue
        spec = DATASETS[group.dataset]
        group_root = source / group.folder
        split_path = next(iter(group_root.rglob("split_info.json")), None)
        split = json.loads(split_path.read_text(encoding="utf-8")) if split_path else {}
        for seed in range(1, 6):
            found = sorted(group_root.rglob(f"seed_{seed}/{group.arch}.pth"))
            errors: list[str] = []
            checkpoint = found[0] if len(found) == 1 else None
            if len(found) != 1:
                errors.append(f"found {len(found)} checkpoint(s), expected 1")
            row: dict[str, Any] = {
                "dataset": group.dataset, "group": group.folder, "arch": group.arch,
                "seed": seed, "checkpoint": str(checkpoint or ""), "sha256": "",
                "metadata_dataset": "", "metadata_classes": "",
                "classifier_classes": "", "status": "", "note": "",
            }
            if checkpoint:
                try:
                    state, meta = load_checkpoint(checkpoint)
                    config = meta.get("config", {})
                    metadata_dataset = str(config.get("dataset", ""))
                    metadata_classes = int(config.get("num_classes", 0) or 0)
                    classifier_classes = head_classes(state)
                    row.update(
                        sha256=sha256(checkpoint),
                        metadata_dataset=metadata_dataset,
                        metadata_classes=metadata_classes,
                        classifier_classes=classifier_classes or "",
                    )
                    if metadata_dataset != group.dataset:
                        errors.append(f"dataset={metadata_dataset}, expected {group.dataset}")
                    if metadata_classes != len(spec.classes):
                        errors.append(
                            f"num_classes={metadata_classes}, expected {len(spec.classes)}"
                        )
                    if classifier_classes != len(spec.classes):
                        errors.append(
                            f"classifier={classifier_classes}, expected {len(spec.classes)}"
                        )
                    if int(meta.get("seed", seed)) != seed:
                        errors.append(f"seed={meta.get('seed')}, expected {seed}")
                    split_classes = tuple(split.get("class_names", ()))
                    if split_classes and split_classes != spec.classes:
                        errors.append("split_info class order mismatch")
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            row["status"] = "FAIL" if errors else "OK"
            row["note"] = " | ".join(errors)
            rows.append(row)

    hashes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["sha256"]:
            hashes.setdefault(str(row["sha256"]), []).append(row)
    for same in hashes.values():
        owners = {str(row["dataset"]) for row in same}
        if len(owners) > 1:
            for row in same:
                row["status"] = "FAIL"
                row["note"] = (str(row["note"]) + " | cross-dataset duplicate").strip(" |")

    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    valid = sum(row["status"] == "OK" for row in rows)
    print(f"[audit] {valid}/{len(rows)} OK -> {output}")
    for row in rows:
        if row["status"] != "OK":
            print(f"[audit] FAIL {row['group']} s{row['seed']}: {row['note']}")
    return rows


def export(rows: list[dict[str, Any]], out_root: Path, size: int, opset: int) -> int:
    valid = [row for row in rows if row["status"] == "OK"]
    records: list[dict[str, Any]] = []
    ok = 0
    for index, row in enumerate(valid, 1):
        checkpoint = Path(str(row["checkpoint"]))
        dataset = str(row["dataset"])
        arch = str(row["arch"])
        model_id = f"{arch}_{dataset}_s{row['seed']}"
        raw = out_root / dataset / "onnx_raw" / f"{model_id}.onnx"
        final = out_root / dataset / "onnx" / f"{model_id}.onnx"
        raw.parent.mkdir(parents=True, exist_ok=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset": dataset, "arch": arch, "seed": row["seed"],
            "model_id": model_id, "checkpoint": str(checkpoint),
            "onnx": str(final), "status": "", "bytes": "", "note": "",
        }
        print(f"[export {index:02d}/{len(valid)}] {model_id}", end=" ", flush=True)
        try:
            state, meta = load_checkpoint(checkpoint)
            name = infer_model_name("auto", meta, checkpoint)
            classes = infer_num_classes(None, meta, state)
            model = build_checkpoint_model(name, classes)
            model.load_state_dict(state, strict=True)
            model.eval()
            if hasattr(model, "convert_to_deploy"):
                model.convert_to_deploy()
                model.eval()
            torch.onnx.export(
                model, torch.zeros(1, 3, size, size), str(raw),
                export_params=True, opset_version=opset, do_constant_folding=True,
                input_names=["input"], output_names=["logits"],
            )
            if arch in {
                "ghostnetv2_100", "ju_lite_mobile_att", "repghostnet_050",
                "ghostnetv1", "ghostnetv1_100",
            }:
                venv_python = HERE / ".venv_tflite" / "Scripts" / "python.exe"
                prepared = subprocess.run(
                    [str(venv_python), str(HERE / "prepare_ghostnetv2_onnx.py"),
                     str(raw), str(final)],
                    cwd=REPO_ROOT, text=True, capture_output=True, check=False,
                )
                if prepared.returncode != 0:
                    raise RuntimeError(
                        "MCU graph rewrite failed: " + prepared.stderr.strip()
                    )
                record["note"] = prepared.stdout.strip().replace("\n", " | ")
            else:
                shutil.copy2(raw, final)
            record.update(status="OK", bytes=final.stat().st_size)
            ok += 1
            print(f"OK {final.stat().st_size / 1e6:.2f} MB")
        except Exception as exc:
            record.update(status="FAIL", note=f"{type(exc).__name__}: {exc}")
            print(f"FAIL {record['note']}")
            traceback.print_exc()
        records.append(record)
    log = out_root / "export_onnx.csv"
    columns = list(records[0]) if records else []
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"[export] {ok}/{len(valid)} OK -> {log}")
    return 0 if ok == len(valid) else 2


def run(command: list[str]) -> int:
    print("[run] " + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def packs(selected: set[str], out_root: Path) -> int:
    rc = 0
    for name in sorted(selected):
        spec = DATASETS[name]
        rc = max(rc, run([
            sys.executable, str(HERE / "make_test_pack.py"),
            "--test-dir", str(spec.root / "test"),
            "--out", str(out_root / name / "SD" / "test" / "test_data.pack"),
            "--input-size", "64", "--class-names", ",".join(spec.classes),
        ]))
    return rc


def convert(selected: set[str], out_root: Path, max_calib: int) -> int:
    python = HERE / ".venv_tflite" / "Scripts" / "python.exe"
    if not python.is_file():
        print(f"[convert] missing {python}")
        return 2
    rc = 0
    for name in sorted(selected):
        spec = DATASETS[name]
        base = out_root / name
        if not any((base / "onnx").glob("*.onnx")):
            print(f"[convert] skip {name}: no ONNX")
            continue
        rc = max(rc, run([
            str(python), str(HERE / "onnx_to_tflite.py"),
            "--onnx-dir", str(base / "onnx"),
            "--out-dir", str(base / "tflite"),
            "--sm-root", str(base / "_savedmodel"),
            "--data-root", str(spec.root),
            "--class-names", ",".join(spec.classes),
            "--input-size", "64", "--max-calib", str(max_calib),
            "--calib-npy", str(base / "calib_64_nhwc_float32.npy"),
        ]))
    return rc


def golden(selected: set[str], out_root: Path) -> int:
    python = HERE / ".venv_tflite" / "Scripts" / "python.exe"
    rc = 0
    for name in sorted(selected):
        spec = DATASETS[name]
        base = out_root / name
        if not any((base / "tflite").glob("*.tflite")):
            print(f"[golden] skip {name}: no TFLite")
            continue
        rc = max(rc, run([
            str(python), str(HERE / "eval_tflite_golden.py"),
            "--tflite-dir", str(base / "tflite"),
            "--pack", str(base / "SD" / "test" / "test_data.pack"),
            "--test-dir", str(spec.root / "test"),
            "--out", str(base / "golden_tflite.csv"),
            "--summary", str(base / "golden_summary.csv"),
            "--num-classes", str(len(spec.classes)), "--input-size", "64",
        ]))
    return rc


def sd(selected: set[str], out_root: Path) -> int:
    rc = 0
    for name in sorted(selected):
        spec = DATASETS[name]
        base = out_root / name
        if not any((base / "tflite").glob("*.tflite")):
            print(f"[sd] skip {name}: no TFLite")
            continue
        rc = max(rc, run([
            sys.executable, str(HERE / "build_sd.py"),
            "--tflite-dir", str(base / "tflite"),
            "--sd", str(base / "SD"), "--num-classes", str(len(spec.classes)),
        ]))
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "export", "pack", "convert", "golden", "sd", "all"))
    parser.add_argument("--source", type=Path, default=Path.home() / "Downloads" / "bổ sung")
    parser.add_argument("--out-root", type=Path, default=HERE / "out" / "supplement")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--architectures", nargs="+",
        choices=tuple(sorted({group.arch for group in GROUPS})),
        default=None,
        help="Only audit/export the selected architectures.",
    )
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--max-calib", type=int, default=0)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    selected = set(args.datasets)
    selected_arches = set(args.architectures or ())
    rows: list[dict[str, Any]] = []
    audit_failed = False
    if args.stage in {"audit", "export", "all"}:
        rows = audit(
            args.source, selected, selected_arches,
            args.out_root / "checkpoint_audit.csv",
        )
        audit_failed = any(row["status"] != "OK" for row in rows)
        if args.stage == "audit":
            return 2 if audit_failed else 0
    rc = 0
    if args.stage in {"export", "all"}:
        rc = max(rc, export(rows, args.out_root, args.input_size, args.opset))
    if args.stage in {"pack", "all"}:
        rc = max(rc, packs(selected, args.out_root))
    if args.stage in {"convert", "all"}:
        rc = max(rc, convert(selected, args.out_root, args.max_calib))
    if args.stage in {"golden", "all"}:
        rc = max(rc, golden(selected, args.out_root))
    if args.stage in {"sd", "all"}:
        rc = max(rc, sd(selected, args.out_root))
    return max(rc, 2 if audit_failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
