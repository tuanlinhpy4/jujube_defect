#!/usr/bin/env python3
"""Prepare an SD card for 4-class RAM-only measurement.

Copies the 11 paper models from weights/jujube_ours/tflite to SD/models, writes a
4-class manifest.csv, and resets results/ram_usage.csv. Existing
manifest/ram_usage files are backed up with a timestamp.

Usage:
  python stm32/host/prepare_ram_sd_4class.py --sd G:/
  python stm32/host/prepare_ram_sd_4class.py --sd G:/ --seeds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "jujube_ours" / "tflite"

INPUT_W = INPUT_H = 64
INPUT_C = 3
NUM_CLASSES = 4
HEADER = ["model_id", "model_bin", "input_w", "input_h", "input_c", "num_classes", "crc32"]


@dataclass(frozen=True)
class ModelSpec:
    report_name: str
    source_arch: str
    manifest_arch: str


MODELS = [
    ModelSpec("TinyNet-E", "tinynet_e", "tinynet_e"),
    ModelSpec("MNASNet0.5", "mnasnet0_5", "mnasnet0_5"),
    ModelSpec("RepGhostNet-0.5", "repghostnet_050", "repghostnet_050"),
    ModelSpec("GhostNetV2", "ghostnetv2_100", "ghostnetv2_100"),
    ModelSpec("GhostNetV1-1.0", "ghostnetv1_100", "ghostnetv1_100"),
    ModelSpec("GhostNetV1-0.5", "ghostnetv1_050", "ghostnetv1_050"),
    ModelSpec("StarNet-S1", "starnet_s1", "starnet_s1"),
    ModelSpec("MobileNetV4-Conv-Small", "mobilenetv4_conv_small", "mobilenetv4_conv_small"),
    ModelSpec("EfficientNet-Lite0", "efficientnet_lite0", "efficientnet_lite0"),
    ModelSpec("RegNetX-002", "regnetx_002", "regnetx_002"),
    ModelSpec("JedgeNet", "jedgenet", "jedgenet"),
]


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def backup(path: Path, stamp: str) -> None:
    if path.exists():
        dst = path.with_name(f"{path.stem}_{stamp}.bak{path.suffix}")
        path.replace(dst)
        print(f"[backup] {path} -> {dst.name}")


def model_source(weights_dir: Path, arch: str, seed: int) -> Path:
    return weights_dir / f"{arch}_s{seed}.tflite"


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seed = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed: {part}") from exc
        if seed < 1:
            raise argparse.ArgumentTypeError("seeds must be positive integers")
        seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sd", type=Path, required=True, help="SD root, e.g. G:/")
    ap.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1"),
                    help="Comma-separated seeds to prepare. Default: 1.")
    ap.add_argument("--clear-models", action="store_true",
                    help="Delete existing SD/models/*.tflite before copying.")
    args = ap.parse_args()

    sd = args.sd
    weights_dir = args.weights_dir
    if not sd.exists():
        ap.error(f"SD path does not exist: {sd}")
    if not weights_dir.is_dir():
        ap.error(f"weights dir does not exist: {weights_dir}")

    missing = []
    for spec in MODELS:
        for seed in args.seeds:
            src = model_source(weights_dir, spec.source_arch, seed)
            if not src.is_file():
                missing.append(str(src))
    if missing:
        ap.error("missing model files:\n  " + "\n  ".join(missing))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup(sd / "manifest.csv", stamp)
    results_dir = sd / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    backup(results_dir / "ram_usage.csv", stamp)

    models_dir = sd / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_models:
        for old in models_dir.glob("*.tflite"):
            old.unlink()

    rows = []
    manifest_rows = []
    for spec in MODELS:
        for seed in args.seeds:
            src = model_source(weights_dir, spec.source_arch, seed)
            dst_name = f"{spec.manifest_arch}_s{seed}.tflite"
            dst = models_dir / dst_name
            shutil.copy2(src, dst)
            crc = crc32_file(dst)
            model_id = f"{spec.manifest_arch}_s{seed}"
            manifest_rows.append({
                "model_id": model_id,
                "model_bin": f"models/{dst_name}",
                "input_w": INPUT_W,
                "input_h": INPUT_H,
                "input_c": INPUT_C,
                "num_classes": NUM_CLASSES,
                "crc32": f"{crc:08x}",
            })
            rows.append({
                "report_name": spec.report_name,
                "model_id": model_id,
                "source": str(src),
                "sd_model": str(dst),
                "bytes": dst.stat().st_size,
                "crc32": f"{crc:08x}",
            })
            print(f"[copy] {model_id:24s} {dst.stat().st_size / 1024:8.1f} KiB  crc={crc:08x}")

    manifest = sd / "manifest.csv"
    with manifest.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(manifest_rows)

    audit = HERE / "out" / "ram_4class_sd_manifest_audit.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["report_name", "model_id", "source", "sd_model", "bytes", "crc32"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] wrote {len(manifest_rows)} models to {sd}")
    print(f"[done] manifest: {manifest}")
    print(f"[done] audit   : {audit}")
    print("[next] build RAM-only firmware, copy tflm_app.bin, then flash/run board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
