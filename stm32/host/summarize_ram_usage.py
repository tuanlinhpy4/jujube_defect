#!/usr/bin/env python3
"""Summarize STM32 RAM-only results by architecture.

Input is the firmware-created results/ram_usage.csv.

Usage:
  python stm32/host/summarize_ram_usage.py --ram-csv G:/results/ram_usage.csv
  python stm32/host/summarize_ram_usage.py --ram-csv G:/results/ram_usage.csv --seed 1
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MID_RE = re.compile(r"^(.*)_s(\d+)$")

DISPLAY = {
    "squeezenet1_1": "SqueezeNet1.1",
    "mobilenet_v2": "MobileNetV2",
    "jedgenet": "JedgeNet",
    "regnetx_002": "RegNetX-002",
    "efficientnet_lite0": "EfficientNet-Lite0",
    "mobilenetv4_conv_small": "MobileNetV4-Conv-Small",
    "starnet_s1": "StarNet-S1",
    "spnasnet_100": "SPNASNet-100",
    "mnasnet0_5": "MNASNet0.5",
    "tinynet_e": "TinyNet-E",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ram-csv", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="Only include rows whose model_id ends with this seed, e.g. 1.")
    ap.add_argument("--out-csv", type=Path, default=HERE / "out" / "ram_usage_4class_summary.csv")
    ap.add_argument("--out-md", type=Path, default=HERE / "out" / "ram_usage_4class_summary.md")
    args = ap.parse_args()

    if not args.ram_csv.is_file():
        ap.error(f"missing ram csv: {args.ram_csv}")

    by_arch = defaultdict(list)
    failures = []
    with args.ram_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model_id = row["model_id"].strip()
            m = MID_RE.match(model_id)
            arch = m.group(1) if m else model_id
            if args.seed is not None:
                if not m or int(m.group(2)) != args.seed:
                    continue
            if row.get("status") != "OK":
                failures.append(row)
                continue
            by_arch[arch].append(row)

    rows = []
    for arch, rs in sorted(by_arch.items()):
        used = [int(r["tflm_arena_used_bytes"]) for r in rs]
        cap = [int(r["tflm_arena_capacity_bytes"]) for r in rs]
        sizes = [int(r["model_size_bytes"]) for r in rs]
        rows.append({
            "model": DISPLAY.get(arch, arch),
            "arch": arch,
            "seeds": len(rs),
            "model_size_kib_mean": round(sum(sizes) / len(sizes) / 1024, 1),
            "arena_used_kib_mean": round(sum(used) / len(used) / 1024, 1),
            "arena_used_kib_max": round(max(used) / 1024, 1),
            "arena_capacity_kib": round(max(cap) / 1024, 1),
        })

    order = {spec: i for i, spec in enumerate(DISPLAY.values())}
    rows.sort(key=lambda r: order.get(r["model"], 999))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "arch", "seeds", "model_size_kib_mean",
            "arena_used_kib_mean", "arena_used_kib_max", "arena_capacity_kib"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# STM32 RAM Usage, 4-Class Models",
        "",
        "| Model | Seeds | TFLite size mean (KiB) | Arena used mean (KiB) | Arena used max (KiB) | Arena capacity (KiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['seeds']} | {r['model_size_kib_mean']:.1f} | "
            f"{r['arena_used_kib_mean']:.1f} | {r['arena_used_kib_max']:.1f} | "
            f"{r['arena_capacity_kib']:.1f} |"
        )
    if failures:
        md.extend(["", f"Failures: {len(failures)} rows. See source CSV for details."])
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[summary] {len(rows)} models -> {args.out_csv}")
    print(f"[summary] markdown -> {args.out_md}")
    if failures:
        print(f"[summary] failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
