#!/usr/bin/env python3
"""Gộp f1_latency.csv theo kiến trúc -> Macro-F1 và latency mean±std 5 seed.

Xếp hạng chính theo Macro-F1. Nếu có golden_summary.csv thì
đối chiếu on-device vs golden (PC). Chạy ENV CHÍNH (pure python).

  python stm32/host/aggregate_results.py --results G:/results/f1_latency.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MID_RE = re.compile(r"^(.*)_s(\d+)$")


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n  # population std (ddof=0), khớp golden
    return m, math.sqrt(var)


def load_golden(path: Path) -> dict:
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["arch"]] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=HERE.parents[1] / "stm32" / "out" / "SD" / "results" / "f1_latency.csv")
    ap.add_argument("--golden", type=Path, default=HERE / "out" / "golden_summary.csv")
    ap.add_argument("--out-csv", type=Path, default=HERE / "out" / "benchmark_table.csv")
    ap.add_argument("--out-md", type=Path, default=HERE / "out" / "benchmark_table.md")
    args = ap.parse_args()

    if not args.results.is_file():
        ap.error(f"không thấy f1_latency.csv: {args.results}")

    by_arch = defaultdict(lambda: defaultdict(list))
    nrows = 0
    with args.results.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = MID_RE.match(row["model_id"].strip())
            arch = m.group(1) if m else row["model_id"].strip()
            by_arch[arch]["f1"].append(float(row["macro_f1"]))
            by_arch[arch]["la"].append(float(row["latency_avg_ms"]))
            by_arch[arch]["lp"].append(float(row["latency_p95_ms"]))
            nrows += 1

    golden = load_golden(args.golden)

    rows = []
    for arch, d in by_arch.items():
        f1m, f1s = mean_std(d["f1"])
        lam, las = mean_std(d["la"])
        lpm, lps = mean_std(d["lp"])
        g = golden.get(arch, {})
        rows.append({
            "arch": arch, "seeds": len(d["f1"]),
            "macro_f1_mean": round(f1m, 4), "macro_f1_std": round(f1s, 4),
            "latency_avg_ms_mean": round(lam, 3), "latency_avg_ms_std": round(las, 3),
            "latency_p95_ms_mean": round(lpm, 3), "latency_p95_ms_std": round(lps, 3),
            "golden_f1_mean": g.get("macro_f1_mean", ""),
            "f1_delta_vs_golden": (round(f1m - float(g["macro_f1_mean"]), 4)
                                   if g.get("macro_f1_mean") else ""),
        })
    rows.sort(key=lambda r: r["macro_f1_mean"], reverse=True)

    cols = ["arch", "seeds", "macro_f1_mean", "macro_f1_std",
            "latency_avg_ms_mean", "latency_avg_ms_std",
            "latency_p95_ms_mean", "latency_p95_ms_std",
            "golden_f1_mean", "f1_delta_vs_golden"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md = ["# Benchmark táo tàu — INT8 trên STM32H750 (TFLite Micro)",
          "",
          f"Gộp {nrows} model-seed theo {len(rows)} kiến trúc. Xếp theo **macro-F1 (mean±std, 5 seed)**.",
          "",
          "| # | Kiến trúc | macro-F1 | latency avg (ms) | p95 (ms) | golden F1 | Δ |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['arch']} | {r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
                  f"| {r['latency_avg_ms_mean']:.3f}±{r['latency_avg_ms_std']:.3f} "
                  f"| {r['latency_p95_ms_mean']:.3f}±{r['latency_p95_ms_std']:.3f} "
                  f"| {r['golden_f1_mean']} "
                  f"| {r['f1_delta_vs_golden']} |")
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[agg] {nrows} dòng -> {len(rows)} kiến trúc")
    print(f"  CSV: {args.out_csv}")
    print(f"  MD : {args.out_md}\n")
    print(f"{'#':<3}{'arch':<34}{'macroF1':>16}{'lat_ms':>18}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}{r['arch']:<34}{r['macro_f1_mean']:>8.4f}±{r['macro_f1_std']:.4f}"
              f"{r['latency_avg_ms_mean']:>10.3f}±{r['latency_avg_ms_std']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
