#!/usr/bin/env python3
"""Count parameters and MACs/FLOPs for every registered MvANet variant."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from models import build_model, list_archs, list_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count MvANet model complexity")
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=4)
    return parser.parse_args()


def require_ptflops():
    try:
        import ptflops
    except ImportError as exc:
        raise SystemExit("ptflops is required: pip install ptflops") from exc
    return ptflops


def parse_gmacs(macs_str: str) -> float:
    parts = macs_str.split()
    value = float(parts[0])
    unit = parts[1] if len(parts) > 1 else ""
    if "G" in unit:
        return value
    if "M" in unit:
        return value / 1_000.0
    if "K" in unit:
        return value / 1_000_000.0
    return value / 1_000_000_000.0


def analyze(arch_name: str, variants: list, input_size: int, in_channels: int,
            num_classes: int, ptflops_module):
    print(f"\n{'=' * 72}")
    print(f"  {arch_name}")
    print(f"{'=' * 72}")
    results = []
    for name in variants:
        model = build_model(name, num_classes=num_classes, in_channels=in_channels)
        model.eval()
        params = sum(p.numel() for p in model.parameters())

        macs_str, _ = ptflops_module.get_model_complexity_info(
            model,
            input_res=(in_channels, input_size, input_size),
            verbose=False,
            print_per_layer_stat=False,
        )
        gmacs = parse_gmacs(macs_str)
        results.append((name, params, gmacs, macs_str))
        print(
            f"  {name:<22} params: {params:>10,} "
            f"({params * 4 / 1024:>8.1f} KB) | "
            f"MACs: {macs_str:<12} GMACs: {gmacs:.4f}"
        )
    return results


def main() -> None:
    args = parse_args()
    ptflops_module = require_ptflops()

    print("=" * 72)
    print(
        f"  Model complexity at {args.in_channels}x"
        f"{args.input_size}x{args.input_size}"
    )
    print("=" * 72)

    all_results = {}
    for arch_name in list_archs():
        variants = list_variants(arch_name)
        all_results[arch_name] = analyze(
            arch_name, variants,
            args.input_size, args.in_channels, args.num_classes,
            ptflops_module,
        )

    archs = list(all_results.keys())
    if len(archs) >= 2:
        a, b = archs[0], archs[1]
        pairs = list(zip(all_results[a], all_results[b]))
        if pairs:
            print(f"\n{'=' * 72}")
            print(f"  Parameter comparison: {a} vs {b}")
            print(f"{'=' * 72}")
            print(f"  {'Variant':<22} {'orig params':>12}  "
                  f"{'Variant':<24} {'opt params':>12}  {'Change':>8}")
            print(f"  {'-'*22} {'-'*12}  {'-'*24} {'-'*12}  {'-'*8}")
            for (na, pa, _, _), (nb, pb, _, _) in pairs:
                change = f"{(1 - pb / pa) * 100:+.1f}%" if pa else "N/A"
                print(f"  {na:<22} {pa:>12,}  {nb:<24} {pb:>12,}  {change:>8}")


if __name__ == "__main__":
    main()
