#!/usr/bin/env python3
"""
FLOPs counter: So sánh tất cả architectures có trong models/
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

try:
    import ptflops
except ImportError:
    print("Cần cài ptflops: pip install ptflops")
    sys.exit(1)

from models import registry


def parse_gflops(macs_str):
    parts = macs_str.split()
    val = float(parts[0])
    unit = parts[1] if len(parts) > 1 else ""
    if "G" in unit:
        return val
    elif "M" in unit:
        return val / 1000
    elif "K" in unit:
        return val / 1e6
    else:
        return val / 1e9


def analyze(arch_name, variants):
    print(f"\n{'=' * 65}")
    print(f"  {arch_name}")
    print(f"{'=' * 65}")
    results = []
    for name in variants:
        model = registry.build(arch_name, name, num_classes=4)
        params = sum(p.numel() for p in model.parameters())
        model.eval()
        macs_str, _ = ptflops.get_model_complexity_info(
            model,
            input_res=(3, 32, 32),
            input_constructor=lambda x: torch.randn(1, 3, 32, 32),
            verbose=False,
        )
        gflops = parse_gflops(macs_str)
        results.append((name, params, gflops, macs_str))
        print(f"  {name:<16}  params: {params:>10,} ({params*4/1024:>8.1f} KB)"
              f"  |  MACs: {macs_str:<12}  GFLOPs: {gflops:.4f}")
    return results


print("=" * 65)
print("  SO SÁNH TẤT CẢ MODELS")
print("=" * 65)

all_results = {}
for arch_name in registry.list_archs():
    variants = registry.list_variants(arch_name)
    all_results[arch_name] = analyze(arch_name, variants)

# Bảng so sánh nếu có >= 2 architectures
archs = list(all_results.keys())
if len(archs) >= 2:
    a, b = archs[0], archs[1]
    res_a, res_b = all_results[a], all_results[b]
    print(f"\n{'=' * 65}")
    print(f"  SO SÁNH: {a} vs {b}")
    print(f"{'=' * 65}")
    print(f"  {'Variant':<16} {a+' params':>14} {b+' params':>14} {'Params Down':>14}")
    print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*14}")
    for (name, pa, ga, _), (_, pb, gb, _) in zip(res_a, res_b):
        if pa > 0:
            pd = f"{(1-pb/pa)*100:+.1f}%"
        else:
            pd = "N/A"
        print(f"  {name:<16} {pa:>14,} {pb:>14,} {pd:>14}")
