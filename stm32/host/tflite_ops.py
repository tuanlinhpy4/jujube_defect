#!/usr/bin/env python3
"""Liệt kê builtin op của mỗi .tflite + HỢP toàn bộ -> phục vụ MicroMutableOpResolver (M4.1).

CHẠY TRONG VENV (cần tensorflow để lấy schema flatbuffer).

  ./.venv_tflite/Scripts/python tflite_ops.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def builtin_name_map():
    from tensorflow.lite.python import schema_py_generated as schema
    bo = schema.BuiltinOperator
    return {v: k for k, v in vars(bo).items() if isinstance(v, int)}


def ops_of(path: Path, names: dict) -> list[str]:
    from tensorflow.lite.python import schema_py_generated as schema
    buf = bytearray(path.read_bytes())
    model = schema.Model.GetRootAsModel(buf, 0)
    out = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        out.append(names.get(code, f"CUSTOM/{oc.CustomCode()}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite-dir", type=Path, default=HERE / "out" / "tflite")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "tflm_ops.txt")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    names = builtin_name_map()
    files = sorted(args.tflite_dir.glob("*.tflite"))
    if not files:
        print(f"[err] không có .tflite trong {args.tflite_dir}")
        return 1

    union: Counter = Counter()
    per_arch: dict[str, set] = {}
    for p in files:
        ops = ops_of(p, names)
        union.update(set(ops))
        arch = p.stem.rsplit("_s", 1)[0]
        per_arch.setdefault(arch, set()).update(ops)

    lines = ["# HỢP builtin op trên toàn bộ tflite (đưa vào MicroMutableOpResolver):"]
    for op, n in sorted(union.items()):
        lines.append(f"  {op:<28} (xuất hiện ở {n}/{len(files)} model)")
    lines.append("")
    lines.append("# Op theo từng arch:")
    for arch, ops in sorted(per_arch.items()):
        lines.append(f"  {arch:<26} {', '.join(sorted(ops))}")
    text = "\n".join(lines)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[done] union {len(union)} op -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
