#!/usr/bin/env python3
"""M4.0 — Khâu 3/3: GOLDEN trên PC — chạy mỗi .tflite trên 519 ảnh test.

CHẠY TRONG VENV: stm32/host/.venv_tflite (tensorflow).

Input GIỐNG HỆT MCU: đọc thẳng test_data.pack (float32 NCHW) -> transpose NHWC -> Invoke.
Ra: accuracy + macro-F1 (mỗi model-seed) + ước lượng arena. Gộp macro-F1 mean±std theo arch.

  ./.venv_tflite/Scripts/python eval_tflite_golden.py
  ./.venv_tflite/Scripts/python eval_tflite_golden.py --only mnasnet0_5_s1
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tflite_common import macro_f1, preprocess_nchw, read_pack  # noqa: E402

REPO_ROOT = HERE.parents[1]
DEFAULT_PACK = REPO_ROOT / "stm32" / "out" / "SD" / "test" / "test_data.pack"
DEFAULT_TESTDIR = REPO_ROOT / "data_jujube" / "data_resize" / "64_split_70_15_15_seed42" / "test"
CLASS_NAMES = ("cracked", "dry", "insect_damaged", "invalid")
MID_RE = re.compile(r"^(.*)_s(\d+)$")


def load_test_set(pack: Path, testdir: Path, input_size: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Trả về (imgs NCHW [N,3,H,W], labels [N], nguồn)."""
    if pack.is_file():
        imgs, labels = read_pack(pack)
        return imgs, labels, f"pack:{pack.name}"
    # fallback: tiền xử lý lại thư mục test (thứ tự lớp alphabet = nhãn)
    xs, ys = [], []
    for lbl, cls in enumerate(CLASS_NAMES):
        for p in sorted((testdir / cls).rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                xs.append(preprocess_nchw(p, input_size))
                ys.append(lbl)
    return np.stack(xs, 0), np.asarray(ys, np.int64), f"dir:{testdir.name}"


def est_arena(interp) -> tuple[int, int]:
    """Ước lượng thô (sum/max byte tensor biến thiên). Arena thật đo ở M4.1 trên board."""
    total = 0
    biggest = 0
    for d in interp.get_tensor_details():
        shape = d.get("shape_signature", d.get("shape"))
        if shape is None or len(shape) == 0:
            continue
        n = 1
        for s in shape:
            n *= int(s) if int(s) > 0 else 1
        nbytes = n * np.dtype(d["dtype"]).itemsize
        total += nbytes
        biggest = max(biggest, nbytes)
    return total, biggest


def eval_one(tflite_path: Path, imgs_nchw: np.ndarray, labels: np.ndarray,
             num_classes: int) -> dict:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]
    in_shape = list(inp["shape"])  # kỳ vọng [1,64,64,3] NHWC
    nhwc = (len(in_shape) == 4 and in_shape[1] == imgs_nchw.shape[2]
            and in_shape[3] == imgs_nchw.shape[1])

    n = imgs_nchw.shape[0]
    preds = np.empty(n, np.int64)
    for i in range(n):
        x = imgs_nchw[i:i + 1]                       # [1,3,H,W]
        x = np.transpose(x, (0, 2, 3, 1)) if nhwc else x
        interp.set_tensor(inp["index"], x.astype(inp["dtype"], copy=False)
                          if np.dtype(inp["dtype"]) != np.float32 else x)
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(outp["index"])[0]))

    acc, mf1, f1s = macro_f1(labels, preds, num_classes)
    arena_sum, arena_max = est_arena(interp)
    return {
        "status": "OK", "n": n, "accuracy": round(acc, 5),
        "macro_f1": round(mf1, 5),
        "f1_per_class": "|".join(f"{x:.3f}" for x in f1s),
        "bytes": tflite_path.stat().st_size,
        "in_layout": "NHWC" if nhwc else ("x".join(map(str, in_shape))),
        "in_dtype": str(np.dtype(inp["dtype"])),
        "arena_sum_KB": round(arena_sum / 1024, 1),
        "arena_max_KB": round(arena_max / 1024, 1),
    }


def summarize(rows: list[dict]) -> list[dict]:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "OK":
            continue
        m = MID_RE.match(r["model_id"])
        arch = m.group(1) if m else r["model_id"]
        by_arch[arch].append(r)
    out = []
    for arch, rs in sorted(by_arch.items()):
        f1 = np.array([x["macro_f1"] for x in rs], float)
        acc = np.array([x["accuracy"] for x in rs], float)
        out.append({
            "arch": arch, "seeds": len(rs),
            "macro_f1_mean": round(float(f1.mean()), 4),
            "macro_f1_std": round(float(f1.std(ddof=0)), 4),
            "acc_mean": round(float(acc.mean()), 4),
            "acc_std": round(float(acc.std(ddof=0)), 4),
            "tflite_KB": round(np.mean([x["bytes"] for x in rs]) / 1024, 1),
            "arena_max_KB": round(max(x["arena_max_KB"] for x in rs), 1),
        })
    out.sort(key=lambda d: d["macro_f1_mean"], reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite-dir", type=Path, default=HERE / "out" / "tflite")
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--test-dir", type=Path, default=DEFAULT_TESTDIR)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "golden_tflite.csv")
    ap.add_argument("--summary", type=Path, default=HERE / "out" / "golden_summary.csv")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    files = sorted(args.tflite_dir.glob("*.tflite"))
    if args.only:
        files = [p for p in files if p.stem in set(args.only)]
    if not files:
        print(f"[err] không có .tflite trong {args.tflite_dir}")
        return 1

    imgs, labels, src = load_test_set(args.pack, args.test_dir, args.input_size)
    print(f"[test] {imgs.shape[0]} ảnh từ {src}, NCHW {imgs.shape[1:]}\n")

    rows, ok = [], 0
    for i, p in enumerate(files, 1):
        mid = p.stem
        print(f"[{i:>2}/{len(files)}] {mid:<32} ", end="", flush=True)
        rec = {"model_id": mid}
        try:
            rec.update(eval_one(p, imgs, labels, args.num_classes))
            ok += 1
            print(f"acc={rec['accuracy']:.4f}  F1={rec['macro_f1']:.4f}  "
                  f"{rec['in_layout']}/{rec['in_dtype']}  arena~{rec['arena_max_KB']}KB")
        except Exception as e:
            rec["status"] = "FAIL"
            rec["note"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAIL  {rec['note']}")
            traceback.print_exc()
        rows.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model_id", "status", "n", "accuracy", "macro_f1", "f1_per_class",
            "bytes", "in_layout", "in_dtype", "arena_sum_KB", "arena_max_KB", "note"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summ = summarize(rows)
    scols = ["arch", "seeds", "macro_f1_mean", "macro_f1_std", "acc_mean", "acc_std",
             "tflite_KB", "arena_max_KB"]
    with args.summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        w.writerows(summ)

    print(f"\n[done] {ok}/{len(files)} OK")
    print(f"  chi tiết : {args.out}")
    print(f"  tóm tắt  : {args.summary}\n")
    if summ:
        print("== Bảng GOLDEN macro-F1 (mean±std theo arch, PC) ==")
        print(f"{'arch':<26}{'seeds':>6}{'macroF1':>12}{'acc':>12}{'tfliteKB':>10}{'arenaKB':>9}")
        for d in summ:
            print(f"{d['arch']:<26}{d['seeds']:>6}"
                  f"{d['macro_f1_mean']:>8.4f}±{d['macro_f1_std']:.3f}"
                  f"{d['acc_mean']:>8.4f}±{d['acc_std']:.3f}"
                  f"{d['tflite_KB']:>10.1f}{d['arena_max_KB']:>9.1f}")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
