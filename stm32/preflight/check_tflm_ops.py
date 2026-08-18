#!/usr/bin/env python3
"""
STM32 INT8 pre-flight check (op-coverage + flash fit).

For each candidate model:
  build (timm / torchvision)  ->  ONNX export @ NxN  ->  op-set analysis vs
  TFLite-Micro / CMSIS-NN INT8 coverage  ->  params / INT8 flash-fit  ->
  (optional, --quantize) real ONNXRuntime static INT8 PTQ to confirm it
  quantizes cleanly and to measure the true INT8 byte size.

This is a HEURISTIC pre-filter to drop obviously-unfit models BEFORE touching
the STM32. The definitive op check is a real TFLite conversion (needs TensorFlow,
not installed here) — add that as a follow-up stage once the green list is small.

Deps: torch, timm, onnx, numpy. Optional: torchvision (shufflenet/squeezenet),
onnxruntime (--quantize).
"""
from __future__ import annotations
import argparse, json, os, sys, traceback, warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Candidate registry. tier: pick = my recommended ~13; contrast = included to
# prove the checker separates green/yellow/red. Edit freely.
# ----------------------------------------------------------------------------
MODELS = [
    # --- recommended 13 (the MCU run) ---
    dict(key="EfficientNet-Lite0",      src="timm", id="tf_efficientnet_lite0",  tier="pick"),
    dict(key="MobileNetV2",             src="timm", id="mobilenetv2_100",        tier="pick"),
    dict(key="MobileNetV3-Small",       src="timm", id="mobilenetv3_small_100",  tier="pick"),
    dict(key="MobileNetV4-Conv-Small",  src="timm", id="mobilenetv4_conv_small", tier="pick"),
    dict(key="MNASNet-05",              src="timm", id="mnasnet_050",            tier="pick"),
    dict(key="FBNet-C100",              src="timm", id="fbnetc_100",             tier="pick"),
    dict(key="ShuffleNetV2-0.5",        src="tv",   id="shufflenet_v2_x0_5",     tier="pick"),
    dict(key="SqueezeNet-1.1",          src="tv",   id="squeezenet1_1",          tier="pick"),
    dict(key="RegNetX-002",             src="timm", id="regnetx_002",            tier="pick"),
    dict(key="RepVGG-A0",               src="timm", id="repvgg_a0",  reparam=True, tier="pick"),
    dict(key="GhostNetV2-100",          src="timm", id="ghostnetv2_100",         tier="pick"),
    dict(key="TinyNet-E",               src="timm", id="tinynet_e",              tier="pick"),
    dict(key="JU-Lite-Mobile-Att-Micro",src="custom", id="",                     tier="pick"),
    # --- contrast set (validate the checker) ---
    dict(key="SPNASNet-100",            src="timm", id="spnasnet_100",           tier="contrast"),
    dict(key="EfficientNet-B0",         src="timm", id="efficientnet_b0",        tier="contrast"),
    dict(key="StarNet-S1",              src="timm", id="starnet_s1",             tier="contrast"),
    dict(key="ConvNeXt-Atto",           src="timm", id="convnext_atto",          tier="contrast"),
    dict(key="ResNet18",                src="timm", id="resnet18",               tier="contrast"),
    dict(key="EfficientViT-M0",         src="timm", id="efficientvit_m0",        tier="contrast"),
    dict(key="FastViT-T8",              src="timm", id="fastvit_t8",             tier="contrast"),
    dict(key="PoolFormer-S12",          src="timm", id="poolformer_s12",         tier="contrast"),
]

# ----------------------------------------------------------------------------
# TFLite-Micro / CMSIS-NN INT8 op coverage (heuristic, keyed by ONNX op_type).
#   OK    : has a solid INT8 kernel, CMSIS-NN friendly
#   FOLD  : disappears at conversion (folded into conv / graph plumbing)
#   RISKY : supported-but-fragile on INT8, OR ambiguous (e.g. MatMul = FC or attention),
#           OR a primitive that usually signals LayerNorm/GELU/attention
#   BAD   : no practical INT8 kernel in TFLM/CMSIS-NN -> will fail or be unusable
# ----------------------------------------------------------------------------
OK_OPS = {
    "Conv", "ConvTranspose", "Gemm", "Relu", "PRelu", "LeakyRelu", "Elu",
    "Clip", "Add", "Sub", "Mul",          # Mul/Add = residual / SE-scale / star-op (INT8 ok)
    "Mean", "ReduceMean",                  # global-avg-pool / SE squeeze -> TFLM MEAN (INT8 ok)
    "Concat", "MaxPool", "AveragePool", "GlobalAveragePool",
    "Pad", "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Split",
    "Slice", "HardSwish", "HardSigmoid", "Resize", "Softmax", "Min", "Max",
}
FOLD_OPS = {
    "BatchNormalization", "Constant", "ConstantOfShape", "Identity", "Shape",
    "Cast", "Dropout", "Unsqueeze",  # shape plumbing usually constant-folded
    # static shape-arithmetic emitted by TF-style 'same' padding & channel-shuffle
    # (operate on shape ints, get constant-folded by the TFLite converter):
    "Gather", "Ceil", "Floor", "Range", "Equal", "Where",
}
RISKY_OPS = {
    "MatMul",      # FC (ok) OR attention/dynamic batch-matmul (bad) — ambiguous
    "Sigmoid",     # SE-gate / swish -> genuine INT8 quality risk
    "Div", "Pow", "Sqrt", "Reciprocal",   # normalization tail (LayerNorm-ish) or real divide
    "ReduceSum", "ReduceMax", "Exp",       # softmax/attention internals
    "ScatterND", "Expand", "Tile", "Mod",
}
BAD_OPS = {
    "LayerNormalization", "InstanceNormalization", "GroupNormalization",
    "Erf", "Gelu", "BiasGelu",            # GELU (vision transformers / convnext)
    "Einsum",                              # attention
    "GridSample", "NonMaxSuppression", "RoiAlign", "TopK", "NonZero",
    "CumSum", "Trilu", "LpNormalization",
}

QSPI_SLOT_MB = 6.0   # plan's current_model_slot
QSPI_TOTAL_MB = 8.0  # whole QSPI chip


def classify_ops(op_counter):
    bad, risky, unknown = {}, {}, {}
    for op, n in op_counter.items():
        if op in BAD_OPS:
            bad[op] = n
        elif op in OK_OPS or op in FOLD_OPS:
            pass
        elif op in RISKY_OPS:
            risky[op] = n
        else:
            unknown[op] = n
    if bad:
        verdict = "RED"
    elif risky or unknown:
        verdict = "YELLOW"
    else:
        verdict = "GREEN"
    return verdict, bad, risky, unknown


def maybe_reparam(m):
    for mod_name, fn_name in [("timm.utils", "reparameterize_model"),
                              ("timm.models", "reparameterize_model")]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            return getattr(mod, fn_name)(m)
        except Exception:
            pass
    if hasattr(m, "reparameterize"):
        try:
            m.reparameterize()
        except Exception:
            pass
    return m


def build_model(spec, num_classes):
    src = spec["src"]
    if src == "timm":
        import timm
        m = timm.create_model(spec["id"], pretrained=False, num_classes=num_classes)
    elif src == "tv":
        import torchvision.models as tvm
        m = getattr(tvm, spec["id"])(weights=None, num_classes=num_classes)
    else:
        raise RuntimeError(f"unsupported source '{src}'")
    if spec.get("reparam"):
        m = maybe_reparam(m)
    m.eval()
    return m


def run_one(spec, args, onnx_dir):
    import torch, onnx
    key = spec["key"]
    rec = dict(model=key, source=spec["src"], id=spec["id"], tier=spec["tier"],
               status="", params_M=None, int8_MB=None, fits_6MB=None, fits_8MB=None,
               n_nodes=None, ops_verdict="", ops_bad="", ops_risky="", ops_unknown="",
               final="", note="")

    if spec["src"] == "custom":
        rec["status"] = "SKIP"
        rec["final"] = "TODO"
        rec["note"] = ("custom model — not in timm. Drop your own .onnx in "
                       f"{os.path.basename(onnx_dir)}/ and re-run with --extra-onnx, "
                       "or add a builder to MODELS.")
        return rec

    # build
    try:
        m = build_model(spec, args.num_classes)
    except Exception as e:
        rec["status"] = "BUILD_FAIL"; rec["final"] = "NO-GO"
        rec["note"] = f"build: {type(e).__name__}: {e}"
        return rec

    # params / flash
    n_params = sum(p.numel() for p in m.parameters())
    rec["params_M"] = round(n_params / 1e6, 3)
    int8_mb = n_params / 1e6  # ~1 byte / weight at INT8
    rec["int8_MB"] = round(int8_mb, 2)
    rec["fits_6MB"] = bool(int8_mb <= QSPI_SLOT_MB)
    rec["fits_8MB"] = bool(int8_mb <= QSPI_TOTAL_MB)

    # export ONNX
    onnx_path = os.path.join(onnx_dir, f"{key}.onnx")
    try:
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
        torch.onnx.export(m, dummy, onnx_path, opset_version=args.opset,
                          do_constant_folding=True, input_names=["input"],
                          output_names=["logits"])
    except Exception as e:
        rec["status"] = "EXPORT_FAIL"; rec["final"] = "NO-GO"
        rec["note"] = f"onnx export: {type(e).__name__}: {str(e)[:160]}"
        return rec

    # op analysis
    try:
        model = onnx.load(onnx_path)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        opc = Counter(n.op_type for n in model.graph.node)
        rec["n_nodes"] = sum(opc.values())
        verdict, bad, risky, unknown = classify_ops(opc)
        rec["ops_verdict"] = verdict
        rec["ops_bad"] = ",".join(f"{k}x{v}" for k, v in sorted(bad.items()))
        rec["ops_risky"] = ",".join(f"{k}x{v}" for k, v in sorted(risky.items()))
        rec["ops_unknown"] = ",".join(f"{k}x{v}" for k, v in sorted(unknown.items()))
    except Exception as e:
        rec["status"] = "ANALYZE_FAIL"; rec["final"] = "NO-GO"
        rec["note"] = f"op analysis: {type(e).__name__}: {e}"
        return rec

    # optional real INT8 quantization (size + clean-quantize signal)
    if args.quantize:
        try:
            int8_path = os.path.join(onnx_dir, f"{key}.int8.onnx")
            sz = quantize_int8(onnx_path, int8_path, args.imgsz)
            rec["int8_MB"] = round(sz / 1e6, 2)
            rec["fits_6MB"] = bool(sz / 1e6 <= QSPI_SLOT_MB)
            rec["fits_8MB"] = bool(sz / 1e6 <= QSPI_TOTAL_MB)
        except Exception as e:
            rec["note"] = (rec["note"] + " | " if rec["note"] else "") + \
                          f"int8 quant FAILED: {type(e).__name__}: {str(e)[:120]}"

    # final verdict
    rec["status"] = "OK"
    if rec["ops_verdict"] == "RED" or not rec["fits_8MB"]:
        rec["final"] = "NO-GO"
    elif rec["ops_verdict"] == "YELLOW" or not rec["fits_6MB"]:
        rec["final"] = "CAUTION"
    else:
        rec["final"] = "GO"

    notes = []
    if not rec["fits_8MB"]:
        notes.append(f"INT8 {rec['int8_MB']}MB > {QSPI_TOTAL_MB:.0f}MB QSPI")
    elif not rec["fits_6MB"]:
        notes.append(f"INT8 {rec['int8_MB']}MB > {QSPI_SLOT_MB:.0f}MB slot (needs SRAM-load or resize slot)")
    if rec["ops_bad"]:
        notes.append("unsupported ops")
    if rec["ops_risky"] and rec["ops_verdict"] != "RED":
        notes.append("INT8/op risk")
    if notes:
        rec["note"] = (rec["note"] + " | " if rec["note"] else "") + "; ".join(notes)
    return rec


def quantize_int8(fp32_path, int8_path, imgsz):
    """Static INT8 PTQ with random calibration (feasibility/size only, not accuracy)."""
    import numpy as np
    from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                          CalibrationDataReader)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32_path + ".pre.onnx"
    quant_pre_process(fp32_path, pre, skip_symbolic_shape=True)

    class RandReader(CalibrationDataReader):
        def __init__(self, n=8):
            self.data = iter([{"input": np.random.randn(1, 3, imgsz, imgsz).astype("float32")}
                              for _ in range(n)])
        def get_next(self):
            return next(self.data, None)

    quantize_static(pre, int8_path, RandReader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8)
    try:
        os.remove(pre)
    except OSError:
        pass
    return os.path.getsize(int8_path)


def main():
    ap = argparse.ArgumentParser(description="STM32 INT8 pre-flight: op coverage + flash fit")
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize", action="store_true",
                    help="also run real ONNXRuntime static INT8 PTQ (true size, clean-quant check)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset by model key substring (e.g. --only Mobile RepVGG)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    try:  # Windows consoles default to cp1252 and choke on emoji / em-dash
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    onnx_dir = os.path.join(args.outdir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    models = MODELS
    if args.only:
        models = [m for m in MODELS if any(s.lower() in m["key"].lower() for s in args.only)]

    print(f"# STM32 INT8 pre-flight  |  input {args.imgsz}x{args.imgsz}  |  "
          f"{len(models)} models  |  quantize={args.quantize}\n")

    rows = []
    for i, spec in enumerate(models, 1):
        print(f"[{i:>2}/{len(models)}] {spec['key']:<24} ", end="", flush=True)
        try:
            rec = run_one(spec, args, onnx_dir)
        except Exception as e:
            rec = dict(model=spec["key"], source=spec["src"], id=spec["id"], tier=spec["tier"],
                       status="CRASH", final="NO-GO", note=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        rows.append(rec)
        tag = rec.get("final", "?")
        extra = ""
        if rec.get("params_M") is not None:
            extra = f"{rec['params_M']:>6}M / {rec.get('int8_MB','?')}MB  ops:{rec.get('ops_verdict','-')}"
        print(f"{tag:<8} {extra}  {rec.get('note','')}")

    # write CSV + JSON
    cols = ["model", "source", "id", "tier", "status", "params_M", "int8_MB",
            "fits_6MB", "fits_8MB", "n_nodes", "ops_verdict", "ops_bad",
            "ops_risky", "ops_unknown", "final", "note"]
    csv_path = os.path.join(args.outdir, "preflight_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join('"' + str(r.get(c, "")).replace('"', "'") + '"' for c in cols) + "\n")
    with open(os.path.join(args.outdir, "preflight_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # summary
    def bucket(v):
        return [r["model"] for r in rows if r.get("final") == v]
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for v, label in [("GO", "✅ GO       (op-clean + fits QSPI slot)"),
                     ("CAUTION", "🟡 CAUTION  (INT8/op risk or tight flash)"),
                     ("NO-GO", "🔴 NO-GO    (unsupported ops or too big)"),
                     ("TODO", "⚪ TODO     (custom / build it yourself)")]:
        b = bucket(v)
        print(f"\n{label}  [{len(b)}]")
        for name in b:
            r = next(x for x in rows if x["model"] == name)
            d = f"{r.get('int8_MB','?')}MB" if r.get("int8_MB") is not None else ""
            why = r.get("note") or r.get("status")
            print(f"   - {name:<24} {d:<8} {why}")
    print(f"\nReport: {csv_path}")
    print(f"ONNX:   {onnx_dir}")


if __name__ == "__main__":
    main()
