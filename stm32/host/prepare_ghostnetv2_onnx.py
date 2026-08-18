#!/usr/bin/env python3
"""Rewrite an exported GhostNetV2 ONNX graph into an equivalent MCU form.

The timm GhostNetV2 graph is mathematically valid, but its dynamic shape
plumbing around GhostModuleV2 confuses onnx2tf's NCHW -> NHWC conversion.
For the fixed 64x64, batch-1 STM32 deployment graph this tool:

* removes channel Slice nodes whose input/output shapes are identical;
* freezes each nearest-neighbour Resize target to its known static shape;
* expresses HardSigmoid as relu6(x + 3) / 6, which TFLite can fuse into
  already-supported arithmetic ops instead of emitting RELU_0_TO_1.

No learned parameter is changed. Use --verify-calib to compare the rewritten
graph against the source graph with ONNX Runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def inferred_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = list(inferred.graph.value_info)
    values += list(inferred.graph.input) + list(inferred.graph.output)
    return {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in values
    }


def resize_targets(model: onnx.ModelProto,
                   shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    targets: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        mul = next(
            (item for item in consumers.get(node.output[0], [])
             if item.op_type == "Mul"),
            None,
        )
        if mul is None:
            raise RuntimeError(f"Resize has no Ghost gate Mul consumer: {node.name}")
        peer_name = mul.input[0] if mul.input[1] == node.output[0] else mul.input[1]
        input_shape = shapes.get(node.input[0])
        peer_shape = shapes.get(peer_name)
        if not input_shape or not peer_shape or len(peer_shape) != 4:
            raise RuntimeError(
                f"Cannot determine static Resize target for {node.name}: "
                f"input={input_shape}, peer={peer_shape}"
            )
        channels = int(input_shape[1] or peer_shape[1])
        targets[node.name] = np.asarray(
            [1, channels, int(peer_shape[2]), int(peer_shape[3])],
            dtype=np.int64,
        )
    return targets


def rewrite(source: Path, output: Path) -> dict[str, int]:
    model = onnx.load(str(source))
    shapes = inferred_shapes(model)
    targets = resize_targets(model, shapes)
    graph = gs.import_onnx(model)

    removed_slices = 0
    for node in list(graph.nodes):
        if node.op != "Slice" or not node.inputs or not node.outputs:
            continue
        input_shape = shapes.get(node.inputs[0].name)
        output_shape = shapes.get(node.outputs[0].name)
        if not input_shape or len(input_shape) != 4 or input_shape != output_shape:
            continue
        old_output = node.outputs[0]
        new_output = node.inputs[0]
        for consumer in list(old_output.outputs):
            consumer.inputs = [
                new_output if item is old_output else item
                for item in consumer.inputs
            ]
        node.outputs.clear()
        removed_slices += 1

    frozen_resizes = 0
    for node in graph.nodes:
        if node.op != "Resize":
            continue
        while len(node.inputs) < 4:
            node.inputs.append(gs.Variable.empty())
        const_name = node.name.replace("/", "_") + "_static_sizes"
        node.inputs[3] = gs.Constant(const_name, targets[node.name])
        frozen_resizes += 1

    rewritten_hardsigmoid = 0
    replacement_nodes: list[gs.Node] = []
    for node in list(graph.nodes):
        if node.op != "HardSigmoid":
            continue
        source_tensor = node.inputs[0]
        output_tensor = node.outputs[0]
        base = node.name.replace("/", "_")
        add_output = gs.Variable(base + "_add3_output", dtype=np.float32)
        clip_output = gs.Variable(base + "_relu6_output", dtype=np.float32)
        replacement_nodes.extend([
            gs.Node(
                op="Add",
                name=base + "_Add3",
                inputs=[source_tensor,
                        gs.Constant(base + "_three", np.asarray(3.0, np.float32))],
                outputs=[add_output],
            ),
            gs.Node(
                op="Clip",
                name=base + "_Relu6",
                inputs=[
                    add_output,
                    gs.Constant(base + "_zero", np.asarray(0.0, np.float32)),
                    gs.Constant(base + "_six", np.asarray(6.0, np.float32)),
                ],
                outputs=[clip_output],
            ),
            gs.Node(
                op="Mul",
                name=base + "_Div6",
                inputs=[
                    clip_output,
                    gs.Constant(base + "_one_sixth",
                                np.asarray(1.0 / 6.0, np.float32)),
                ],
                outputs=[output_tensor],
            ),
        ])
        node.outputs.clear()
        rewritten_hardsigmoid += 1

    graph.nodes.extend(replacement_nodes)
    graph.cleanup().toposort()
    output.parent.mkdir(parents=True, exist_ok=True)
    rewritten = gs.export_onnx(graph)
    onnx.checker.check_model(rewritten)
    onnx.save(rewritten, str(output))
    return {
        "removed_noop_slices": removed_slices,
        "frozen_resizes": frozen_resizes,
        "rewritten_hardsigmoid": rewritten_hardsigmoid,
        "nodes": len(graph.nodes),
    }


def verify(source: Path, output: Path, calib_path: Path, samples: int) -> None:
    import onnxruntime as ort

    data = np.load(calib_path, mmap_mode="r")[:samples]
    source_session = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"])
    output_session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"])
    source_name = source_session.get_inputs()[0].name
    output_name = output_session.get_inputs()[0].name
    max_abs = 0.0
    for image in data:
        nchw = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        expected = source_session.run(None, {source_name: nchw})[0]
        actual = output_session.run(None, {output_name: nchw})[0]
        max_abs = max(max_abs, float(np.max(np.abs(expected - actual))))
    print(f"verify_samples={len(data)} max_abs={max_abs:.9g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-calib", type=Path)
    parser.add_argument("--verify-samples", type=int, default=5)
    args = parser.parse_args()

    stats = rewrite(args.source, args.output)
    print(" ".join(f"{key}={value}" for key, value in stats.items()))
    print(f"output={args.output} bytes={args.output.stat().st_size}")
    if args.verify_calib:
        verify(args.source, args.output, args.verify_calib, args.verify_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
