"""Find the fastest working ONNX Runtime INT8 configuration.

ORT's CPU backend only reaches its fast INT8 kernels when the quantized graph
fuses into QLinearConv, and whether that happens depends on the format,
per-channel setting and reduce_range. Rather than report whichever config was
tried first, this sweeps the plausible ones on a short latency run.

    python benchmarks/onnx_int8_sweep.py
"""

import os
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from bench_common import ROOT, calibration_images, preprocess, test_images

FP32_MODEL = os.path.join(ROOT, "best.onnx")
PREPROC_MODEL = os.path.join(ROOT, "best_preproc.onnx")
SCRATCH = os.path.join(ROOT, "benchmarks", "_sweep")


class Reader(CalibrationDataReader):
    def __init__(self, input_name, paths):
        self.input_name, self.paths, self.i = input_name, list(paths), 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        b = preprocess(self.paths[self.i])
        self.i += 1
        return {self.input_name: b}

    def rewind(self):
        self.i = 0


CONFIGS = [
    ("QDQ  per-channel", dict(quant_format=QuantFormat.QDQ, per_channel=True)),
    ("QDQ  per-tensor", dict(quant_format=QuantFormat.QDQ, per_channel=False)),
    ("QOp  per-channel", dict(quant_format=QuantFormat.QOperator, per_channel=True)),
    ("QOp  per-tensor", dict(quant_format=QuantFormat.QOperator, per_channel=False)),
    ("QDQ  per-tensor reduce_range",
     dict(quant_format=QuantFormat.QDQ, per_channel=False, reduce_range=True)),
]


def time_model(path, n=8):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    paths = test_images()[:n]
    batch = preprocess(paths[0])
    for _ in range(3):
        sess.run(None, {name: batch})

    times = []
    for p in paths:
        b = preprocess(p)
        t = time.perf_counter()
        sess.run(None, {name: b})
        times.append((time.perf_counter() - t) * 1000)
    return float(np.mean(times))


def count_ops(path):
    import onnx
    m = onnx.load(path)
    kinds = {}
    for node in m.graph.node:
        kinds[node.op_type] = kinds.get(node.op_type, 0) + 1
    return kinds


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    sess = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    fp32_ms = time_model(FP32_MODEL)
    print(f"{'config':<32} {'mean ms':>9}  {'vs FP32':>8}   key ops")
    print("-" * 86)
    print(f"{'FP32 baseline':<32} {fp32_ms:>9.1f}  {'1.00x':>8}")

    if not os.path.exists(PREPROC_MODEL):
        quant_pre_process(FP32_MODEL, PREPROC_MODEL, skip_symbolic_shape=True)
    paths = calibration_images(50)

    for label, kwargs in CONFIGS:
        out = os.path.join(SCRATCH, label.replace(" ", "_") + ".onnx")
        if not os.path.exists(out):
            quantize_static(
                PREPROC_MODEL, out, Reader(input_name, paths),
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv"], **kwargs)
        ms = time_model(out)
        ops = count_ops(out)
        key = " ".join(f"{k}={v}" for k, v in sorted(ops.items())
                       if k in ("Conv", "QLinearConv", "ConvInteger",
                                "QuantizeLinear", "DequantizeLinear"))
        print(f"{label:<32} {ms:>9.1f}  {fp32_ms / ms:>7.2f}x   {key}")


if __name__ == "__main__":
    main()
