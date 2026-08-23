"""ONNX Runtime INT8: quantize best.onnx statically, then benchmark it.

Uses post-training static quantization (QDQ format, per-channel weights,
asymmetric activations) calibrated on the same 100 training images the custom
engine's own calibration uses, so the comparison is like-for-like.

    python benchmarks/onnx_int8_benchmark.py
"""

import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from bench_common import ROOT, benchmark, calibration_images, preprocess

FP32_MODEL = os.path.join(ROOT, "best.onnx")
PREPROC_MODEL = os.path.join(ROOT, "best_preproc.onnx")
INT8_MODEL = os.path.join(ROOT, "best_int8.onnx")


class FireCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name, paths):
        self.input_name = input_name
        self.paths = list(paths)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        batch = preprocess(self.paths[self.i])
        self.i += 1
        return {self.input_name: batch}

    def rewind(self):
        self.i = 0


def main():
    if not os.path.exists(FP32_MODEL):
        raise SystemExit(f"{FP32_MODEL} missing -- run benchmarks/export_onnx.py first")

    session = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if not os.path.exists(INT8_MODEL):
        # Shape inference + constant folding; static quantization needs it.
        print("Pre-processing model for quantization...")
        quant_pre_process(FP32_MODEL, PREPROC_MODEL, skip_symbolic_shape=True)

        paths = calibration_images(100)
        print(f"Quantizing with {len(paths)} calibration images "
              f"(this takes a few minutes)...")
        quantize_static(
            PREPROC_MODEL,
            INT8_MODEL,
            FireCalibrationReader(input_name, paths),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            # Quantize only the convolutions. The Detect tail concatenates box
            # coordinates (0..640) with class scores (0..1) into one tensor; a
            # single INT8 scale over that range is ~2.5 per step, which rounds
            # every class score to zero and yields mAP 0. Production INT8
            # pipelines leave the decode in float for exactly this reason.
            op_types_to_quantize=["Conv"],
        )
        size_mb = os.path.getsize(INT8_MODEL) / 1e6
        print(f"Wrote {INT8_MODEL} ({size_mb:.1f} MB)")
    else:
        print(f"Reusing existing {INT8_MODEL}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    int8_session = ort.InferenceSession(INT8_MODEL, opts,
                                        providers=["CPUExecutionProvider"])
    int8_input = int8_session.get_inputs()[0].name

    def run(batch):
        return int8_session.run(None, {int8_input: batch})[0]

    benchmark(run, "ONNX Runtime INT8 (CPU)")


if __name__ == "__main__":
    main()
