"""OpenVINO INT8: quantize best.onnx with NNCF, then benchmark it.

Calibrated on the same 100 training images as everything else.

NNCF's default is to quantize the whole graph. For a detection model that
includes the Detect tail, where box coordinates (0..640) and class scores (0..1)
are concatenated into one tensor -- a single INT8 scale across that range rounds
every class score to zero. `--ignore-tail` keeps that subgraph in FP32, which is
what a production INT8 detection pipeline does.

    python benchmarks/openvino_int8_benchmark.py
"""

import argparse
import os

import nncf
import numpy as np
import openvino as ov

from bench_common import ROOT, benchmark, calibration_images, preprocess

FP32_MODEL = os.path.join(ROOT, "best.onnx")
INT8_DIR = os.path.join(ROOT, "benchmarks", "_ov_int8")
INT8_XML = os.path.join(INT8_DIR, "best_int8_ov.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-ignore-tail", dest="ignore_tail", action="store_false",
                    help="quantize the Detect tail too (expect mAP to collapse)")
    args = ap.parse_args()

    core = ov.Core()

    if args.rebuild or not os.path.exists(INT8_XML):
        model = core.read_model(FP32_MODEL)

        paths = calibration_images(100)
        print(f"Calibrating on {len(paths)} images...")
        dataset = nncf.Dataset(paths, lambda p: preprocess(p))

        ignored = None
        if args.ignore_tail:
            # Keep the box/score concatenation and everything after it in FP32.
            ignored = nncf.IgnoredScope(
                subgraphs=[
                    nncf.Subgraph(
                        inputs=["__module.model.22/aten::cat/Concat",
                                "__module.model.22/aten::cat/Concat_1",
                                "__module.model.22/aten::cat/Concat_2"],
                        outputs=["__module.model.22/aten::cat/Concat_5"],
                    )
                ]
            )

        print("Quantizing (this takes a few minutes)...")
        quantized = nncf.quantize(
            model, dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(paths),
            ignored_scope=ignored,
        )

        os.makedirs(INT8_DIR, exist_ok=True)
        ov.save_model(quantized, INT8_XML)
        print(f"Wrote {INT8_XML}")
    else:
        print(f"Reusing existing {INT8_XML}")

    compiled = core.compile_model(core.read_model(INT8_XML), "CPU")
    output = compiled.output(0)

    def run(batch):
        return compiled([batch])[output]

    benchmark(run, "OpenVINO INT8 (CPU)")


if __name__ == "__main__":
    main()
