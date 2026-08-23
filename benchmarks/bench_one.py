"""Benchmark exactly one runtime, in its own process, and print JSON.

Isolation is the point. PyTorch, ONNX Runtime and OpenVINO each create a thread
pool sized to the core count; loading several into one process oversubscribes the
CPU and inflates every measurement (ONNX FP32 measured 360 ms sharing a process
with the others, versus 38 ms alone). run_all_benchmarks.py drives this script
once per runtime and aggregates the results.

    python benchmarks/bench_one.py onnx_fp32
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from bench_common import ROOT, decode, preprocess, score, test_images

ONNX_FP32 = os.path.join(ROOT, "best.onnx")
ONNX_INT8 = os.path.join(ROOT, "best_int8.onnx")
OV_INT8_XML = os.path.join(ROOT, "benchmarks", "_ov_int8", "best_int8_ov.xml")

LABELS = {
    "pytorch": "PyTorch (Ultralytics) FP32",
    "onnx_fp32": "ONNX Runtime FP32",
    "onnx_int8": "ONNX Runtime INT8",
    "ov_fp32": "OpenVINO FP32",
    "ov_int8": "OpenVINO INT8",
}


def build(key):
    if key == "pytorch":
        import torch
        from ultralytics import YOLO

        net = YOLO(os.path.join(ROOT, "best.pt")).model
        net.eval()

        def run(batch):
            with torch.no_grad():
                out = net(torch.from_numpy(batch))
            return out[0] if isinstance(out, (tuple, list)) else out

        return run

    if key in ("onnx_fp32", "onnx_int8"):
        import onnxruntime as ort

        path = ONNX_FP32 if key == "onnx_fp32" else ONNX_INT8
        if not os.path.exists(path):
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return lambda batch: sess.run(None, {name: batch})[0]

    if key in ("ov_fp32", "ov_int8"):
        import openvino as ov

        path = ONNX_FP32 if key == "ov_fp32" else OV_INT8_XML
        if not os.path.exists(path):
            return None
        core = ov.Core()
        compiled = core.compile_model(core.read_model(path), "CPU")
        output = compiled.output(0)
        return lambda batch: compiled([batch])[output]

    raise SystemExit(f"Unknown runtime '{key}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runtime", choices=sorted(LABELS))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    fn = build(args.runtime)
    if fn is None:
        print(json.dumps({"label": LABELS[args.runtime], "skipped": True}))
        return

    paths = test_images()
    batches = [preprocess(p) for p in paths[:args.images]]

    for _ in range(10):
        fn(batches[0])

    samples = []
    for _ in range(args.rounds):
        for b in batches:
            start = time.perf_counter()
            fn(b)
            samples.append((time.perf_counter() - start) * 1000.0)

    preds = {os.path.basename(p): decode(fn(preprocess(p))) for p in paths}
    m, per_class = score(preds)

    s = np.array(samples)
    result = {
        "label": LABELS[args.runtime],
        "min_ms": float(s.min()),
        "median_ms": float(np.median(s)),
        "mean_ms": float(s.mean()),
        "map50": m,
        "per_class": per_class,
        "n_samples": int(s.size),
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
