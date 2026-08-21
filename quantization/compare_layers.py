"""Validate the C++ engine against PyTorch, layer by layer and end to end.

The engine is compared against the FP32 model on *exactly* the same input: the
INT8 blob the engine consumes is dequantized and fed to PyTorch, so nothing in
the comparison depends on image decoding or resizing.

For each of the 23 top-level modules, the FP32 activation is quantized with the
same scale/zero-point the engine used for that tensor, and the two INT8 tensors
are compared. Any gap is quantization error plus whatever the engine gets wrong;
the "ceiling" column shows how close a *perfect* INT8 engine could get, which is
the interesting baseline -- 100% is not achievable at 8 bits.

Usage:
    # from the build dir, first produce the dumps:
    ./custom_engine --input-bin ../../test_input.bin --dump-dir dumps
    # then:
    python quantization/compare_layers.py --dumps cpp_engine/build/dumps
"""

import argparse
import json
import os

import numpy as np
import torch

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def quantize(x, scale, zero_point):
    q = np.round(x / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def layer_output_qparams(meta):
    """Reproduce the (scale, zero_point) the engine holds for each top-level layer."""
    convs = meta["conv_layers"]
    qp = {}
    for arch in meta["architecture"]:
        i, kind = arch["layer_id"], arch["type"]
        if kind == "Conv":
            c = convs[arch["conv"]]
            qp[i] = (c["out_scale"], c["out_zero_point"])
        elif kind in ("C2f", "SPPF"):
            c = convs[arch["cv2"]]
            qp[i] = (c["out_scale"], c["out_zero_point"])
        elif kind == "Concat":
            qp[i] = (arch["out_scale"], arch["out_zero_point"])
        elif kind == "Upsample":
            # Nearest-neighbour inherits its source's quantization exactly.
            qp[i] = qp[arch["input_from"]]
    return qp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.path.join("cpp_engine", "build", "dumps"))
    ap.add_argument("--model", default=os.path.join("quantization", "model_int8.json"))
    ap.add_argument("--input-bin", default=os.path.join("cpp_engine", "build", "..", "..",
                                                        "test_input.bin"))
    args = ap.parse_args()

    dumps_dir = args.dumps if os.path.isabs(args.dumps) else os.path.join(ROOT, args.dumps)
    input_bin = (args.input_bin if os.path.isabs(args.input_bin)
                 else os.path.join(ROOT, args.input_bin))

    model_json = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    with open(model_json) as f:
        meta = json.load(f)
    qparams = layer_output_qparams(meta)

    in_scale = meta["input"]["scale"]
    in_zp = meta["input"]["zero_point"]

    # Dequantize the engine's own input so both sides see identical numbers.
    x_q = np.fromfile(input_bin, dtype=np.int8).reshape(1, 3, 640, 640)
    x_fp32 = torch.from_numpy((x_q.astype(np.float32) - in_zp) * in_scale)

    model = YOLO(os.path.join(ROOT, "best.pt"))
    net = model.model
    net.eval()

    captured = {}

    def make_hook(i):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(t):
                captured[i] = t.detach().cpu().numpy()[0]
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(net.model)]
    with torch.no_grad():
        net(x_fp32)
    for h in handles:
        h.remove()

    print(f"{'Layer':<7} {'Type':<9} {'N':>9} {'exact':>8} {'within1':>9} "
          f"{'within2':>9} {'MAE':>7} {'max':>5}")
    print("-" * 66)

    worst = []
    for arch in meta["architecture"]:
        i, kind = arch["layer_id"], arch["type"]
        path = os.path.join(dumps_dir, f"L{i:02d}_cpp.bin")
        if i not in qparams or not os.path.exists(path):
            continue

        scale, zp = qparams[i]
        ref_fp32 = captured[i]
        ref_q = quantize(ref_fp32, scale, zp).astype(np.int32).ravel()
        cpp_q = np.fromfile(path, dtype=np.int8).astype(np.int32)

        if ref_q.size != cpp_q.size:
            print(f"L{i:02d}     {kind:<9} SIZE MISMATCH ref={ref_q.size} cpp={cpp_q.size}")
            continue

        d = np.abs(ref_q - cpp_q)
        exact = (d == 0).mean() * 100
        within1 = (d <= 1).mean() * 100
        within2 = (d <= 2).mean() * 100
        mae = d.mean()

        print(f"L{i:02d}     {kind:<9} {ref_q.size:>9} {exact:>7.2f}% {within1:>8.2f}% "
              f"{within2:>8.2f}% {mae:>7.3f} {d.max():>5}")
        worst.append((within1, i, kind))

    worst.sort()
    print()
    if worst:
        w1, i, kind = worst[0]
        print(f"Worst layer: L{i:02d} ({kind}) at {w1:.2f}% within +/-1")

    # End-to-end: what does the FP32 model actually predict on this input?
    print("\n=== FP32 reference detections (same input) ===")
    with torch.no_grad():
        results = model.predict(x_fp32, verbose=False, conf=0.25, iou=0.45)
    boxes = results[0].boxes
    names = meta["class_names"]
    if len(boxes) == 0:
        print("  (none)")
    for b in boxes:
        cls = int(b.cls.item())
        xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
        print(f"  {names[cls]:<6} {float(b.conf.item()):.3f}  box={xyxy}")


if __name__ == "__main__":
    main()
