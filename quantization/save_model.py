"""Serialize the quantized YOLOv8n model for the C++ engine.

Writes two files:

  model_int8.bin   raw INT8 weights, one blob per Conv2d, back to back
  model_int8.json  everything the engine needs to interpret those bytes

Weights stay **symmetric per-tensor INT8**, exactly as before -- the .bin is
byte-for-byte identical to the previous version. What is new is the metadata
that makes the engine numerically correct:

* **Folded BatchNorm.** YOLOv8's ``Conv`` is ``Conv2d(bias=False) -> BatchNorm2d
  -> SiLU``. The previous export dropped the BatchNorm entirely. Folding it into
  a per-output-channel gain and bias

      gain[oc] = gamma[oc] / sqrt(var[oc] + eps)
      bias[oc] = beta[oc] - gamma[oc] * mean[oc] / sqrt(var[oc] + eps)

  lets the engine apply BatchNorm as part of requantization, at no extra cost:
  the requant multiplier simply becomes per-channel. At layer 0 alone this gain
  spans 13x..151x across channels, so omitting it is not a rounding error.

* **Real conv geometry** (stride / padding / groups) read off the modules
  instead of being hardcoded in C++.

* **Per-conv output scales**, so convs inside C2f/SPPF use their own calibrated
  range rather than borrowing the block's.

* **Detect head metadata**, so the engine can decode boxes instead of returning
  a feature map.

Run from anywhere:  python quantization/save_model.py
"""

import argparse
import json
import os

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, Detect, SPPF

from quantization import compute_scale_zeropoint, quantize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALES_PATH = os.path.join(ROOT, "quantization", "activation_scales.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--per-channel", action="store_true",
    help="quantize weights with one scale per output channel instead of one per "
         "tensor. Costs 4 bytes per output channel of metadata and nothing at "
         "inference time, since the requant multiplier is already per-channel.")
parser.add_argument("--prefix", default=None,
                    help="output basename (default: model_int8, or "
                         "model_int8_pc with --per-channel)")
args = parser.parse_args()

PER_CHANNEL = args.per_channel
prefix = args.prefix or ("model_int8_pc" if PER_CHANNEL else "model_int8")
BIN_PATH = os.path.join(ROOT, "quantization", prefix + ".bin")
JSON_PATH = os.path.join(ROOT, "quantization", prefix + ".json")

# The engine feeds images in as INT8 with this fixed quantization: a pixel in
# [0,1] maps to [-128,127]. Must match preprocessing in the C++ engine.
INPUT_SCALE = 1.0 / 255.0
INPUT_ZERO_POINT = -128

model = YOLO(os.path.join(ROOT, "best.pt"))
pytorch_model = model.model
pytorch_model.eval()

with open(SCALES_PATH) as f:
    act_scales = json.load(f)


def scale_of(key):
    """Look up a calibrated (scale, zero_point); raises if calibration is stale."""
    if key not in act_scales:
        raise KeyError(f"No calibrated scale for '{key}'. Re-run calibrate.py.")
    e = act_scales[key]
    return float(e["scale"]), int(e["zero_point"])


# ---------------------------------------------------------------------------
# Map each Conv2d to its enclosing ultralytics Conv module (which owns the BN
# and the SiLU). Anything left over -- the Detect head's final 1x1 convs and the
# DFL conv -- is a bare Conv2d with no BatchNorm.
# ---------------------------------------------------------------------------
owner_of = {}  # id(Conv2d) -> module path of the owning Conv module
for name, module in pytorch_model.named_modules():
    if isinstance(module, Conv):
        owner_of[id(module.conv)] = name

conv_layers = []
conv_id_by_path = {}
weight_blobs = []
byte_offset = 0

for name, module in pytorch_model.named_modules():
    if not isinstance(module, torch.nn.Conv2d):
        continue

    w_fp32 = module.weight.detach().cpu().numpy()
    out_ch = w_fp32.shape[0]

    if PER_CHANNEL:
        # One symmetric scale per output filter. Filters in the same layer can
        # differ in magnitude by an order of magnitude or more, and a single
        # per-tensor scale forces the quietest filters into only a handful of
        # the 256 available levels.
        abs_max = np.abs(w_fp32).reshape(out_ch, -1).max(axis=1)
        s_vec = np.maximum(abs_max / 127.0, 1e-12).astype(np.float32)
        w_int8 = np.clip(
            np.round(w_fp32 / s_vec[:, None, None, None]), -128, 127
        ).astype(np.int8)
        weight_scale = [float(v) for v in s_vec]
    else:
        # Symmetric per-tensor on the raw weight -- the original scheme.
        s, z = compute_scale_zeropoint(w_fp32, symmetric=True)
        w_int8 = quantize(w_fp32, s, z)
        weight_scale = float(s)

    blob = w_int8.tobytes()
    owner_path = owner_of.get(id(module))

    if owner_path is not None:
        # Conv + BatchNorm + SiLU: fold the BN into a per-channel gain and bias.
        bn = pytorch_model.get_submodule(owner_path).bn
        inv_std = 1.0 / torch.sqrt(bn.running_var + bn.eps)
        gain = (bn.weight * inv_std).detach().cpu().numpy()
        bias = (bn.bias - bn.weight * bn.running_mean * inv_std).detach().cpu().numpy()
        act = "silu"
        out_scale, out_zp = scale_of(owner_path)
        module_path = owner_path
    else:
        # Bare Conv2d (Detect head 1x1 predictors, DFL): no BN. Its own bias, if
        # any, plays the role of the folded bias. These feed box decoding, so
        # the engine keeps their output in FP32 and no output scale is needed.
        gain = np.ones(out_ch, dtype=np.float32)
        if module.bias is not None:
            bias = module.bias.detach().cpu().numpy()
        else:
            bias = np.zeros(out_ch, dtype=np.float32)
        act = "none"
        out_scale, out_zp = None, None
        module_path = name

    conv_layers.append({
        "layer_id": len(conv_layers),
        "path": name,
        "module_path": module_path,
        "weight_shape": list(w_int8.shape),
        "byte_offset": byte_offset,
        "byte_length": len(blob),
        "weight_scale": weight_scale,
        "weight_zero_point": 0,
        "quantization_scheme": ("symmetric_per_channel" if PER_CHANNEL
                                else "symmetric_per_tensor"),
        "stride": int(module.stride[0]),
        "padding": int(module.padding[0]),
        "groups": int(module.groups),
        "bn_gain": [float(v) for v in gain],
        "bn_bias": [float(v) for v in bias],
        "act": act,
        "out_scale": out_scale,
        "out_zero_point": out_zp,
    })
    conv_id_by_path[name] = len(conv_layers) - 1
    weight_blobs.append(blob)
    byte_offset += len(blob)

with open(BIN_PATH, "wb") as f:
    for blob in weight_blobs:
        f.write(blob)
print(f"Wrote {os.path.basename(BIN_PATH)} ({byte_offset} bytes, "
      f"{byte_offset / 1024:.1f} KB)")

# ---------------------------------------------------------------------------
# Top-level architecture. `input_from` comes from yolov8.yaml and is the one
# thing that genuinely has to be hardcoded; everything else is read off the
# live modules so it cannot drift from the checkpoint.
# ---------------------------------------------------------------------------
INPUT_FROM = [
    "input", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    [10, 6], 11, 12, [13, 4], 14, 15, [16, 12], 17, 18, [19, 9], 20,
    [15, 18, 21],
]

architecture = []
for idx, layer in enumerate(pytorch_model.model):
    kind = type(layer).__name__
    path = f"model.{idx}"
    entry = {
        "layer_id": idx,
        "type": kind,
        "input_from": INPUT_FROM[idx],
    }

    if kind == "Conv":
        entry["conv"] = conv_id_by_path[f"{path}.conv"]

    elif kind == "C2f":
        entry["cv1"] = conv_id_by_path[f"{path}.cv1.conv"]
        entry["cv2"] = conv_id_by_path[f"{path}.cv2.conv"]
        entry["bottlenecks"] = [
            {
                "cv1": conv_id_by_path[f"{path}.m.{i}.cv1.conv"],
                "cv2": conv_id_by_path[f"{path}.m.{i}.cv2.conv"],
                "shortcut": bool(layer.m[i].add),
            }
            for i in range(len(layer.m))
        ]
        # Common scale that every branch is requantized to before the concat
        # that feeds cv2. Without this the branches are concatenated on
        # different scales, which was the drift the README blamed.
        cs, czp = scale_of(f"{path}.cv2_input")
        entry["concat_scale"] = cs
        entry["concat_zero_point"] = czp

    elif kind == "SPPF":
        entry["cv1"] = conv_id_by_path[f"{path}.cv1.conv"]
        entry["cv2"] = conv_id_by_path[f"{path}.cv2.conv"]
        entry["k"] = int(layer.m.kernel_size)
        # MaxPool preserves its input's scale, so all four concat branches are
        # already on cv1's scale -- no requantization needed here.

    elif kind == "Concat":
        s, zp = scale_of(f"layer.{idx}")
        entry["out_scale"] = s
        entry["out_zero_point"] = zp

    elif kind == "Upsample":
        # Nearest-neighbour copies values verbatim, so it inherits the input's
        # scale exactly. Nothing to store.
        pass

    elif kind == "Detect":
        entry["nc"] = int(layer.nc)
        entry["reg_max"] = int(layer.reg_max)
        entry["no"] = int(layer.no)
        entry["strides"] = [float(v) for v in layer.stride.tolist()]
        entry["cv2"] = [
            [conv_id_by_path[f"{path}.cv2.{i}.0.conv"],
             conv_id_by_path[f"{path}.cv2.{i}.1.conv"],
             conv_id_by_path[f"{path}.cv2.{i}.2"]]
            for i in range(3)
        ]
        entry["cv3"] = [
            [conv_id_by_path[f"{path}.cv3.{i}.0.conv"],
             conv_id_by_path[f"{path}.cv3.{i}.1.conv"],
             conv_id_by_path[f"{path}.cv3.{i}.2"]]
            for i in range(3)
        ]
        entry["dfl"] = conv_id_by_path[f"{path}.dfl.conv"]

    else:
        raise RuntimeError(f"Unhandled module type at layer {idx}: {kind}")

    architecture.append(entry)

names = pytorch_model.names
metadata = {
    "num_bits": 8,
    "input": {"scale": INPUT_SCALE, "zero_point": INPUT_ZERO_POINT},
    "class_names": [names[i] for i in range(len(names))],
    "conv_layers": conv_layers,
    "activation_scales": act_scales,
    "architecture": architecture,
}

with open(JSON_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

n_folded = sum(1 for c in conv_layers if c["act"] == "silu")
print(f"Wrote {os.path.basename(JSON_PATH)} ({len(conv_layers)} conv layers, "
      f"{n_folded} with folded BatchNorm, {len(architecture)} top-level modules, "
      f"weights={'per-channel' if PER_CHANNEL else 'per-tensor'})")

print("\nArchitecture map:")
for a in architecture:
    extra = ""
    if a["type"] == "C2f":
        extra = f" cv1={a['cv1']} cv2={a['cv2']} n={len(a['bottlenecks'])}"
    elif a["type"] in ("Conv",):
        extra = f" conv={a['conv']}"
    elif a["type"] == "SPPF":
        extra = f" cv1={a['cv1']} cv2={a['cv2']} k={a['k']}"
    print(f"  Layer {a['layer_id']:>2}: {a['type']:<9} from={a['input_from']}{extra}")
