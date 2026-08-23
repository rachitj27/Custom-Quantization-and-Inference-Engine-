"""Activation calibration for the custom INT8 engine.

Records the FP32 dynamic range of every tensor the C++ engine has to represent
as INT8, then converts each range into an asymmetric scale / zero-point.

Three groups of tensors get calibrated:

1. Every ``Conv`` module output (57 of them). These are post-SiLU. Calibrating
   per Conv module -- rather than only at the 23 top-level modules -- is what
   lets convs *inside* C2f/SPPF blocks use their own scale instead of borrowing
   the block's output scale.
2. Every top-level module output (23). Needed for the Concat and Upsample nodes
   in the neck, which are not Conv modules but still hold INT8 tensors.
3. The *input* to each C2f's ``cv2`` (via a forward-pre-hook). That tensor is the
   concatenation of feature maps that each arrive on a different scale, so the
   engine needs one common scale to requantize them all to before concatenating.

Run from anywhere:  python quantization/calibrate.py
"""

import glob
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

from quantization import compute_scale_zeropoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "quantization", "activation_scales.json")
N_CALIB_IMAGES = 100

model = YOLO(os.path.join(ROOT, "best.pt"))
pytorch_model = model.model
pytorch_model.eval()

calib_paths = sorted(
    glob.glob(os.path.join(ROOT, "YOLOv8-Fire-and-Smoke-Detection",
                           "datasets", "fire-8", "train", "images", "*.jpg"))
)[:N_CALIB_IMAGES]
if not calib_paths:
    raise SystemExit("No calibration images found -- check the dataset path.")
print(f"Using {len(calib_paths)} calibration images")

# key -> {"min": float, "max": float}
ranges = {}


def observe(key, tensor):
    """Fold a tensor's min/max into the running range for `key`."""
    if not torch.is_tensor(tensor):
        return
    lo = tensor.detach().min().item()
    hi = tensor.detach().max().item()
    if key not in ranges:
        ranges[key] = {"min": lo, "max": hi}
    else:
        ranges[key]["min"] = min(ranges[key]["min"], lo)
        ranges[key]["max"] = max(ranges[key]["max"], hi)


def make_output_hook(key):
    def hook(module, inputs, output):
        observe(key, output[0] if isinstance(output, tuple) else output)
    return hook


def make_input_hook(key):
    def hook(module, inputs):
        if inputs:
            observe(key, inputs[0])
    return hook


hooks = []

# (1) every Conv module output -- post-SiLU
conv_module_paths = []
for name, module in pytorch_model.named_modules():
    if isinstance(module, Conv):
        conv_module_paths.append(name)
        hooks.append(module.register_forward_hook(make_output_hook(name)))

# (2) every top-level module output -- covers Concat / Upsample
for i, layer in enumerate(pytorch_model.model):
    hooks.append(layer.register_forward_hook(make_output_hook(f"layer.{i}")))

# (3) the concatenated tensor feeding each C2f's cv2
c2f_concat_paths = []
for name, module in pytorch_model.named_modules():
    if isinstance(module, C2f):
        key = f"{name}.cv2_input"
        c2f_concat_paths.append(key)
        hooks.append(module.cv2.register_forward_pre_hook(make_input_hook(key)))

print(f"Hooked {len(conv_module_paths)} Conv modules, "
      f"{len(pytorch_model.model)} top-level modules, "
      f"{len(c2f_concat_paths)} C2f concat inputs")


def preprocess_image(img_path):
    """Match Ultralytics' letterbox-free eval path: resize, BGR->RGB, /255, CHW."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(np.expand_dims(img, 0))


print("Running calibration...")
with torch.no_grad():
    for i, img_path in enumerate(calib_paths):
        pytorch_model(preprocess_image(img_path))
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(calib_paths)}")

for h in hooks:
    h.remove()

print(f"\nCalibration complete. Captured ranges for {len(ranges)} tensors.")

# Convert observed ranges into asymmetric scale / zero-point pairs.
# The range is widened to always include 0 so that zero is exactly representable
# -- padding and cleared accumulators depend on this.
scales = {}
for key, r in ranges.items():
    lo = min(r["min"], 0.0)
    hi = max(r["max"], 0.0)
    if hi == lo:
        hi = lo + 1e-6
    scale, zero_point = compute_scale_zeropoint(
        np.array([lo, hi], dtype=np.float32), symmetric=False
    )
    scales[key] = {
        "scale": float(scale),
        "zero_point": int(zero_point),
        "min": float(r["min"]),
        "max": float(r["max"]),
    }

print("\nSample scales:")
for key in ["model.0", "model.2.cv1", "model.2.m.0.cv2", "model.2.cv2_input", "layer.11"]:
    if key in scales:
        s = scales[key]
        print(f"  {key:<22} scale={s['scale']:.5f} zp={s['zero_point']:<5d} "
              f"range=[{s['min']:.3f}, {s['max']:.3f}]")

with open(OUT_PATH, "w") as f:
    json.dump(scales, f, indent=2)

print(f"\nSaved {len(scales)} activation scales to {OUT_PATH}")
