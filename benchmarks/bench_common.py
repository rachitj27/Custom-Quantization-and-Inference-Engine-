"""Shared plumbing so every runtime is measured the same way.

All benchmarks use identical preprocessing (plain 640x640 resize, no letterbox),
identical NMS settings, and the same mAP implementation the custom C++ engine is
scored with -- otherwise the numbers in the README would not be comparable.
"""

import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "quantization"))

from eval_map import average_precision, load_ground_truth  # noqa: E402

NET = 640
CONF = 0.25
IOU = 0.45
CLASS_NAMES = ["fire", "smoke"]


def preprocess(img_path):
    """BGR file -> normalised NCHW float32 batch, matching calibrate.py."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (NET, NET))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), 0)


def test_images(split="test"):
    return sorted(glob.glob(os.path.join(
        ROOT, "YOLOv8-Fire-and-Smoke-Detection", "datasets", "fire-8", split,
        "images", "*.jpg")))


def calibration_images(n=100):
    return sorted(glob.glob(os.path.join(
        ROOT, "YOLOv8-Fire-and-Smoke-Detection", "datasets", "fire-8", "train",
        "images", "*.jpg")))[:n]


def decode(raw):
    """Ultralytics ONNX/OpenVINO output (1, 4+nc, 8400) -> list of detections.

    Returns (cls, conf, x1, y1, x2, y2) in 640x640 pixels.
    """
    import torch

    # non_max_suppression moved from ultralytics.utils.ops to ultralytics.utils.nms
    # in 8.4; support both so this runs on either version.
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    if not isinstance(raw, torch.Tensor):
        raw = torch.from_numpy(np.asarray(raw))
    if raw.ndim == 2:
        raw = raw.unsqueeze(0)

    kept = non_max_suppression(raw, conf_thres=CONF, iou_thres=IOU,
                               nc=len(CLASS_NAMES))[0]
    out = []
    for row in kept.tolist():
        x1, y1, x2, y2, conf, cls = row[:6]
        out.append((int(cls), float(conf), x1, y1, x2, y2))
    return out


def score(preds_by_image, split="test"):
    """preds_by_image: {basename: [(cls, conf, x1, y1, x2, y2), ...]} -> mAP@0.5."""
    labels_dir = os.path.join(ROOT, "YOLOv8-Fire-and-Smoke-Detection", "datasets",
                              "fire-8", split, "labels")
    names = list(preds_by_image.keys())
    gt = load_ground_truth(labels_dir, names)

    by_cls = {}
    for img, dets in preds_by_image.items():
        for cls, conf, x1, y1, x2, y2 in dets:
            by_cls.setdefault(cls, []).append((img, conf, x1, y1, x2, y2))

    aps = []
    per_class = {}
    for cls, name in enumerate(CLASS_NAMES):
        ap, n_gt = average_precision(by_cls.get(cls, []), gt, cls)
        if ap is None:
            continue
        per_class[name] = ap
        aps.append(ap)
    return (float(np.mean(aps)) if aps else 0.0), per_class


def benchmark(run_fn, label, warmup=5, split="test"):
    """Time `run_fn(batch)` over the test set and score its detections.

    run_fn must return the raw (1, 4+nc, 8400) output for a preprocessed batch.
    """
    paths = test_images(split)
    if not paths:
        raise SystemExit("No test images found")

    batch = preprocess(paths[0])
    for _ in range(warmup):
        run_fn(batch)

    latencies = []
    preds = {}
    for path in paths:
        batch = preprocess(path)
        start = time.perf_counter()
        raw = run_fn(batch)
        latencies.append((time.perf_counter() - start) * 1000.0)
        preds[os.path.basename(path)] = decode(raw)

    latencies = np.array(latencies)
    m, per_class = score(preds, split)

    print(f"\n=== {label} ===")
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    for name, ap in per_class.items():
        print(f"  AP@0.5 {name:<6} = {ap:.4f}")
    print(f"  mAP@0.5        = {m:.4f}")
    return {"label": label, "mean_ms": float(latencies.mean()),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "map50": m, "per_class": per_class}
