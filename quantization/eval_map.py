"""Score the custom C++ engine against ground truth, and against FP32 PyTorch.

Both sides are evaluated with the *same* mAP implementation and the *same*
preprocessing (plain 640x640 resize, no letterbox) so the comparison is
apples-to-apples. That matters: Ultralytics' own `model.val()` letterboxes and
uses a slightly different matching rule, so its numbers are not directly
comparable to a hand-rolled evaluator.

Usage:
    python quantization/eval_map.py --csv cpp_engine/build/preds/detections.csv
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = 640


def load_ground_truth(labels_dir, image_names):
    """YOLO-format labels -> {image: [(cls, x1, y1, x2, y2), ...]} in network pixels."""
    gt = {}
    for name in image_names:
        stem = os.path.splitext(name)[0]
        path = os.path.join(labels_dir, stem + ".txt")
        boxes = []
        if os.path.exists(path):
            for line in open(path):
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, w, h = int(parts[0]), *[float(v) for v in parts[1:5]]
                boxes.append((c,
                              (cx - w / 2) * NET, (cy - h / 2) * NET,
                              (cx + w / 2) * NET, (cy + h / 2) * NET))
        gt[name] = boxes
    return gt


def iou_matrix(box, others):
    if not others:
        return np.zeros(0)
    b = np.array(others, dtype=np.float32)
    x1 = np.maximum(box[0], b[:, 0])
    y1 = np.maximum(box[1], b[:, 1])
    x2 = np.minimum(box[2], b[:, 2])
    y2 = np.minimum(box[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def average_precision(preds, gt, cls, iou_thr=0.5):
    """VOC all-point-interpolated AP for one class.

    preds: list of (image, conf, x1, y1, x2, y2)
    gt:    {image: [(cls, x1, y1, x2, y2)]}
    """
    targets = {img: [b[1:] for b in boxes if b[0] == cls] for img, boxes in gt.items()}
    n_gt = sum(len(v) for v in targets.values())
    if n_gt == 0:
        return None, 0

    preds = sorted(preds, key=lambda p: -p[1])
    matched = {img: np.zeros(len(v), dtype=bool) for img, v in targets.items()}

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img, _conf, *box) in enumerate(preds):
        candidates = targets.get(img, [])
        if not candidates:
            fp[i] = 1
            continue
        ious = iou_matrix(box, candidates)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not matched[img][best]:
            tp[i] = 1
            matched[img][best] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # All-point interpolation.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), n_gt


def report(title, preds_by_cls, gt, names):
    print(f"\n=== {title} ===")
    aps = []
    for cls, name in enumerate(names):
        ap, n_gt = average_precision(preds_by_cls.get(cls, []), gt, cls)
        if ap is None:
            print(f"  {name:<6} (no ground truth)")
            continue
        n_pred = len(preds_by_cls.get(cls, []))
        print(f"  {name:<6} AP@0.5 = {ap:.4f}   ({n_pred} preds, {n_gt} labels)")
        aps.append(ap)
    m = float(np.mean(aps)) if aps else 0.0
    print(f"  {'mAP@0.5':<6} = {m:.4f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("cpp_engine", "build", "preds",
                                                  "detections.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(ROOT, args.csv)
    split_dir = os.path.join(ROOT, "YOLOv8-Fire-and-Smoke-Detection", "datasets",
                             "fire-8", args.split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    with open(os.path.join(ROOT, "quantization", "model_int8.json")) as f:
        names = json.load(f)["class_names"]

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    image_names = [os.path.basename(p) for p in image_paths]
    gt = load_ground_truth(labels_dir, image_names)
    print(f"{len(image_names)} images, "
          f"{sum(len(v) for v in gt.values())} ground-truth boxes")

    # --- custom engine, read back from the CSV it wrote ---
    engine_preds = {}
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 8:
                continue
            img, cls, _name, conf, x1, y1, x2, y2 = parts
            engine_preds.setdefault(int(cls), []).append(
                (img, float(conf), float(x1), float(y1), float(x2), float(y2)))

    # --- FP32 PyTorch on identical inputs ---
    model = YOLO(os.path.join(ROOT, "best.pt"))
    fp32_preds = {}
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (NET, NET))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            res = model.predict(tensor, verbose=False, conf=args.conf, iou=args.iou)
        base = os.path.basename(path)
        for b in res[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            fp32_preds.setdefault(int(b.cls.item()), []).append(
                (base, float(b.conf.item()), x1, y1, x2, y2))

    m_fp32 = report("FP32 PyTorch (same preprocessing)", fp32_preds, gt, names)
    m_engine = report("Custom C++ INT8 engine", engine_preds, gt, names)

    print(f"\nRetained {m_engine / m_fp32 * 100:.1f}% of FP32 mAP@0.5"
          if m_fp32 > 0 else "")


if __name__ == "__main__":
    main()
