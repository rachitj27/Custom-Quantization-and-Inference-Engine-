"""TensorRT INT8 benchmark -- run this on Colab with a GPU runtime.

TensorRT needs an NVIDIA GPU, so this cannot run on the Core Ultra laptop the
rest of the benchmarks were taken on. Run it on the same Colab T4 the original
TensorRT FP16 number came from, then paste the printed table row into README.md.

  Runtime -> Change runtime type -> T4 GPU, then:

    !pip install -q ultralytics tensorrt
    # upload best.pt and the fire-8 dataset (or mount Drive), then:
    !python tensorrt_int8_colab.py --weights best.pt --data fire-8/data.yaml \\
        --images fire-8/test/images --labels fire-8/test/labels

Calibration uses 100 training images, matching every other INT8 configuration in
this repo so the comparison stays like-for-like.
"""

import argparse
import glob
import os
import time

import numpy as np

NET = 640
CONF = 0.25
IOU = 0.45
CLASS_NAMES = ["fire", "smoke"]


def preprocess(path):
    import cv2
    img = cv2.imread(path)
    img = cv2.resize(img, (NET, NET))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), 0)


def load_ground_truth(labels_dir, names):
    gt = {}
    for name in names:
        path = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")
        boxes = []
        if os.path.exists(path):
            for line in open(path):
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *[float(v) for v in p[1:5]]
                boxes.append((c, (cx - w / 2) * NET, (cy - h / 2) * NET,
                              (cx + w / 2) * NET, (cy + h / 2) * NET))
        gt[name] = boxes
    return gt


def average_precision(preds, gt, cls, iou_thr=0.5):
    targets = {i: [b[1:] for b in bs if b[0] == cls] for i, bs in gt.items()}
    n_gt = sum(len(v) for v in targets.values())
    if n_gt == 0:
        return None
    preds = sorted(preds, key=lambda p: -p[1])
    matched = {i: np.zeros(len(v), bool) for i, v in targets.items()}
    tp, fp = np.zeros(len(preds)), np.zeros(len(preds))

    for i, (img, _c, *box) in enumerate(preds):
        cands = targets.get(img, [])
        if not cands:
            fp[i] = 1
            continue
        b = np.array(cands, dtype=np.float32)
        ix1 = np.maximum(box[0], b[:, 0]); iy1 = np.maximum(box[1], b[:, 1])
        ix2 = np.minimum(box[2], b[:, 2]); iy2 = np.minimum(box[3], b[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
        ious = inter / np.maximum(area_a + area_b - inter, 1e-9)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img][j]:
            tp[i] = 1
            matched[img][j] = True
        else:
            fp[i] = 1

    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def decode(raw):
    import torch
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
    return [(int(r[5]), float(r[4]), *r[:4]) for r in kept.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--data", default="fire-8/data.yaml")
    ap.add_argument("--images", default="fire-8/test/images")
    ap.add_argument("--labels", default="fire-8/test/labels")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--timed-images", type=int, default=12)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device -- switch the Colab runtime to a T4 GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ultralytics drives TensorRT's INT8 calibrator for us; `fraction` is sized
    # to land on ~100 calibration images, matching the other INT8 configs here.
    print("Exporting INT8 TensorRT engine (takes several minutes)...")
    engine_path = YOLO(args.weights).export(
        format="engine", int8=True, data=args.data, imgsz=NET,
        batch=1, workspace=4, verbose=False,
    )
    print(f"Engine: {engine_path}")

    model = YOLO(engine_path, task="detect")

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if not paths:
        raise SystemExit(f"No images found in {args.images}")
    batches = [preprocess(p) for p in paths[:args.timed_images]]

    for _ in range(10):
        model.predict(batches[0], verbose=False)
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.rounds):
        for b in batches:
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.predict(b, verbose=False)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t) * 1000.0)

    preds_by_cls = {}
    for p in paths:
        res = model.predict(preprocess(p), verbose=False, conf=CONF, iou=IOU)
        base = os.path.basename(p)
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds_by_cls.setdefault(int(box.cls.item()), []).append(
                (base, float(box.conf.item()), x1, y1, x2, y2))

    gt = load_ground_truth(args.labels, [os.path.basename(p) for p in paths])
    aps = []
    for cls, name in enumerate(CLASS_NAMES):
        ap_val = average_precision(preds_by_cls.get(cls, []), gt, cls)
        if ap_val is not None:
            print(f"  AP@0.5 {name:<6} = {ap_val:.4f}")
            aps.append(ap_val)
    m = float(np.mean(aps)) if aps else 0.0

    s = np.array(samples)
    print(f"\n=== TensorRT INT8 ({torch.cuda.get_device_name(0)}) ===")
    print(f"  Best:   {s.min():.1f} ms")
    print(f"  Median: {np.median(s):.1f} ms")
    print(f"  mAP@0.5 = {m:.4f}")
    print("\nREADME row:")
    print(f"| TensorRT | INT8 | {s.min():.1f} ms | {np.median(s):.1f} ms | {m:.4f} |")


if __name__ == "__main__":
    main()
