"""TensorRT INT8 with quantization restricted to the convolutions.

Run this on Colab with a T4 GPU runtime, after setup_colab.py has prepared
/content/fire-8.

The default Ultralytics `int8=True` export quantizes 200 nodes, including the
Add and Mul operations that turn raw network output into box coordinates. That
scores mAP 0.0833 against 0.8859 for full precision. This script does the same
job but tells the quantizer to touch convolutions only, leaving the detection
head in full precision, which is the same scoping that took ONNX Runtime from
0.0000 to 0.8556.

It deliberately avoids Ultralytics at inference time. Ultralytics' TensorRT
loader wants metadata that only its own exporter writes, and its warmup imports
torchvision, which breaks after ModelOpt upgrades torch mid-session. Running the
engine directly through the TensorRT API sidesteps both problems, and torch is
used only as a convenient GPU buffer allocator.

    python tensorrt_int8_scoped_colab.py --weights best.pt \\
        --images fire-8/test/images --labels fire-8/test/labels \\
        --calib fire-8/calib/images
"""

import argparse
import glob
import inspect
import os
import time

import numpy as np

NET = 640
CONF = 0.25
IOU = 0.45
CLASS_NAMES = ["fire", "smoke"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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
    """(1, 4+nc, 8400) -> [(cls, conf, x1, y1, x2, y2)] using Ultralytics' NMS.

    Uses the same NMS as every other row in the benchmark table so the numbers
    stay comparable. Falls back to a plain greedy implementation if importing
    Ultralytics fails, which can happen after ModelOpt shuffles torch versions.
    """
    import torch
    if not isinstance(raw, torch.Tensor):
        raw = torch.from_numpy(np.asarray(raw))
    if raw.ndim == 2:
        raw = raw.unsqueeze(0)

    try:
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression
        kept = non_max_suppression(raw, conf_thres=CONF, iou_thres=IOU,
                                   nc=len(CLASS_NAMES))[0]
        return [(int(r[5]), float(r[4]), *r[:4]) for r in kept.tolist()]
    except Exception as e:
        print(f"  (Ultralytics NMS unavailable, using fallback: {type(e).__name__})")
        return _fallback_nms(raw[0].cpu().numpy())


def _fallback_nms(pred):
    """pred: (4+nc, 8400) with xywh boxes, per-class greedy NMS."""
    boxes = pred[:4].T
    scores = pred[4:].T
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= CONF
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]

    xy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                   boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)

    out = []
    for c in np.unique(cls):
        m = cls == c
        b, s = xy[m], conf[m]
        order = s.argsort()[::-1]
        while len(order):
            i = order[0]
            out.append((int(c), float(s[i]), *b[i].tolist()))
            if len(order) == 1:
                break
            rest = b[order[1:]]
            ix1 = np.maximum(b[i, 0], rest[:, 0]); iy1 = np.maximum(b[i, 1], rest[:, 1])
            ix2 = np.minimum(b[i, 2], rest[:, 2]); iy2 = np.minimum(b[i, 3], rest[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            a = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            ar = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            iou = inter / np.maximum(a + ar - inter, 1e-9)
            order = order[1:][iou <= IOU]
    return out


# ---------------------------------------------------------------------------
# Quantize and build
# ---------------------------------------------------------------------------

def export_onnx(weights, out_path):
    if os.path.exists(out_path):
        print(f"Reusing {out_path}")
        return out_path
    from ultralytics import YOLO
    print("Exporting FP32 ONNX...")
    produced = YOLO(weights).export(format="onnx", imgsz=NET, batch=1)
    if os.path.abspath(produced) != os.path.abspath(out_path):
        os.replace(produced, out_path)
    return out_path


def quantize_conv_only(onnx_path, calib_dir, out_path, n_calib=100):
    """Quantize with ModelOpt, restricted to Conv nodes."""
    if os.path.exists(out_path):
        print(f"Reusing {out_path}")
        return out_path

    from modelopt.onnx.quantization import quantize

    paths = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))[:n_calib]
    if not paths:
        raise SystemExit(f"No calibration images in {calib_dir}")
    print(f"Building calibration array from {len(paths)} images...")
    calib = np.concatenate([preprocess(p) for p in paths], 0)

    sig = inspect.signature(quantize)
    kwargs = dict(onnx_path=onnx_path, calibration_data={"images": calib},
                  output_path=out_path, quantize_mode="int8")

    # The whole point of this script. Name differs across ModelOpt releases, so
    # pick whichever this one accepts and report if neither exists.
    if "op_types_to_quantize" in sig.parameters:
        kwargs["op_types_to_quantize"] = ["Conv"]
        print("Quantizing Conv nodes only (op_types_to_quantize)")
    elif "op_types_to_exclude" in sig.parameters:
        kwargs["op_types_to_exclude"] = ["Add", "Mul", "Resize", "MaxPool", "Concat"]
        print("Quantizing with the detection tail op types excluded")
    else:
        print("ModelOpt quantize() accepts:")
        for name in sig.parameters:
            print(f"  {name}")
        raise SystemExit("No way to scope quantization in this ModelOpt version")

    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or k == "onnx_path"}
    quantize(**kwargs)
    print(f"Wrote {out_path}")
    return out_path


def build_engine(onnx_path, engine_path, workspace_gb=4):
    if os.path.exists(engine_path):
        print(f"Reusing {engine_path}")
        return engine_path

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # TensorRT 10 and later dropped EXPLICIT_BATCH, since every network is
    # explicit batch now. Older versions still require the flag.
    try:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    except AttributeError:
        network = builder.create_network()

    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("Failed to parse the quantized ONNX")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace_gb * (1 << 30))
    except AttributeError:
        config.max_workspace_size = workspace_gb * (1 << 30)

    print("Building TensorRT engine (several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Wrote {engine_path}")
    return engine_path


class TrtRunner:
    """Runs an engine directly, using torch tensors as GPU buffers."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        print(f"Engine IO: {self.input_name} -> {self.output_name} {self.output_shape}")

    def __call__(self, batch):
        torch = self.torch
        inp = torch.from_numpy(np.ascontiguousarray(batch)).cuda()
        out = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.input_name, inp.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.cpu().numpy()


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--images", default="fire-8/test/images")
    ap.add_argument("--labels", default="fire-8/test/labels")
    ap.add_argument("--calib", default="fire-8/calib/images")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--timed-images", type=int, default=12)
    ap.add_argument("--onnx", default="best_fp32.onnx")
    ap.add_argument("--int8-onnx", default="best_conv_int8.onnx")
    ap.add_argument("--engine", default="best_conv_int8.engine")
    args, _unknown = ap.parse_known_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Switch the Colab runtime to a T4 GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    onnx_path = export_onnx(args.weights, args.onnx)
    int8_path = quantize_conv_only(onnx_path, args.calib, args.int8_onnx)
    engine_path = build_engine(int8_path, args.engine)
    run = TrtRunner(engine_path)

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if not paths:
        raise SystemExit(f"No images in {args.images}")
    batches = [preprocess(p) for p in paths[:args.timed_images]]

    for _ in range(10):
        run(batches[0])

    samples = []
    for _ in range(args.rounds):
        for b in batches:
            t = time.perf_counter()
            run(b)
            samples.append((time.perf_counter() - t) * 1000.0)

    preds_by_cls = {}
    for p in paths:
        for cls, conf, x1, y1, x2, y2 in decode(run(preprocess(p))):
            preds_by_cls.setdefault(cls, []).append(
                (os.path.basename(p), conf, x1, y1, x2, y2))

    gt = load_ground_truth(args.labels, [os.path.basename(p) for p in paths])
    aps = []
    for cls, name in enumerate(CLASS_NAMES):
        ap_val = average_precision(preds_by_cls.get(cls, []), gt, cls)
        if ap_val is not None:
            print(f"  AP@0.5 {name:<6} = {ap_val:.4f}")
            aps.append(ap_val)
    m = float(np.mean(aps)) if aps else 0.0

    s = np.array(samples)
    print(f"\n=== TensorRT INT8, convolutions only ({torch.cuda.get_device_name(0)}) ===")
    print(f"  Best:   {s.min():.1f} ms")
    print(f"  Median: {np.median(s):.1f} ms")
    print(f"  mAP@0.5 = {m:.4f}")
    print("\nREADME row:")
    print(f"| TensorRT, T4 (convolutions only) | INT8 | {s.min():.1f} ms | "
          f"{np.median(s):.1f} ms | {m:.4f} |")


if __name__ == "__main__":
    main()
