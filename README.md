# Custom AI Hardware Inference

A from-scratch inference and quantization pipeline for a YOLOv8n fire and smoke detection model. Instead of treating quantization and inference as black-box library calls, this project implements the underlying math from scratch, then benchmarks the result against production runtimes.

The goal is to understand what production inference systems do internally by building a smaller version by hand.

## Status

- [x] **M1: FP32 baseline** — trained YOLOv8n on the fire-8 dataset, benchmarked across four production runtimes
- [x] **M2: From-scratch INT8 quantization** — implemented affine quantization, quantized matmul, and activation calibration end-to-end. Preserved 91% of FP32 mAP@0.5.
- [x] **M3: Custom C++ inference engine** — hand-written engine that loads the quantized model and runs full YOLOv8n inference end-to-end. Conv2d operator validated bit-level against PyTorch reference; end-to-end drift documented below.

## Model

YOLOv8n fine-tuned on the fire-8 dataset (2 classes: fire, smoke). Trained for 50 epochs on a Tesla T4 in Google Colab.

- Parameters: 3.0M
- Model size (FP32): 6.0 MB
- Model size (INT8): 2.9 MB (~4× compression)

## Results

### Accuracy

| Configuration | mAP@0.5 | Drop from FP32 |
|---------------|---------|----------------|
| FP32 baseline | 0.9253 | — |
| INT8 weights (symmetric per-tensor) | 0.8467 | -8.5% |
| INT8 weights + activations (asymmetric per-tensor) | 0.8445 | -8.7% |

From-scratch quantization preserves 91% of the FP32 mAP@0.5.

### Latency

Comparison of full inference on the same 640×640 image:

| Runtime | Hardware | Precision | Latency |
|---------|----------|-----------|---------|
| PyTorch (Ultralytics) | Intel Core Ultra 7 256V | FP32 | 46 ms |
| ONNX Runtime | Intel Core Ultra 7 256V | FP32 | 40 ms |
| OpenVINO | Intel Core Ultra 7 256V | FP32 | 31 ms |
| TensorRT | NVIDIA T4 (Colab) | FP16 | 12 ms |
| Custom C++ engine | Intel Core Ultra 7 256V | INT8 | 3145 ms |

Production runtimes are significantly faster because they use hardware-specific optimizations that my hand-written naive C++ does not. Closing this gap would require techniques like multi-threading and vectorization.

The custom C++ engine performs a full quantized forward pass but accumulates numerical drift end-to-end; see the following section for details.

### End-to-End Numerical Accuracy

The C++ engine was validated at two levels: per-operator and end-to-end.

**Per-operator (Layer 0 conv2d)** was compared bit-level against a PyTorch reference across 1.6M output values: 84.0% exact matches, 98.1% within ±1. This confirms the conv2d operator itself is numerically correct.

**End-to-end** was measured by dumping intermediate activations at every network layer and comparing against the fake-quantized Python reference on the same test image:

| Layer | % Within ±1 | MAE |
|-------|-------------|-----|
| L00 (first conv) | 38.3% | 9.0 |
| L01 (second conv) | 33.6% | 6.6 |
| L02 (first C2f) | 10.2% | 80.2 |
| L15 (P3 output) | 17.6% | 15.3 |
| L18 (P4 output) | 2.8% | 83.8 |
| L21 (P5 output) | 1.1% | 132.4 |

Drift grows sharply at layers containing C2f blocks. Two design decisions cause this:

1. **Concat scale mixing.** YOLOv8n uses concat operations to merge feature maps at different points in the network. Each input to a concat is a quantized tensor at its own scale. The engine concatenates raw INT8 bytes without requantizing to a common scale — a production runtime like TensorRT inserts a requantization step here.
2. **Per-module activation calibration.** Activation scales are calibrated at the output of each of the 22 top-level modules, not at every internal conv inside C2f and SPPF blocks. Internal convs use the block's output scale as an approximation. Per-conv calibration was attempted and did not materially reduce drift — Concat scale-mixing is the dominant source.

The engine is intended as a portfolio piece to demonstrate the mechanics of quantized inference from first principles; closing this drift would require implementing common-scale requantization at every concat boundary.

## Approach

**M2 (Quantization)** implements post-training static quantization from first principles rather than calling `torch.quantization`:

- **Weights**: symmetric per-tensor INT8, computed directly from weight min/max
- **Activations**: asymmetric per-tensor INT8, calibrated on 100 training images using PyTorch forward hooks
- **Matmul**: INT32 accumulation to prevent overflow, requantization scale `M = s_A × s_B / s_C` applied per output

**M3 (C++ Engine)** implements the operators needed to run YOLOv8n:

- Custom Tensor class with heap-allocated INT8 buffers
- Model loader that reads a custom binary + JSON format
- Quantized conv2d with INT32 accumulation and requantization
- Activation function (SiLU) via 256-entry precomputed lookup table
- Element-wise ops (Concat, Upsample, MaxPool)
- Compound blocks (Bottleneck, C2f, SPPF) built from base operators
- Forward pass runner that walks the model architecture

## Repo Contents

- `quantization/` — from-scratch INT8 quantization pipeline (Python)
  - `quantization.py` — quantize/dequantize/quantized_matmul functions
  - `calibrate.py` — activation calibration via forward hooks
  - `quantized_inference.py` — end-to-end fake-quantized inference
  - `save_model.py` — serialize the quantized model with architecture metadata
- `cpp_engine/` — custom C++ inference engine (M3)
  - `tensor.h/cpp` — Tensor class
  - `model.h/cpp` — model loader
  - `ops.h/cpp` — all operators and forward pass runner
  - `main.cpp` — entry point and benchmarking harness
- `benchmarks/` — M1 baselines against production runtimes
- `Results.md` — full benchmark results and notes

## Dataset

[fire-8 from Abonia1](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection) — 2-class fire and smoke detection dataset.