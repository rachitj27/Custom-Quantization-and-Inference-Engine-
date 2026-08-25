# Benchmark results

CPU rows, Intel Core Ultra 7 256V on AC power, each runtime in its own process
with a settle gap between them, reporting the fastest observed time. Library
rows are 10 rounds x 12 images. Engine rows are 5 rounds x 4 passes, which is
the same protocol scaled to a runtime that takes seconds rather than
milliseconds per image. GPU rows, Colab Tesla T4. mAP is scored over the same
49 test images with one shared implementation.

| Runtime | Precision | Best latency | Median | mAP@0.5 |
|---------|-----------|--------------|--------|---------|
| PyTorch (Ultralytics) | FP32 | 42.4 ms | 58.5 ms | 0.8859 |
| ONNX Runtime | FP32 | 24.1 ms | 28.4 ms | 0.8859 |
| ONNX Runtime | INT8 | 30.8 ms | 35.8 ms | 0.8556 |
| OpenVINO | FP32 | 28.5 ms | 33.3 ms | 0.8859 |
| OpenVINO | INT8 | 13.9 ms | 16.4 ms | 0.8089 |
| Custom C++ engine, scalar | FP32 arithmetic | 3105.9 ms | 3127.6 ms | 0.8836 |
| Custom C++ engine, scalar | INT8 | 3578.7 ms | 3776.1 ms | 0.8826 |
| Custom C++ engine, AVX-VNNI | INT8 | 233.5 ms | 238.7 ms | 0.8826 |
| Custom C++ engine (per-tensor), scalar | INT8 | 3418.1 ms | 3461.4 ms | 0.7680 |
| PyTorch (Ultralytics), T4 | FP32 | 8.0 ms | 8.1 ms | 0.8859 |
| TensorRT, T4, default export | INT8 | 4.4 ms | 5.1 ms | 0.0833 |
| TensorRT, T4, convolutions only | INT8 | 4.9 ms | 6.4 ms | 0.8656 |

## What INT8 actually did to latency

Pairing each runtime against its own FP32 measurement, which is the only way to
attribute a change to precision rather than to a change of runtime.

| Runtime | FP32 | INT8 | Effect |
|---------|------|------|--------|
| ONNX Runtime | 24.1 ms | 30.8 ms | 1.28x slower |
| OpenVINO | 28.5 ms | 13.9 ms | 2.05x faster |
| Custom engine, scalar loop | 3105.9 ms | 3578.7 ms | 1.15x slower |
| Custom engine, AVX-VNNI | 3105.9 ms | 233.5 ms | 13.3x faster |

Both engine rows share the same FP32 baseline because they are the same engine
with the same weights, differing only in how the multiply-accumulates are
issued.

INT8 is not intrinsically faster to compute. It is faster only when the kernel
issues an instruction that consumes more 8-bit lanes per cycle than the FP32
equivalent. OpenVINO fuses convolution, bias and activation into VNNI kernels
and collects that. ONNX Runtime here quantizes only Conv, so the graph converts
format between nearly every layer and the conversions cost more than the faster
convolutions save. The engine's scalar loop issues one multiply at a time and
collects nothing, which is why quantizing it alone made it slightly slower.

There is no TensorRT precision pair. The 4.9 ms INT8 figure is measured against
PyTorch FP32 at 8.0 ms, so that ratio mixes a runtime change with a precision
change and is not comparable to the rows above.

## Remaining gap

OpenVINO INT8 at 13.9 ms is still 16.8x faster than the vectorized engine at
233.5 ms. The engine runs on one core against eight, which accounts for most of
it. The rest is cache blocking and fusing the activation into the convolution.
