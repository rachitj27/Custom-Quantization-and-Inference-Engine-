# Benchmark results

Intel Core Ultra 7 256V, on AC power. 10 rounds x 12 images per runtime, each in its own process with a 10s settle gap, reporting the fastest observed time. mAP scored over the full 49-image test set.

| Runtime | Precision | Best latency | Median | mAP@0.5 |
|---------|-----------|--------------|--------|---------|
| PyTorch (Ultralytics) | FP32 | 42.4 ms | 58.5 ms | 0.8859 |
| ONNX Runtime | FP32 | 24.1 ms | 28.4 ms | 0.8859 |
| ONNX Runtime | INT8 | 30.8 ms | 35.8 ms | 0.8556 |
| OpenVINO | FP32 | 28.5 ms | 33.3 ms | 0.8859 |
| OpenVINO | INT8 | 13.9 ms | 16.4 ms | 0.8089 |
| Custom C++ engine (per-channel) | INT8 | 3516.4 ms | 3537.1 ms | 0.8826 |
| Custom C++ engine (per-tensor) | INT8 | 3418.1 ms | 3461.4 ms | 0.7680 |
