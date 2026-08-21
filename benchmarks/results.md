# Benchmark results

4 rounds x 12 images per runtime, each in its own process with a 8s settle gap; mAP scored over the full test set.

| Runtime | Precision | Best latency | Median | mAP@0.5 |
|---------|-----------|--------------|--------|---------|
| PyTorch (Ultralytics) | FP32 | 71.4 ms | 76.8 ms | 0.8859 |
| ONNX Runtime | FP32 | 36.4 ms | 44.1 ms | 0.8859 |
| ONNX Runtime | INT8 | 38.5 ms | 51.1 ms | 0.8556 |
| OpenVINO | FP32 | 30.4 ms | 31.7 ms | 0.8859 |
| OpenVINO | INT8 | 16.0 ms | 18.4 ms | 0.8089 |
| Custom C++ engine (per-channel) | INT8 | 5055.8 ms | 5257.9 ms | 0.8826 |
| Custom C++ engine (per-tensor) | INT8 | 5182.1 ms | 5407.5 ms | 0.7680 |
