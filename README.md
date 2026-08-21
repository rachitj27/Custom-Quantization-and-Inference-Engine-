# Custom AI Hardware Inference

A from-scratch inference and quantization pipeline for a YOLOv8n fire and smoke detection model. Instead of treating quantization and inference as black-box library calls, this project implements the underlying math from scratch, then benchmarks the result against production runtimes.

The engine takes a JPEG and draws boxes around fire and smoke, running the entire quantized forward pass — convolutions, activations, the Detect head, DFL box decoding and NMS — in hand-written C++.

## Status

- [x] **M1: FP32 baseline** — trained YOLOv8n on the fire-8 dataset, benchmarked across four production runtimes
- [x] **M2: From-scratch INT8 quantization** — affine quantization, quantized matmul, and activation calibration implemented end to end
- [x] **M3: Custom C++ inference engine** — loads the quantized model, runs full INT8 inference, decodes boxes, and retains **99.6% of FP32 mAP@0.5**
- [ ] **M4: Hardware accelerator** — Verilog RTL FPGA accelerator (stretch goal)

## Usage

```bash
cd cpp_engine && mkdir -p build && cd build
cmake .. && make

./custom_engine path/to/fire.jpg              # writes fire_pred.jpg
./custom_engine fire.jpg -o out.jpg --conf 0.3
./custom_engine fire.jpg --model-json ../../quantization/model_int8_pc.json \
                         --model-bin  ../../quantization/model_int8_pc.bin
```

It prints each detection and writes an annotated copy of the image:

```
Loaded fire.jpg (640x640)
Inference: 5155.8 ms
Detections: 1
  fire   0.830  box=[329.2, 158.6, 532.6, 585.8]
Wrote fire_pred.jpg
```

Requires `cmake`, a C++17 compiler, and `nlohmann-json`. JPEG/PNG decoding uses [stb](https://github.com/nothings/stb), vendored in `cpp_engine/third_party/`.

## Model

YOLOv8n fine-tuned on the fire-8 dataset (2 classes: fire, smoke). Trained for 50 epochs on a Tesla T4 in Google Colab.

- Parameters: 3.0M
- Model size (FP32): 6.0 MB
- Model size (INT8): 2.9 MB (~4× compression)

## Results

### Accuracy

Every number below was produced by `quantization/eval_map.py` over the 49-image test set, using identical preprocessing (plain 640×640 resize) and a single VOC all-point-interpolation AP implementation, so the rows are directly comparable to each other.

| Configuration | fire AP@0.5 | smoke AP@0.5 | mAP@0.5 | vs FP32 |
|---|---|---|---|---|
| FP32 (PyTorch / ONNX / OpenVINO — all identical) | 0.7742 | 0.9976 | **0.8859** | — |
| **Custom C++ engine, per-channel weights** | 0.7742 | 0.9909 | **0.8826** | **99.6%** |
| ONNX Runtime INT8 | 0.7352 | 0.9759 | 0.8556 | 96.6% |
| OpenVINO INT8 (NNCF) | 0.7126 | 0.9051 | 0.8089 | 91.3% |
| Custom C++ engine, per-tensor weights | 0.5935 | 0.9426 | 0.7680 | 86.7% |

The hand-written engine is the **most accurate INT8 configuration measured here** — ahead of both production INT8 pipelines. That is not because the arithmetic is cleverer; it is because the engine keeps the Detect head's box decoding in FP32 and calibrates every convolution individually, where the automated pipelines apply a single policy across the whole graph.

> **Note on the 0.9253 figure in earlier versions of this README:** that came from Ultralytics' `model.val()`, which letterboxes inputs and uses its own AP implementation. This table uses a self-contained evaluator so the custom engine can be scored the same way as everything else. The two protocols are not interchangeable — the FP32 model is unchanged.

### Latency

Intel Core Ultra 7 256V, **on AC power**. Each runtime measured in its own process with a 10 s settle gap between runs, 10 rounds × 12 images (120 samples each), reporting the fastest observed time. See *Benchmark methodology* below — this took several attempts to get right.

| Runtime | Precision | Best latency | Median | mAP@0.5 |
|---------|-----------|--------------|--------|---------|
| PyTorch (Ultralytics) | FP32 | 42.4 ms | 58.5 ms | 0.8859 |
| ONNX Runtime | FP32 | 24.1 ms | 28.4 ms | 0.8859 |
| ONNX Runtime | INT8 | 30.8 ms | 35.8 ms | 0.8556 |
| OpenVINO | FP32 | 28.5 ms | 33.3 ms | 0.8859 |
| **OpenVINO** | **INT8** | **13.9 ms** | 16.4 ms | 0.8089 |
| Custom C++ engine (per-channel) | INT8 | 3516.4 ms | 3537.1 ms | 0.8826 |
| Custom C++ engine (per-tensor) | INT8 | 3418.1 ms | 3461.4 ms | 0.7680 |

GPU numbers are kept separate because a Tesla T4 is not comparable to a laptop CPU. Measured on Colab with the same preprocessing, NMS and mAP code:

| Runtime | Precision | Best latency | Median | mAP@0.5 |
|---------|-----------|--------------|--------|---------|
| PyTorch (Ultralytics), T4 | FP32 | 8.0 ms | 8.1 ms | 0.8859 |
| **TensorRT, T4** | **INT8** | **4.4 ms** | 5.1 ms | **0.0833** |
| TensorRT, T4 | FP16 | 12.4 ms | — | *carried over from M1, not re-measured* |

Reading the table:

- **OpenVINO INT8 is the clear winner at 13.9 ms**, a 2.05× speedup over its own FP32 build. This is what INT8 is supposed to buy you.
- **ONNX Runtime INT8 is *slower* than its FP32 build** (30.8 ms vs 24.1 ms) — consistently, across every measurement run. Only the convolutions are quantized, because the Detect tail has to stay in FP32 (see below), so the graph pays for a `QuantizeLinear`/`DequantizeLinear` pair around all 64 convs without ever fusing into a fully-integer subgraph. INT8 is not automatically faster; it depends on whether the runtime's kernels can actually consume the quantized graph.
- **The custom engine is ~250× slower than OpenVINO.** It is six nested loops with no vectorization, no blocking, no threading, and an FP32 requantization per output element. Closing that gap is what M4 is for. It is worth noting it is only ~12% slower than the *original, incorrect* engine (3145 ms) despite now also folding BatchNorm, correcting zero-points, and running a full Detect head with DFL decoding and NMS.
- The two custom-engine rows differ by ~3%, confirming that per-channel weights cost essentially nothing at inference time — the requantization multiplier was already per-channel.
- **TensorRT INT8 is the fastest thing here at 4.4 ms — and the least accurate, at mAP 0.0833.** A default `int8=True` export destroys this model. Details below; it is the same failure that gave ONNX Runtime mAP 0.0000.

### Layer-by-layer accuracy

Each of the 22 backbone layers' INT8 output compared against the FP32 model quantized on the same grid, on one test image (`quantization/compare_layers.py`, per-channel weights):

| Layer | Type | within ±1 | MAE | | Layer | Type | within ±1 | MAE |
|---|---|---|---|---|---|---|---|---|
| L00 | Conv | 99.91% | 0.081 | | L09 | SPPF | 68.65% | 1.683 |
| L01 | Conv | 99.91% | 0.119 | | L15 | C2f | 99.40% | 0.241 |
| L02 | C2f | 77.46% | 0.954 | | L18 | C2f | 94.92% | 0.393 |
| L06 | C2f | 90.71% | 0.535 | | L21 | C2f | 94.94% | 0.410 |

The key property is that **error no longer compounds with depth** — L21 (94.94%) is more accurate than L02 (77.46%). Residual error is ordinary INT8 rounding noise, not a systematic defect.

## What made it accurate

An earlier version of this engine drifted badly, and the drift grew with depth — layer 0 matched the reference on 38.3% of values, layer 21 on 1.1%, with a mean absolute error of 132 INT8 steps. The README at the time blamed concat scale mixing. That was a real bug, but it was not the main one.

| Layer | within ±1 (before → after) | MAE (before → after) |
|---|---|---|
| L00 | 38.3% → **99.91%** | 9.0 → **0.081** |
| L02 | 10.2% → **77.46%** | 80.2 → **0.954** |
| L15 | 17.6% → **99.40%** | 15.3 → **0.241** |
| L18 | 2.8% → **94.92%** | 83.8 → **0.393** |
| L21 | 1.1% → **94.94%** | 132.4 → **0.410** |

Five defects, in the order they mattered:

**1. BatchNorm was never folded.** The exporter walked `named_modules()` and quantized each `Conv2d.weight` directly. But YOLOv8's `Conv` block is `Conv2d(bias=False) → BatchNorm2d → SiLU`, so the engine was silently omitting an entire operator. At layer 0 alone, BatchNorm applies a per-channel gain spanning **13.0× to 150.8×** plus a per-channel bias from −3.0 to +6.7. This was wrong from the very first layer.

It hid because the per-operator validation compared against `F.conv2d(input, weight)` — a bare convolution with no BatchNorm either. Both sides omitted the same operator, so the test passed at 84%.

The fix folds BatchNorm into a per-output-channel gain and bias at export time:

```
gain[oc] = gamma[oc] / sqrt(var[oc] + eps)
bias[oc] = beta[oc] - gamma[oc] * mean[oc] / sqrt(var[oc] + eps)
```

which the engine applies during requantization. The multiplier was already per-output-channel, so this costs nothing at inference time.

**2. Convolution ignored the input zero-point.** Activations are asymmetric, so the accumulator must be `Σ(q_in − z_in)·q_w`, not `Σ q_in·q_w`. Every convolution in the network was affected. Subtracting the zero-point inside the accumulation also makes zero-padding exact for free: a padded position holds real 0, i.e. `q == z_in`, so it contributes nothing — where the old code skipped padded positions and silently treated them as `q = 0`.

**3. Activations were clipped before SiLU ran.** The engine requantized each convolution's output into the layer's calibrated range and *then* applied SiLU. But that range is calibrated on the **post**-SiLU tensor — layer 0's is `[−0.278, 56.08]` — while the pre-activation values run far more negative. The negative tail was clamped away before SiLU ever saw it. Fusing convolution and activation through the INT32 accumulator removes the intermediate rounding entirely.

**4. Concat mixed scales.** Feature maps arriving on different scales were concatenated as raw INT8 bytes. Each input is now requantized to one common scale first — for the C2f blocks, a scale calibrated on the tensor that actually feeds `cv2`. (SPPF needs no requantization: max-pooling preserves its input's scale, so all four branches already agree.)

**5. Residual connections dropped the zero-point.** `q_y + q_in` should be `q_y + q_in − z`. The skip connection is now added in the real domain and quantized once, so the two tensors' scales need not agree.

Fixing 1–3 alone brought layer 0 from 38.3% to a **bit-exact match** against the reference.

### Per-tensor vs per-channel weights

Both schemes are implemented (`python quantization/save_model.py [--per-channel]`) and both are in the table above. Per-channel is worth it here: **0.7680 → 0.8826 mAP@0.5**, recovering nearly all the remaining gap to FP32, with fire AP going from 0.5935 to 0.7742 — exactly matching FP32.

The reason is the same BatchNorm gain spread that caused defect 1. Filters within one layer differ in magnitude by more than an order of magnitude, and a single per-tensor scale forces the quietest filters into a handful of the 256 available levels. Per-channel costs 4 bytes per output channel of metadata and nothing at inference time, because the requantization multiplier is already per-channel.

### Why the Detect head stays in FP32

This turned out to be the most interesting result in the project, because **two independent production toolchains both destroy this model when asked to quantize it with default settings** — and for the same structural reason as defect **4** above.

**ONNX Runtime** scored **mAP 0.0000** on the first attempt while producing perfectly reasonable-looking box coordinates. The exported graph's final `Concat` merges box coordinates (0–640) with class scores (0–1) into one tensor. A single INT8 scale across that range is ~2.5 per step, so every class score rounds to zero. Restricting quantization to convolutions fixed it: **0.0000 → 0.8556**.

**TensorRT** does the same thing and cannot be talked out of it as easily. Ultralytics' `int8=True` export hands the graph to ModelOpt, which quantizes `Add`, `Mul`, `Conv`, `Resize` and `MaxPool` — 200 nodes in all. That sweeps in the arithmetic of the Detect tail itself: the DFL expectation and the `anchor ± distance` box decoding are `Add`/`Mul` ops operating on values spanning 0–640. The engine builds fine, runs fast, and is useless:

| | fire AP@0.5 | smoke AP@0.5 | mAP@0.5 |
|---|---|---|---|
| PyTorch FP32 on the same T4 | 0.7742 | 0.9976 | **0.8859** |
| TensorRT INT8 on the same T4 | 0.0000 | 0.1667 | **0.0833** |

That FP32 row is a control, run through the *identical* preprocessing, NMS and scoring code on the same machine, and it reproduces the laptop CPU result exactly (0.7742 / 0.9976). So the collapse is real, not a harness artifact — a 4.4 ms inference that finds essentially nothing.

The custom engine keeps the whole Detect head — DFL softmax, distance-to-box decoding, NMS — in FP32 by construction, which is precisely why it out-scores every production INT8 pipeline measured here. Quantizing a detector is not one decision applied uniformly to a graph; the box-decode arithmetic has a dynamic range that INT8 cannot hold, and a pipeline that does not know where the network stops being convolutions will quantize straight through it.

### A note on the dataset labels

Worth recording because it cost a debugging cycle: the upstream fire-8 dataset ships **3-class** labels (0=Fire, 1=default, 2=smoke), but this model is trained on the 2-class scheme produced by `datasets/fire-8/remap.py` (0 stays fire, 1 is dropped, 2 becomes smoke). Scoring against un-remapped labels yields mAP 0.0000 with no smoke ground truth at all — which looks exactly like a quantization failure and is not one. `benchmarks/setup_colab.py` applies the remap so a fresh clone reproduces the local labels byte-for-byte (31 fire + 20 smoke across the 49 test images).

## Benchmark methodology

The latency numbers took four corrections to get right, which is worth recording because the wrong versions all looked plausible:

1. **One process per runtime.** PyTorch, ONNX Runtime and OpenVINO each build a thread pool sized to the core count. Loaded into one process they oversubscribe the CPU — ONNX FP32 measured **360 ms** sharing a process, versus **34 ms** alone.
2. **A settle gap between runtimes.** Even in separate processes, launching back to back leaves the previous runtime's threads winding down. ONNX FP32 measured **341 ms** immediately after a PyTorch run, versus **34 ms** after an 8-second pause.
3. **Report the minimum.** Interference can only make a run slower, so the fastest observed time is the best estimate of real cost. An early run reported a *mean* of 272 ms against a *median* of 413 ms — a distribution that shape only happens when something else is stealing the CPU.
4. **Enough samples, on AC power.** On battery the CPU throttles and run-to-run variance was large enough to reorder the table: one pass had OpenVINO FP32 at 30.4 ms and the next at 34.8 ms, flipping it either side of ONNX FP32. The final numbers are 120 samples per runtime on mains power, where the spread closes to a few percent.

The result validates itself against the original M1 run on the same machine: PyTorch FP32 measures **42.4 ms** against **45.77 ms**, and OpenVINO FP32 **28.5 ms** against **30.57 ms**. (ONNX FP32 comes out faster than the original 40.10 ms, most likely a newer `onnxruntime` build.)

## Approach

**M2 (Quantization)** implements post-training static quantization from first principles rather than calling `torch.quantization`:

- **Weights**: symmetric INT8, per-tensor or per-output-channel
- **Activations**: asymmetric INT8, calibrated on 100 training images via forward hooks, at **every** `Conv` module rather than only the 23 top-level ones
- **Requantization**: INT32 accumulation, per-channel multiplier `M[oc] = s_in · s_w[oc] · bn_gain[oc] / s_out`

**M3 (C++ Engine)** implements the operators needed to run YOLOv8n:

- `Tensor` carrying its own scale and zero-point, so ops cannot silently mix scales
- Model loader for a custom binary + JSON format
- Quantized `conv2d` with INT32 accumulation, folded BatchNorm, and fused activation
- Element-wise ops (Concat with requantization, Upsample, MaxPool)
- Compound blocks (Bottleneck, C2f, SPPF)
- Detect head: DFL expectation, distance-to-box decoding, per-class NMS
- JPEG in, annotated JPEG out

## Reproducing

```bash
python quantization/calibrate.py                    # activation ranges -> activation_scales.json
python quantization/save_model.py --per-channel     # -> model_int8_pc.{bin,json}
cd cpp_engine && mkdir -p build && cd build && cmake .. && make

# layer-by-layer validation
./custom_engine --input-bin ../../test_input.bin --dump-dir dumps_pc \
    --model-json ../../quantization/model_int8_pc.json \
    --model-bin  ../../quantization/model_int8_pc.bin
python quantization/compare_layers.py --dumps cpp_engine/build/dumps_pc \
    --model quantization/model_int8_pc.json

# end-to-end mAP
bash cpp_engine/run_both.sh
python quantization/eval_map.py --csv cpp_engine/build/preds_pc/detections.csv

# runtime comparison
python benchmarks/run_all_benchmarks.py
```

## Repo contents

- `quantization/` — INT8 quantization pipeline (Python)
  - `quantization.py` — quantize / dequantize / quantized_matmul
  - `calibrate.py` — per-module activation calibration via forward hooks
  - `save_model.py` — BatchNorm folding and serialization (`--per-channel`)
  - `compare_layers.py` — layer-by-layer validation against PyTorch
  - `eval_map.py` — mAP scoring for the engine and FP32 reference
- `cpp_engine/` — the inference engine (M3)
  - `tensor.h/cpp` — INT8 tensor carrying its quantization parameters
  - `model.h/cpp` — model loader
  - `ops.h/cpp` — convolution, blocks, Detect head, NMS
  - `image_io.h/cpp` — JPEG decode, preprocessing, box drawing
  - `main.cpp` — CLI
  - `run_testset.sh`, `run_both.sh` — batch runners
- `benchmarks/` — runtime comparison
  - `bench_common.py` — shared preprocessing, NMS and scoring
  - `bench_one.py` — benchmark a single runtime in isolation
  - `run_all_benchmarks.py` — orchestrator, emits the table above
  - `onnx_int8_benchmark.py`, `openvino_int8_benchmark.py` — INT8 quantization + benchmark
  - `tensorrt_int8_colab.py` — TensorRT INT8, for a Colab GPU runtime

## Dataset

[fire-8 from Abonia1](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection) — 2-class fire and smoke detection dataset.
