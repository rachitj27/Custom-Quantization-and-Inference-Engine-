# Custom AI Hardware Inference

A fire and smoke detector that runs on hand written C++ instead of a machine learning library.

The starting point is a YOLOv8n model trained to find fire and smoke in photos. Normally you would run it with PyTorch or ONNX Runtime, which does all the math for you. This project rebuilds that math from scratch. First it shrinks the model so it uses 8 bit whole numbers instead of 32 bit decimals, then it implements the code that actually runs it, including the convolutions, the detection head, box decoding and duplicate removal.

Give it a photo and it draws boxes around the fire and smoke it finds.

```bash
./custom_engine fire.jpg      # writes fire_pred.jpg
```

```
Loaded fire.jpg (640x640)
Inference: 3516.4 ms
Detections: 1
  fire   0.830  box=[329.2, 158.6, 532.6, 585.8]
```

## Results

**It keeps 99.6% of the original model's accuracy.** The metric is mAP@0.5, a standard detection score where higher is better and 1.0 is perfect. Every row below was produced by the same scoring code over the same 49 test images, so they compare directly.

| Configuration | mAP@0.5 | vs full precision |
|---|---|---|
| Original 32 bit model | 0.8859 | reference |
| **This engine** | **0.8826** | **99.6%** |
| TensorRT 8 bit, quantization scoped correctly | 0.8656 | 97.7% |
| ONNX Runtime 8 bit | 0.8556 | 96.6% |
| OpenVINO 8 bit | 0.8089 | 91.3% |
| TensorRT 8 bit, default settings | 0.0833 | 9.4% |

Speed is the honest weak point and the current focus. The engine is a correctness first implementation with no vectorization, no threading and no cache blocking.

| Runtime | Precision | Best latency |
|---|---|---|
| PyTorch, laptop CPU | FP32 | 42.4 ms |
| ONNX Runtime, laptop CPU | FP32 | 24.1 ms |
| OpenVINO, laptop CPU | INT8 | 13.9 ms |
| TensorRT, Tesla T4 | INT8 | 4.9 ms |
| This engine, laptop CPU | INT8 | 3516 ms |

## Key findings

**Two production toolchains silently destroy this model with default settings.** ONNX Runtime scored 0.0000 and TensorRT scored 0.0833 out of the box. Neither one errors or warns. They produce a model that builds cleanly, runs faster than anything else measured, and detects almost nothing.

The cause is the same in both cases. At the end of the network the box coordinates and the confidence scores get merged into one block of numbers. Coordinates run from 0 to 640 and scores run from 0 to 1. Forced to cover both ranges with a single 8 bit scale, each step is about 2.5, so every score rounds down to zero. Reading the raw output made it unambiguous.

| | box values | confidence scores |
|---|---|---|
| Full precision | 6.70 to 634.43 | 0.0 to **0.751** |
| 8 bit, everything converted | 7.60 to 635.58 | 0.0 to **0.0** |

**Restricting the conversion to convolutions fixes it, and costs almost nothing.** Driving NVIDIA's quantization toolkit directly and converting only the convolutions took TensorRT from 0.0833 to 0.8656, which is 97.7% of full precision. It cost half a millisecond of latency and still runs 1.6 times faster than the full precision model. Same weights, same hardware, same calibration images. The only difference is which parts of the network were converted.

This is also why the hand written engine scores highest. It keeps the detection head in full precision by design.

**The intuitive explanation for that failure is wrong.** Coarse box coordinates are not the problem. Shifting every edge of every true box by 2.51 pixels still leaves an average overlap above 0.92, and not one box falls below the threshold that counts as a match. 8 bit precision ruins detectors through the confidence scores, not through blurry boxes.

**Going to 8 bit does not automatically make anything faster.** ONNX Runtime's 8 bit build is slower than its own full precision build, 30.8 ms against 24.1 ms, consistently across every run. Only the convolutions can be converted safely, so the graph pays the cost of switching number formats around all 64 of them without ever running one long stretch of integer arithmetic.

## What is actually 8 bit

Worth stating precisely. The weights are 8 bit, stored and loaded as raw bytes. The activations between every layer are 8 bit. The convolution is a true integer multiply accumulate into a 32 bit accumulator. The conversion between layers and the detection head decoding are floating point, the latter deliberately.

## Current focus, closing the speed gap

The model needs 4.04 billion multiply accumulates per image. This engine gets through 1.15 billion of them per second. OpenVINO manages 291 billion, and the ceiling for this laptop's processor is somewhere around 1500 billion. That gap is the next phase of the project, and it breaks down into named techniques rather than mystery.

| Technique | What it addresses |
|---|---|
| Loop reordering for a contiguous inner loop | cache behaviour, currently close to worst case |
| im2col plus a blocked matrix multiply | how real engines implement convolution |
| AVX2 VNNI intrinsics | the actual reason 8 bit is fast |
| Multithreading | using more than one of eight cores |
| Fixed point requantization | the conversion path, once convolution is fast |

Profiling the model first turned up something useful. The detection head accounts for 36.6% of all the arithmetic, and its four highest resolution convolutions are the four most expensive operations in the whole network. The backbone is not the bottleneck here.

Accuracy is protected while this work happens. Layer by layer comparison against PyTorch plus end to end scoring over the test set means any optimization that breaks correctness shows up immediately.

## Reproducing

```bash
python quantization/calibrate.py                    # measure activation ranges
python quantization/save_model.py --per-channel     # write the quantized model
cd cpp_engine && mkdir -p build && cd build && cmake .. && make

python quantization/compare_layers.py --dumps cpp_engine/build/dumps_pc \
    --model quantization/model_int8_pc.json         # layer by layer check
bash cpp_engine/run_both.sh
python quantization/eval_map.py --csv cpp_engine/build/preds_pc/detections.csv
python benchmarks/run_all_benchmarks.py             # compare against the libraries
```

Requires `cmake`, a C++17 compiler and `nlohmann-json`. Image handling uses [stb](https://github.com/nothings/stb), included in `cpp_engine/third_party/`.

## Layout

- `quantization/`, the Python side. Conversion math, calibration, BatchNorm folding, and the two validation harnesses.
- `cpp_engine/`, the engine. Tensor type, model loader, operators, detection head, image handling.
- `benchmarks/`, comparisons against PyTorch, ONNX Runtime, OpenVINO and TensorRT.

## Dataset

[fire-8 from Abonia1](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection), a two class fire and smoke detection dataset. The model is YOLOv8n fine tuned for 50 epochs, 3.0M parameters, 6.0 MB at full precision and 2.9 MB once converted.
