# Custom AI Hardware Inference

A fire and smoke detector that runs on hand written C++ instead of a machine learning library.

The starting point is a YOLOv8n model trained to find fire and smoke in photos. Normally you would run that model with a library like PyTorch or ONNX Runtime, which does all the math for you. This project rebuilds that math from scratch, first shrinking the model so it uses 8 bit whole numbers instead of 32 bit decimals, then writing the code that actually runs it.

Give it a photo and it draws boxes around the fire and smoke it finds.

## Status

- [x] **M1, FP32 baseline.** Trained YOLOv8n on the fire-8 dataset and benchmarked it across four production libraries.
- [x] **M2, INT8 quantization from scratch.** Wrote the math that converts the model from decimals to 8 bit whole numbers.
- [x] **M3, custom C++ engine.** Loads the shrunken model, runs it end to end, decodes the boxes, and keeps **99.6% of the original accuracy**.
- [ ] **M4, hardware accelerator.** An FPGA version written in Verilog. Stretch goal.

## What quantization means here

A trained neural network stores its numbers as 32 bit decimals. Quantization replaces each one with an 8 bit whole number, which can only hold 256 distinct values. You store a scale factor alongside it so the original value can be approximated as `scale x (stored_value - offset)`.

The payoff is a model that is four times smaller and, on hardware built for it, several times faster. The cost is precision, since you are rounding every number onto a grid of 256 steps. Most of the time that rounding is harmless. Figuring out where it is *not* harmless turned out to be the most interesting part of this project, and there is a section on it below.

## Usage

```bash
cd cpp_engine && mkdir -p build && cd build
cmake .. && make

./custom_engine path/to/fire.jpg              # writes fire_pred.jpg
./custom_engine fire.jpg -o out.jpg --conf 0.3
./custom_engine fire.jpg --model-json ../../quantization/model_int8_pc.json \
                         --model-bin  ../../quantization/model_int8_pc.bin
```

It prints what it found and saves a copy of the image with boxes drawn on it.

```
Loaded fire.jpg (640x640)
Inference: 3516.4 ms
Detections: 1
  fire   0.830  box=[329.2, 158.6, 532.6, 585.8]
```

You need `cmake`, a C++17 compiler, and `nlohmann-json`. Reading and writing JPEGs uses [stb](https://github.com/nothings/stb), which is included in `cpp_engine/third_party/`.

## The model

YOLOv8n fine tuned on the fire-8 dataset, which has two classes, fire and smoke. Trained for 50 epochs on a Tesla T4 in Google Colab.

- 3.0M parameters
- 6.0 MB as 32 bit decimals
- 2.9 MB as 8 bit whole numbers, so about four times smaller

## Results

### Accuracy

The score below is mAP@0.5, a standard detection metric. Roughly speaking it asks how often the model draws a box in the right place with the right label, where a box counts as correct if it overlaps the true box by at least half. Higher is better and 1.0 is perfect.

Every number here comes from the same scoring script (`quantization/eval_map.py`) run over the same 49 test images with the same image preparation, so the rows can be compared directly.

| Configuration | fire | smoke | mAP@0.5 | vs FP32 |
|---|---|---|---|---|
| Original 32 bit model (PyTorch, ONNX and OpenVINO all agree) | 0.7742 | 0.9976 | **0.8859** | reference |
| **This engine, per channel weights** | 0.7742 | 0.9909 | **0.8826** | **99.6%** |
| ONNX Runtime INT8 | 0.7352 | 0.9759 | 0.8556 | 96.6% |
| OpenVINO INT8 | 0.7126 | 0.9051 | 0.8089 | 91.3% |
| This engine, per tensor weights | 0.5935 | 0.9426 | 0.7680 | 86.7% |

The hand written engine is the **most accurate 8 bit version measured here**, ahead of both production libraries. Not because the arithmetic is cleverer, but because of one design choice explained in the section on the detection head further down.

> An earlier version of this README quoted 0.9253. That figure came from Ultralytics' own scoring tool, which prepares images differently and counts matches slightly differently. This table uses one self contained scorer so the custom engine can be measured the same way as everything else. The model itself has not changed.

### Speed

Measured on an Intel Core Ultra 7 256V laptop running on mains power. Each library was timed in its own process with a pause in between, 10 rounds of 12 images each, reporting the fastest time seen. The section on benchmark methodology explains why all of that matters.

| Runtime | Precision | Best | Median | mAP@0.5 |
|---------|-----------|------|--------|---------|
| PyTorch (Ultralytics) | FP32 | 42.4 ms | 58.5 ms | 0.8859 |
| ONNX Runtime | FP32 | 24.1 ms | 28.4 ms | 0.8859 |
| ONNX Runtime | INT8 | 30.8 ms | 35.8 ms | 0.8556 |
| OpenVINO | FP32 | 28.5 ms | 33.3 ms | 0.8859 |
| **OpenVINO** | **INT8** | **13.9 ms** | 16.4 ms | 0.8089 |
| This engine, per channel | INT8 | 3516.4 ms | 3537.1 ms | 0.8826 |
| This engine, per tensor | INT8 | 3418.1 ms | 3461.4 ms | 0.7680 |

GPU results are listed separately because a data centre GPU and a laptop CPU are not comparable. These ran on a Tesla T4 in Colab using the same image preparation and scoring code.

| Runtime | Precision | Best | Median | mAP@0.5 |
|---------|-----------|------|--------|---------|
| PyTorch (Ultralytics), T4 | FP32 | 8.0 ms | 8.1 ms | 0.8859 |
| **TensorRT, T4** | **INT8** | **4.4 ms** | 5.1 ms | **0.0833** |
| TensorRT, T4 | FP16 | 12.4 ms | not re-measured | carried over from M1 |

What the numbers say.

- **OpenVINO INT8 wins on CPU at 13.9 ms**, just over twice as fast as the same library running the full precision model. This is what 8 bit is supposed to buy you.
- **ONNX Runtime INT8 is slightly slower than its own full precision build**, 30.8 ms against 24.1 ms, and that held across every measurement run. Only the convolutions could be quantized safely, so the graph pays the cost of converting numbers back and forth around all 64 of them without ever getting to run one long stretch of pure integer math. Going to 8 bit does not automatically make things faster. It depends on whether the library can actually use the result.
- **This engine is roughly 250 times slower than OpenVINO.** It is six nested loops with no vectorization, no threading, and no memory tricks. Speeding it up is what M4 is for. Worth noting though that it is only about 12% slower than the earlier, broken version of itself, despite now doing considerably more work.
- The two rows for this engine differ by about 3%, which confirms that per channel weights are effectively free at runtime.
- **TensorRT INT8 is the fastest result anywhere in this project at 4.4 ms, and also by far the least accurate at 0.0833.** A default export destroys this model. That story is below.

## Fixing the accuracy

An earlier version of this engine drifted badly, and the drift compounded the deeper you went. At the first layer, 38.3% of its numbers were within one step of the reference. By the last layer only 1.1% were. The README at the time blamed one particular cause. That cause was real but it was not the main one.

| Layer | Within one step, before and after | Average error, before and after |
|---|---|---|
| L00 | 38.3% to **99.91%** | 9.0 to **0.081** |
| L02 | 10.2% to **77.46%** | 80.2 to **0.954** |
| L15 | 17.6% to **99.40%** | 15.3 to **0.241** |
| L18 | 2.8% to **94.92%** | 83.8 to **0.393** |
| L21 | 1.1% to **94.94%** | 132.4 to **0.410** |

There were five separate bugs. In order of how much they mattered.

**1. An entire operation was missing.** YOLOv8 layers are built from three pieces stacked together, a convolution, then a normalization step called BatchNorm, then an activation function. The export script only saved the convolution. BatchNorm was silently dropped.

This is not a rounding error. BatchNorm multiplies each output channel by its own factor, and at the very first layer those factors range from 13x to 151x. The engine was skipping all of it.

The bug hid because the test that was supposed to catch it compared the engine against a reference that *also* left BatchNorm out. Both sides were wrong in the same way, so the test passed at 84%.

The fix folds BatchNorm into a per channel multiplier and offset that get applied while converting numbers back from integers, so it costs nothing extra at runtime.

**2. The offset was ignored during convolution.** Each 8 bit number represents a real value through a scale and an offset. The convolution code used the stored numbers directly without subtracting the offset first, which throws off every convolution in the network. Handling it properly also fixes image border handling for free, since a padded edge pixel represents a real zero and now correctly contributes nothing.

**3. Values were clipped before the activation function ran.** The engine squeezed each convolution's output into the range recorded for that layer, then applied the activation function. But that recorded range was measured *after* the activation, which never outputs anything below about -0.278. The raw convolution output goes far more negative than that, so the entire negative tail was cut off before the activation function ever saw it. Doing the convolution and the activation together, before rounding, removes the problem.

**4. Feature maps on different scales were glued together.** When the network merges two sets of features, each set arrives with its own scale factor. The old code copied the raw bytes together as if the scales matched. They now get converted to a common scale first.

**5. Skip connections dropped the offset.** Adding two 8 bit numbers is not as simple as adding the stored values, because the offset gets counted twice. The addition now happens on the real values and gets converted back once.

Fixing the first three alone was enough to make layer 0 match the reference exactly.

### Per tensor against per channel weights

Both versions are built in, using `python quantization/save_model.py [--per-channel]`, and both are in the results table.

Per channel is clearly worth it here, taking the score from **0.7680 to 0.8826**. The fire class in particular goes from 0.5935 to 0.7742, which is exactly the full precision number.

The reason connects back to bug 1. A single layer contains many filters, and their strengths can differ by more than a factor of ten. One shared scale for the whole layer forces the quietest filters into a handful of the 256 available steps, and their detail is lost. Giving each filter its own scale costs four extra bytes per filter and nothing at all at runtime.

## Why the detection head stays in full precision

This turned out to be the most interesting finding in the project, because **two separate professional toolchains both wreck this model when you ask them to quantize it with default settings**, and for the same underlying reason as bug 4 above.

**ONNX Runtime** scored **0.0000** on the first attempt. Looking at the raw output showed exactly where the damage was.

| | box values, min and max | confidence scores, min and max |
|---|---|---|
| Full precision | 6.70 to 634.43 | 0.0 to 0.751 |
| 8 bit, everything quantized | 7.60 to 635.58 | **0.0 to 0.0** |

The boxes were fine. **Every confidence score was exactly zero.** At the very end the network glues the box coordinates and the confidence scores into a single block of numbers. Box coordinates run from 0 to 640. Confidence scores run from 0 to 1. Forced to cover both with one shared scale, each step is about 2.5, so every score, all of which are below 1, rounds down to zero. Nothing passes the confidence threshold and the model reports nothing at all. Quantizing only the convolutions fixed it, taking the score from **0.0000 to 0.8556**.

It is worth being clear about what is *not* the problem here, because the obvious guess is wrong. Rounding the box coordinates onto that same coarse grid is survivable. Shifting every edge of every true box by 2.51 pixels still leaves an average overlap of 0.925 for fire and 0.963 for smoke, and **not a single box** falls below the 0.5 threshold that counts as a match. Eight bit precision ruins detectors through the confidence scores, not through blurry box coordinates.

**TensorRT** fails the same way and is harder to talk out of it. The standard export path quantizes 200 operations including the arithmetic that turns raw network output into box coordinates. The result builds cleanly, runs faster than anything else measured here, and finds almost nothing.

| | fire | smoke | mAP@0.5 |
|---|---|---|---|
| Full precision on the same T4 | 0.7742 | 0.9976 | **0.8859** |
| TensorRT 8 bit on the same T4 | 0.0000 | 0.1667 | **0.0833** |

That first row is a control. It ran through the *same* image preparation and scoring code on the *same* machine, and it reproduces the laptop result exactly. So the collapse is real and not a measurement mistake.

The precise mechanism inside TensorRT was not pinned down the way it was for ONNX Runtime, and its toolchain handles that final merge differently, which fits a score of 0.0833 rather than a clean zero. The shape of the failure is suggestive though. Fire scores nothing while smoke still manages 0.1667, and smoke boxes cover about 70% of the image on average against fire's 16%. That is what it looks like when labels get swapped and a mislabelled box is simply big enough to overlap something by accident. Confirming that would mean using the quantization toolkit directly instead of through Ultralytics.

None of this means TensorRT is bad. It means the default settings are wrong for this kind of model. This engine keeps the entire detection head in full precision by design, which is exactly why it scores higher than every professional 8 bit pipeline measured here. Quantizing a detector is not one decision applied evenly across a network. The final box decoding step needs a range of values that 8 bits cannot hold, and a tool that does not know where the convolutions stop will quantize straight through it.

## A trap in the dataset labels

Worth writing down because it cost a debugging session. The public fire-8 dataset ships labels with three categories, numbered 0 for fire, 1 for an unused category, and 2 for smoke. This model was trained on a two category version produced by `datasets/fire-8/remap.py`, where 0 stays fire, 1 is thrown away, and 2 becomes 1 for smoke.

Score the model against the original labels and you get 0.0000 with no smoke targets at all, which looks exactly like a quantization disaster and is nothing of the sort. `benchmarks/setup_colab.py` applies the same conversion so a fresh download matches the local labels exactly, 31 fire and 20 smoke boxes across the 49 test images.

## How the speed was measured

Getting believable timings took four corrections, and the wrong versions all looked perfectly reasonable.

1. **One library per process.** PyTorch, ONNX Runtime and OpenVINO each start a pool of worker threads sized to the number of CPU cores. Load them together and they fight over the same cores. ONNX measured **360 ms** sharing a process against **34 ms** on its own.
2. **A pause between libraries.** Even in separate processes, starting one immediately after another catches the previous one still shutting down its threads. ONNX measured **341 ms** straight after a PyTorch run against **34 ms** after an eight second pause.
3. **Report the fastest run, not the average.** Interference can only ever slow a run down, so the quickest time is the closest thing to the true cost. One early run reported an average of 272 ms against a median of 413 ms, a pattern that only happens when something else is stealing the processor.
4. **Enough samples, on mains power.** On battery the processor throttles and the numbers moved enough to reorder the table, with one pass putting OpenVINO at 30.4 ms and the next at 34.8 ms. The final figures use 120 samples per library on mains power, where the spread narrows to a few percent.

The result checks out against the original M1 measurements on the same laptop. PyTorch comes in at 42.4 ms against 45.77 ms, and OpenVINO at 28.5 ms against 30.57 ms. ONNX comes out faster than its original 40.10 ms, most likely a newer build of the library.

## How it works

**M2, the quantization.** Written from first principles rather than calling a library function.

- Weights are converted using one scale per filter, or optionally one scale for the whole layer
- Activation ranges are measured by running 100 training images through the model and recording the highest and lowest values at **every** convolution, rather than only at the 23 top level blocks
- Sums are accumulated in 32 bit whole numbers to avoid overflow, then converted back using a per channel multiplier that also carries the folded BatchNorm

**M3, the C++ engine.**

- A tensor type that carries its own scale and offset, so operations cannot silently mix incompatible numbers
- A loader for a custom binary and JSON format
- Convolution with 32 bit accumulation, folded BatchNorm, and the activation applied before rounding
- Supporting operations for merging, upscaling and pooling feature maps
- The compound blocks YOLOv8 is built from
- The detection head, including box decoding and removal of duplicate boxes
- JPEG in, annotated JPEG out

## Reproducing

```bash
python quantization/calibrate.py                    # measure activation ranges
python quantization/save_model.py --per-channel     # write the quantized model
cd cpp_engine && mkdir -p build && cd build && cmake .. && make

# check the engine layer by layer against PyTorch
./custom_engine --input-bin ../../test_input.bin --dump-dir dumps_pc \
    --model-json ../../quantization/model_int8_pc.json \
    --model-bin  ../../quantization/model_int8_pc.bin
python quantization/compare_layers.py --dumps cpp_engine/build/dumps_pc \
    --model quantization/model_int8_pc.json

# score it on the test set
bash cpp_engine/run_both.sh
python quantization/eval_map.py --csv cpp_engine/build/preds_pc/detections.csv

# compare against the production libraries
python benchmarks/run_all_benchmarks.py
```

## What is in here

- `quantization/`, the Python side
  - `quantization.py`, the core conversion math
  - `calibrate.py`, measures activation ranges
  - `save_model.py`, folds BatchNorm and writes the model files
  - `compare_layers.py`, checks the engine layer by layer against PyTorch
  - `eval_map.py`, scores detections against the true boxes
- `cpp_engine/`, the engine itself
  - `tensor.h/cpp`, the tensor type
  - `model.h/cpp`, the model loader
  - `ops.h/cpp`, convolution, blocks, detection head and duplicate removal
  - `image_io.h/cpp`, JPEG handling, image preparation and box drawing
  - `main.cpp`, the command line interface
  - `run_testset.sh` and `run_both.sh`, batch runners
- `benchmarks/`, comparisons against production libraries
  - `bench_common.py`, shared image preparation and scoring
  - `bench_one.py`, times a single library in isolation
  - `run_all_benchmarks.py`, runs them all and prints the table
  - `onnx_int8_benchmark.py` and `openvino_int8_benchmark.py`, 8 bit conversion and timing
  - `tensorrt_int8_colab.py` and `setup_colab.py`, the TensorRT run, for a Colab GPU

## Dataset

[fire-8 from Abonia1](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection), a two class fire and smoke detection dataset.
