#ifndef MODEL_H
#define MODEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensor.h"

// One Conv2d: INT8 weights plus everything needed to turn an INT32 accumulator
// back into real numbers.
struct Layer {
    int layer_id = -1;
    std::string path;         // e.g. "model.2.cv1.conv"
    std::string module_path;  // enclosing Conv module, e.g. "model.2.cv1"

    std::vector<int> weight_shape;  // (out_ch, in_ch, kh, kw)
    size_t byte_offset = 0;
    size_t byte_length = 0;

    // One entry per output channel. A per-tensor export is expanded on load, so
    // the convolution reads weight_scales[oc] either way and costs nothing extra.
    std::vector<float> weight_scales;
    int weight_zero_point = 0;  // always 0: weights are symmetric

    int stride = 1;
    int padding = 0;
    int groups = 1;

    // BatchNorm folded into a per-output-channel affine transform, applied
    // during requantization. For bare Conv2d layers (the Detect head's 1x1
    // predictors) gain is 1 and bias is the layer's own bias.
    std::vector<float> bn_gain;
    std::vector<float> bn_bias;

    bool silu = false;  // whether a SiLU follows this conv

    // Calibrated output quantization. Absent for layers whose output the engine
    // keeps in FP32 (the Detect head predictors).
    bool has_out_scale = false;
    float out_scale = 1.0f;
    int out_zero_point = 0;

    std::unique_ptr<Tensor> weights;

    // ---- Alternate kernel representations, built once at load time ----
    //
    // The same convolution can run three ways. Only the arrays the selected
    // kernel needs are ever filled, so the unused ones cost nothing.

    // ScalarFp32: the identical weights, dequantized, in the same
    // [oc][ic][kh][kw] order as the INT8 blob. Keeping the layout identical is
    // the point: the FP32 and INT8 scalar kernels then differ in arithmetic
    // width and nothing else, which is what makes the comparison mean anything.
    std::vector<float> weights_fp32;

    // VnniInt8: weights repacked to [oc][kh][kw][ic_padded] so that input
    // channels -- the axis the convolution reduces over -- are contiguous.
    // VPDPBUSD consumes 32 adjacent bytes from each operand, which the native
    // [oc][ic][kh][kw] layout cannot supply, since there consecutive channels
    // sit a whole feature map apart.
    //
    // ic is padded up to a multiple of 32 with zero weights. A zero weight adds
    // nothing to the dot product and nothing to weight_sums, so the padding is
    // arithmetically invisible and removes the need for a scalar tail loop.
    std::vector<int8_t> weights_hwc;

    // Sum of every weight in one output channel's window. VPDPBUSD computes
    // sum(a * w) over raw bytes, but the convolution needs sum((a - z) * w).
    // Because sum((a - z) * w) == sum(a * w) - z * sum(w), precomputing sum(w)
    // turns the zero-point correction into one subtraction per output pixel
    // instead of one per multiply-accumulate.
    std::vector<int32_t> weight_sums;

    int ic_padded = 0;

    int out_channels() const { return weight_shape[0]; }
};

struct BottleneckSpec {
    int cv1 = -1;
    int cv2 = -1;
    bool shortcut = false;
};

// One of the 23 top-level modules in the YOLOv8n graph.
struct ArchLayer {
    int layer_id = -1;
    std::string type;

    bool input_is_raw = false;  // true only for layer 0, which reads the image
    std::vector<int> inputs;    // producing layer ids, in order

    // Conv
    int conv = -1;

    // C2f / SPPF
    int cv1 = -1;
    int cv2 = -1;
    std::vector<BottleneckSpec> bottlenecks;
    float concat_scale = 1.0f;  // C2f: common scale for the pre-cv2 concat
    int concat_zero_point = 0;
    int k = 5;  // SPPF pooling kernel

    // Concat
    float out_scale = 1.0f;
    int out_zero_point = 0;

    // Detect
    int nc = 0;
    int reg_max = 0;
    int no = 0;
    std::vector<float> strides;
    std::vector<std::vector<int>> det_cv2;  // [level][3] box branch
    std::vector<std::vector<int>> det_cv3;  // [level][3] class branch
    int dfl = -1;
};

struct Model {
    int num_bits = 8;
    float input_scale = 1.0f / 255.0f;
    int input_zero_point = -128;

    std::vector<Layer> conv_layers;
    std::vector<ArchLayer> architecture;
};

// A decoded detection, in pixel coordinates of the 640x640 network input.
struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    int cls = 0;
};

// Which convolution implementation the engine runs.
//
//   ScalarInt8  the original loop, one INT8 multiply-accumulate at a time
//   ScalarFp32  the same loop in FP32, as an arithmetic-width baseline
//   VnniInt8    INT8 through AVX-VNNI, 32 multiply-accumulates per instruction
enum class Kernel { ScalarInt8, ScalarFp32, VnniInt8 };

const char* kernel_name(Kernel k);
bool parse_kernel(const std::string& name, Kernel& out);

// Whether this CPU actually has the AVX-VNNI instructions.
bool vnni_supported();

// Build whichever weight representation the kernel needs. Safe to call twice.
void prepare_kernel(Model& model, Kernel k);

Model load_model_metadata(const std::string& json_path);
void load_model_weights(Model& model, const std::string& bin_path);

#endif
