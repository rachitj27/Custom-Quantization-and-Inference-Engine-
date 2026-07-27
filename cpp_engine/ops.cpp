#include "ops.h"
#include "model.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>

std::unique_ptr<Tensor> conv2d(
    const Tensor& input,
    const Tensor& weights,
    int stride,
    int padding,
    float input_scale,
    float weight_scale,
    float output_scale,
    int output_zero_point
) {
    int in_channels  = input.shape[0];
    int in_height    = input.shape[1];
    int in_width     = input.shape[2];

    int out_channels = weights.shape[0];
    int weight_in_ch = weights.shape[1];
    int kernel_h     = weights.shape[2];
    int kernel_w     = weights.shape[3];

    if (in_channels != weight_in_ch) {
        throw std::runtime_error("Input channels do not match weight in_channels");
    }

    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width  = (in_width  + 2 * padding - kernel_w) / stride + 1;

    auto output = std::make_unique<Tensor>(std::vector<int>{out_channels, out_height, out_width});
    //main 6 nested for loops to do the convolution
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                int32_t acc = 0;

                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
                                continue;
                            }

                            int input_idx = ic * (in_height * in_width) + ih * in_width + iw;
                            int weight_idx = oc * (in_channels * kernel_h * kernel_w)
                                           + ic * (kernel_h * kernel_w)
                                           + kh * kernel_w
                                           + kw;

                            acc += (int32_t)input.data[input_idx] * (int32_t)weights.data[weight_idx];
                        }
                    }
                }

            // Compute the combined scale factor
            float M = input_scale * weight_scale / output_scale;
            // Requantization math with no scale factor
            int32_t requantized = (int32_t)std::round(M * acc) + output_zero_point;

            // Clamp to INT8 range just in case of floating point rounding mistakes or calibration values out of range of what quantizatio was done with
            if (requantized > 127) requantized = 127;
            if (requantized < -128) requantized = -128;

            int output_idx = oc * (out_height * out_width) + oh * out_width + ow;
            output->data[output_idx] = (int8_t)requantized;
            }
        }
    }

    return output;
}
void silu_inplace(
    Tensor& tensor,
    float input_scale,
    int input_zero_point,
    float output_scale,
    int output_zero_point
) {
    
    // For each of the 256 possible input values, compute the SiLU result
    int8_t lookup[256];
    for (int i = -128; i <= 127; i++) {
        // Dequantize input value to FP32
        float x_fp32 = input_scale * (i - input_zero_point);
        
        // Apply SiLU: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + std::exp(-x_fp32));
        float silu = x_fp32 * sigmoid;
        
        // Requantize to INT8
        int32_t q = (int32_t)std::round(silu / output_scale) + output_zero_point;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        
        lookup[i + 128] = (int8_t)q;
    }
    
    // Apply lookup to every element in the tensor
    for (size_t i = 0; i < tensor.num_elements; i++) {
        int idx = tensor.data[i] + 128;  // shift to 0-255 range
        tensor.data[i] = lookup[idx];
    }
}
std::unique_ptr<Tensor> upsample2x(const Tensor& input) {
    // Input shape: (channels, height, width)
    // Output shape: (channels, height * 2, width * 2)
    int channels = input.shape[0];
    int in_h = input.shape[1];
    int in_w = input.shape[2];
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    
    auto output = std::make_unique<Tensor>(std::vector<int>{channels, out_h, out_w});
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                // Nearest neighbor: divide by 2 to get input position
                int ih = oh / 2;
                int iw = ow / 2;
                
                int in_idx  = c * (in_h * in_w)  + ih * in_w + iw;
                int out_idx = c * (out_h * out_w) + oh * out_w + ow;
                
                output->data[out_idx] = input.data[in_idx];
            }
        }
    }
    
    return output;
}

std::unique_ptr<Tensor> concat(const std::vector<const Tensor*>& tensors) {
    // All tensors must have same height and width, we concat along channel dim
    if (tensors.empty()) {
        throw std::runtime_error("Concat needs at least one tensor");
    }
    
    // Verify all tensors have the same spatial shape
    int height = tensors[0]->shape[1];
    int width  = tensors[0]->shape[2];
    int total_channels = 0;
    for (const Tensor* t : tensors) {
        if (t->shape[1] != height || t->shape[2] != width) {
            throw std::runtime_error("Concat tensors must have matching spatial dims");
        }
        total_channels += t->shape[0];
    }
    
    auto output = std::make_unique<Tensor>(std::vector<int>{total_channels, height, width});
    
    // Copy each input tensor's data into the output at the right channel offset
    int channel_offset = 0;
    for (const Tensor* t : tensors) {
        int t_channels = t->shape[0];
        size_t bytes_per_channel = height * width;  // INT8 = 1 byte per element
        
        for (int c = 0; c < t_channels; c++) {
            const int8_t* src = t->data + c * bytes_per_channel;
            int8_t* dst = output->data + (channel_offset + c) * bytes_per_channel;
            std::memcpy(dst, src, bytes_per_channel);
        }
        
        channel_offset += t_channels;
    }
    
    return output;
}
std::unique_ptr<Tensor> maxpool2d(const Tensor& input, int kernel_size) {
    // Input shape: (channels, height, width)
    // Output shape: same as input (stride=1, padding = kernel_size/2)
    int channels = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    int padding = kernel_size / 2;  // integer division; for kernel=5, padding=2
    
    auto output = std::make_unique<Tensor>(std::vector<int>{channels, height, width});
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                // Find max in the kernel_size x kernel_size window centered at (oh, ow)
                int8_t max_val = -128;  // INT8 minimum, so anything real beats this
                
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh + kh - padding;
                        int iw = ow + kw - padding;
                        
                        // Skip positions outside the input
                        if (ih < 0 || ih >= height || iw < 0 || iw >= width) {
                            continue;
                        }
                        
                        int in_idx = c * (height * width) + ih * width + iw;
                        if (input.data[in_idx] > max_val) {
                            max_val = input.data[in_idx];
                        }
                    }
                }
                
                int out_idx = c * (height * width) + oh * width + ow;
                output->data[out_idx] = max_val;
            }
        }
    }
    
    return output;
}
std::unique_ptr<Tensor> bottleneck(
    const Tensor& input,
    const Layer& conv1, float input_scale, int input_zp, float mid_scale, int mid_zp,
    const Layer& conv2, float out_scale, int out_zp,
    bool shortcut
) {
    auto x = conv2d(input, *conv1.weights, 1, 1,
                    input_scale, conv1.weight_scale, mid_scale, mid_zp);
    silu_inplace(*x, mid_scale, mid_zp, mid_scale, mid_zp);
    
    auto y = conv2d(*x, *conv2.weights, 1, 1,
                    mid_scale, conv2.weight_scale, out_scale, out_zp);
    silu_inplace(*y, out_scale, out_zp, out_scale, out_zp);
    
    if (shortcut) {
        for (size_t i = 0; i < y->num_elements; i++) {
            int32_t sum = (int32_t)y->data[i] + (int32_t)input.data[i];
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            y->data[i] = (int8_t)sum;
        }
    }
    return y;
}
std::unique_ptr<Tensor> c2f_block(
    const Tensor& input,
    const std::vector<const Layer*>& convs,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int n_bottlenecks,
    bool shortcut
) {
    float in_s = scales[0];
    float out_s = scales[1];
    int out_z = zero_points[1];
    
    // conv[0] is cv1, conv[1] is cv2 (final output conv)
    // convs[2..] are the bottleneck convs, 2 per bottleneck
    
    auto x = conv2d(input, *convs[0]->weights, 1, 0,
                    in_s, convs[0]->weight_scale, out_s, out_z);
    silu_inplace(*x, out_s, out_z, out_s, out_z);
    
    int channels = x->shape[0];
    int half_channels = channels / 2;
    int height = x->shape[1];
    int width = x->shape[2];
    
    Tensor a({half_channels, height, width});
    Tensor b({half_channels, height, width});
    size_t half_bytes = half_channels * height * width;
    std::memcpy(a.data, x->data, half_bytes);
    std::memcpy(b.data, x->data + half_bytes, half_bytes);
    
    std::vector<std::unique_ptr<Tensor>> owned_tensors;
    std::vector<const Tensor*> concat_inputs;
    concat_inputs.push_back(&a);
    concat_inputs.push_back(&b);
    
    const Tensor* current = &b;
    for (int i = 0; i < n_bottlenecks; i++) {
        // Bottleneck convs start at index 2, two per bottleneck
        int c1_idx = 2 + i * 2;
        int c2_idx = 3 + i * 2;
        
        auto next = bottleneck(*current,
                               *convs[c1_idx], out_s, out_z, out_s, out_z,
                               *convs[c2_idx], out_s, out_z,
                               shortcut);
        current = next.get();
        concat_inputs.push_back(current);
        owned_tensors.push_back(std::move(next));
    }
    
    auto concat_out = concat(concat_inputs);
    
    // Final conv is conv[1] (cv2)
    auto result = conv2d(*concat_out, *convs[1]->weights, 1, 0,
                         out_s, convs[1]->weight_scale, out_s, out_z);
    silu_inplace(*result, out_s, out_z, out_s, out_z);
    return result;
}

std::unique_ptr<Tensor> sppf_block(
    const Tensor& input,
    const Layer& cv1, float input_scale, int input_zp, float mid_scale, int mid_zp,
    const Layer& cv2, float out_scale, int out_zp,
    int kernel_size
) {
    auto x = conv2d(input, *cv1.weights, 1, 0,
                    input_scale, cv1.weight_scale, mid_scale, mid_zp);
    silu_inplace(*x, mid_scale, mid_zp, mid_scale, mid_zp);
    
    auto y1 = maxpool2d(*x, kernel_size);
    auto y2 = maxpool2d(*y1, kernel_size);
    auto y3 = maxpool2d(*y2, kernel_size);
    
    std::vector<const Tensor*> to_concat = {x.get(), y1.get(), y2.get(), y3.get()};
    auto merged = concat(to_concat);
    
    auto result = conv2d(*merged, *cv2.weights, 1, 0,
                         mid_scale, cv2.weight_scale, out_scale, out_zp);
    silu_inplace(*result, out_scale, out_zp, out_scale, out_zp);
    return result;
}
std::unique_ptr<Tensor> conv_block(
    const Tensor& input,
    const Layer& conv,
    int stride,
    int padding,
    float input_scale,
    float output_scale,
    int output_zp
) {
    auto out = conv2d(input, *conv.weights, stride, padding,
                      input_scale, conv.weight_scale, output_scale, output_zp);
    silu_inplace(*out, output_scale, output_zp, output_scale, output_zp);
    return out;
}

std::vector<std::unique_ptr<Tensor>> run_forward(const Model& model, const Tensor& input) {
    std::vector<std::unique_ptr<Tensor>> outputs;
    outputs.resize(model.architecture.size());
    
    float input_scale = 1.0f / 255.0f;
    
    for (size_t i = 0; i < model.architecture.size(); i++) {
        const auto& arch = model.architecture[i];
        
        float out_scale = model.activation_scales.count(arch.layer_id)
            ? model.activation_scales.at(arch.layer_id).scale : 1.0f;
        int out_zp = model.activation_scales.count(arch.layer_id)
            ? model.activation_scales.at(arch.layer_id).zero_point : 0;
        
        const Tensor* in_tensor = nullptr;
        float in_scale = input_scale;
        
        if (!arch.is_multi_input) {
            if (arch.input_from_single == -1) {
                in_tensor = &input;
                in_scale = input_scale;
            } else {
                in_tensor = outputs[arch.input_from_single].get();
                in_scale = model.activation_scales.count(arch.input_from_single)
                    ? model.activation_scales.at(arch.input_from_single).scale : 1.0f;
            }
        }
        
        std::unique_ptr<Tensor> result;
        
        if (arch.type == "Conv") {
            int stride = (arch.layer_id == 0 || arch.layer_id == 1 ||
                          arch.layer_id == 3 || arch.layer_id == 5 ||
                          arch.layer_id == 7 || arch.layer_id == 16 ||
                          arch.layer_id == 19) ? 2 : 1;
            int padding = 1;
            const Layer& conv = model.conv_layers[arch.conv_ids[0]];
            result = conv_block(*in_tensor, conv, stride, padding,
                                in_scale, out_scale, out_zp);
        }
        else if (arch.type == "C2f") {
            std::vector<const Layer*> convs;
            for (int cid : arch.conv_ids) convs.push_back(&model.conv_layers[cid]);
            std::vector<float> scales = {in_scale, out_scale};
            std::vector<int> zps = {0, out_zp};
            result = c2f_block(*in_tensor, convs, scales, zps,
                               arch.n_bottlenecks, arch.shortcut);
        }
        else if (arch.type == "SPPF") {
            const Layer& cv1 = model.conv_layers[arch.conv_ids[0]];
            const Layer& cv2 = model.conv_layers[arch.conv_ids[1]];
            result = sppf_block(*in_tensor, cv1, in_scale, 0, out_scale, out_zp,
                                cv2, out_scale, out_zp, arch.kernel_size);
        }
        else if (arch.type == "Upsample") {
            result = upsample2x(*in_tensor);
        }
        else if (arch.type == "Concat") {
            std::vector<const Tensor*> to_concat;
            for (int layer_idx : arch.input_from_multi) {
                to_concat.push_back(outputs[layer_idx].get());
            }
            result = concat(to_concat);
        }
       else if (arch.type == "Detect") {
            // Return the deepest feature map (from layer 21) as final output
            // In production, this would feed into box decoding + NMS
            int last_source = arch.input_from_multi.back();
            result = std::make_unique<Tensor>(outputs[last_source]->shape);
            std::memcpy(result->data, outputs[last_source]->data,
                        outputs[last_source]->num_elements);
        }
        
        outputs[i] = std::move(result);
    }
    
    return outputs;
}