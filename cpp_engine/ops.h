#ifndef OPS_H
#define OPS_H

#include <memory>
#include <vector>

#include "model.h"
#include "tensor.h"

// ---------------------------------------------------------------------------
// Core convolution
//
// Both variants share the same INT8 x INT8 -> INT32 accumulation. They differ
// only in what they do with the result:
//
//   acc  = sum over the receptive field of (q_in - z_in) * q_w
//   real = in.scale * w_scale * acc * bn_gain[oc] + bn_bias[oc]
//   real = silu(real)                                    (if the layer has one)
//
// Subtracting the input zero-point inside the accumulation is what makes
// asymmetric activations correct, and it makes zero-padding exact for free:
// a padded position holds real 0, i.e. q == z_in, so it contributes nothing.
//
// Folding BatchNorm into bn_gain/bn_bias means the activation is applied to the
// true pre-activation value. Requantizing before the SiLU -- as the engine used
// to -- clipped the conv output into the *post*-SiLU calibrated range, which
// discarded the negative tail before SiLU ever saw it.
// ---------------------------------------------------------------------------

// Convolve and requantize to (out_scale, out_zp).
// If `residual` is non-null its dequantized value is added after the activation,
// which is how a Bottleneck's skip connection stays exact across differing
// scales -- the add happens in the real domain, then quantizes once.
std::unique_ptr<Tensor> conv2d_quant(const Tensor& input,
                                     const Layer& layer,
                                     float out_scale,
                                     int out_zp,
                                     bool apply_silu,
                                     const Tensor* residual = nullptr);

// Convolve and keep the result in FP32 (Detect head predictors).
FloatTensor conv2d_float(const Tensor& input, const Layer& layer, bool apply_silu);

// ---------------------------------------------------------------------------
// Tensor plumbing
// ---------------------------------------------------------------------------

// Re-express a tensor on a different scale/zero-point. A no-op when the
// parameters already match.
std::unique_ptr<Tensor> requantize(const Tensor& input, float scale, int zero_point);

// Concatenate along channels, requantizing every input to (scale, zero_point)
// first. Concatenating raw bytes from tensors on different scales was the
// dominant source of drift in the neck.
std::unique_ptr<Tensor> concat(const std::vector<const Tensor*>& tensors,
                               float scale,
                               int zero_point);

std::unique_ptr<Tensor> upsample2x(const Tensor& input);
std::unique_ptr<Tensor> maxpool2d(const Tensor& input, int kernel_size);

// ---------------------------------------------------------------------------
// Compound blocks
// ---------------------------------------------------------------------------
std::unique_ptr<Tensor> c2f_block(const Tensor& input, const Model& model, const ArchLayer& arch);
std::unique_ptr<Tensor> sppf_block(const Tensor& input, const Model& model, const ArchLayer& arch);

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

// Runs layers 0..21 and returns each layer's INT8 output. Layer 22 (Detect) is
// left empty; use detect_head() on the result.
std::vector<std::unique_ptr<Tensor>> run_backbone(const Model& model, const Tensor& input);

// Decodes the Detect head into boxes: DFL expectation -> distance-to-box ->
// class sigmoid -> per-class NMS. Boxes are in 640x640 input pixels.
std::vector<Detection> detect_head(const Model& model,
                                   const std::vector<std::unique_ptr<Tensor>>& features,
                                   float conf_threshold,
                                   float iou_threshold);

// Convenience: backbone + head.
std::vector<Detection> run_inference(const Model& model,
                                     const Tensor& input,
                                     float conf_threshold,
                                     float iou_threshold);

#endif
