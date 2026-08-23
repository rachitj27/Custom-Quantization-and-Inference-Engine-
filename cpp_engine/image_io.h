#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>
#include <vector>

#include "model.h"
#include "tensor.h"

// An 8-bit RGB image, HWC interleaved.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;  // size = width * height * 3
};

// Decode a JPEG/PNG/BMP into RGB. Throws on failure.
Image load_image(const std::string& path);

// Write RGB out as a JPEG. Throws on failure.
void save_jpeg(const std::string& path, const Image& img, int quality = 92);

// Resize to `size` x `size` with bilinear interpolation, then quantize to INT8
// using the model's input scale/zero-point.
//
// The sampling convention matches cv2.resize(..., INTER_LINEAR) -- source
// coordinate (d + 0.5) * scale - 0.5 -- because that is what calibrate.py used
// to derive the activation ranges.
std::unique_ptr<Tensor> preprocess(const Image& img, int size, const Model& model);

// Draw detection boxes in place. Colour is chosen per class.
void draw_detections(Image& img,
                     const std::vector<Detection>& dets,
                     int net_size,
                     const std::vector<std::string>& class_names);

#endif
