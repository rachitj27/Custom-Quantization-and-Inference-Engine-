#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <fstream>
#include <vector>

// A quantized INT8 tensor in CHW layout.
//
// A stored value q represents the real number  scale * (q - zero_point).
//
// The quantization parameters live on the tensor itself. That is deliberate:
// previously they were passed around as loose function arguments, which made it
// easy for an op to combine two tensors that were on different scales without
// anything catching it. Carrying them here means every op can see the scale of
// what it was handed.
class Tensor {
public:
    std::vector<int> shape;
    int8_t* data;
    size_t num_elements;

    float scale;
    int zero_point;

    explicit Tensor(const std::vector<int>& shape, float scale = 1.0f, int zero_point = 0);
    ~Tensor();

    // Owning raw buffer: copying would double-free.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void print_info() const;
    int8_t at(size_t index) const;
    void load_from_stream(std::ifstream& file, size_t byte_length);

    // Dequantized value at a flat index.
    float real_at(size_t i) const {
        return scale * static_cast<float>(static_cast<int>(data[i]) - zero_point);
    }

    int channels() const { return shape[0]; }
    int height() const { return shape[1]; }
    int width() const { return shape[2]; }
};

// A plain FP32 tensor in CHW layout. Used only for the Detect head, where the
// engine leaves the INT8 domain to decode boxes.
struct FloatTensor {
    std::vector<int> shape;
    std::vector<float> data;

    FloatTensor() = default;

    explicit FloatTensor(const std::vector<int>& s) : shape(s) {
        size_t n = 1;
        for (int d : s) n *= static_cast<size_t>(d);
        data.assign(n, 0.0f);
    }

    int channels() const { return shape[0]; }
    int height() const { return shape[1]; }
    int width() const { return shape[2]; }
};

#endif
