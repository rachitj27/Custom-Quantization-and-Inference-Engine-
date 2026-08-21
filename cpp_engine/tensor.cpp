#include "tensor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape_in, float scale_in, int zero_point_in)
    : shape(shape_in), scale(scale_in), zero_point(zero_point_in) {
    num_elements = 1;
    for (int dim : shape) {
        num_elements *= static_cast<size_t>(dim);
    }
    data = new int8_t[num_elements];
}

Tensor::~Tensor() {
    delete[] data;
}

void Tensor::print_info() const {
    std::cout << "Tensor shape: (";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << ")  elements: " << num_elements
              << "  scale: " << scale << "  zp: " << zero_point << std::endl;

    std::cout << "First values: ";
    size_t n = std::min(num_elements, static_cast<size_t>(8));
    for (size_t i = 0; i < n; i++) {
        std::cout << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::endl;
}

int8_t Tensor::at(size_t index) const {
    return data[index];
}

void Tensor::load_from_stream(std::ifstream& file, size_t byte_length) {
    if (byte_length != num_elements) {
        throw std::runtime_error("Byte length does not match tensor size");
    }
    file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(byte_length));
    if (!file) {
        throw std::runtime_error("Failed to read from file");
    }
}
