#include "model.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

Model load_model_metadata(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + json_path);
    }

    json metadata;
    file >> metadata;

    Model model;
    model.num_bits = metadata["num_bits"];

    if (metadata.contains("input")) {
        model.input_scale = metadata["input"]["scale"];
        model.input_zero_point = metadata["input"]["zero_point"];
    }

    for (const auto& j : metadata["conv_layers"]) {
        Layer layer;
        layer.layer_id = j["layer_id"];
        layer.path = j["path"];
        layer.module_path = j.value("module_path", layer.path);
        layer.weight_shape = j["weight_shape"].get<std::vector<int>>();
        layer.byte_offset = j["byte_offset"];
        layer.byte_length = j["byte_length"];
        const int out_ch = layer.weight_shape[0];
        if (j["weight_scale"].is_array()) {
            layer.weight_scales = j["weight_scale"].get<std::vector<float>>();
            if (static_cast<int>(layer.weight_scales.size()) != out_ch) {
                throw std::runtime_error("Per-channel weight scale count mismatch for " +
                                         layer.path);
            }
        } else {
            layer.weight_scales.assign(out_ch, j["weight_scale"].get<float>());
        }
        layer.weight_zero_point = j["weight_zero_point"];
        layer.stride = j.value("stride", 1);
        layer.padding = j.value("padding", 0);
        layer.groups = j.value("groups", 1);
        layer.bn_gain = j["bn_gain"].get<std::vector<float>>();
        layer.bn_bias = j["bn_bias"].get<std::vector<float>>();
        layer.silu = (j.value("act", std::string("none")) == "silu");

        if (j.contains("out_scale") && !j["out_scale"].is_null()) {
            layer.has_out_scale = true;
            layer.out_scale = j["out_scale"];
            layer.out_zero_point = j["out_zero_point"];
        }

        if (static_cast<int>(layer.bn_gain.size()) != layer.out_channels() ||
            static_cast<int>(layer.bn_bias.size()) != layer.out_channels()) {
            throw std::runtime_error("BN parameter count does not match out_channels for " +
                                     layer.path);
        }

        model.conv_layers.push_back(std::move(layer));
    }

    for (const auto& j : metadata["architecture"]) {
        ArchLayer a;
        a.layer_id = j["layer_id"];
        a.type = j["type"];

        const auto& from = j["input_from"];
        if (from.is_string()) {
            a.input_is_raw = true;
        } else if (from.is_array()) {
            a.inputs = from.get<std::vector<int>>();
        } else {
            a.inputs.push_back(from.get<int>());
        }

        if (a.type == "Conv") {
            a.conv = j["conv"];
        } else if (a.type == "C2f") {
            a.cv1 = j["cv1"];
            a.cv2 = j["cv2"];
            a.concat_scale = j["concat_scale"];
            a.concat_zero_point = j["concat_zero_point"];
            for (const auto& b : j["bottlenecks"]) {
                BottleneckSpec spec;
                spec.cv1 = b["cv1"];
                spec.cv2 = b["cv2"];
                spec.shortcut = b["shortcut"];
                a.bottlenecks.push_back(spec);
            }
        } else if (a.type == "SPPF") {
            a.cv1 = j["cv1"];
            a.cv2 = j["cv2"];
            a.k = j["k"];
        } else if (a.type == "Concat") {
            a.out_scale = j["out_scale"];
            a.out_zero_point = j["out_zero_point"];
        } else if (a.type == "Detect") {
            a.nc = j["nc"];
            a.reg_max = j["reg_max"];
            a.no = j["no"];
            a.strides = j["strides"].get<std::vector<float>>();
            a.det_cv2 = j["cv2"].get<std::vector<std::vector<int>>>();
            a.det_cv3 = j["cv3"].get<std::vector<std::vector<int>>>();
            a.dfl = j["dfl"];
        }

        model.architecture.push_back(std::move(a));
    }

    return model;
}

void load_model_weights(Model& model, const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + bin_path);
    }

    for (auto& layer : model.conv_layers) {
        layer.weights = std::make_unique<Tensor>(layer.weight_shape, layer.weight_scales[0], 0);
        file.seekg(static_cast<std::streamoff>(layer.byte_offset));
        layer.weights->load_from_stream(file, layer.byte_length);
    }

    std::cout << "Loaded weights for " << model.conv_layers.size() << " conv layers"
              << std::endl;
}
