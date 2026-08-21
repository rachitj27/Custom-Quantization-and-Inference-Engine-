#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

namespace {

inline unsigned char clamp_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return static_cast<unsigned char>(std::lround(v));
}

// Colour per class, cycled if there are more classes than entries.
const unsigned char kPalette[][3] = {
    {255, 56, 56},    // fire  - red
    {56, 168, 255},   // smoke - blue
    {66, 217, 110},
    {255, 178, 29},
    {170, 106, 255},
};
constexpr int kPaletteSize = sizeof(kPalette) / sizeof(kPalette[0]);

void set_pixel(Image& img, int x, int y, const unsigned char* colour) {
    if (x < 0 || x >= img.width || y < 0 || y >= img.height) return;
    const size_t i = (static_cast<size_t>(y) * img.width + x) * 3;
    img.rgb[i + 0] = colour[0];
    img.rgb[i + 1] = colour[1];
    img.rgb[i + 2] = colour[2];
}

}  // namespace

Image load_image(const std::string& path) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (pixels == nullptr) {
        throw std::runtime_error("Could not read image '" + path + "': " +
                                 std::string(stbi_failure_reason() ? stbi_failure_reason()
                                                                  : "unknown"));
    }

    Image img;
    img.width = w;
    img.height = h;
    img.rgb.assign(pixels, pixels + static_cast<size_t>(w) * h * 3);
    stbi_image_free(pixels);
    return img;
}

void save_jpeg(const std::string& path, const Image& img, int quality) {
    const int ok = stbi_write_jpg(path.c_str(), img.width, img.height, 3,
                                  img.rgb.data(), quality);
    if (!ok) {
        throw std::runtime_error("Could not write image '" + path + "'");
    }
}

std::unique_ptr<Tensor> preprocess(const Image& img, int size, const Model& model) {
    auto out = std::make_unique<Tensor>(std::vector<int>{3, size, size},
                                        model.input_scale, model.input_zero_point);

    const float scale_x = static_cast<float>(img.width) / static_cast<float>(size);
    const float scale_y = static_cast<float>(img.height) / static_cast<float>(size);
    const size_t plane = static_cast<size_t>(size) * size;

    for (int y = 0; y < size; y++) {
        float sy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        float fy = sy - static_cast<float>(y0);
        int y1 = y0 + 1;
        y0 = std::clamp(y0, 0, img.height - 1);
        y1 = std::clamp(y1, 0, img.height - 1);

        for (int x = 0; x < size; x++) {
            float sx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            float fx = sx - static_cast<float>(x0);
            int x1 = x0 + 1;
            x0 = std::clamp(x0, 0, img.width - 1);
            x1 = std::clamp(x1, 0, img.width - 1);

            for (int c = 0; c < 3; c++) {
                const float p00 = img.rgb[(static_cast<size_t>(y0) * img.width + x0) * 3 + c];
                const float p01 = img.rgb[(static_cast<size_t>(y0) * img.width + x1) * 3 + c];
                const float p10 = img.rgb[(static_cast<size_t>(y1) * img.width + x0) * 3 + c];
                const float p11 = img.rgb[(static_cast<size_t>(y1) * img.width + x1) * 3 + c];

                const float top = p00 + (p01 - p00) * fx;
                const float bottom = p10 + (p11 - p10) * fx;
                const float value = top + (bottom - top) * fy;

                // Pixels are normalised to [0,1] before quantization, matching
                // the preprocessing used during calibration.
                const float real = value / 255.0f;
                int32_t q = static_cast<int32_t>(std::lround(real / model.input_scale)) +
                            model.input_zero_point;
                q = std::clamp(q, -128, 127);
                out->data[static_cast<size_t>(c) * plane + static_cast<size_t>(y) * size + x] =
                    static_cast<int8_t>(q);
            }
        }
    }

    return out;
}

void draw_detections(Image& img, const std::vector<Detection>& dets, int net_size,
                     const std::vector<std::string>& class_names) {
    (void)class_names;  // labels are reported on stdout; the box is colour-coded

    // Boxes come back in network pixels; map them onto the original image.
    const float sx = static_cast<float>(img.width) / static_cast<float>(net_size);
    const float sy = static_cast<float>(img.height) / static_cast<float>(net_size);

    // Thicker outline on bigger images so it stays visible.
    const int thickness = std::max(2, std::min(img.width, img.height) / 250);

    for (const Detection& d : dets) {
        const unsigned char* colour = kPalette[d.cls % kPaletteSize];

        int x1 = static_cast<int>(std::lround(d.x1 * sx));
        int y1 = static_cast<int>(std::lround(d.y1 * sy));
        int x2 = static_cast<int>(std::lround(d.x2 * sx));
        int y2 = static_cast<int>(std::lround(d.y2 * sy));

        x1 = std::clamp(x1, 0, img.width - 1);
        y1 = std::clamp(y1, 0, img.height - 1);
        x2 = std::clamp(x2, 0, img.width - 1);
        y2 = std::clamp(y2, 0, img.height - 1);
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);

        for (int t = 0; t < thickness; t++) {
            for (int x = x1; x <= x2; x++) {
                set_pixel(img, x, y1 + t, colour);
                set_pixel(img, x, y2 - t, colour);
            }
            for (int y = y1; y <= y2; y++) {
                set_pixel(img, x1 + t, y, colour);
                set_pixel(img, x2 - t, y, colour);
            }
        }

        // Solid tag in the top-left corner of the box: its length scales with
        // confidence, so a glance shows how sure the engine is.
        const int tag_h = std::max(4, thickness * 3);
        const int tag_w = static_cast<int>((x2 - x1) * std::clamp(d.score, 0.0f, 1.0f));
        for (int y = y1; y < y1 + tag_h; y++) {
            for (int x = x1; x < x1 + tag_w; x++) {
                set_pixel(img, x, y, colour);
            }
        }
    }
}
