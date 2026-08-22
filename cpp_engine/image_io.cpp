#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

void fill_rect(Image& img, int x0, int y0, int x1, int y1, const unsigned char* colour) {
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) set_pixel(img, x, y, colour);
    }
}

// A 5x7 bitmap font, one byte per column, bit 0 is the top row. Covers ASCII 32
// to 90, which is space through 'Z'. Lowercase is drawn as uppercase, which is
// all the labels need.
constexpr int kGlyphW = 5;
constexpr int kGlyphH = 7;
const unsigned char kFont[59][5] = {
    {0x00,0x00,0x00,0x00,0x00}, // space
    {0x00,0x00,0x5F,0x00,0x00}, // !
    {0x00,0x07,0x00,0x07,0x00}, // "
    {0x14,0x7F,0x14,0x7F,0x14}, // #
    {0x24,0x2A,0x7F,0x2A,0x12}, // $
    {0x23,0x13,0x08,0x64,0x62}, // %
    {0x36,0x49,0x55,0x22,0x50}, // &
    {0x00,0x05,0x03,0x00,0x00}, // '
    {0x00,0x1C,0x22,0x41,0x00}, // (
    {0x00,0x41,0x22,0x1C,0x00}, // )
    {0x14,0x08,0x3E,0x08,0x14}, // *
    {0x08,0x08,0x3E,0x08,0x08}, // +
    {0x00,0x50,0x30,0x00,0x00}, // ,
    {0x08,0x08,0x08,0x08,0x08}, // -
    {0x00,0x60,0x60,0x00,0x00}, // .
    {0x20,0x10,0x08,0x04,0x02}, // /
    {0x3E,0x51,0x49,0x45,0x3E}, // 0
    {0x00,0x42,0x7F,0x40,0x00}, // 1
    {0x42,0x61,0x51,0x49,0x46}, // 2
    {0x21,0x41,0x45,0x4B,0x31}, // 3
    {0x18,0x14,0x12,0x7F,0x10}, // 4
    {0x27,0x45,0x45,0x45,0x39}, // 5
    {0x3C,0x4A,0x49,0x49,0x30}, // 6
    {0x01,0x71,0x09,0x05,0x03}, // 7
    {0x36,0x49,0x49,0x49,0x36}, // 8
    {0x06,0x49,0x49,0x29,0x1E}, // 9
    {0x00,0x36,0x36,0x00,0x00}, // :
    {0x00,0x56,0x36,0x00,0x00}, // ;
    {0x08,0x14,0x22,0x41,0x00}, // <
    {0x14,0x14,0x14,0x14,0x14}, // =
    {0x00,0x41,0x22,0x14,0x08}, // >
    {0x02,0x01,0x51,0x09,0x06}, // ?
    {0x32,0x49,0x79,0x41,0x3E}, // @
    {0x7E,0x11,0x11,0x11,0x7E}, // A
    {0x7F,0x49,0x49,0x49,0x36}, // B
    {0x3E,0x41,0x41,0x41,0x22}, // C
    {0x7F,0x41,0x41,0x22,0x1C}, // D
    {0x7F,0x49,0x49,0x49,0x41}, // E
    {0x7F,0x09,0x09,0x09,0x01}, // F
    {0x3E,0x41,0x49,0x49,0x7A}, // G
    {0x7F,0x08,0x08,0x08,0x7F}, // H
    {0x00,0x41,0x7F,0x41,0x00}, // I
    {0x20,0x40,0x41,0x3F,0x01}, // J
    {0x7F,0x08,0x14,0x22,0x41}, // K
    {0x7F,0x40,0x40,0x40,0x40}, // L
    {0x7F,0x02,0x0C,0x02,0x7F}, // M
    {0x7F,0x04,0x08,0x10,0x7F}, // N
    {0x3E,0x41,0x41,0x41,0x3E}, // O
    {0x7F,0x09,0x09,0x09,0x06}, // P
    {0x3E,0x41,0x51,0x21,0x5E}, // Q
    {0x7F,0x09,0x19,0x29,0x46}, // R
    {0x46,0x49,0x49,0x49,0x31}, // S
    {0x01,0x01,0x7F,0x01,0x01}, // T
    {0x3F,0x40,0x40,0x40,0x3F}, // U
    {0x1F,0x20,0x40,0x20,0x1F}, // V
    {0x3F,0x40,0x38,0x40,0x3F}, // W
    {0x63,0x14,0x08,0x14,0x63}, // X
    {0x07,0x08,0x70,0x08,0x07}, // Y
    {0x61,0x51,0x49,0x45,0x43}, // Z
};

int text_width(const std::string& s, int scale) {
    if (s.empty()) return 0;
    return static_cast<int>(s.size()) * (kGlyphW + 1) * scale - scale;
}

void draw_text(Image& img, int x, int y, const std::string& s, int scale,
               const unsigned char* colour) {
    int cx = x;
    for (char raw : s) {
        char ch = raw;
        if (ch >= 'a' && ch <= 'z') ch -= 32;  // the font is uppercase only
        const int idx = ch - 32;
        if (idx >= 0 && idx < static_cast<int>(sizeof(kFont) / sizeof(kFont[0]))) {
            for (int col = 0; col < kGlyphW; col++) {
                const unsigned char bits = kFont[idx][col];
                for (int row = 0; row < kGlyphH; row++) {
                    if (bits & (1u << row)) {
                        fill_rect(img, cx + col * scale, y + row * scale,
                                  cx + (col + 1) * scale, y + (row + 1) * scale, colour);
                    }
                }
            }
        }
        cx += (kGlyphW + 1) * scale;
    }
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
    // Boxes come back in network pixels; map them onto the original image.
    const float sx = static_cast<float>(img.width) / static_cast<float>(net_size);
    const float sy = static_cast<float>(img.height) / static_cast<float>(net_size);

    // Scale the outline and the label with the image so both stay readable.
    const int shorter = std::min(img.width, img.height);
    const int thickness = std::max(2, shorter / 200);
    const int scale = std::max(2, shorter / 180);

    const unsigned char white[3] = {255, 255, 255};
    const unsigned char black[3] = {0, 0, 0};

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

        // Label reads "FIRE 83%".
        std::string name = (d.cls < static_cast<int>(class_names.size()))
                               ? class_names[d.cls]
                               : ("CLASS" + std::to_string(d.cls));
        const int pct = static_cast<int>(std::lround(std::clamp(d.score, 0.0f, 1.0f) * 100.0f));
        const std::string label = name + " " + std::to_string(pct) + "%";

        const int pad = std::max(2, scale);
        const int box_w = text_width(label, scale) + pad * 2;
        const int box_h = kGlyphH * scale + pad * 2;

        // Sit the label above the box, or just inside it when there is no room.
        int lx = x1;
        int ly = y1 - box_h;
        if (ly < 0) ly = y1 + thickness;
        lx = std::clamp(lx, 0, std::max(0, img.width - box_w));

        fill_rect(img, lx, ly, lx + box_w, ly + box_h, colour);

        // Dark text on light fills, light text on dark ones, so it stays legible
        // whichever colour the class happens to be.
        const int luma = (colour[0] * 299 + colour[1] * 587 + colour[2] * 114) / 1000;
        draw_text(img, lx + pad, ly + pad, label, scale, luma > 140 ? black : white);
    }
}
