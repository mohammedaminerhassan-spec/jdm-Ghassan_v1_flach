#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace gai {

struct ImageBuffer {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<float> chw_data;
    bool valid = false;
};

struct VideoBuffer {
    int num_frames = 0;
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<std::vector<float>> frames;
    bool valid = false;
};

ImageBuffer load_image_any(const std::string& path, int target_w = 336, int target_h = 336,
                           bool keep_aspect_ratio = true);

ImageBuffer load_image_from_memory(const uint8_t* data, size_t len, int target_w = 336, int target_h = 336,
                                   bool keep_aspect_ratio = true);

bool resize_bilinear_rgb(const uint8_t* src, int src_w, int src_h,
                         float* dst_chw, int dst_w, int dst_h,
                         bool keep_aspect_ratio = true);

VideoBuffer load_video_any(const std::string& path, int max_frames = 16, int target_w = 336, int target_h = 336,
                           bool keep_aspect_ratio = true);

bool load_ppm_raw(const std::string& path, float* out, int C, int H, int W);

inline bool load_ppm(const std::string& path, float* out, int C, int H, int W) {
    return load_ppm_raw(path, out, C, H, W);
}

}
