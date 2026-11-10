#pragma once

#include "core/common.h"
#include "core/tensor.h"

#include <map>
#include <string>
#include <vector>

namespace gai {

// A ViT image encoder that feeds the language model.
//
// This is the "body" half of a multimodal model: the training side (Python)
// trains a vision tower plus a projector, exports them next to the language
// model, and the engine runs them here so an image can be part of a prompt.
//
// The tower follows the usual SigLIP / CLIP layout:
//
//   patch_embed   [dim, patch*patch*3]      (a matmul over the patches)
//   pos_embed     [n_patches, dim]
//   layers.N      norm1, qkv (q,k,v), proj, norm2, fc1, fc2
//   post_ln       [dim]
//   projector     fc1 [out, dim], fc2 [out, out_hidden]   (0 or 2 layers)
//
// The weights are loaded from a GGUF file, so the same tokenizer/vision
// pair travels with the language model.
//
// Every tensor is shape checked when the tower is built: a checkpoint whose
// weights do not match the declared config is rejected instead of reading
// out of bounds during the forward pass.

struct VisionConfig {
    int image_size = 336;
    int patch_size = 14;
    int dim = 1024;          // the tower's hidden size
    int layers = 24;
    int heads = 16;
    int mlp_ratio = 4;
    int out_dim = 2048;      // the language model's hidden size
    int proj_hidden = 0;     // 0 = a single linear projector layer
    int channels = 3;

    int patches_per_side() const {
        return image_size / patch_size;
    }
    int n_patches() const { return patches_per_side() * patches_per_side(); }
    int patch_dim() const { return patch_size * patch_size * channels; }
    int head_dim() const { return dim / heads; }
    int mlp_hidden() const { return dim * mlp_ratio; }
};

class VisionTower {
public:
    VisionTower() = default;
    VisionTower(VisionConfig cfg, const std::map<std::string, Tensor>& weights);

    bool valid() const { return valid_; }
    const VisionConfig& config() const { return cfg_; }

    // The names the loader looks for, in order.
    static std::vector<std::string> weight_names(const VisionConfig& cfg);

    // Encodes a normalized CHW image into `out` (n_patches * out_dim floats).
    // pixels must already be resized and normalized.
    void encode(const float* pixels, std::vector<float>& out) const;

    // Encodes and returns only the patch features (no projector).
    void encode_patches(const float* pixels, std::vector<float>& out) const;

private:
    VisionConfig cfg_;
    bool valid_ = false;

    Tensor patch_embed_;   // [dim, patch_dim]
    Tensor pos_embed_;     // [n_patches, dim]
    Tensor post_ln_;       // [dim]
    std::vector<Tensor> norm1_, norm2_, qkv_, proj_, fc1_, fc2_;
    std::vector<Tensor> norm1_b_, norm2_b_, qkv_b_, proj_b_, fc1_b_, fc2_b_;
    Tensor proj_w_;        // projector
    Tensor proj_bias_;
    Tensor proj2_w_;       // optional second projector layer
    Tensor proj2_bias_;
};

// Reads a binary PPM/PGM image (P6/P5) and returns normalized CHW floats.
// The Python side converts PNG/JPEG to this format, so the engine needs no
// image decoder.
bool load_ppm(const std::string& path, float* out, int C, int H, int W);

}  // namespace gai
