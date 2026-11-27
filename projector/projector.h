#pragma once

#include "core/common.h"
#include "core/tensor.h"
#include "core/device.h"
#include "projector/image_video_loader.h"

#include <map>
#include <string>
#include <vector>
#include <memory>

#if defined(_MSC_VER) || defined(__MINGW32__)
#  include <malloc.h>
#  ifndef gai_aligned_free
#    define gai_aligned_free(ptr) _aligned_free(ptr)
#  endif
#else
#  ifndef gai_aligned_free
#    define gai_aligned_free(ptr) free(ptr)
#  endif
#endif

namespace gai {

struct VisionConfig {
    int image_size  = 336;
    int patch_size  = 14;
    int dim         = 1024;
    int layers      = 24;
    int heads       = 16;
    int mlp_ratio   = 4;
    int out_dim     = 1536;
    int proj_hidden = 2048;
    int channels    = 3;
    int max_frames  = 32;
    bool spatial_pool = true;

    int patches_per_side() const { return image_size / patch_size; }
    int n_patches() const { return patches_per_side() * patches_per_side(); }
    int pooled_patches() const {
        return spatial_pool ? (patches_per_side() / 2) * (patches_per_side() / 2) : n_patches();
    }
    int patch_dim() const { return patch_size * patch_size * channels; }
    int head_dim() const { return dim / heads; }
    int mlp_hidden() const { return dim * mlp_ratio; }
};

struct ProjectorGradients {
    Tensor d_w1;
    Tensor d_b1;
    Tensor d_w2;
    Tensor d_b2;
};

struct ProjectorAdamWState {
    Tensor m_w1, v_w1;
    Tensor m_b1, v_b1;
    Tensor m_w2, v_w2;
    Tensor m_b2, v_b2;
    int step = 0;
};

class alignas(64) VisionWorkspace {
public:
    explicit VisionWorkspace(size_t capacity_bytes = (64ULL << 20));
    ~VisionWorkspace();

    float* alloc(size_t num_floats);
    void reset() {
        for (float* p : overflow_) gai_aligned_free(p);
        overflow_.clear();
        offset_ = 0;
    }

private:
    float* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    std::vector<float*> overflow_;
};

class VisionTower {
public:
    VisionTower() = default;
    VisionTower(VisionConfig cfg, const std::map<std::string, Tensor>& weights, Device dev = Device::CPU);

    bool valid() const { return valid_; }
    const VisionConfig& config() const { return cfg_; }
    Device device() const { return dev_; }

    static std::vector<std::string> weight_names(const VisionConfig& cfg);

    void to(Device dev);

    void encode(const float* pixels, std::vector<float>& out) const;

    void encode_patches(const float* pixels, std::vector<float>& out) const;

    void encode_video(const std::vector<const float*>& frames_chw, std::vector<float>& out) const;

    void encode_patches_tensor(const float* pixels, Tensor& out) const;
    void encode_pooled_features(const float* pixels, Tensor& out_pooled) const;
    void encode_video_pooled_features(const std::vector<const float*>& frames_chw, Tensor& out_pooled) const;
    void encode_tensor(const float* pixels, Tensor& out) const;
    void encode_video_tensor(const std::vector<const float*>& frames_chw, Tensor& out) const;

    bool encode_image_file(const std::string& path, std::vector<float>& out) const;

    bool encode_video_file(const std::string& path, int max_frames, std::vector<float>& out) const;

    void forward_projector(Device dev, const float* in_features, float* out_tokens, int num_tokens) const;

    void projector_forward_train(const float* in_features, int num_tokens,
                                 float* out_tokens, float* saved_h1, float* saved_act) const;
    void projector_forward_train_dev(const float* in_features, int num_tokens,
                                     float* out_tokens, float* saved_h1, float* saved_act) const;

    void projector_backward(const float* in_features, const float* saved_h1, const float* saved_act,
                            const float* d_out, int num_tokens, ProjectorGradients& grads) const;
    void projector_backward_dev(const float* in_features, const float* saved_h1, const float* saved_act,
                                const float* d_out, int num_tokens, ProjectorGradients& grads) const;

    void projector_adamw_step(ProjectorGradients& grads, ProjectorAdamWState& opt,
                              float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                              float eps = 1e-8f, float weight_decay = 0.01f);

    bool save_projector_checkpoint(const std::string& path) const;
    bool load_projector_checkpoint(const std::string& path);

private:
    void apply_spatial_pooling(const float* in_patches, float* out_pooled, int num_patches) const;

    VisionConfig cfg_;
    Device dev_ = Device::CPU;
    bool valid_ = false;

    Tensor patch_embed_;
    Tensor pos_embed_;
    Tensor time_embed_;
    Tensor post_ln_;
    Tensor post_ln_b_;

    std::vector<Tensor> norm1_, norm2_, qkv_, proj_, fc1_, fc2_;
    std::vector<Tensor> norm1_b_, norm2_b_, qkv_b_, proj_b_, fc1_b_, fc2_b_;

    Tensor proj_w_;
    Tensor proj_bias_;
    Tensor proj2_w_;
    Tensor proj2_bias_;

    mutable VisionWorkspace workspace_;
};

std::unique_ptr<VisionTower> load_vision_from_gguf(const std::string& path, Device dev);

}
