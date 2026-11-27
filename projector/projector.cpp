#include "projector/projector.h"
#include "projector/image_video_loader.h"
#include "core/ops.h"
#include "core/device.h"
#include "format/gguf_codec.h"

#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

#ifdef GAI_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#define gai_aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define gai_aligned_free(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
static inline void* gai_aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
}
#define gai_aligned_free(ptr) free(ptr)
#endif

namespace gai {

namespace {

void linear_bias(Device dev, const float* x, const float* w, const float* b,
                 float* y, int M, int K, int N) {
    ops::linear_forward(dev, x, w, y, M, K, N);
    if (b) ops::add_bias(dev, b, y, M, N);
}

bool has_shape(const Tensor& t, i64 n) {
    return t.defined() && t.numel() == n;
}

}

VisionWorkspace::VisionWorkspace(size_t capacity_bytes) {
    capacity_ = capacity_bytes / sizeof(float);
    buffer_ = static_cast<float*>(gai_aligned_alloc(64, capacity_ * sizeof(float)));
    offset_ = 0;
}

VisionWorkspace::~VisionWorkspace() {
    for (float* p : overflow_) {
        if (p) gai_aligned_free(p);
    }
    overflow_.clear();
    if (buffer_) {
        gai_aligned_free(buffer_);
        buffer_ = nullptr;
    }
}

float* VisionWorkspace::alloc(size_t num_floats) {
    const size_t aligned_floats = (num_floats + 15) & ~15;
    if (offset_ + aligned_floats > capacity_) {
        float* p = static_cast<float*>(gai_aligned_alloc(64, num_floats * sizeof(float)));
        if (p) overflow_.push_back(p);
        return p;
    }
    float* ptr = buffer_ + offset_;
    offset_ += aligned_floats;
    return ptr;
}

std::vector<std::string> VisionTower::weight_names(const VisionConfig& cfg) {
    std::vector<std::string> names;
    names.push_back("vision.patch_embed.weight");
    names.push_back("vision.pos_embed.weight");
    names.push_back("vision.time_embed.weight");
    names.push_back("vision.post_ln.weight");
    names.push_back("vision.post_ln.bias");
    for (int i = 0; i < cfg.layers; ++i) {
        const std::string p = "vision.layers." + std::to_string(i) + ".";
        names.push_back(p + "norm1.weight");
        names.push_back(p + "norm1.bias");
        names.push_back(p + "qkv.weight");
        names.push_back(p + "qkv.bias");
        names.push_back(p + "proj.weight");
        names.push_back(p + "proj.bias");
        names.push_back(p + "norm2.weight");
        names.push_back(p + "norm2.bias");
        names.push_back(p + "fc1.weight");
        names.push_back(p + "fc1.bias");
        names.push_back(p + "fc2.weight");
        names.push_back(p + "fc2.bias");
    }
    names.push_back("vision.projector.weight");
    names.push_back("vision.projector.bias");
    if (cfg.proj_hidden > 0) {
        names.push_back("vision.projector.2.weight");
        names.push_back("vision.projector.2.bias");
    }
    return names;
}

VisionTower::VisionTower(VisionConfig cfg, const std::map<std::string, Tensor>& weights, Device dev)
    : cfg_(cfg), dev_(Device::CPU), workspace_(64ULL << 20) {
    (void)dev;

    auto get = [&](const std::string& n, Tensor* dst) {
        auto it = weights.find(n);
        if (it == weights.end()) return false;
        if (!it->second.defined()) return false;
        *dst = it->second;
        return true;
    };

    if (!get("vision.patch_embed.weight", &patch_embed_)) return;
    if (!get("vision.pos_embed.weight", &pos_embed_)) return;
    get("vision.time_embed.weight", &time_embed_);
    if (!get("vision.post_ln.weight", &post_ln_)) return;
    get("vision.post_ln.bias", &post_ln_b_);

    norm1_.resize(static_cast<size_t>(cfg_.layers));
    norm2_.resize(static_cast<size_t>(cfg_.layers));
    qkv_.resize(static_cast<size_t>(cfg_.layers));
    proj_.resize(static_cast<size_t>(cfg_.layers));
    fc1_.resize(static_cast<size_t>(cfg_.layers));
    fc2_.resize(static_cast<size_t>(cfg_.layers));
    norm1_b_.resize(static_cast<size_t>(cfg_.layers));
    norm2_b_.resize(static_cast<size_t>(cfg_.layers));
    qkv_b_.resize(static_cast<size_t>(cfg_.layers));
    proj_b_.resize(static_cast<size_t>(cfg_.layers));
    fc1_b_.resize(static_cast<size_t>(cfg_.layers));
    fc2_b_.resize(static_cast<size_t>(cfg_.layers));

    for (int i = 0; i < cfg_.layers; ++i) {
        const size_t k = static_cast<size_t>(i);
        const std::string p = "vision.layers." + std::to_string(i) + ".";
        const bool ok =
            get(p + "norm1.weight", &norm1_[k]) &&
            get(p + "norm1.bias", &norm1_b_[k]) &&
            get(p + "qkv.weight", &qkv_[k]) &&
            get(p + "qkv.bias", &qkv_b_[k]) &&
            get(p + "proj.weight", &proj_[k]) &&
            get(p + "proj.bias", &proj_b_[k]) &&
            get(p + "norm2.weight", &norm2_[k]) &&
            get(p + "norm2.bias", &norm2_b_[k]) &&
            get(p + "fc1.weight", &fc1_[k]) &&
            get(p + "fc1.bias", &fc1_b_[k]) &&
            get(p + "fc2.weight", &fc2_[k]) &&
            get(p + "fc2.bias", &fc2_b_[k]);
        if (!ok) return;
    }

    if (!get("vision.projector.weight", &proj_w_)) return;
    if (!get("vision.projector.bias", &proj_bias_)) return;
    if (cfg_.proj_hidden > 0) {
        if (!get("vision.projector.2.weight", &proj2_w_)) return;
        if (!get("vision.projector.2.bias", &proj2_bias_)) return;
    }

    const i64 D = cfg_.dim;
    const i64 P = cfg_.n_patches();
    const i64 hidden = cfg_.mlp_hidden();
    const i64 O = cfg_.out_dim;
    const i64 inter = (cfg_.proj_hidden > 0) ? cfg_.proj_hidden : O;

    if (!has_shape(patch_embed_, D * cfg_.patch_dim())) return;
    if (!has_shape(pos_embed_, P * D)) return;
    if (!has_shape(post_ln_, D)) return;
    if (!has_shape(proj_w_, inter * D)) return;
    if (!has_shape(proj_bias_, inter)) return;

    for (int i = 0; i < cfg_.layers; ++i) {
        const size_t k = static_cast<size_t>(i);
        if (!has_shape(norm1_[k], D) || !has_shape(norm1_b_[k], D)) return;
        if (!has_shape(qkv_[k], 3 * D * D) || !has_shape(qkv_b_[k], 3 * D)) return;
        if (!has_shape(proj_[k], D * D) || !has_shape(proj_b_[k], D)) return;
        if (!has_shape(norm2_[k], D) || !has_shape(norm2_b_[k], D)) return;
        if (!has_shape(fc1_[k], hidden * D) || !has_shape(fc1_b_[k], hidden)) return;
        if (!has_shape(fc2_[k], D * hidden) || !has_shape(fc2_b_[k], D)) return;
    }

    if (cfg_.proj_hidden > 0) {
        if (!has_shape(proj2_w_, O * inter)) return;
        if (!has_shape(proj2_bias_, O)) return;
    }

    valid_ = true;
    if (dev != Device::CPU) {
        to(dev);
    }
}

void VisionTower::to(Device dev) {
#ifndef GAI_CUDA
    if (dev == Device::CUDA) {
        log_warn("[vision] CUDA support not compiled into this binary; keeping vision tower on CPU");
        return;
    }
#endif
    if (dev_ == dev) return;
    dev_ = dev;

    auto move_t = [dev](Tensor& t) {
        if (t.defined() && t.device() != dev) t = t.to(dev);
    };
    move_t(patch_embed_);
    move_t(pos_embed_);
    move_t(time_embed_);
    move_t(post_ln_);
    move_t(post_ln_b_);
    for (size_t i = 0; i < norm1_.size(); ++i) {
        move_t(norm1_[i]);  move_t(norm1_b_[i]);
        move_t(qkv_[i]);    move_t(qkv_b_[i]);
        move_t(proj_[i]);   move_t(proj_b_[i]);
        move_t(norm2_[i]);  move_t(norm2_b_[i]);
        move_t(fc1_[i]);    move_t(fc1_b_[i]);
        move_t(fc2_[i]);    move_t(fc2_b_[i]);
    }
    move_t(proj_w_);
    move_t(proj_bias_);
    move_t(proj2_w_);
    move_t(proj2_bias_);
}

void VisionTower::apply_spatial_pooling(const float* in_patches, float* out_pooled, int num_patches) const {
    const int D = cfg_.dim;
    const int S = cfg_.patches_per_side();
    if (!cfg_.spatial_pool || S % 2 != 0 || num_patches != S * S) {
        device_copy(out_pooled, dev_, in_patches, dev_,
                    static_cast<size_t>(num_patches) * D * sizeof(float));
        return;
    }
    ops::vit_spatial_pool_2x2(dev_, in_patches, out_pooled, S, D);
}

void VisionTower::encode_patches_tensor(const float* pixels, Tensor& out) const {
    const int P      = cfg_.n_patches();
    const int D      = cfg_.dim;
    const int pd     = cfg_.patch_dim();
    const int ps     = cfg_.patch_size;
    const int C      = cfg_.channels;
    const int H      = cfg_.heads;
    const int hd     = cfg_.head_dim();
    const int hidden = cfg_.mlp_hidden();

    Tensor pixels_dev;
    const float* pix_ptr = pixels;
    if (dev_ == Device::CUDA) {
#ifdef GAI_CUDA
        if (!cuda::is_device_memory(pixels)) {
            const i64 pix_n = static_cast<i64>(C) * cfg_.image_size * cfg_.image_size;
            pixels_dev = Tensor::empty({pix_n}, DType::F32, Device::CUDA);
            device_copy(pixels_dev.f32(), Device::CUDA, pixels, Device::CPU,
                        static_cast<size_t>(pix_n) * sizeof(float));
            pix_ptr = pixels_dev.f32();
        }
#endif
    }

    Tensor patches_t  = Tensor::empty({static_cast<i64>(P), pd}, DType::F32, dev_);
    Tensor x_t        = Tensor::empty({static_cast<i64>(P), D}, DType::F32, dev_);
    Tensor h_t        = Tensor::empty({static_cast<i64>(P), D}, DType::F32, dev_);
    Tensor qkv_t      = Tensor::empty({static_cast<i64>(P), 3 * D}, DType::F32, dev_);
    Tensor ctx_t      = Tensor::empty({static_cast<i64>(P), D}, DType::F32, dev_);
    Tensor proj_out_t = Tensor::empty({static_cast<i64>(P), D}, DType::F32, dev_);
    Tensor mlp_t      = Tensor::empty({static_cast<i64>(P), hidden}, DType::F32, dev_);
    Tensor scores_t   = Tensor::empty({static_cast<i64>(H), P, P}, DType::F32, dev_);

    float* patches  = patches_t.f32();
    float* x        = x_t.f32();
    float* h        = h_t.f32();
    float* qkv      = qkv_t.f32();
    float* ctx      = ctx_t.f32();
    float* proj_out = proj_out_t.f32();
    float* mlp      = mlp_t.f32();
    float* scores   = scores_t.f32();

    ops::vit_extract_patches(dev_, pix_ptr, patches, cfg_.image_size, ps, C);
    linear_bias(dev_, patches, patch_embed_.f32(), nullptr, x, P, pd, D);
    ops::add_inplace(dev_, x, pos_embed_.f32(), static_cast<i64>(P) * D);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int l = 0; l < cfg_.layers; ++l) {
        const size_t k = static_cast<size_t>(l);
        ops::layernorm_forward(dev_, x, norm1_[k].f32(), norm1_b_[k].f32(), h, P, D, 1e-5f);
        linear_bias(dev_, h, qkv_[k].f32(), qkv_b_[k].f32(), qkv, P, D, 3 * D);
        ops::vit_attention_forward(dev_, qkv, scores, ctx, P, H, hd, scale);
        linear_bias(dev_, ctx, proj_[k].f32(), proj_b_[k].f32(), proj_out, P, D, D);
        ops::add_inplace(dev_, x, proj_out, static_cast<i64>(P) * D);

        ops::layernorm_forward(dev_, x, norm2_[k].f32(), norm2_b_[k].f32(), h, P, D, 1e-5f);
        linear_bias(dev_, h, fc1_[k].f32(), fc1_b_[k].f32(), mlp, P, D, hidden);
        ops::gelu_forward(dev_, mlp, mlp, static_cast<i64>(P) * hidden);
        linear_bias(dev_, mlp, fc2_[k].f32(), fc2_b_[k].f32(), proj_out, P, hidden, D);
        ops::add_inplace(dev_, x, proj_out, static_cast<i64>(P) * D);
    }

    out = Tensor::empty({static_cast<i64>(P), D}, DType::F32, dev_);
    ops::layernorm_forward(dev_, x, post_ln_.f32(),
                           post_ln_b_.defined() ? post_ln_b_.f32() : nullptr,
                           out.f32(), P, D, 1e-5f);
}

void VisionTower::encode_patches(const float* pixels, std::vector<float>& out) const {
    Tensor patches_dev;
    encode_patches_tensor(pixels, patches_dev);
    const size_t n = static_cast<size_t>(patches_dev.numel());
    out.resize(n);
    device_copy(out.data(), Device::CPU, patches_dev.f32(), dev_, n * sizeof(float));
}

void VisionTower::encode_pooled_features(const float* pixels, Tensor& out_pooled) const {
    Tensor patches_dev;
    encode_patches_tensor(pixels, patches_dev);
    const int P_out = cfg_.pooled_patches();
    const int D = cfg_.dim;
    out_pooled = Tensor::empty({static_cast<i64>(P_out), D}, DType::F32, dev_);
    apply_spatial_pooling(patches_dev.f32(), out_pooled.f32(), cfg_.n_patches());
}

void VisionTower::encode_video_pooled_features(const std::vector<const float*>& frames_chw, Tensor& out_pooled) const {
    const int F = static_cast<int>(frames_chw.size());
    if (F <= 0) return;
    const int P_out = cfg_.pooled_patches();
    const int P_all = cfg_.n_patches();
    const int D     = cfg_.dim;
    const float* te = time_embed_.defined() ? time_embed_.f32() : nullptr;

    out_pooled = Tensor::empty({static_cast<i64>(F) * P_out, D}, DType::F32, dev_);
    for (int f = 0; f < F; ++f) {
        Tensor frame_patches;
        encode_patches_tensor(frames_chw[static_cast<size_t>(f)], frame_patches);
        if (te) {
            const int ti = std::min(f, cfg_.max_frames - 1);
            const float* t_vec = te + static_cast<size_t>(ti) * D;
            ops::add_bias(dev_, t_vec, frame_patches.f32(), P_all, D);
        }
        float* dst = out_pooled.f32() + static_cast<size_t>(f) * P_out * D;
        apply_spatial_pooling(frame_patches.f32(), dst, P_all);
    }
}

void VisionTower::encode_tensor(const float* pixels, Tensor& out) const {
    Tensor pooled;
    encode_pooled_features(pixels, pooled);
    const int P_out = cfg_.pooled_patches();
    const int O = cfg_.out_dim;
    out = Tensor::empty({static_cast<i64>(P_out), O}, DType::F32, dev_);
    forward_projector(dev_, pooled.f32(), out.f32(), P_out);
}

void VisionTower::encode_video_tensor(const std::vector<const float*>& frames_chw, Tensor& out) const {
    const int F = static_cast<int>(frames_chw.size());
    if (F <= 0) return;
    Tensor pooled;
    encode_video_pooled_features(frames_chw, pooled);
    const int total_tokens = F * cfg_.pooled_patches();
    const int O = cfg_.out_dim;
    out = Tensor::empty({static_cast<i64>(total_tokens), O}, DType::F32, dev_);
    forward_projector(dev_, pooled.f32(), out.f32(), total_tokens);
}

void VisionTower::encode(const float* pixels, std::vector<float>& out) const {
    Tensor out_t;
    encode_tensor(pixels, out_t);
    const size_t n = static_cast<size_t>(out_t.numel());
    out.resize(n);
    device_copy(out.data(), Device::CPU, out_t.f32(), dev_, n * sizeof(float));
}

void VisionTower::encode_video(const std::vector<const float*>& frames_chw, std::vector<float>& out) const {
    const int F = static_cast<int>(frames_chw.size());
    if (F <= 0) return;
    Tensor out_t;
    encode_video_tensor(frames_chw, out_t);
    const size_t n = static_cast<size_t>(out_t.numel());
    out.resize(n);
    device_copy(out.data(), Device::CPU, out_t.f32(), dev_, n * sizeof(float));
}

bool VisionTower::encode_image_file(const std::string& path, std::vector<float>& out) const {
    ImageBuffer img = load_image_any(path, cfg_.image_size, cfg_.image_size);
    if (!img.valid) return false;
    encode(img.chw_data.data(), out);
    return true;
}

bool VisionTower::encode_video_file(const std::string& path, int max_frames, std::vector<float>& out) const {
    VideoBuffer vid = load_video_any(path, max_frames, cfg_.image_size, cfg_.image_size);
    if (!vid.valid || vid.frames.empty()) return false;
    std::vector<const float*> ptrs;
    ptrs.reserve(vid.frames.size());
    for (const auto& f : vid.frames) ptrs.push_back(f.data());
    encode_video(ptrs, out);
    return true;
}

void VisionTower::forward_projector(Device dev, const float* in_features, float* out_tokens, int num_tokens) const {
    const int D = cfg_.dim;
    const int O = cfg_.out_dim;
    const int inter = (cfg_.proj_hidden > 0) ? cfg_.proj_hidden : O;

    Tensor pw1 = proj_w_.device() == dev ? proj_w_ : proj_w_.to(dev);
    Tensor pb1 = proj_bias_.device() == dev ? proj_bias_ : proj_bias_.to(dev);

    if (cfg_.proj_hidden > 0) {
        Tensor pw2 = proj2_w_.device() == dev ? proj2_w_ : proj2_w_.to(dev);
        Tensor pb2 = proj2_bias_.device() == dev ? proj2_bias_ : proj2_bias_.to(dev);

        if (dev == Device::CUDA) {
            Tensor in_dev;
            const float* in_ptr = in_features;
#ifdef GAI_CUDA
            if (!cuda::is_device_memory(in_features)) {
                in_dev = Tensor({static_cast<i64>(num_tokens) * D}, DType::F32, Device::CUDA);
                device_copy(in_dev.f32(), Device::CUDA, in_features, Device::CPU,
                            sizeof(float) * static_cast<size_t>(num_tokens) * D);
                in_ptr = in_dev.f32();
            }
#endif
            Tensor tmp_dev = Tensor::empty({static_cast<i64>(num_tokens) * inter}, DType::F32, Device::CUDA);
            linear_bias(Device::CUDA, in_ptr, pw1.f32(), pb1.defined() ? pb1.f32() : nullptr,
                        tmp_dev.f32(), num_tokens, D, inter);
            ops::gelu_forward(Device::CUDA, tmp_dev.f32(), tmp_dev.f32(), static_cast<i64>(num_tokens) * inter);
#ifdef GAI_CUDA
            const bool out_is_dev = cuda::is_device_memory(out_tokens);
#else
            const bool out_is_dev = false;
#endif
            if (out_is_dev) {
                linear_bias(Device::CUDA, tmp_dev.f32(), pw2.f32(), pb2.defined() ? pb2.f32() : nullptr,
                            out_tokens, num_tokens, inter, O);
            } else {
                Tensor out_dev = Tensor::empty({static_cast<i64>(num_tokens) * O}, DType::F32, Device::CUDA);
                linear_bias(Device::CUDA, tmp_dev.f32(), pw2.f32(), pb2.defined() ? pb2.f32() : nullptr,
                            out_dev.f32(), num_tokens, inter, O);
                device_copy(out_tokens, Device::CPU, out_dev.f32(), Device::CUDA,
                            sizeof(float) * static_cast<size_t>(num_tokens) * O);
            }
        } else {
            std::vector<float> tmp(static_cast<size_t>(num_tokens) * inter);
            linear_bias(Device::CPU, in_features, pw1.f32(), pb1.defined() ? pb1.f32() : nullptr,
                        tmp.data(), num_tokens, D, inter);
            ops::gelu_forward(Device::CPU, tmp.data(), tmp.data(), static_cast<i64>(num_tokens) * inter);
            linear_bias(Device::CPU, tmp.data(), pw2.f32(), pb2.defined() ? pb2.f32() : nullptr,
                        out_tokens, num_tokens, inter, O);
        }
    } else {
        linear_bias(dev, in_features, pw1.f32(), pb1.defined() ? pb1.f32() : nullptr,
                    out_tokens, num_tokens, D, O);
    }
}

void VisionTower::projector_forward_train(const float* in_features, int num_tokens,
                                          float* out_tokens, float* saved_h1, float* saved_act) const {
    const int D = cfg_.dim;
    const int O = cfg_.out_dim;
    const int inter = cfg_.proj_hidden;

    Tensor pw1 = proj_w_.device() == Device::CPU ? proj_w_ : proj_w_.to(Device::CPU);
    Tensor pb1 = proj_bias_.device() == Device::CPU ? proj_bias_ : proj_bias_.to(Device::CPU);
    Tensor pw2 = proj2_w_.device() == Device::CPU ? proj2_w_ : proj2_w_.to(Device::CPU);
    Tensor pb2 = proj2_bias_.device() == Device::CPU ? proj2_bias_ : proj2_bias_.to(Device::CPU);

    linear_bias(Device::CPU, in_features, pw1.f32(), pb1.defined() ? pb1.f32() : nullptr,
                saved_h1, num_tokens, D, inter);
    ops::gelu_forward(Device::CPU, saved_h1, saved_act, static_cast<i64>(num_tokens) * inter);
    linear_bias(Device::CPU, saved_act, pw2.f32(), pb2.defined() ? pb2.f32() : nullptr,
                out_tokens, num_tokens, inter, O);
}

void VisionTower::projector_backward(const float* in_features, const float* saved_h1, const float* saved_act,
                                     const float* d_out, int num_tokens, ProjectorGradients& grads) const {
    const int D = cfg_.dim;
    const int O = cfg_.out_dim;
    const int inter = cfg_.proj_hidden;

    if (!grads.d_w2.defined()) grads.d_w2 = Tensor({O, inter}, DType::F32, Device::CPU);
    if (!grads.d_b2.defined()) grads.d_b2 = Tensor({O}, DType::F32, Device::CPU);
    if (!grads.d_w1.defined()) grads.d_w1 = Tensor({inter, D}, DType::F32, Device::CPU);
    if (!grads.d_b1.defined()) grads.d_b1 = Tensor({inter}, DType::F32, Device::CPU);

    float* dw2 = grads.d_w2.f32();
    float* db2 = grads.d_b2.f32();
    float* dw1 = grads.d_w1.f32();
    float* db1 = grads.d_b1.f32();

    grads.d_w2.zero_();
    grads.d_b2.zero_();
    grads.d_w1.zero_();
    grads.d_b1.zero_();

    ops::gemm(Device::CPU, true, false, O, inter, num_tokens, 1.0f,
              d_out, O, saved_act, inter, 0.0f, dw2, inter);
    ops::bias_backward(Device::CPU, d_out, db2, num_tokens, O);

    Tensor pw2 = proj2_w_.device() == Device::CPU ? proj2_w_ : proj2_w_.to(Device::CPU);
    std::vector<float> d_act(static_cast<size_t>(num_tokens) * inter);
    ops::gemm(Device::CPU, false, false, num_tokens, inter, O, 1.0f,
              d_out, O, pw2.f32(), inter, 0.0f, d_act.data(), inter);

    std::vector<float> d_h1(static_cast<size_t>(num_tokens) * inter, 0.0f);
    ops::gelu_backward(Device::CPU, saved_h1, d_act.data(), d_h1.data(), static_cast<i64>(num_tokens) * inter);

    ops::gemm(Device::CPU, true, false, inter, D, num_tokens, 1.0f,
              d_h1.data(), inter, in_features, D, 0.0f, dw1, D);
    ops::bias_backward(Device::CPU, d_h1.data(), db1, num_tokens, inter);
}

void VisionTower::projector_forward_train_dev(const float* in_features, int num_tokens,
                                              float* out_tokens, float* saved_h1, float* saved_act) const {
    const int D = cfg_.dim;
    const int O = cfg_.out_dim;
    const int inter = (cfg_.proj_hidden > 0) ? cfg_.proj_hidden : O;

    linear_bias(dev_, in_features, proj_w_.f32(),
                proj_bias_.defined() ? proj_bias_.f32() : nullptr,
                saved_h1, num_tokens, D, inter);
    if (cfg_.proj_hidden > 0) {
        ops::gelu_forward(dev_, saved_h1, saved_act, static_cast<i64>(num_tokens) * inter);
        linear_bias(dev_, saved_act, proj2_w_.f32(),
                    proj2_bias_.defined() ? proj2_bias_.f32() : nullptr,
                    out_tokens, num_tokens, inter, O);
    } else {
        device_copy(out_tokens, dev_, saved_h1, dev_,
                    static_cast<size_t>(num_tokens) * O * sizeof(float));
    }
}

void VisionTower::projector_backward_dev(const float* in_features, const float* saved_h1, const float* saved_act,
                                         const float* d_out, int num_tokens, ProjectorGradients& grads) const {
    const int D = cfg_.dim;
    const int O = cfg_.out_dim;
    const int inter = (cfg_.proj_hidden > 0) ? cfg_.proj_hidden : O;

    if (!grads.d_w1.defined() || grads.d_w1.device() != dev_) grads.d_w1 = Tensor({inter, D}, DType::F32, dev_);
    if (!grads.d_b1.defined() || grads.d_b1.device() != dev_) grads.d_b1 = Tensor({inter}, DType::F32, dev_);
    grads.d_w1.zero_();
    grads.d_b1.zero_();

    if (cfg_.proj_hidden > 0) {
        if (!grads.d_w2.defined() || grads.d_w2.device() != dev_) grads.d_w2 = Tensor({O, inter}, DType::F32, dev_);
        if (!grads.d_b2.defined() || grads.d_b2.device() != dev_) grads.d_b2 = Tensor({O}, DType::F32, dev_);
        grads.d_w2.zero_();
        grads.d_b2.zero_();

        ops::gemm(dev_, true, false, O, inter, num_tokens, 1.0f,
                  d_out, O, saved_act, inter, 0.0f, grads.d_w2.f32(), inter);
        ops::bias_backward(dev_, d_out, grads.d_b2.f32(), num_tokens, O);

        Tensor d_act = Tensor::empty({static_cast<i64>(num_tokens), inter}, DType::F32, dev_);
        ops::gemm(dev_, false, false, num_tokens, inter, O, 1.0f,
                  d_out, O, proj2_w_.f32(), inter, 0.0f, d_act.f32(), inter);

        Tensor d_h1 = Tensor::empty({static_cast<i64>(num_tokens), inter}, DType::F32, dev_);
        ops::gelu_backward(dev_, saved_h1, d_act.f32(), d_h1.f32(), static_cast<i64>(num_tokens) * inter);

        ops::gemm(dev_, true, false, inter, D, num_tokens, 1.0f,
                  d_h1.f32(), inter, in_features, D, 0.0f, grads.d_w1.f32(), D);
        ops::bias_backward(dev_, d_h1.f32(), grads.d_b1.f32(), num_tokens, inter);
    } else {
        ops::gemm(dev_, true, false, O, D, num_tokens, 1.0f,
                  d_out, O, in_features, D, 0.0f, grads.d_w1.f32(), D);
        ops::bias_backward(dev_, d_out, grads.d_b1.f32(), num_tokens, O);
    }
}

void VisionTower::projector_adamw_step(ProjectorGradients& grads, ProjectorAdamWState& opt,
                                      float lr, float beta1, float beta2, float eps, float weight_decay) {
    ++opt.step;
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(opt.step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(opt.step));

    auto update_param = [&](Tensor& w, Tensor& dw, Tensor& m, Tensor& v, bool decay) {
        if (!w.defined() || !dw.defined()) return;
        if (dw.device() != w.device()) dw = dw.to(w.device());
        if (!m.defined() || m.device() != w.device()) m = Tensor::zeros(w.shape(), DType::F32, w.device());
        if (!v.defined() || v.device() != w.device()) v = Tensor::zeros(w.shape(), DType::F32, w.device());

        ops::adamw_step(w.device(), w.f32(), dw.f32(), m.f32(), v.f32(),
                        w.numel(), lr, beta1, beta2, eps,
                        decay ? weight_decay : 0.0f, bc1, bc2, 1.0f);
    };

    update_param(proj_w_, grads.d_w1, opt.m_w1, opt.v_w1, true);
    update_param(proj_bias_, grads.d_b1, opt.m_b1, opt.v_b1, false);
    if (cfg_.proj_hidden > 0) {
        update_param(proj2_w_, grads.d_w2, opt.m_w2, opt.v_w2, true);
        update_param(proj2_bias_, grads.d_b2, opt.m_b2, opt.v_b2, false);
    }
}

bool VisionTower::save_projector_checkpoint(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) return false;

    auto write_tensor = [&](const Tensor& t) {
        Tensor cpu_t = (t.device() == Device::CPU) ? t : t.to(Device::CPU);
        const i64 n = cpu_t.numel();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(cpu_t.f32()), static_cast<std::streamsize>(n * sizeof(float)));
    };

    write_tensor(proj_w_);
    write_tensor(proj_bias_);
    if (cfg_.proj_hidden > 0) {
        write_tensor(proj2_w_);
        write_tensor(proj2_bias_);
    }
    return true;
}

bool VisionTower::load_projector_checkpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    auto read_tensor = [&](Tensor& t) -> bool {
        i64 n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n != t.numel()) return false;
        Tensor cpu_t = Tensor::empty(t.shape(), DType::F32, Device::CPU);
        f.read(reinterpret_cast<char*>(cpu_t.f32()), static_cast<std::streamsize>(n * sizeof(float)));
        t = (dev_ == Device::CPU) ? cpu_t : cpu_t.to(dev_);
        return true;
    };

    if (!read_tensor(proj_w_)) return false;
    if (!read_tensor(proj_bias_)) return false;
    if (cfg_.proj_hidden > 0) {
        if (!read_tensor(proj2_w_)) return false;
        if (!read_tensor(proj2_bias_)) return false;
    }
    return true;
}

std::unique_ptr<VisionTower> load_vision_from_gguf(const std::string& path, Device dev) {
    GGUFReader r;
    if (!r.open(path)) return nullptr;

    VisionConfig cfg;
    cfg.image_size  = static_cast<int>(r.get_int("vision.image_size",  cfg.image_size));
    cfg.patch_size  = static_cast<int>(r.get_int("vision.patch_size",  cfg.patch_size));
    cfg.dim         = static_cast<int>(r.get_int("vision.hidden_size",  cfg.dim));
    cfg.layers      = static_cast<int>(r.get_int("vision.num_hidden_layers", cfg.layers));
    cfg.heads       = static_cast<int>(r.get_int("vision.num_attention_heads", cfg.heads));
    cfg.mlp_ratio   = static_cast<int>(r.get_int("vision.mlp_ratio",   cfg.mlp_ratio));
    cfg.out_dim     = static_cast<int>(r.get_int("vision.projection_dim", cfg.out_dim));
    cfg.proj_hidden = static_cast<int>(r.get_int("vision.proj_hidden",  cfg.proj_hidden));
    cfg.channels    = static_cast<int>(r.get_int("vision.num_channels", cfg.channels));
    cfg.max_frames  = static_cast<int>(r.get_int("vision.max_frames",   cfg.max_frames));

    std::map<std::string, Tensor> weights;
    const auto needed = VisionTower::weight_names(cfg);
    for (const auto& name : needed) {
        if (r.find_tensor(name)) {
            weights[name] = r.read_tensor_f32(name);
        }
    }

    if (weights.empty()) return nullptr;

    auto vt = std::make_unique<VisionTower>(cfg, weights, dev);
    if (!vt->valid()) return nullptr;
    return vt;
}

}
