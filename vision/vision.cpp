#include "vision/vision.h"

#include "core/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gai {

namespace {

// The tower runs on the CPU: a single image is a one-off cost per request
// (a few milliseconds), and keeping the weights in host memory avoids a
// device round trip that would cost more than the encoding itself.
constexpr Device kDevice = Device::CPU;

// LayerNorm over the last dimension.
void layer_norm(const float* x, const float* w, const float* b,
                float* y, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<size_t>(r) * dim;
        float* yr = y + static_cast<size_t>(r) * dim;
        double mean = 0.0;
        for (int i = 0; i < dim; ++i) mean += xr[i];
        mean /= dim;
        double var = 0.0;
        for (int i = 0; i < dim; ++i) {
            const double d = static_cast<double>(xr[i]) - mean;
            var += d * d;
        }
        var /= dim;
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (int i = 0; i < dim; ++i) {
            float v = (static_cast<float>(xr[i]) - static_cast<float>(mean)) * inv;
            if (w) v *= w[i];
            if (b) v += b[i];
            yr[i] = v;
        }
    }
}

inline float gelu(float x) {
    // The tanh approximation, as in the usual ViT implementations.
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x3)));
}

void softmax_rows(float* x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<size_t>(r) * cols;
        float mx = row[0];
        for (int i = 1; i < cols; ++i) if (row[i] > mx) mx = row[i];
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            row[i] = std::exp(row[i] - mx);
            sum += row[i];
        }
        const float inv = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
        for (int i = 0; i < cols; ++i) row[i] *= inv;
    }
}

// y[M,N] = x[M,K] * w[N,K]^T + bias[N]
void linear_bias(const float* x, const float* w, const float* b,
                 float* y, int M, int K, int N) {
    ops::linear_forward(kDevice, x, w, y, M, K, N);
    if (!b) return;
    for (int m = 0; m < M; ++m) {
        float* row = y + static_cast<size_t>(m) * N;
        for (int n = 0; n < N; ++n) row[n] += b[n];
    }
}

bool has_shape(const Tensor& t, i64 n) {
    return t.defined() && t.numel() == n;
}

}  // namespace

std::vector<std::string> VisionTower::weight_names(const VisionConfig& cfg) {
    std::vector<std::string> names;
    names.push_back("vision.patch_embed.weight");
    names.push_back("vision.pos_embed.weight");
    names.push_back("vision.post_ln.weight");
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

VisionTower::VisionTower(VisionConfig cfg, const std::map<std::string, Tensor>& weights) {
    cfg_ = cfg;

    auto get = [&](const std::string& n, Tensor* dst) {
        auto it = weights.find(n);
        if (it == weights.end()) return false;
        if (!it->second.defined()) return false;
        *dst = it->second;
        return true;
    };

    if (!get("vision.patch_embed.weight", &patch_embed_)) return;
    if (!get("vision.pos_embed.weight", &pos_embed_)) return;
    if (!get("vision.post_ln.weight", &post_ln_)) return;

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

    // Shape checks: a mismatch here means the weights belong to another
    // vision config and would silently produce garbage (or read out of
    // bounds). Reject the checkpoint instead.
    const i64 D = cfg_.dim;
    const i64 P = cfg_.n_patches();
    const i64 hidden = cfg_.mlp_hidden();
    const i64 O = cfg_.out_dim;

    if (!has_shape(patch_embed_, D * cfg_.patch_dim())) return;
    if (!has_shape(pos_embed_, P * D)) return;
    if (!has_shape(post_ln_, D)) return;
    if (!has_shape(proj_w_, O * D)) return;
    if (!has_shape(proj_bias_, O)) return;

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
        const i64 inter = cfg_.proj_hidden;
        if (!has_shape(proj2_w_, O * inter)) return;
        if (!has_shape(proj2_bias_, O)) return;
    }

    valid_ = true;
}

void VisionTower::encode_patches(const float* pixels, std::vector<float>& out) const {
    const int P = cfg_.n_patches();
    const int D = cfg_.dim;
    const int pd = cfg_.patch_dim();
    const int ps = cfg_.patch_size;
    const int C = cfg_.channels;
    const int H = cfg_.heads;
    const int hd = cfg_.head_dim();
    const int hidden = cfg_.mlp_hidden();

    out.assign(static_cast<size_t>(P) * D, 0.0f);

    // --- patchify ---
    // The patches are laid out as a [P, patch_dim] matrix so the projection
    // is a single GEMM instead of one per patch.
    std::vector<float> patches(static_cast<size_t>(P) * pd);
    for (int p = 0; p < P; ++p) {
        const int py = (p / cfg_.patches_per_side()) * ps;
        const int px = (p % cfg_.patches_per_side()) * ps;
        float* dst = patches.data() + static_cast<size_t>(p) * pd;
        int k = 0;
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < ps; ++y) {
                for (int x = 0; x < ps; ++x) {
                    dst[k++] = pixels[static_cast<size_t>(
                        (static_cast<size_t>(c) * cfg_.image_size +
                         static_cast<size_t>(py + y)) * cfg_.image_size +
                        static_cast<size_t>(px + x))];
                }
            }
        }
    }

    // --- patch projection + position embedding ---
    std::vector<float> x(static_cast<size_t>(P) * D, 0.0f);
    linear_bias(patches.data(), patch_embed_.f32(), nullptr, x.data(), P, pd, D);

    const float* pe = pos_embed_.f32();
    for (int p = 0; p < P; ++p) {
        float* row = x.data() + static_cast<size_t>(p) * D;
        const float* per = pe + static_cast<size_t>(p) * D;
        for (int i = 0; i < D; ++i) row[i] += per[i];
    }

    // --- transformer blocks ---
    std::vector<float> h(static_cast<size_t>(P) * D);
    std::vector<float> qkv(static_cast<size_t>(P) * 3 * D);
    std::vector<float> ctx(static_cast<size_t>(P) * D, 0.0f);
    std::vector<float> scores(static_cast<size_t>(P) * P);
    std::vector<float> proj_out(static_cast<size_t>(P) * D);
    std::vector<float> mlp(static_cast<size_t>(P) * hidden);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int l = 0; l < cfg_.layers; ++l) {
        const size_t k = static_cast<size_t>(l);

        // --- attention ---
        layer_norm(x.data(), norm1_[k].f32(), norm1_b_[k].f32(),
                   h.data(), P, D, 1e-5f);
        linear_bias(h.data(), qkv_[k].f32(), qkv_b_[k].f32(),
                    qkv.data(), P, D, 3 * D);

        // Bidirectional attention: every patch sees every patch (a vision
        // tower has no causal mask). The scores and the context are two
        // GEMMs per head.
        std::fill(ctx.begin(), ctx.end(), 0.0f);
        for (int hh = 0; hh < H; ++hh) {
            const float* qh = qkv.data() + hh * hd;
            const float* kh = qkv.data() + D + hh * hd;
            const float* vh = qkv.data() + 2 * D + hh * hd;

            // scores[p, j] = dot(q_p, k_j) * scale
            ops::gemm(kDevice, false, true, P, P, hd, scale,
                      qh, 3 * D, kh, 3 * D, 0.0f, scores.data(), P);
            softmax_rows(scores.data(), P, P);
            // ctx[p, :] = sum_j scores[p, j] * v_j
            ops::gemm(kDevice, false, false, P, hd, P, 1.0f,
                      scores.data(), P, vh, 3 * D, 0.0f, ctx.data() + hh * hd, hd);
        }

        // output projection + residual
        linear_bias(ctx.data(), proj_[k].f32(), proj_b_[k].f32(),
                    proj_out.data(), P, D, D);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_out[i];

        // --- MLP ---
        layer_norm(x.data(), norm2_[k].f32(), norm2_b_[k].f32(),
                   h.data(), P, D, 1e-5f);
        linear_bias(h.data(), fc1_[k].f32(), fc1_b_[k].f32(),
                    mlp.data(), P, D, hidden);
        for (float& v : mlp) v = gelu(v);
        linear_bias(mlp.data(), fc2_[k].f32(), fc2_b_[k].f32(),
                    proj_out.data(), P, hidden, D);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_out[i];
    }

    // final norm
    layer_norm(x.data(), post_ln_.f32(), nullptr, out.data(), P, D, 1e-5f);
}

void VisionTower::encode(const float* pixels, std::vector<float>& out) const {
    std::vector<float> patches;
    encode_patches(pixels, patches);

    const int P = cfg_.n_patches();
    const int D = cfg_.dim;
    const int O = cfg_.out_dim;
    const int inter = (cfg_.proj_hidden > 0) ? cfg_.proj_hidden : O;

    out.assign(static_cast<size_t>(P) * O, 0.0f);

    std::vector<float> tmp(static_cast<size_t>(P) * inter, 0.0f);
    linear_bias(patches.data(), proj_w_.f32(), proj_bias_.f32(),
                tmp.data(), P, D, inter);

    if (cfg_.proj_hidden > 0) {
        for (float& v : tmp) v = gelu(v);
        linear_bias(tmp.data(), proj2_w_.f32(), proj2_bias_.f32(),
                    out.data(), P, inter, O);
    } else {
        std::memcpy(out.data(), tmp.data(),
                    sizeof(float) * static_cast<size_t>(P) * O);
    }
}

bool load_ppm(const std::string& path, float* out, int C, int H, int W) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    auto skip_ws = [&]() {
        while (true) {
            const int c = f.peek();
            if (c == '#') {
                std::string junk;
                std::getline(f, junk);
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { f.get(); continue; }
            break;
        }
    };
    auto read_int = [&](int& v) {
        skip_ws();
        std::string s;
        while (true) {
            const int c = f.peek();
            if (c < '0' || c > '9') break;
            s.push_back(static_cast<char>(f.get()));
        }
        if (s.empty()) return false;
        v = std::atoi(s.c_str());
        return true;
    };

    char magic[3] = {0, 0, 0};
    f.read(magic, 2);
    if (magic[0] != 'P') return false;
    const bool color = (magic[1] == '6');
    if (!color && magic[1] != '5') return false;

    int w = 0, h = 0, maxv = 0;
    if (!read_int(w) || !read_int(h) || !read_int(maxv)) return false;
    if (w != W || h != H) return false;
    if (maxv <= 0) return false;

    // The header ends with exactly one whitespace character, but a writer
    // that uses CRLF puts two. Consume one, and the second only when it is
    // the other half of the same line ending.
    {
        const int c = f.get();
        if (c == '\r' && f.peek() == '\n') f.get();
        else if (c < 0) return false;
    }

    const int src_channels = color ? 3 : 1;
    const float scale = 1.0f / static_cast<float>(maxv);

    auto store = [&](int x, int y, int c, float v) {
        // A grayscale image is broadcast to every channel, so the engine
        // always receives a proper CHW tensor.
        const size_t base = static_cast<size_t>(y) * W + x;
        if (src_channels == 3) {
            out[static_cast<size_t>(c) * H * W + base] = v;
        } else {
            for (int cc = 0; cc < C; ++cc) out[static_cast<size_t>(cc) * H * W + base] = v;
        }
    };

    if (magic[1] == '6') {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const int v = f.get();
                    if (v < 0) return false;
                    store(x, y, c, static_cast<float>(v) * scale);
                }
            }
        }
    } else {
        // P5: one sample per pixel, in ASCII.
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int v = 0;
                if (!read_int(v)) return false;
                store(x, y, 0, static_cast<float>(v) * scale);
            }
        }
    }
    return true;
}

}  // namespace gai
