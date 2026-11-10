#include "vision/vision.h"

#include "core/ops.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gai {

namespace {

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

void softmax_row(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int i = 0; i < n; ++i) x[i] *= inv;
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

    norm1_.resize(cfg_.layers);
    norm2_.resize(cfg_.layers);
    qkv_.resize(cfg_.layers);
    proj_.resize(cfg_.layers);
    fc1_.resize(cfg_.layers);
    fc2_.resize(cfg_.layers);

    for (int i = 0; i < cfg_.layers; ++i) {
        const std::string p = "vision.layers." + std::to_string(i) + ".";
        const bool ok =
            get(p + "norm1.weight", &norm1_[static_cast<size_t>(i)]) &&
            get(p + "qkv.weight", &qkv_[static_cast<size_t>(i)]) &&
            get(p + "proj.weight", &proj_[static_cast<size_t>(i)]) &&
            get(p + "norm2.weight", &norm2_[static_cast<size_t>(i)]) &&
            get(p + "fc1.weight", &fc1_[static_cast<size_t>(i)]) &&
            get(p + "fc2.weight", &fc2_[static_cast<size_t>(i)]);
        if (!ok) return;
    }

    if (!get("vision.projector.weight", &proj_w_)) return;
    if (cfg_.proj_hidden > 0) {
        if (!get("vision.projector.2.weight", &proj2_w_)) return;
    }

    // Shape checks: a mismatch here means the weights belong to another
    // vision config and would silently produce garbage.
    if (patch_embed_.numel() != static_cast<i64>(cfg_.dim) * cfg_.patch_dim()) return;
    if (pos_embed_.numel() != static_cast<i64>(cfg_.n_patches()) * cfg_.dim) return;
    if (proj_w_.numel() != static_cast<i64>(cfg_.out_dim) * cfg_.dim) return;

    valid_ = true;
}

void VisionTower::encode_patches(const float* pixels, std::vector<float>& out) const {
    const int P = cfg_.n_patches();
    const int D = cfg_.dim;
    const int pd = cfg_.patch_dim();
    const int ps = cfg_.patch_size;
    const int C = cfg_.channels;

    out.assign(static_cast<size_t>(P) * D, 0.0f);

    // Patchify and project.
    std::vector<float> patch(pd);
    std::vector<float> row(D);
    for (int p = 0; p < P; ++p) {
        const int py = (p / cfg_.patches_per_side()) * ps;
        const int px = (p % cfg_.patches_per_side()) * ps;
        int k = 0;
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < ps; ++y) {
                for (int x = 0; x < ps; ++x) {
                    patch[static_cast<size_t>(k++)] =
                        pixels[static_cast<size_t>((static_cast<size_t>(c) * cfg_.image_size +
                                                   static_cast<size_t>(py + y)) * cfg_.image_size +
                                                   static_cast<size_t>(px + x))];
                }
            }
        }
        // row = W * patch  (the weight is [dim, patch_dim], row major)
        const float* w = patch_embed_.f32();
        for (int i = 0; i < D; ++i) {
            const float* wr = w + static_cast<size_t>(i) * pd;
            float s = 0.0f;
            for (int j = 0; j < pd; ++j) s += wr[j] * patch[static_cast<size_t>(j)];
            row[static_cast<size_t>(i)] = s;
        }
        // + position embedding
        const float* pe = pos_embed_.f32() + static_cast<size_t>(p) * D;
        float* dst = out.data() + static_cast<size_t>(p) * D;
        for (int i = 0; i < D; ++i) dst[i] = row[static_cast<size_t>(i)] + pe[i];
    }

    // Transformer blocks.
    const int H = cfg_.heads;
    const int hd = D / H;
    const int hidden = D * cfg_.mlp_ratio;

    std::vector<float> x = out;
    std::vector<float> h(D), qkv(3 * D), att(P), ctx(D), y(D), mlp(hidden);

    for (int l = 0; l < cfg_.layers; ++l) {
        // --- attention ---
        layer_norm(x.data(), norm1_[static_cast<size_t>(l)].f32(), nullptr,
                   h.data(), P, D, 1e-5f);
        {
            const float* w = qkv_[static_cast<size_t>(l)].f32();
            const float* b = nullptr;
            for (int p = 0; p < P; ++p) {
                const float* hr = h.data() + static_cast<size_t>(p) * D;
                float* qr = qkv.data() + static_cast<size_t>(p) * 3 * D;
                for (int i = 0; i < 3 * D; ++i) {
                    const float* wr = w + static_cast<size_t>(i) * D;
                    float s = 0.0f;
                    for (int j = 0; j < D; ++j) s += wr[j] * hr[j];
                    qr[i] = s;
                }
            }
            (void)b;
        }
        // Bidirectional attention: every patch sees every patch (a vision
        // tower has no causal mask).
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        for (int hh = 0; hh < H; ++hh) {
            const float* qbase = qkv.data() + hh * hd;
            const float* kbase = qkv.data() + D + hh * hd;
            const float* vbase = qkv.data() + 2 * D + hh * hd;
            std::fill(ctx.begin(), ctx.end(), 0.0f);
            for (int i = 0; i < P; ++i) {
                const float* qi = qbase + static_cast<size_t>(i) * 3 * D;
                for (int j = 0; j < P; ++j) {
                    const float* kj = kbase + static_cast<size_t>(j) * 3 * D;
                    float s = 0.0f;
                    for (int c = 0; c < hd; ++c) s += qi[c] * kj[c];
                    att[static_cast<size_t>(j)] = s * scale;
                }
                softmax_row(att.data(), P);
                for (int j = 0; j < P; ++j) {
                    const float* vj = vbase + static_cast<size_t>(j) * 3 * D;
                    const float aj = att[static_cast<size_t>(j)];
                    float* ci = ctx.data() + static_cast<size_t>(i) * hd;
                    for (int c = 0; c < hd; ++c) ci[c] += aj * vj[c];
                }
            }
            // write the head output back into the context rows
            for (int i = 0; i < P; ++i) {
                float* ci = ctx.data() + static_cast<size_t>(i) * hd;
                float* xi = h.data() + static_cast<size_t>(i) * D + hh * hd;
                for (int c = 0; c < hd; ++c) xi[c] = ci[c];
            }
        }
        // output projection + residual
        {
            const float* w = proj_[static_cast<size_t>(l)].f32();
            for (int p = 0; p < P; ++p) {
                const float* hr = h.data() + static_cast<size_t>(p) * D;
                float* yr = y.data() + static_cast<size_t>(p) * D;
                for (int i = 0; i < D; ++i) {
                    const float* wr = w + static_cast<size_t>(i) * D;
                    float s = 0.0f;
                    for (int j = 0; j < D; ++j) s += wr[j] * hr[j];
                    yr[i] = s;
                }
            }
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];

        // --- MLP ---
        layer_norm(x.data(), norm2_[static_cast<size_t>(l)].f32(), nullptr,
                   h.data(), P, D, 1e-5f);
        {
            const float* w = fc1_[static_cast<size_t>(l)].f32();
            for (int p = 0; p < P; ++p) {
                const float* hr = h.data() + static_cast<size_t>(p) * D;
                float* mr = mlp.data() + static_cast<size_t>(p) * hidden;
                for (int i = 0; i < hidden; ++i) {
                    const float* wr = w + static_cast<size_t>(i) * D;
                    float s = 0.0f;
                    for (int j = 0; j < D; ++j) s += wr[j] * hr[j];
                    mr[i] = gelu(s);
                }
            }
        }
        {
            const float* w = fc2_[static_cast<size_t>(l)].f32();
            for (int p = 0; p < P; ++p) {
                const float* mr = mlp.data() + static_cast<size_t>(p) * hidden;
                float* yr = y.data() + static_cast<size_t>(p) * D;
                for (int i = 0; i < D; ++i) {
                    const float* wr = w + static_cast<size_t>(i) * hidden;
                    float s = 0.0f;
                    for (int j = 0; j < hidden; ++j) s += wr[j] * mr[j];
                    yr[i] = s;
                }
            }
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
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

    out.assign(static_cast<size_t>(P) * O, 0.0f);

    const float* w = proj_w_.f32();
    std::vector<float> tmp;
    const int inter = (cfg_.proj_hidden > 0) ? cfg_.proj_hidden : O;

    tmp.assign(static_cast<size_t>(P) * inter, 0.0f);
    for (int p = 0; p < P; ++p) {
        const float* pr = patches.data() + static_cast<size_t>(p) * D;
        float* tr = tmp.data() + static_cast<size_t>(p) * inter;
        for (int i = 0; i < inter; ++i) {
            const float* wr = w + static_cast<size_t>(i) * D;
            float s = 0.0f;
            for (int j = 0; j < D; ++j) s += wr[j] * pr[j];
            tr[i] = s;
        }
    }
    if (cfg_.proj_hidden > 0) {
        for (float& v : tmp) v = gelu(v);
        const float* w2 = proj2_w_.f32();
        for (int p = 0; p < P; ++p) {
            const float* tr = tmp.data() + static_cast<size_t>(p) * inter;
            float* orow = out.data() + static_cast<size_t>(p) * O;
            for (int i = 0; i < O; ++i) {
                const float* wr = w2 + static_cast<size_t>(i) * inter;
                float s = 0.0f;
                for (int j = 0; j < inter; ++j) s += wr[j] * tr[j];
                orow[i] = s;
            }
        }
    } else {
        for (int p = 0; p < P; ++p) {
            const float* tr = tmp.data() + static_cast<size_t>(p) * O;
            float* orow = out.data() + static_cast<size_t>(p) * O;
            std::memcpy(orow, tr, sizeof(float) * static_cast<size_t>(O));
        }
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
    const bool binary = (magic[1] == '6');
    if (!binary && magic[1] != '5') return false;

    int w = 0, h = 0, maxv = 0;
    if (!read_int(w) || !read_int(h) || !read_int(maxv)) return false;
    if (w != W || h != H) return false;

    if (binary) {
        f.get();  // the single whitespace after the header
        const int chans = (magic[1] == '6') ? 3 : 1;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < chans; ++c) {
                    const int v = f.get();
                    if (v < 0) return false;
                    const float norm = static_cast<float>(v) / static_cast<float>(maxv ? maxv : 255);
                    // to CHW
                    const int cc = (chans == 3) ? c : 0;
                    out[static_cast<size_t>(cc) * H * W + static_cast<size_t>(y) * W + x] = norm;
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    int v = 0;
                    if (!read_int(v)) return false;
                    const float norm = static_cast<float>(v) / static_cast<float>(maxv ? maxv : 255);
                    out[static_cast<size_t>(c) * H * W + static_cast<size_t>(y) * W + x] = norm;
                }
            }
        }
    }
    return true;
}

}  // namespace gai
