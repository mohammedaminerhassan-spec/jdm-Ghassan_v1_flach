// Vision tower unit tests.
//
// Covers the parts of the multimodal path that can be checked without a
// checkpoint on disk:
//   * the weight names the loader looks for
//   * the shape validation (a mismatched checkpoint must be rejected, not
//     read out of bounds)
//   * the forward pass: the output size, the finiteness, the determinism
//     and the fact that a different image gives a different embedding
//   * the PPM reader (P6 colour, P5 grayscale, a wrong size)
//
// The end-to-end path (a GGUF that carries both a language model and a
// vision tower, an image in a prompt, the HTTP server) is exercised by
// tools/py/make_test_model.py and the engine's own CLI.

#include "core/common.h"
#include "core/tensor.h"
#include "vision/vision.h"
#include "tools/cli_common.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace gai;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::cout << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << msg  \
                      << "  [" #cond "]\n";                                     \
        }                                                                       \
    } while (0)

#define CHECK_EQ(a, b, msg)                                                     \
    do {                                                                        \
        ++g_checks;                                                             \
        const auto _a = (a);                                                    \
        const auto _b = (b);                                                    \
        if (!(_a == _b)) {                                                      \
            ++g_failures;                                                       \
            std::cout << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << msg  \
                      << "\n    want: " << _b << "\n    got : " << _a << "\n";   \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------

static VisionConfig tiny_config() {
    VisionConfig c;
    c.image_size = 8;
    c.patch_size = 4;      // 2x2 patches
    c.dim = 8;
    c.layers = 2;
    c.heads = 2;
    c.mlp_ratio = 2;
    c.out_dim = 16;
    c.proj_hidden = 0;
    c.channels = 3;
    return c;
}

static Tensor rand_tensor(i64 n, std::mt19937& rng, float scale = 0.2f) {
    Tensor t(std::vector<i64>{n}, DType::F32);
    std::uniform_real_distribution<float> d(-scale, scale);
    float* p = t.f32();
    for (i64 i = 0; i < n; ++i) p[i] = d(rng);
    return t;
}

// A tower whose weights match `cfg`.
static std::map<std::string, Tensor> make_weights(const VisionConfig& cfg, std::mt19937& rng) {
    std::map<std::string, Tensor> w;
    const i64 D = cfg.dim;
    const i64 hidden = cfg.mlp_hidden();
    w["vision.patch_embed.weight"] = rand_tensor(D * cfg.patch_dim(), rng);
    w["vision.pos_embed.weight"] = rand_tensor(cfg.n_patches() * D, rng);
    w["vision.post_ln.weight"] = rand_tensor(D, rng, 1.0f);
    for (int i = 0; i < cfg.layers; ++i) {
        const std::string p = "vision.layers." + std::to_string(i) + ".";
        w[p + "norm1.weight"] = rand_tensor(D, rng, 1.0f);
        w[p + "norm1.bias"] = rand_tensor(D, rng, 0.05f);
        w[p + "qkv.weight"] = rand_tensor(3 * D * D, rng);
        w[p + "qkv.bias"] = rand_tensor(3 * D, rng, 0.05f);
        w[p + "proj.weight"] = rand_tensor(D * D, rng);
        w[p + "proj.bias"] = rand_tensor(D, rng, 0.05f);
        w[p + "norm2.weight"] = rand_tensor(D, rng, 1.0f);
        w[p + "norm2.bias"] = rand_tensor(D, rng, 0.05f);
        w[p + "fc1.weight"] = rand_tensor(hidden * D, rng);
        w[p + "fc1.bias"] = rand_tensor(hidden, rng, 0.05f);
        w[p + "fc2.weight"] = rand_tensor(D * hidden, rng);
        w[p + "fc2.bias"] = rand_tensor(D, rng, 0.05f);
    }
    w["vision.projector.weight"] = rand_tensor(cfg.out_dim * D, rng);
    w["vision.projector.bias"] = rand_tensor(cfg.out_dim, rng, 0.05f);
    if (cfg.proj_hidden > 0) {
        w["vision.projector.2.weight"] = rand_tensor(cfg.out_dim * cfg.proj_hidden, rng);
        w["vision.projector.2.bias"] = rand_tensor(cfg.out_dim, rng, 0.05f);
    }
    return w;
}

static bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------

static void test_weight_names() {
    VisionConfig c = tiny_config();
    const auto names = VisionTower::weight_names(c);

    // 3 global tensors + 12 per layer + 2 projector tensors.
    CHECK_EQ(names.size(), static_cast<size_t>(3 + 12 * c.layers + 2),
             "the weight list covers every tensor of the tower");

    bool has_patch = false, has_proj = false;
    for (const auto& n : names) {
        if (n == "vision.patch_embed.weight") has_patch = true;
        if (n == "vision.projector.weight") has_proj = true;
    }
    CHECK(has_patch, "the patch embedding is in the list");
    CHECK(has_proj, "the projector is in the list");

    // A two layer projector adds its own pair.
    c.proj_hidden = 32;
    const auto names2 = VisionTower::weight_names(c);
    CHECK_EQ(names2.size(), names.size() + 2, "a hidden projector adds two tensors");
}

static void test_shape_validation() {
    std::mt19937 rng(7);
    VisionConfig c = tiny_config();

    {
        auto w = make_weights(c, rng);
        VisionTower t(c, w);
        CHECK(t.valid(), "a matching checkpoint is accepted");
    }
    {
        // A weight from another config: the tower must refuse it instead of
        // reading past the end of the tensor.
        auto w = make_weights(c, rng);
        w["vision.patch_embed.weight"] = rand_tensor(17, rng);
        VisionTower t(c, w);
        CHECK(!t.valid(), "a mismatched patch embedding is rejected");
    }
    {
        auto w = make_weights(c, rng);
        w.erase("vision.layers.1.fc2.weight");
        VisionTower t(c, w);
        CHECK(!t.valid(), "a missing weight is rejected");
    }
    {
        // The bias must not be optional: the forward pass uses it.
        auto w = make_weights(c, rng);
        w.erase("vision.layers.0.qkv.bias");
        VisionTower t(c, w);
        CHECK(!t.valid(), "a missing bias is rejected");
    }
    {
        auto w = make_weights(c, rng);
        w["vision.projector.weight"] = rand_tensor(c.out_dim * c.dim + 1, rng);
        VisionTower t(c, w);
        CHECK(!t.valid(), "a mismatched projector is rejected");
    }
}

static void test_forward() {
    std::mt19937 rng(11);
    VisionConfig c = tiny_config();
    const auto w = make_weights(c, rng);
    VisionTower t(c, w);
    CHECK(t.valid(), "the tower is built");

    const size_t npx = static_cast<size_t>(c.channels) * c.image_size * c.image_size;
    std::vector<float> img(npx);
    std::mt19937 irng(3);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    for (float& v : img) v = d(irng);

    std::vector<float> out;
    t.encode(img.data(), out);
    CHECK_EQ(out.size(), static_cast<size_t>(c.n_patches()) * c.out_dim,
             "the projector returns one vector per patch");
    CHECK(all_finite(out), "the embedding has no NaN or Inf");

    // The same image gives the same embedding.
    std::vector<float> out2;
    t.encode(img.data(), out2);
    bool same = out.size() == out2.size();
    for (size_t i = 0; same && i < out.size(); ++i) same = (out[i] == out2[i]);
    CHECK(same, "the forward pass is deterministic");

    // A different image gives a different embedding.
    std::vector<float> img2 = img;
    img2[0] = 1.0f - img2[0];
    std::vector<float> out3;
    t.encode(img2.data(), out3);
    bool differs = false;
    for (size_t i = 0; i < out.size(); ++i) {
        if (std::fabs(out[i] - out3[i]) > 1e-6f) { differs = true; break; }
    }
    CHECK(differs, "a different image gives a different embedding");

    // The patch features keep the tower width.
    std::vector<float> patches;
    t.encode_patches(img.data(), patches);
    CHECK_EQ(patches.size(), static_cast<size_t>(c.n_patches()) * c.dim,
             "the patch features keep the tower width");
    CHECK(all_finite(patches), "the patch features have no NaN or Inf");
}

static void test_two_layer_projector() {
    std::mt19937 rng(13);
    VisionConfig c = tiny_config();
    c.proj_hidden = 12;
    const auto w = make_weights(c, rng);
    VisionTower t(c, w);
    CHECK(t.valid(), "a two layer projector is accepted");

    const size_t npx = static_cast<size_t>(c.channels) * c.image_size * c.image_size;
    std::vector<float> img(npx, 0.25f);
    std::vector<float> out;
    t.encode(img.data(), out);
    CHECK_EQ(out.size(), static_cast<size_t>(c.n_patches()) * c.out_dim,
             "the projector keeps the output width");
    CHECK(all_finite(out), "the embedding has no NaN or Inf");
}

// ---------------------------------------------------------------------------

static std::string temp_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

static void test_load_ppm() {
    const int S = 4;
    const int C = 3;

    // --- P6 (colour) ---
    {
        const std::string path = temp_path("_gai_test.ppm");
        {
            std::ofstream f(path, std::ios::binary);
            f << "P6\n" << S << " " << S << "\n255\n";
            for (int y = 0; y < S; ++y) {
                for (int x = 0; x < S; ++x) {
                    const unsigned char rgb[3] = {
                        static_cast<unsigned char>(x * 60),
                        static_cast<unsigned char>(y * 60),
                        128};
                    f.write(reinterpret_cast<const char*>(rgb), 3);
                }
            }
        }
        std::vector<float> px(static_cast<size_t>(C) * S * S, -1.0f);
        CHECK(load_ppm(path, px.data(), C, S, S), "a P6 image is read");

        // CHW layout: the channel planes are contiguous.
        const float r = px[0 * S * S + 1 * S + 1];
        const float g = px[1 * S * S + 1 * S + 1];
        const float b = px[2 * S * S + 1 * S + 1];
        CHECK(std::fabs(r - 60.0f / 255.0f) < 1e-6f, "the red plane is right");
        CHECK(std::fabs(g - 60.0f / 255.0f) < 1e-6f, "the green plane is right");
        CHECK(std::fabs(b - 128.0f / 255.0f) < 1e-6f, "the blue plane is right");
        std::remove(path.c_str());
    }

    // --- P5 (grayscale): the value is broadcast to every channel ---
    {
        const std::string path = temp_path("_gai_test.pgm");
        {
            std::ofstream f(path, std::ios::binary);
            f << "P5\n" << S << " " << S << "\n255\n";
            for (int y = 0; y < S; ++y) {
                for (int x = 0; x < S; ++x) {
                    f << (x + y * S) * 16 << " ";
                }
            }
        }
        std::vector<float> px(static_cast<size_t>(C) * S * S, -1.0f);
        CHECK(load_ppm(path, px.data(), C, S, S), "a P5 image is read");
        const float v = px[0];
        CHECK(std::fabs(v - 0.0f / 255.0f) < 1e-6f, "the first pixel is right");
        for (int c = 0; c < C; ++c) {
            CHECK(std::fabs(px[static_cast<size_t>(c) * S * S + 3] -
                            px[0 * S * S + 3]) < 1e-6f,
                  "a grayscale image is broadcast to every channel");
        }
        std::remove(path.c_str());
    }

    // --- a size that does not match is refused ---
    {
        const std::string path = temp_path("_gai_test_small.ppm");
        {
            std::ofstream f(path, std::ios::binary);
            f << "P6\n2 2\n255\n";
            for (int i = 0; i < 12; ++i) f.put(static_cast<char>(i * 10));
        }
        std::vector<float> px(static_cast<size_t>(C) * S * S, -1.0f);
        CHECK(!load_ppm(path, px.data(), C, S, S), "a wrong size is refused");
        std::remove(path.c_str());
    }

    // --- a file that is not an image at all ---
    {
        const std::string path = temp_path("_gai_test_bad.ppm");
        {
            std::ofstream f(path, std::ios::binary);
            f << "not an image";
        }
        std::vector<float> px(static_cast<size_t>(C) * S * S, -1.0f);
        CHECK(!load_ppm(path, px.data(), C, S, S), "a broken file is refused");
        std::remove(path.c_str());
    }
}

int main() {
    enable_utf8_console();

    test_weight_names();
    test_shape_validation();
    test_forward();
    test_two_layer_projector();
    test_load_ppm();

    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << (g_checks - g_failures)
              << "/" << g_checks << " checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
