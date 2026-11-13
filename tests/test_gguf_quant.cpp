// The GGUF quantization codec and the export dtype policy.
//
// Both are checked here because a silent error in either one produces a file
// that loads, runs, and speaks nonsense - the hardest kind of bug to find
// from the outside:
//
//   1. The Q4_0 writer must follow GGML's quantize_row_q4_0_ref: the block
//      scale comes from the SIGNED extreme (the value with the largest |x|,
//      sign kept), not from the absolute maximum. With the absolute maximum
//      the scale is always negative, so a block whose values are mostly
//      negative (granite's mamba.dt_bias spans -11.94 .. -0.58) falls half
//      outside the representable range and loses 12.5% of it instead of
//      6.25%. The bytes must also match llama.cpp's, or another GGUF reader
//      decodes them differently.
//
//   2. The export policy must keep the numerically fragile tensors at full
//      precision. dt = softplus(dt_raw + dt_bias) cancels most of the value,
//      so a 4-bit dt_bias moves the recurrence's time step by a factor of
//      exp(1.49) - the model then emits noise instead of text.

#include "core/common.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "format/gguf_codec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace gai;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok    %s\n", what.c_str());
    } else {
        printf("  FAIL  %s\n", what.c_str());
        ++g_failures;
    }
}

// GGML's reference encoder (ggml-quants.c, quantize_row_q4_0_ref): the block
// scale comes from the signed extreme. The dequantization is GGML's too.
std::vector<float> ggml_q4_0_roundtrip(const std::vector<float>& src, bool signed_extreme) {
    const size_t n = src.size();
    const size_t nb = (n + 31) / 32;
    std::vector<uint8_t> bytes(nb * 18, 0);
    std::vector<float> out(n, 0.0f);
    for (size_t b = 0; b < nb; ++b) {
        const size_t start = b * 32;
        const size_t end   = std::min(start + 32, n);
        uint8_t* block = bytes.data() + b * 18;
        float amax = 0.0f, vmax = 0.0f;
        for (size_t i = start; i < end; ++i) {
            const float v = src[i];
            if (std::fabs(v) > amax) {
                amax = std::fabs(v);
                vmax = v;
            }
        }
        if (amax == 0.0f) continue;
        const float d  = (signed_extreme ? vmax : amax) / -8.0f;
        const float id = 1.0f / d;
        const uint16_t sh = fp32_to_fp16(d);
        std::memcpy(block, &sh, 2);
        // The block stores the scale as f16, so every reader dequantizes
        // with the ROUNDED scale - exactly what the reference does here.
        const float ds = fp16_to_fp32(sh);
        float nib[32];
        for (int j = 0; j < 16; ++j) {
            const float f0 = (start + j      < end) ? src[start + j]      : 0.0f;
            const float f1 = (start + j + 16 < end) ? src[start + j + 16] : 0.0f;
            const int xi0 = std::max(0, std::min(15, static_cast<int>(f0 * id + 8.5f)));
            const int xi1 = std::max(0, std::min(15, static_cast<int>(f1 * id + 8.5f)));
            block[2 + j] = static_cast<uint8_t>((xi1 << 4) | xi0);
            nib[j]      = static_cast<float>(xi0 - 8) * ds;
            nib[j + 16] = static_cast<float>(xi1 - 8) * ds;
        }
        for (size_t i = 0; i < 32; ++i)
            if (start + i < n) out[start + i] = nib[i];
    }
    return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(a[i] - b[i])));
    return worst;
}

// A deterministic pseudo-random generator (no <random>, so the test is
// reproducible on every platform).
struct Rng {
    u64 s = 0x9E3779B97F4A7C15ull;
    float next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<float>(static_cast<double>(s >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
    }
};

// Writes a tensor into a GGUF and reads it back, so the writer's Q4_0 block
// layout is checked end to end.
std::vector<float> roundtrip_through_gguf(const std::string& path,
                                          const std::vector<float>& src,
                                          GGMLType type) {
    {
        GGUFWriter w(path);
        w.set_arch("ghassan");
        w.set_model_name("test");
        Tensor t({(i64)src.size()}, DType::F32, Device::CPU);
        std::memcpy(t.f32(), src.data(), src.size() * sizeof(float));
        w.add_tensor("probe", t, type);
        w.write();
    }
    GGUFReader r;
    if (!r.open(path)) return {};
    Tensor t = r.read_tensor_f32("probe");
    std::vector<float> out(src.size(), 0.0f);
    if (t.defined() && t.numel() == (i64)src.size())
        std::memcpy(out.data(), t.f32(), src.size() * sizeof(float));
    return out;
}


// GGML's block_q4_1 reference: f16 d, f16 m, then 16 bytes of nibbles (the
// low nibble of byte j holds element j, the high nibble element j+16) and
// the value is d*q + m.
std::vector<float> ggml_q4_1_roundtrip(const std::vector<float>& src) {
    const size_t n = src.size();
    const size_t nb = (n + 31) / 32;
    std::vector<uint8_t> bytes(nb * 20, 0);
    std::vector<float> out(n, 0.0f);
    for (size_t b = 0; b < nb; ++b) {
        const size_t start = b * 32;
        const size_t end   = std::min(start + 32, n);
        uint8_t* block = bytes.data() + b * 20;
        float mn = src[start], mx = src[start];
        for (size_t i = start; i < end; ++i) {
            mn = std::min(mn, src[i]);
            mx = std::max(mx, src[i]);
        }
        const float d   = (mx - mn) / 15.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        const uint16_t sh_d = fp32_to_fp16(d);
        const uint16_t sh_m = fp32_to_fp16(mn);
        std::memcpy(block, &sh_d, 2);
        std::memcpy(block + 2, &sh_m, 2);
        for (int j = 0; j < 16; ++j) {
            const float f0 = (start + j      < end) ? src[start + j]      : mn;
            const float f1 = (start + j + 16 < end) ? src[start + j + 16] : mn;
            const int xi0 = std::max(0, std::min(15, static_cast<int>(std::lround((f0 - mn) * inv))));
            const int xi1 = std::max(0, std::min(15, static_cast<int>(std::lround((f1 - mn) * inv))));
            block[4 + j] = static_cast<uint8_t>((xi1 << 4) | xi0);
        }
        const float ds = fp16_to_fp32(sh_d);
        const float ms = fp16_to_fp32(sh_m);
        for (int j = 0; j < 16; ++j) {
            const int xi0 = block[4 + j] & 0x0F;
            const int xi1 = block[4 + j] >> 4;
            if (start + j < end)      out[start + j]      = ds * xi0 + ms;
            if (start + j + 16 < end)  out[start + j + 16] = ds * xi1 + ms;
        }
    }
    return out;
}

void test_q4_0_writer() {
    printf("the Q4_0 writer (GGML block layout)\n");
    Rng rng;

    // A gaussian block, like a real weight matrix.
    std::vector<float> gauss(32 * 16);
    for (auto& v : gauss) v = rng.next() * 0.05f;

    const std::vector<float> mine = roundtrip_through_gguf("test_q4_probe.gguf", gauss, GGMLType::Q4_0);
    check(mine.size() == gauss.size(), "the GGUF round trip returns the tensor");

    const std::vector<float> ref  = ggml_q4_0_roundtrip(gauss, true);   // GGML's rule
    const std::vector<float> bug  = ggml_q4_0_roundtrip(gauss, false);  // the old rule
    const double err_mine = max_abs_diff(mine, ref);
    const double err_bug  = max_abs_diff(mine, bug);
    printf("        the writer vs GGML's reference: %.8f | vs the absolute-maximum rule: %.8f\n",
           err_mine, err_bug);
    check(err_mine < 1e-6,
          "the writer reproduces GGML's Q4_0 exactly (the bytes are llama.cpp's)");
    check(err_bug > 1e-4,
          "the absolute-maximum rule really is different (the regression is caught)");

    // A block whose values are all NEGATIVE: the case that broke Granite.
    std::vector<float> neg(32);
    for (int i = 0; i < 32; ++i) neg[i] = -11.9375f + 0.45f * static_cast<float>(i);
    const std::vector<float> n_mine = roundtrip_through_gguf("test_q4_neg.gguf", neg, GGMLType::Q4_0);
    const std::vector<float> n_ref  = ggml_q4_0_roundtrip(neg, true);
    const std::vector<float> n_bug  = ggml_q4_0_roundtrip(neg, false);
    const double n_err_mine = max_abs_diff(n_mine, n_ref);
    const double n_err_bug  = max_abs_diff(n_mine, n_bug);
    printf("        a negative-only block: the writer vs GGML %.6f | vs the old rule %.6f\n",
           n_err_mine, n_err_bug);
    check(n_err_mine < 1e-6,
          "a negative-only block follows GGML too");
    check(n_err_bug > 0.5,
          "the old rule lost more than half a step on a negative block");

    // The absolute error against the original: 6.25% of the range is the
    // theoretical bound of Q4_0; the old rule reached 12.5%.
    const double err_vs_src = max_abs_diff(n_mine, neg);
    printf("        the negative block's absolute error: %.4f (the range is %.4f)\n",
           err_vs_src, 11.9375f);
    check(err_vs_src < 0.80,
          "a negative block keeps its error under 0.8 (a quarter of its range)");
}


void test_q4_1_writer() {
    printf("the Q4_1 writer (GGML block layout, 5 bits)\n");
    Rng rng;
    std::vector<float> gauss(32 * 16);
    for (auto& v : gauss) v = rng.next() * 0.05f;

    const std::vector<float> mine = roundtrip_through_gguf("test_q4_1_probe.gguf", gauss, GGMLType::Q4_1);
    const std::vector<float> ref  = ggml_q4_1_roundtrip(gauss);
    const double err = max_abs_diff(mine, ref);
    printf("        the writer vs GGML's reference: %.8f\n", err);
    check(err < 1e-6, "the writer reproduces GGML's Q4_1 exactly");

    // Q4_1 keeps more of the tensor than Q4_0 (5 bits + the block's real
    // min/max instead of 4 bits with a symmetric range).
    const std::vector<float> q40 = ggml_q4_0_roundtrip(gauss, true);
    const double err40 = max_abs_diff(q40, gauss);
    const double err41 = max_abs_diff(mine, gauss);
    printf("        the error vs the source: Q4_0 %.6f | Q4_1 %.6f\n", err40, err41);
    check(err41 < err40,
          "Q4_1 keeps more of the tensor than Q4_0 (the same ballpark of size)");
}

void test_dtype_policy() {
    printf("the export dtype policy\n");
    ExportProfileGGUF p;
    p.name = "q4_0";
    p.default_type = GGMLType::Q4_0;
    p.embedding_type = GGMLType::Q8_0;
    p.norm_type = GGMLType::F32;
    p.sensitive_type = GGMLType::Q8_0;
    p.hybrid = true;   // granite

    check(gguf_dtype_for("layers.0.mamba.A_log", p) == GGMLType::F32,
          "mamba.A_log stays F32 (the scan uses A = -exp(A_log))");
    check(gguf_dtype_for("layers.0.mamba.dt_bias", p) == GGMLType::F32,
          "mamba.dt_bias stays F32 (dt = softplus(dt_raw + dt_bias))");
    check(gguf_dtype_for("layers.0.mamba.D", p) == GGMLType::F32,
          "mamba.D stays F32 (the SSM skip path)");
    check(gguf_dtype_for("layers.0.mamba.conv1d.weight", p) == GGMLType::F32,
          "mamba.conv1d.weight stays F32 (it feeds the scan directly)");
    check(gguf_dtype_for("layers.0.mamba.conv1d.bias", p) == GGMLType::F32,
          "mamba.conv1d.bias stays F32");
    check(gguf_dtype_for("layers.0.moe_router", p) == GGMLType::F32,
          "moe_router stays F32 (a coarse router changes which experts run)");
    check(gguf_dtype_for("layers.0.mamba.in_proj", p) == GGMLType::F16,
          "mamba.in_proj is promoted to F16 (dt_raw comes from it)");
    check(gguf_dtype_for("layers.0.mamba.out_proj", p) == GGMLType::Q8_0,
          "mamba.out_proj is promoted to Q8_0 (it is on every token)");
    check(gguf_dtype_for("layers.0.moe_sh_gate", p) == GGMLType::Q8_0,
          "the shared expert is promoted to Q8_0 (it is on every token)");
    check(gguf_dtype_for("layers.0.moe_gate", p) == GGMLType::Q4_0,
          "the routed experts keep the profile's 4-bit default");
    check(gguf_dtype_for("layers.0.attn_norm", p) == GGMLType::F32,
          "the norms stay F32");
    check(gguf_dtype_for("tok_embeddings", p) == GGMLType::Q8_0,
          "the embedding keeps the embedding precision");
    check(gguf_dtype_for("final_norm", p) == GGMLType::F32,
          "the final norm stays F32");

    // A non-hybrid model (no Mamba) keeps the plain profile: the promotion
    // must not change the other architectures' files.
    ExportProfileGGUF q = p;
    q.hybrid = false;
    check(gguf_dtype_for("layers.0.mamba.in_proj", q) == GGMLType::Q4_0,
          "a non-hybrid model keeps the 4-bit default everywhere");
    check(gguf_dtype_for("layers.0.moe_sh_gate", q) == GGMLType::Q4_0,
          "a non-hybrid model keeps the 4-bit default for the shared expert too");
    check(gguf_dtype_for("layers.0.moe_router", q) == GGMLType::F32,
          "the router stays F32 for every architecture");
}

}  // namespace

int main() {
    printf("== the GGUF codec and the export policy ==\n");
    test_dtype_policy();
    test_q4_0_writer();
    test_q4_1_writer();
    if (g_failures == 0) {
        printf("\nPASS\n");
        return 0;
    }
    printf("\nFAIL: %d check(s)\n", g_failures);
    return 1;
}
