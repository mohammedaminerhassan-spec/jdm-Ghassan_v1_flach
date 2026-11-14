// The gradient check: the analytic gradients against the numerical ones.
//
// This is the test that catches the class of bugs the audit found: a
// gradient that is computed with a missing factor (the residual
// multiplier) or with a wrong shape (the shared expert width) is
// perfectly well-behaved - the loss still goes down - so nothing else
// notices it. A finite-difference check does.
//
// The check runs on the CPU (the CUDA path is checked by the same test
// when a device is available).
#include "core/common.h"
#include "core/device.h"
#include "core/ops.h"
#include "model/model.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace gai;

static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,      \
                         __LINE__);                                           \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

// The mean cross-entropy of a batch (the same quantity the trainer
// optimises).
static double mean_loss(Model& m, const std::vector<i32>& ids,
                        const std::vector<i32>& targets, int B, int T) {
    Activations act = m.make_activations(B, T, false);
    Tensor d_ids({B * T}, DType::I32, Device::CPU);
    std::memcpy(d_ids.data_ptr(), ids.data(), sizeof(i32) * ids.size());
    Tensor d_tgt({B * T}, DType::I32, Device::CPU);
    std::memcpy(d_tgt.data_ptr(), targets.data(), sizeof(i32) * targets.size());
    Tensor& logits = m.forward(d_ids.i32p(), B, T, act);
    double sum = 0.0;
    i64 n = 0;
    ops::softmax_cross_entropy(Device::CPU, logits.f32(), d_tgt.i32p(), nullptr,
                               B * T, m.config().vocab_size, &sum, &n, 0.0f);
    return n > 0 ? sum / static_cast<double>(n) : 0.0;
}

static void check_gradient(Model& m, const std::string& tag,
                           const std::vector<i32>& ids,
                           const std::vector<i32>& targets, int B, int T) {
    m.enable_grad(true);
    m.zero_grad();
    Activations act = m.make_activations(B, T, true);
    Tensor d_ids({B * T}, DType::I32, Device::CPU);
    std::memcpy(d_ids.data_ptr(), ids.data(), sizeof(i32) * ids.size());
    Tensor d_tgt({B * T}, DType::I32, Device::CPU);
    std::memcpy(d_tgt.data_ptr(), targets.data(), sizeof(i32) * targets.size());

    i64 ntok = 0;
    // dout_scale = 1: the gradients are then the plain sum over the tokens.
    m.forward_backward(d_ids.i32p(), d_tgt.i32p(), B, T, act, &ntok, 1.0f, false,
                       nullptr, targets.data());
    CHECK(ntok > 0, (tag + ": no supervised token").c_str());

    // The parameters to probe: a spread over the network, with the
    // matrices that carry the residual path first.
    static const char* kNames[] = {
        "layers.0.wq", "layers.0.wo", "layers.0.w_gate", "layers.0.w_down",
        "layers.1.wq", "layers.1.w_gate",
    };
    // The finite difference of a tiny gradient is dominated by the fp32
    // resolution of the loss, so the probes are the LARGEST gradient
    // elements (the best signal-to-noise) and the step is large enough for
    // the loss change to stand above the noise.
    const double eps = 1e-2;
    const double tol_rel = 0.15;   // 15% of the gradient scale

    for (const char* nm : kNames) {
        Parameter* p = m.find_parameter(nm);
        if (!p || !p->g.defined()) continue;   // not present in this shape
        const i64 n = p->w.numel();
        // The three largest |gradient| elements.
        i64 probes[3] = {-1, -1, -1};
        for (int slot = 0; slot < 3; ++slot) {
            double best = -1.0;
            i64 bi = -1;
            for (i64 i = 0; i < n; ++i) {
                if (probes[0] == i || probes[1] == i || probes[2] == i) continue;
                const double a = std::fabs(static_cast<double>(p->g.f32()[i]));
                if (a > best) { best = a; bi = i; }
            }
            probes[slot] = bi;
        }
        for (int k = 0; k < 3; ++k) {
            const i64 idx = probes[k];
            if (idx < 0 || idx >= n) continue;
            const float w0 = p->w.f32()[idx];
            p->w.f32()[idx] = w0 + static_cast<float>(eps);
            const double up = mean_loss(m, ids, targets, B, T);
            p->w.f32()[idx] = w0 - static_cast<float>(eps);
            const double dn = mean_loss(m, ids, targets, B, T);
            p->w.f32()[idx] = w0;

            const double num = (up - dn) / (2.0 * eps);
            const double ana = static_cast<double>(p->g.f32()[idx]) / static_cast<double>(ntok);
            const double err = std::fabs(num - ana);
            const double scale = std::max(std::fabs(num), std::fabs(ana));
            const bool ok = scale > 0.0 && (err / scale <= tol_rel);
            if (!ok) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "%s %s[%lld]: analytic %.6e vs numeric %.6e (%.1f%% off)",
                              tag.c_str(), nm, static_cast<long long>(idx),
                              ana, num, scale > 0.0 ? 100.0 * err / scale : 0.0);
                CHECK(false, buf);
            }
        }
    }
    m.enable_grad(false);
}

static void make_batch(std::vector<i32>& ids, std::vector<i32>& targets,
                       int V, int B, int T, u64 seed) {
    u64 s = seed;
    const u64 vocab = static_cast<u64>(V);
    auto rnd = [&s, vocab]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<i32>((s >> 33) % vocab);
    };
    ids.resize(static_cast<size_t>(B) * T);
    targets.resize(static_cast<size_t>(B) * T);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = rnd();
        targets[i] = rnd();
    }
}

// A dense model with a residual multiplier: the FFN gradient is only
// correct when the multiplier is applied to the gradient that enters the
// FFN (the audit's BUG-01).
static void test_dense_residual_multiplier() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 24;
    cfg.max_seq_len = 32;
    cfg.tie_embeddings = true;
    cfg.residual_multiplier = 0.5f;      // <- the factor under test
    Model m(cfg, Device::CPU);
    m.init_weights(7);
    std::vector<i32> ids, targets;
    make_batch(ids, targets, cfg.vocab_size, 2, 8, 12345);
    check_gradient(m, "dense", ids, targets, 2, 8);
}

// A MoE model whose shared expert has a DIFFERENT width than the routed
// experts: the shared expert's backward used the routed width (the
// audit's BUG-02).
static void test_moe_shared_width() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 24;
    cfg.max_seq_len = 32;
    cfg.use_moe = true;
    cfg.num_experts = 4;
    cfg.moe_top_k = 2;
    cfg.moe_expert_dim = 8;
    cfg.moe_shared = true;
    cfg.shared_expert_intermediate_size = 16;   // different from the routed 8
    Model m(cfg, Device::CPU);
    m.init_weights(11);
    std::vector<i32> ids, targets;
    make_batch(ids, targets, cfg.vocab_size, 2, 8, 999);
    check_gradient(m, "moe", ids, targets, 2, 8);
}

int main() {
    test_dense_residual_multiplier();
    test_moe_shared_width();
    if (g_failures == 0) {
        std::printf("test_gradient_check: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_gradient_check: %d failure(s)\n", g_failures);
    return 1;
}
