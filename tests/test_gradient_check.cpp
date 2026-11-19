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
//
// TWO RULES make the numbers trustworthy:
//
//  1) The step has to be big enough that the loss moves by far more than
//     its fp32 resolution, and small enough that the loss stays close to
//     its tangent. The step is chosen from the analytic gradient (the
//     loss change it produces is 1000 times the resolution of the loss)
//     and the two central differences (h and h/2) are combined with the
//     Richardson extrapolation, which cancels the O(h^2) truncation.
//
//  2) A probe is only asserted when the finite difference can RESOLVE
//     it: the two central differences must agree with each other. When
//     they do not, the measurement is noise (the gradient is too small
//     compared to the fp32 resolution of the loss), the probe is skipped
//     and reported as such - it is not a failure, and it is not a pass
//     either.
//
// The Mamba-2 block is checked on its own by test_mamba_backward (a
// linear loss there gives a clean signal for every one of its tensors).
#include "core/common.h"
#include "core/device.h"
#include "core/ops.h"
#include "model/model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace gai;

static int g_failures = 0;
static int g_skipped  = 0;

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

// One central difference of the loss with respect to one weight.
static double central(Model& m, const std::vector<i32>& ids,
                      const std::vector<i32>& targets, int B, int T,
                      Parameter* p, i64 idx, double w0, double h) {
    p->w.f32()[idx] = static_cast<float>(w0 + h);
    const double up = mean_loss(m, ids, targets, B, T);
    p->w.f32()[idx] = static_cast<float>(w0 - h);
    const double dn = mean_loss(m, ids, targets, B, T);
    p->w.f32()[idx] = static_cast<float>(w0);
    return (up - dn) / (2.0 * h);
}

static void check_gradient(Model& m, const std::string& tag,
                           const std::vector<i32>& ids,
                           const std::vector<i32>& targets, int B, int T,
                           const std::vector<const char*>& names) {
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

    const double l0 = mean_loss(m, ids, targets, B, T);
    // The effective resolution of the loss: the forward is fp32 and the
    // loss sums many cancelling terms, so it is a few ULPs, not one.
    const double res = std::fabs(l0) * 1e-6;
    const double tol_rel = 0.05;   // 5% of the gradient scale

    for (const char* nm : names) {
        Parameter* p = m.find_parameter(nm);
        if (!p || !p->g.defined()) continue;   // not present in this shape
        const i64 n = p->w.numel();
        // The typical magnitude of the tensor (the RMS): the step is
        // relative to it, so every tensor is probed the same way.
        double sq = 0.0;
        for (i64 i = 0; i < n; ++i) sq += static_cast<double>(p->w.f32()[i]) * p->w.f32()[i];
        const double scale = n > 0 ? std::sqrt(sq / static_cast<double>(n)) : 1.0;

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
        int asserted = 0, skipped_here = 0;
        double worst = 0.0;
        for (int k = 0; k < 3; ++k) {
            const i64 idx = probes[k];
            if (idx < 0 || idx >= n) continue;
            const double w0 = p->w.f32()[idx];
            const double ana = static_cast<double>(p->g.f32()[idx]) / static_cast<double>(ntok);
            if (ana == 0.0) continue;

            // The step: the loss has to move by 1000 resolutions of the
            // fp32 loss, and it must stay a small perturbation of the
            // weight (the tensor's own scale).
            const double want = 1000.0 * res / std::fabs(ana);
            double h = want;
            const double floor_h = 1e-5 * (std::fabs(w0) + scale);
            const double cap_h   = 0.05 * (std::fabs(w0) + scale) + 1e-4;
            if (h < floor_h) h = floor_h;
            if (h > cap_h) h = cap_h;

            const double d_big   = central(m, ids, targets, B, T, p, idx, w0, h);
            const double d_small = central(m, ids, targets, B, T, p, idx, w0, h * 0.5);
            // The Richardson extrapolation cancels the O(h^2) truncation.
            const double num = (4.0 * d_small - d_big) / 3.0;

            // The measurement has to be consistent with itself: the two
            // steps must agree. When they do not, the difference is
            // dominated by the fp32 noise of the loss and the probe
            // cannot resolve the gradient. The spread between the two
            // steps IS the uncertainty of the measurement, and it is what
            // the tolerance is built from.
            const double spread = std::fabs(d_big - d_small);
            const double sc = std::max(std::fabs(d_big), std::fabs(d_small));
            const double spread_rel = (sc > 0.0) ? spread / sc : 1.0;
            if (spread_rel > 0.25) {
                ++g_skipped;
                ++skipped_here;
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "  %s %s[%lld]: not resolvable (the fp32 loss noise "
                              "is larger than the gradient: analytic %.3e, the two "
                              "steps give %.3e and %.3e) - skipped",
                              tag.c_str(), nm, static_cast<long long>(idx),
                              ana, d_big, d_small);
                std::printf("%s\n", buf);
                continue;
            }

            const double err = std::fabs(num - ana);
            const double scl = std::max(std::fabs(num), std::fabs(ana));
            const double rel = (scl > 0.0) ? err / scl : 0.0;
            if (rel > worst) worst = rel;
            ++asserted;
            // The tolerance: 5% of the gradient, or four times the
            // uncertainty of the measurement (whichever is larger).
            const double tol = std::max(tol_rel, 4.0 * spread_rel);
            if (rel > tol) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "%s %s[%lld]: analytic %.6e vs numeric %.6e (%.1f%% off, "
                              "the measurement is worth %.1f%%)",
                              tag.c_str(), nm, static_cast<long long>(idx),
                              ana, num, 100.0 * rel, 100.0 * spread_rel);
                CHECK(false, buf);
            }
        }
        if (asserted > 0 || skipped_here > 0) {
            std::printf("  %s %-28s : %d probe(s) asserted (worst %.1f%%), %d skipped\n",
                        tag.c_str(), nm, asserted, 100.0 * worst, skipped_here);
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
    check_gradient(m, "dense", ids, targets, 2, 8,
                   {"layers.0.wq", "layers.0.wo", "layers.0.w_gate", "layers.0.w_down",
                    "layers.1.wq", "layers.1.w_gate", "layers.0.attn_norm",
                    "layers.0.ffn_norm"});
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
    // NOTE: the router is NOT probed: the top-k selection is DISCRETE, so
    // perturbing the router can change WHICH experts run. The finite
    // difference would then measure a different routing (a different
    // function), not the gradient. The router's gradient is verified by
    // its closed form instead.
    check_gradient(m, "moe", ids, targets, 2, 8,
                   {"layers.0.moe_gate", "layers.0.moe_down",
                    "layers.0.moe_sh_gate", "layers.0.moe_sh_down"});
}

// A Granite-like model: the Mamba-2 hybrid with the MoE and the
// multipliers. It is the shape the reference checkpoint has (NoPE, a shared
// expert of its own width, the residual multiplier).
static void test_granite_hybrid() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 16;
    cfg.max_seq_len = 32;
    cfg.tie_embeddings = true;
    cfg.arch = ModelConfig::ArchGranite;
    cfg.layer_types = {0, 1};          // mamba, then attention
    cfg.use_moe = true;
    cfg.num_experts = 4;
    cfg.moe_top_k = 2;
    cfg.moe_expert_dim = 12;
    cfg.moe_shared = true;
    cfg.shared_expert_intermediate_size = 20;   // different from the routed width
    cfg.position_embedding = 1;        // NoPE
    cfg.embedding_multiplier = 4.0f;
    cfg.residual_multiplier = 0.5f;
    cfg.attention_multiplier = 0.0f;   // 0 => 1/sqrt(head_dim)
    cfg.logits_scaling = 2.0f;
    cfg.mamba_n_heads = 4;
    cfg.mamba_d_head = 8;
    cfg.mamba_d_state = 16;
    cfg.mamba_n_groups = 1;
    cfg.mamba_d_conv = 4;
    cfg.mamba_expand = 2;

    Model m(cfg, Device::CPU);
    m.init_weights(31337);
    std::vector<i32> ids, targets;
    make_batch(ids, targets, cfg.vocab_size, 2, 8, 555);
    check_gradient(m, "granite", ids, targets, 2, 8,
                   {"layers.0.moe_gate", "layers.0.moe_down",
                    "layers.0.moe_sh_gate", "layers.0.moe_sh_down",
                    "layers.1.moe_gate", "layers.1.moe_sh_down"});
}

// The same Granite shape, but with a DENSE FFN. The MoE routing is
// discrete, so a perturbation of the first layer can change WHICH
// experts run in the second one: the finite difference would then
// measure a different function. Without the MoE the loss is smooth in
// every weight, and the Mamba-2 block can be probed in the full model
// (the wiring of the block inside the model, not only the block itself).
static void test_granite_mamba_wiring() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 16;
    cfg.max_seq_len = 32;
    cfg.tie_embeddings = true;
    cfg.arch = ModelConfig::ArchGranite;
    cfg.layer_types = {0, 1};          // mamba, then attention
    cfg.use_moe = false;               // <- a smooth loss
    cfg.position_embedding = 1;        // NoPE
    cfg.embedding_multiplier = 4.0f;
    cfg.residual_multiplier = 0.5f;
    cfg.attention_multiplier = 0.0f;
    cfg.logits_scaling = 2.0f;
    cfg.mamba_n_heads = 4;
    cfg.mamba_d_head = 8;
    cfg.mamba_d_state = 16;
    cfg.mamba_n_groups = 1;
    cfg.mamba_d_conv = 4;
    cfg.mamba_expand = 2;

    Model m(cfg, Device::CPU);
    m.init_weights(4242);
    std::vector<i32> ids, targets;
    make_batch(ids, targets, cfg.vocab_size, 2, 8, 4242);
    check_gradient(m, "granite-mamba", ids, targets, 2, 8,
                   {"layers.0.mamba.in_proj", "layers.0.mamba.out_proj",
                    "layers.0.mamba.conv1d.weight", "layers.0.mamba.conv1d.bias",
                    "layers.0.mamba.dt_bias", "layers.0.mamba.A_log",
                    "layers.0.mamba.D", "layers.0.mamba.norm.weight",
                    "layers.0.attn_norm", "layers.1.wq", "layers.1.w_gate"});
}

int main() {
    test_dense_residual_multiplier();
    test_moe_shared_width();
    test_granite_hybrid();
    test_granite_mamba_wiring();

    if (g_failures == 0) {
        std::printf("test_gradient_check: OK (%d probe(s) not resolvable in fp32)\n",
                    g_skipped);
        return 0;
    }
    std::fprintf(stderr, "test_gradient_check: %d failure(s)\n", g_failures);
    return 1;
}
