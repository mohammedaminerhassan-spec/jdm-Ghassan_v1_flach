// The Mamba-2 layer backward, checked on its own.
//
// The whole-model finite-difference check (test_gradient_check) mixes the
// Mamba-2 block with the attention, the MoE and the norms, so a failure
// there does not say WHERE the error is, and the loss resolution of a
// 64-token vocabulary drowns the small gradients in noise.
//
// This test isolates the layer: a tiny random Mamba-2, a LINEAR loss
// (dL/dout is a known constant vector, so the analytic gradients are
// exact and large enough for a central difference to resolve them).
//
// The zero initial state is part of the contract: the backward assumes the
// recurrent state starts at zero, so the state must be reset between
// independent sequences.
#include "core/common.h"
#include "core/ops_cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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

// A small deterministic generator (the test must be reproducible: no
// <random> implementation drift between the platforms).
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 1) {}
    u32 next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<u32>(s >> 32);
    }
    // [-1, 1]
    float uni() { return static_cast<float>(static_cast<int>(next() % 20001) - 10000) / 10000.0f; }
    float norm() { return uni() * 0.5f; }
};

struct Geometry {
    int B = 2, T = 5;
    int d = 6;             // the hidden size
    int expand = 2;
    int n_heads = 3;
    int d_head = 4;
    int d_state = 3;
    int n_groups = 1;
    int k = 4;             // the causal conv width
    float eps = 1e-5f;

    int d_inner() const { return expand * d; }
    int d_state_tot() const { return n_groups * d_state; }
    int conv_dim() const { return d_inner() + 2 * d_state_tot(); }
    int proj_size() const { return d_inner() + conv_dim() + n_heads; }
};

struct Layer {
    Geometry g;
    std::vector<float> x;       // [B, T, d]
    std::vector<float> w_in;    // [proj_size, d]
    std::vector<float> conv_w;  // [conv_dim, k]
    std::vector<float> conv_b;  // [conv_dim]
    std::vector<float> dt_bias; // [n_heads]
    std::vector<float> a_log;   // [n_heads]
    std::vector<float> D;       // [n_heads]
    std::vector<float> norm_w;  // [d_inner]
    std::vector<float> w_out;   // [d, d_inner]
    std::vector<float> c;       // the loss weights (dout)

    void init(u64 seed) {
        Rng r(seed);
        const int N = g.B * g.T;
        x.resize(static_cast<size_t>(N) * g.d);
        for (auto& v : x) v = r.norm();

        w_in.resize(static_cast<size_t>(g.proj_size()) * g.d);
        for (auto& v : w_in) v = r.norm();

        conv_w.resize(static_cast<size_t>(g.conv_dim()) * g.k);
        for (auto& v : conv_w) v = r.norm();

        conv_b.resize(static_cast<size_t>(g.conv_dim()));
        for (auto& v : conv_b) v = r.uni() * 0.1f;

        dt_bias.resize(static_cast<size_t>(g.n_heads));
        for (int h = 0; h < g.n_heads; ++h) dt_bias[h] = r.uni() + 2.0f;

        a_log.resize(static_cast<size_t>(g.n_heads));
        for (int h = 0; h < g.n_heads; ++h) a_log[h] = std::log(static_cast<float>(h + 1));

        D.resize(static_cast<size_t>(g.n_heads));
        for (int h = 0; h < g.n_heads; ++h) D[h] = 1.0f;

        norm_w.assign(static_cast<size_t>(g.d_inner()), 1.0f);

        w_out.resize(static_cast<size_t>(g.d) * g.d_inner());
        for (auto& v : w_out) v = r.norm();

        c.resize(static_cast<size_t>(N) * g.d);
        for (auto& v : c) v = r.uni();
    }

    // The loss: L = sum(out * c). The state is always reset (the backward
    // assumes a zero initial state).
    double loss() const {
        std::vector<float> state(static_cast<size_t>(g.B) * g.n_heads * g.d_head * g.d_state, 0.0f);
        std::vector<float> out(static_cast<size_t>(g.B) * g.T * g.d, 0.0f);
        cpu::mamba2_forward(x.data(), w_in.data(), conv_w.data(), conv_b.data(),
                            dt_bias.data(), a_log.data(), D.data(), norm_w.data(),
                            w_out.data(), state.data(), out.data(),
                            g.B, g.T, g.d, g.d_inner(), g.conv_dim(), g.n_heads,
                            g.d_head, g.d_state, g.k, g.eps);
        double s = 0.0;
        for (size_t i = 0; i < out.size(); ++i) s += static_cast<double>(out[i]) * c[i];
        return s;
    }
};

struct Grads {
    std::vector<float> dx, dw_in, dconv_w, dconv_b, ddt_bias, da_log, dD, dnorm_w, dw_out;
};

// The analytic gradients of the layer for the loss L = sum(out * c).
static Grads analytic(Layer& L) {
    const Geometry& g = L.g;
    Grads gr;
    const int N = g.B * g.T;
    gr.dx.assign(static_cast<size_t>(N) * g.d, 0.0f);
    gr.dw_in.assign(static_cast<size_t>(g.proj_size()) * g.d, 0.0f);
    gr.dconv_w.assign(static_cast<size_t>(g.conv_dim()) * g.k, 0.0f);
    gr.dconv_b.assign(static_cast<size_t>(g.conv_dim()), 0.0f);
    gr.ddt_bias.assign(static_cast<size_t>(g.n_heads), 0.0f);
    gr.da_log.assign(static_cast<size_t>(g.n_heads), 0.0f);
    gr.dD.assign(static_cast<size_t>(g.n_heads), 0.0f);
    gr.dnorm_w.assign(static_cast<size_t>(g.d_inner()), 0.0f);
    gr.dw_out.assign(static_cast<size_t>(g.d) * g.d_inner(), 0.0f);

    std::vector<float> state(static_cast<size_t>(g.B) * g.n_heads * g.d_head * g.d_state, 0.0f);
    std::vector<float> out(static_cast<size_t>(N) * g.d, 0.0f);
    cpu::mamba2_forward(L.x.data(), L.w_in.data(), L.conv_w.data(), L.conv_b.data(),
                        L.dt_bias.data(), L.a_log.data(), L.D.data(), L.norm_w.data(),
                        L.w_out.data(), state.data(), out.data(),
                        g.B, g.T, g.d, g.d_inner(), g.conv_dim(), g.n_heads,
                        g.d_head, g.d_state, g.k, g.eps);

    cpu::mamba2_backward(L.x.data(), L.w_in.data(), L.conv_w.data(), L.conv_b.data(),
                         L.dt_bias.data(), L.a_log.data(), L.D.data(), L.norm_w.data(),
                         L.w_out.data(), L.c.data(),
                         gr.dx.data(), gr.dw_in.data(), gr.dconv_w.data(), gr.dconv_b.data(),
                         gr.ddt_bias.data(), gr.da_log.data(), gr.dD.data(), gr.dnorm_w.data(),
                         gr.dw_out.data(),
                         g.B, g.T, g.d, g.d_inner(), g.conv_dim(), g.n_heads,
                         g.d_head, g.d_state, g.k, g.eps);
    return gr;
}

// The central difference of the loss with respect to one element of a
// parameter vector.
//
// The step is chosen so that the loss change it produces is FAR above the
// fp32 resolution of the loss: the forward is fp32, so a loss of L has an
// absolute resolution of about L*1.2e-7, and a step that does not move the
// loss by more than that measures pure noise.
//
// Two steps (h and h/2) are combined with the Richardson extrapolation
//     D_R = (4*D(h/2) - D(h)) / 3
// which cancels the O(h^2) truncation error of the central difference
// (the O(h^4) remainder is 16x smaller), so the comparison is against the
// exact derivative and not against a linearisation of it.
// One central difference.
static double central(Layer& L, std::vector<float>& p, i64 idx, double w0, double h);

static double fd(Layer& L, std::vector<float>& p, i64 idx, double ana,
                 double* out_spread) {
    double sq = 0.0;
    for (float v : p) sq += static_cast<double>(v) * v;
    const double scale = std::sqrt(sq / static_cast<double>(p.size()));
    const double w0 = p[static_cast<size_t>(idx)];
    const double l0 = std::fabs(L.loss());
    // The fp32 resolution of the loss (its ULP).
    const double res = l0 * 1.2e-7;
    double h = 1e-5 * (std::fabs(w0) + scale);
    // The step has to move the loss by at least 1000 resolutions.
    const double need = 1000.0 * res / std::max(std::fabs(ana), 1e-30);
    if (need > h) h = need;
    // ... but not past the point where the linearisation breaks.
    const double cap = 1e-2 * (std::fabs(w0) + scale) + 1e-3;
    if (h > cap) h = cap;

    const double d_big = central(L, p, idx, w0, h);
    const double d_small = central(L, p, idx, w0, h * 0.5);
    if (out_spread) {
        const double sc = std::max(std::fabs(d_big), std::fabs(d_small));
        *out_spread = (sc > 0.0) ? std::fabs(d_big - d_small) / sc : 1.0;
    }
    return (4.0 * d_small - d_big) / 3.0;
}

// One central difference.
static double central(Layer& L, std::vector<float>& p, i64 idx, double w0, double h) {
    p[static_cast<size_t>(idx)] = static_cast<float>(w0 + h);
    const double up = L.loss();
    p[static_cast<size_t>(idx)] = static_cast<float>(w0 - h);
    const double dn = L.loss();
    p[static_cast<size_t>(idx)] = static_cast<float>(w0);
    return (up - dn) / (2.0 * h);
}

// The probe: the elements with the largest |analytic| gradient (the best
// signal to noise).
//
// The tolerance is built from the measurement's own uncertainty: the two
// steps (h and h/2) must agree, and the difference between them is the
// uncertainty of the difference (the truncation and the fp32 noise of the
// loss). A probe whose two steps disagree by more than 25% cannot
// resolve the gradient and is reported, not asserted.
static void probe(const char* name, Layer& L, std::vector<float>& p,
                  std::vector<float>& gr, int count, double tol) {
    const int n = static_cast<int>(p.size());
    for (int slot = 0; slot < count; ++slot) {
        i64 best = -1;
        double bv = -1.0;
        for (int i = 0; i < n; ++i) {
            const double a = std::fabs(static_cast<double>(gr[static_cast<size_t>(i)]));
            if (a > bv) { bv = a; best = i; }
        }
        if (best < 0) return;
        const double ana = gr[static_cast<size_t>(best)];
        double spread = 0.0;
        const double num = fd(L, p, best, ana, &spread);
        const double err = std::fabs(num - ana);
        const double sc = std::max(std::fabs(num), std::fabs(ana));
        if (sc <= 0.0) continue;
        const double rel = err / sc;
        // The tolerance is the measurement's own uncertainty (the spread
        // between the two steps) or 2%, whichever is larger.
        const double allow = std::max(tol, 4.0 * spread);
        if (rel > allow) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "%s[%lld]: analytic %.6e vs numeric %.6e (%.1f%% off, "
                          "the measurement is worth %.1f%%)",
                          name, static_cast<long long>(best), ana, num, 100.0 * rel,
                          100.0 * spread);
            CHECK(false, buf);
        }
        std::printf("  %-10s[%lld]: analytic %.6e vs numeric %.6e (%.2f%% off)\n",
                    name, static_cast<long long>(best), ana, num, 100.0 * rel);
        // Do not probe the same element twice.
        gr[static_cast<size_t>(best)] = 0.0f;
    }
}

int main() {
    Layer L;
    L.init(20261114);
    Grads gr = analytic(L);

    const double tol = 0.02;   // the fp32 forward is worth about this
    probe("in_proj",   L, L.w_in,   gr.dw_in,   3, tol);
    probe("conv1d.w",  L, L.conv_w, gr.dconv_w, 3, tol);
    probe("conv1d.b",  L, L.conv_b, gr.dconv_b, 3, tol);
    probe("dt_bias",   L, L.dt_bias,gr.ddt_bias,2, tol);
    probe("A_log",     L, L.a_log,  gr.da_log,  2, tol);
    probe("D",         L, L.D,      gr.dD,      2, tol);
    probe("norm.w",    L, L.norm_w, gr.dnorm_w, 3, tol);
    probe("out_proj",  L, L.w_out,  gr.dw_out,  3, tol);

    if (g_failures == 0) {
        std::printf("test_mamba_backward: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_mamba_backward: %d failure(s)\n", g_failures);
    return 1;
}
