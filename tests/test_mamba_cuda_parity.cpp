// The Mamba-2 layer: the CPU <-> CUDA parity, on its own.
//
// The whole-model parity (test_cuda_parity) tells that the Granite case
// disagrees, but not WHERE: the Mamba-2 block, the attention, the MoE or
// the norms. This test runs the LAYER on both devices with the same
// inputs and compares:
//
//   1. the forward output (the layer computes the same function?),
//   2. every gradient (the backward of the same function?).
//
// It is skipped (and it says so) when the build has no CUDA.

#include "core/common.h"
#include "core/device.h"
#include "core/ops.h"
#include "core/tensor.h"

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

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 1) {}
    u32 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<u32>(s >> 32); }
    float uni() { return static_cast<float>(static_cast<int>(next() % 20001) - 10000) / 10000.0f; }
    float norm() { return uni() * 0.4f; }
};

// The geometry of the Granite-4.0 block, small enough to be fast.
struct Geo {
    int B = 2, T = 7;
    int d = 8;             // the hidden size
    int expand = 2;
    int n_heads = 3;
    int d_head = 4;
    int d_state = 5;
    int n_groups = 1;
    int k = 4;
    float eps = 1e-5f;
    int d_inner() const { return expand * d; }
    int d_state_tot() const { return n_groups * d_state; }
    int conv_dim() const { return d_inner() + 2 * d_state_tot(); }
    int proj_size() const { return d_inner() + conv_dim() + n_heads; }
};

struct Weights {
    Geo g;
    std::vector<float> x, w_in, conv_w, conv_b, dt_bias, a_log, D, norm_w, w_out, dout;

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
        for (auto& h : dt_bias) h = r.uni() + 1.0f;
        a_log.resize(static_cast<size_t>(g.n_heads));
        for (int h = 0; h < g.n_heads; ++h) a_log[h] = std::log(static_cast<float>(h + 1));
        D.resize(static_cast<size_t>(g.n_heads));
        for (auto& v : D) v = 1.0f;
        norm_w.assign(static_cast<size_t>(g.d_inner()), 1.0f);
        w_out.resize(static_cast<size_t>(g.d) * g.d_inner());
        for (auto& v : w_out) v = r.norm();
        dout.resize(static_cast<size_t>(N) * g.d);
        for (auto& v : dout) v = r.uni();
    }
};

struct Grads {
    std::vector<float> dx, dw_in, dconv_w, dconv_b, ddt_bias, da_log, dD, dnorm_w, dw_out;
};

// The forward + the backward of the layer on one device.
static void run(Device dev, Weights& w, std::vector<float>& out, Grads& gr) {
    const Geo& g = w.g;
    const int N = g.B * g.T;
    const size_t n = static_cast<size_t>(N) * g.d;

    // The inputs: built on the host, then moved to the device.
    auto put = [&](const std::vector<float>& src) {
        Tensor t({static_cast<i64>(src.size())}, DType::F32, Device::CPU);
        std::memcpy(t.data_ptr(), src.data(), src.size() * sizeof(float));
        return dev == Device::CPU ? t : t.to(dev);
    };
    const Tensor tx = put(w.x);
    const Tensor tw_in = put(w.w_in);
    const Tensor tconv_w = put(w.conv_w);
    const Tensor tconv_b = put(w.conv_b);
    const Tensor tdt_bias = put(w.dt_bias);
    const Tensor ta_log = put(w.a_log);
    const Tensor tD = put(w.D);
    const Tensor tnorm_w = put(w.norm_w);
    const Tensor tw_out = put(w.w_out);
    const Tensor tdout = put(w.dout);

    Tensor tstate({static_cast<i64>(g.B) * g.n_heads * g.d_head * g.d_state}, DType::F32, dev);
    Tensor tout({static_cast<i64>(n)}, DType::F32, dev);
    ops::zero(dev, tstate.f32(), tstate.numel());

    ops::mamba2_forward(dev, tx.f32(), tw_in.f32(), tconv_w.f32(), tconv_b.f32(),
                        tdt_bias.f32(), ta_log.f32(), tD.f32(), tnorm_w.f32(),
                        tw_out.f32(), tstate.f32(), tout.f32(),
                        g.B, g.T, g.d, g.d_inner(), g.conv_dim(), g.n_heads,
                        g.d_head, g.d_state, g.k, g.eps);
    {
        Tensor h = tout.to(Device::CPU);
        out.resize(static_cast<size_t>(h.numel()));
        std::memcpy(out.data(), h.f32(), out.size() * sizeof(float));
    }

    // The gradients (the outputs are written, so they are not const).
    Tensor tdx({static_cast<i64>(n)}, DType::F32, dev);
    Tensor tdw_in({static_cast<i64>(g.proj_size()) * g.d}, DType::F32, dev);
    Tensor tdconv_w({static_cast<i64>(g.conv_dim()) * g.k}, DType::F32, dev);
    Tensor tdconv_b({static_cast<i64>(g.conv_dim())}, DType::F32, dev);
    Tensor tddt_bias({static_cast<i64>(g.n_heads)}, DType::F32, dev);
    Tensor tda_log({static_cast<i64>(g.n_heads)}, DType::F32, dev);
    Tensor tdD({static_cast<i64>(g.n_heads)}, DType::F32, dev);
    Tensor tdnorm_w({static_cast<i64>(g.d_inner())}, DType::F32, dev);
    Tensor tdw_out({static_cast<i64>(g.d) * g.d_inner()}, DType::F32, dev);
    // The CUDA kernels ACCUMULATE into the gradient buffers (beta = 1 and
    // the atomicAdds), so they have to start at zero - exactly like the
    // model's zero_grad() does before the backward.
    ops::zero(dev, tdx.f32(), tdx.numel());
    ops::zero(dev, tdw_in.f32(), tdw_in.numel());
    ops::zero(dev, tdconv_w.f32(), tdconv_w.numel());
    ops::zero(dev, tdconv_b.f32(), tdconv_b.numel());
    ops::zero(dev, tddt_bias.f32(), tddt_bias.numel());
    ops::zero(dev, tda_log.f32(), tda_log.numel());
    ops::zero(dev, tdD.f32(), tdD.numel());
    ops::zero(dev, tdnorm_w.f32(), tdnorm_w.numel());
    ops::zero(dev, tdw_out.f32(), tdw_out.numel());

    ops::mamba2_backward(dev, tx.f32(), tw_in.f32(), tconv_w.f32(), tconv_b.f32(),
                         tdt_bias.f32(), ta_log.f32(), tD.f32(), tnorm_w.f32(),
                         tw_out.f32(), tdout.f32(),
                         tdx.f32(), tdw_in.f32(), tdconv_w.f32(), tdconv_b.f32(),
                         tddt_bias.f32(), tda_log.f32(), tdD.f32(), tdnorm_w.f32(),
                         tdw_out.f32(),
                         g.B, g.T, g.d, g.d_inner(), g.conv_dim(), g.n_heads,
                         g.d_head, g.d_state, g.k, g.eps);

    struct { std::vector<float>* v; const Tensor* t; } pairs[] = {
        {&gr.dx, &tdx}, {&gr.dw_in, &tdw_in}, {&gr.dconv_w, &tdconv_w},
        {&gr.dconv_b, &tdconv_b}, {&gr.ddt_bias, &tddt_bias}, {&gr.da_log, &tda_log},
        {&gr.dD, &tdD}, {&gr.dnorm_w, &tdnorm_w}, {&gr.dw_out, &tdw_out}};
    for (auto& p : pairs) {
        Tensor h = p.t->to(Device::CPU);
        p.v->resize(static_cast<size_t>(h.numel()));
        std::memcpy(p.v->data(), h.f32(), p.v->size() * sizeof(float));
    }
}

// The comparison of two vectors (the worst absolute difference and the
// scale of the reference).
static void compare(const char* what, const std::vector<float>& a,
                    const std::vector<float>& b) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(a[i])));
    }
    const double tol = 1e-3 * scale + 1e-6;
    if (worst > tol) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "%s: CPU vs CUDA differ by %.3e (the reference scale is %.3e)",
                      what, worst, scale);
        CHECK(false, buf);
    } else {
        std::printf("  %-12s : ok (the worst difference is %.3e, the scale is %.3e)\n",
                    what, worst, scale);
    }
}

int main() {
    if (!cuda_available()) {
        std::printf("test_mamba_cuda_parity: no CUDA device -> skipped\n");
        return 0;
    }
    Weights w;
    w.init(987654321);

    std::vector<float> out_cpu, out_gpu;
    Grads gc, gg;
    run(Device::CPU, w, out_cpu, gc);
    run(Device::CUDA, w, out_gpu, gg);

    std::printf("test_mamba_cuda_parity: the forward\n");
    compare("the output", out_cpu, out_gpu);
    std::printf("test_mamba_cuda_parity: the backward\n");
    compare("dx", gc.dx, gg.dx);
    compare("d(in_proj)", gc.dw_in, gg.dw_in);
    compare("d(conv1d.w)", gc.dconv_w, gg.dconv_w);
    compare("d(conv1d.b)", gc.dconv_b, gg.dconv_b);
    compare("d(dt_bias)", gc.ddt_bias, gg.ddt_bias);
    compare("d(A_log)", gc.da_log, gg.da_log);
    compare("d(D)", gc.dD, gg.dD);
    compare("d(norm.w)", gc.dnorm_w, gg.dnorm_w);
    compare("d(out_proj)", gc.dw_out, gg.dw_out);

    if (g_failures == 0) {
        std::printf("test_mamba_cuda_parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_mamba_cuda_parity: %d failure(s)\n", g_failures);
    return 1;
}
