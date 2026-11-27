// ══════════════════════════════════════════════════════════════════════════════
// backend_sycl.cpp — Intel GPU (oneAPI/SYCL) backend implementation
// ══════════════════════════════════════════════════════════════════════════════
#ifdef GAI_SYCL

#include "backends/backend_sycl.h"
#include "core/common.h"

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

namespace gai {
namespace sycl_backend {

static std::unique_ptr<sycl::queue> g_queue;

sycl::queue& queue() {
    GAI_CHECK(g_queue, "sycl_backend: not initialised — call init() first");
    return *g_queue;
}

void init() {
    if (g_queue) return;
    // Select GPU device; fall back to CPU if no GPU found
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (...) {
        log_warn("[sycl] no GPU found — falling back to CPU device");
        dev = sycl::device(sycl::cpu_selector_v);
    }
    g_queue = std::make_unique<sycl::queue>(dev,
                  sycl::property::queue::in_order{}); // serialised queue

    log_info(strfmt("[sycl] device: %s  vendor: %s  max CUs: %d",
                    dev.get_info<sycl::info::device::name>().c_str(),
                    dev.get_info<sycl::info::device::vendor>().c_str(),
                    dev.get_info<sycl::info::device::max_compute_units>()));
}

void finalize() { g_queue.reset(); }

// ── GEMM (FP32) via oneMKL ──────────────────────────────────────────────────
void gemm_fp32(bool trans_a, bool trans_b,
               int M, int N, int K, float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    using namespace oneapi::mkl;
    auto opA = trans_a ? transpose::trans : transpose::nontrans;
    auto opB = trans_b ? transpose::trans : transpose::nontrans;
    // oneMKL blas::gemm: C = alpha*A*B + beta*C (row-major via col-major swap)
    blas::gemm(queue(), opB, opA, N, M, K,
               alpha, B, ldb, A, lda, beta, C, ldc);
    queue().wait();
}

// ── GEMM (FP16) via oneMKL ──────────────────────────────────────────────────
void gemm_fp16(int M, int N, int K, float alpha,
               const sycl::half* A, const sycl::half* B,
               float beta, float* C) {
    using namespace oneapi::mkl;
    // oneMKL gemm_ext supports mixed-precision: half inputs, float output
    blas::gemm(queue(), transpose::trans, transpose::nontrans, N, M, K,
               alpha, B, K, A, K, beta, C, N);
    queue().wait();
}

// ── Flash Attention (tiled, SYCL nd_range) ───────────────────────────────────
// Work-group = one (batch, head): sub-groups process one query token at a time.
static constexpr int FA_WG = 64;  // sub-group / work-group size (Intel EU: 8/16/32)

void attention_forward_sycl(
        const float* q, const float* k, const float* v, float* out,
        int B, int T, int H, int KV, int hd, float scale)
{
    const int group = H / KV;
    sycl::range<2> global{static_cast<size_t>(B * H),
                           static_cast<size_t>(T)};
    sycl::range<2> local{1, FA_WG};  // 1 (b,h) × 64 threads per token

    // Each work-item handles: one query token q[tq], iterates over KV
    queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global * local, local),
            [=](sycl::nd_item<2> it) {
                const int bh  = it.get_global_id(0);
                const int tq  = it.get_global_id(1);
                const int b   = bh / H;
                const int h   = bh % H;
                const int kvh = h / group;
                if (b >= B || tq >= T) return;

                const size_t stride_q  = static_cast<size_t>(T) * H  * hd;
                const size_t stride_kv = static_cast<size_t>(T) * KV * hd;

                const float* qh = q   + b * stride_q  + (tq * H + h) * hd;
                const float* kh = k   + b * stride_kv + kvh * T * hd;
                const float* vh = v   + b * stride_kv + kvh * T * hd;
                float*       oh = out + b * stride_q  + (tq * H + h) * hd;

                // Online softmax
                float m = -1e30f, lse = 0.f;
                // local accumulator (stack-allocated, stays in registers for hd≤128)
                float acc[128] = {};
                for (int d = 0; d < hd && d < 128; ++d) acc[d] = 0.f;

                for (int j = 0; j <= tq; ++j) {
                    float dot = 0.f;
                    for (int d = 0; d < hd; ++d) dot += qh[d] * kh[j * hd + d];
                    dot *= scale;
                    float m_new = sycl::fmax(m, dot);
                    float alpha_v = sycl::exp(m - m_new);
                    lse = lse * alpha_v + sycl::exp(dot - m_new);
                    for (int d = 0; d < hd; ++d)
                        acc[d] = acc[d] * alpha_v + sycl::exp(dot - m_new) * vh[j * hd + d];
                    m = m_new;
                }
                for (int d = 0; d < hd; ++d) oh[d] = acc[d] / lse;
            });
    });
    queue().wait();
}

// ── Fused SwiGLU FFN (gate · SiLU(up) → down projection) ───────────────────
// Uses oneDNN for fused MatMul + elementwise on Intel GPU
void fused_ffn_swiglu(const float* x, const float* w_gate, const float* w_up,
                      const float* w_down, float* out,
                      int M, int K, int E, int d_out) {
    // For now: sequential matmuls (future: replace with oneDNN fusion API)
    // gate = x @ w_gate.T  [M, E]
    // up   = x @ w_up.T    [M, E]
    // act[i] = gate[i] * sigmoid(gate[i]) * up[i]  (SwiGLU)
    // out  = act @ w_down.T [M, d_out]

    std::vector<float> gate(static_cast<size_t>(M) * E);
    std::vector<float> up_v(static_cast<size_t>(M) * E);
    std::vector<float> act(static_cast<size_t>(M) * E);

    gemm_fp32(false, true, M, E, K, 1.f, x, K, w_gate, K, 0.f, gate.data(), E);
    gemm_fp32(false, true, M, E, K, 1.f, x, K, w_up,   K, 0.f, up_v.data(), E);

    for (size_t i = 0; i < static_cast<size_t>(M) * E; ++i) {
        const float g = gate[i];
        act[i] = g / (1.f + std::exp(-g)) * up_v[i]; // SwiGLU
    }

    gemm_fp32(false, true, M, d_out, E, 1.f, act.data(), E, w_down, E, 0.f, out, d_out);
}

} // namespace sycl_backend
} // namespace gai

#endif // GAI_SYCL
