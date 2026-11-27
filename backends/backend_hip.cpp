// ══════════════════════════════════════════════════════════════════════════════
// backend_hip.cpp — AMD GPU (ROCm/HIP) backend implementation
// ══════════════════════════════════════════════════════════════════════════════
#ifdef GAI_HIP

#include "backends/backend_hip.h"
#include "core/common.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gai {
namespace hip_backend {

static hipblasHandle_t g_hipblas = nullptr;

hipblasHandle_t& hipblas_handle() { return g_hipblas; }

void init() {
    if (g_hipblas) return;
    HIPBLAS_CHECK(hipblasCreate(&g_hipblas));
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    log_info(strfmt("[hip] device: %s  gcnArchName: %s  VRAM: %.1f GB",
                    prop.name, prop.gcnArchName,
                    prop.totalGlobalMem / 1e9));
}

void finalize() {
    if (g_hipblas) { hipblasDestroy(g_hipblas); g_hipblas = nullptr; }
}

void gemm_fp32(bool trans_a, bool trans_b,
               int M, int N, int K, float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc) {
    hipblasOperation_t opA = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    // hipBLAS is column-major; swap A↔B and M↔N
    HIPBLAS_CHECK(hipblasSgemm(g_hipblas,
        opB, opA,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc));
}

void gemm_fp16(int M, int N, int K, float alpha,
               const __half* A, const __half* B,
               float beta, float* C) {
    // hipBLAS HGEMM: accumulate in FP32 via hipblasSgemmEx
    HIPBLAS_CHECK(hipblasSgemmEx(g_hipblas,
        HIPBLAS_OP_T, HIPBLAS_OP_N,
        N, M, K,
        &alpha,
        B, HIPBLAS_R_16F, K,
        A, HIPBLAS_R_16F, K,
        &beta,
        C, HIPBLAS_R_32F, N));
}

// ── Attention kernel: warp-parallel softmax (warpSize=64 on RDNA) ────────────
// Each wavefront (64 threads) handles one query token.
__global__ void k_flash_attn_hip(
        const float* __restrict__ q,   // [B, H, T, hd]
        const float* __restrict__ k_,  // [B, KV, T, hd]
        const float* __restrict__ v_,  // [B, KV, T, hd]
        float*       __restrict__ out, // [B, H, T, hd]
        int T, int H, int KV, int hd, float scale, int B)
{
    // One block = one (batch, head, query_token)
    const int bh  = blockIdx.x;
    const int tq  = blockIdx.y;
    const int b   = bh / H;
    const int h   = bh % H;
    const int kvh = h / (H / KV);
    const int tid = threadIdx.x; // 0..63 on RDNA (wavefront=64)

    if (b >= B || tq >= T) return;

    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

    const float* qh = q   + b * qs  + (static_cast<size_t>(tq) * H  + h)   * hd;
    const float* kh = k_  + b * kvs + static_cast<size_t>(kvh) * T * hd;
    const float* vh = v_  + b * kvs + static_cast<size_t>(kvh) * T * hd;
    float*       oh = out + b * qs  + (static_cast<size_t>(tq) * H  + h)   * hd;

    // Online softmax per token (causal)
    float m = -1e30f, lse = 0.f;
    // Accumulate output in registers (hd <= 128 typical)
    float acc[128] = {};

    for (int j = 0; j <= tq; ++j) {
        // dot(q, k[j]) — each thread handles one dimension
        float dot = 0.f;
        for (int d = tid; d < hd; d += WARP_SIZE)
            dot += qh[d] * kh[j * hd + d];
        // Warp reduction
        for (int off = WARP_SIZE/2; off >= 1; off >>= 1)
            dot += __shfl_down(dot, off);
        dot = __shfl(dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = expf(m - m_new);
        lse  = lse * alpha + expf(dot - m_new);
        for (int d = tid; d < hd; d += WARP_SIZE)
            acc[d] = acc[d] * alpha + expf(dot - m_new) * vh[j * hd + d];
        m = m_new;
    }

    // Write normalised output
    for (int d = tid; d < hd; d += WARP_SIZE)
        oh[d] = acc[d] / lse;
}

void attention_forward_hip(
        const float* q, const float* k, const float* v, float* out,
        int B, int T, int H, int KV, int hd, float scale)
{
    dim3 grid(B * H, T);
    dim3 block(WARP_SIZE); // one wavefront per token
    hipLaunchKernelGGL(k_flash_attn_hip, grid, block, 0, 0,
                       q, k, v, out, T, H, KV, hd, scale, B);
    HIP_CHECK(hipGetLastError());
}

} // namespace hip_backend
} // namespace gai

#endif // GAI_HIP
