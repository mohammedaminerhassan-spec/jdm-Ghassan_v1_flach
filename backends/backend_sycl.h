// ══════════════════════════════════════════════════════════════════════════════
// backend_sycl.h — Intel GPU (oneAPI/SYCL) backend
// ══════════════════════════════════════════════════════════════════════════════
// Compile with: icpx -fsycl -fsycl-targets=spir64_gen
//               -Xs "-device xe-hpg" (Intel Arc / Iris Xe)
//
// CMake flag: -DGAI_ENABLE_SYCL=ON
// Requires:   Intel oneAPI Base Toolkit (oneMKL, oneDNN, SYCL runtime)
// ══════════════════════════════════════════════════════════════════════════════
#pragma once

#ifdef GAI_SYCL

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "core/common.h"

namespace gai {
namespace sycl_backend {

// ── Global SYCL queue (GPU device) ──────────────────────────────────────────
sycl::queue& queue();
void         init();      // selects GPU device, prints info
void         finalize();

// ── Memory helpers ───────────────────────────────────────────────────────────
inline void* alloc(size_t nbytes) {
    return sycl::malloc_device(nbytes, queue());
}
inline void free_ptr(void* p) { sycl::free(p, queue()); }

inline void memcpy_h2d(void* dst, const void* src, size_t n) {
    queue().memcpy(dst, src, n).wait();
}
inline void memcpy_d2h(void* dst, const void* src, size_t n) {
    queue().memcpy(dst, src, n).wait();
}
inline void sync() { queue().wait(); }

// ── oneMKL GEMM (FP32) ───────────────────────────────────────────────────────
// Wraps oneapi::mkl::blas::gemm with the same semantics as cuBLAS wrapper.
void gemm_fp32(bool trans_a, bool trans_b,
               int M, int N, int K, float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

// ── oneMKL GEMM (FP16) ───────────────────────────────────────────────────────
// Accumulates in FP32 on Intel GPUs via compute_mode::float_to_bf16
void gemm_fp16(int M, int N, int K, float alpha,
               const sycl::half* A, const sycl::half* B,
               float beta, float* C);

// ── Tiled Flash Attention using SYCL nd_range ─────────────────────────────
void attention_forward_sycl(
        const float* q, const float* k, const float* v, float* out,
        int B, int T, int H, int KV, int hd, float scale);

// ── oneDNN helper: fused MatMul + ReLU/GELU for FFN ─────────────────────────
// Avoids separate kernel launches for gate and up projections.
void fused_ffn_swiglu(const float* x, const float* w_gate, const float* w_up,
                      const float* w_down, float* out,
                      int M, int K, int E, int d_out);

} // namespace sycl_backend
} // namespace gai

#endif // GAI_SYCL
