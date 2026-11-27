// ══════════════════════════════════════════════════════════════════════════════
// backend_hip.h — AMD GPU (ROCm/HIP) backend abstraction
// ══════════════════════════════════════════════════════════════════════════════
// Compile with: hipcc or clang++ --offload-arch=gfx1100 (RDNA3)
//               or amdclang++ (ROCm 5.x+)
//
// CMake flag: -DGAI_ENABLE_HIP=ON
// ══════════════════════════════════════════════════════════════════════════════
#pragma once

#ifdef GAI_HIP

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "core/common.h"

#define HIP_CHECK(x) do {                                                     \
    hipError_t _e = (x);                                                      \
    if (_e != hipSuccess)                                                     \
        GAI_FAIL(std::string("HIP ") + #x + ": " + hipGetErrorString(_e));   \
} while (0)

#define HIPBLAS_CHECK(x) do {                                                  \
    hipblasStatus_t _s = (x);                                                  \
    if (_s != HIPBLAS_STATUS_SUCCESS)                                          \
        GAI_FAIL(std::string("hipBLAS ") + #x + " failed (status=" +          \
                 std::to_string(static_cast<int>(_s)) + ")");                  \
} while (0)

namespace gai {
namespace hip_backend {

// ── Global handle (one per process) ─────────────────────────────────────────
hipblasHandle_t& hipblas_handle();
void             init();
void             finalize();

// ── Memory helpers mirroring CUDA backend ───────────────────────────────────
inline void* alloc(size_t nbytes) {
    void* p = nullptr;
    HIP_CHECK(hipMalloc(&p, nbytes));
    return p;
}
inline void free_ptr(void* p) { HIP_CHECK(hipFree(p)); }

inline void memcpy_h2d(void* dst, const void* src, size_t n) {
    HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
}
inline void memcpy_d2h(void* dst, const void* src, size_t n) {
    HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyDeviceToHost));
}
inline void memcpy_d2d(void* dst, const void* src, size_t n) {
    HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice));
}
inline void memset_zero(void* p, size_t n) { HIP_CHECK(hipMemset(p, 0, n)); }
inline void sync() { HIP_CHECK(hipDeviceSynchronize()); }

// ── GEMM: hipBLAS SGEMM (FP32) ──────────────────────────────────────────────
// Wraps hipblasSgemm with the same interface as the CUDA cuBLAS wrapper.
void gemm_fp32(bool trans_a, bool trans_b,
               int M, int N, int K, float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta, float* C, int ldc);

// ── GEMM: hipBLAS HGEMM (FP16, uses hipblasSgemmEx internally) ──────────────
void gemm_fp16(int M, int N, int K, float alpha,
               const __half* A, const __half* B,
               float beta, float* C);

// ── RoCm-specific: warpSize = 64 on GCN/RDNA ────────────────────────────────
// (vs. 32 on NVIDIA) — kernels must be aware of this.
static constexpr int WARP_SIZE = 64;

} // namespace hip_backend
} // namespace gai

#endif // GAI_HIP
