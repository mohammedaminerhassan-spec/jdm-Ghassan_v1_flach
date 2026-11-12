#pragma once

// The SIMD helpers shared by the CPU kernels (the GEMM, the attention, the
// MoE). They live in a header because several translation units use them:
// every one of them is `inline`, so there is a single definition.

#include <cstdlib>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(_MSC_VER)
#define GAI_HAVE_AVX2_DISPATCH 1
#include <immintrin.h>
namespace gai {
namespace cpu {
namespace simd {

inline bool supported() {
    static const bool v = [] {
        const char* e = std::getenv("GAI_NO_SIMD");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return v;
}

__attribute__((target("avx2,fma"))) inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    const __m128 lo = _mm256_castps256_ps128(s);
    const __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 h = _mm_add_ps(lo, hi);
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float out = _mm_cvtss_f32(h);
    for (; i < n; ++i) out += a[i] * b[i];
    return out;
}

__attribute__((target("avx2,fma"))) inline void axpy_avx2(float* y, const float* x, float a, int n) {
    const __m256 va = _mm256_set1_ps(a);
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        y0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),      y0);
        y1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8),  y1);
        y2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 16), y2);
        y3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 24), y3);
        _mm256_storeu_ps(y + i,      y0);
        _mm256_storeu_ps(y + i + 8,  y1);
        _mm256_storeu_ps(y + i + 16, y2);
        _mm256_storeu_ps(y + i + 24, y3);
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

}  // namespace simd
}  // namespace cpu
}  // namespace gai
#else
#define GAI_HAVE_AVX2_DISPATCH 0
#endif

namespace gai {
namespace cpu {

// The dot product of two rows, with an AVX2/FMA path when the CPU has it.
inline float dot_f32(const float* a, const float* b, int n) {
#if GAI_HAVE_AVX2_DISPATCH
    if (n >= 32 && simd::supported()) return simd::dot_avx2(a, b, n);
#endif
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// y += a * x
inline void axpy_f32(float* y, const float* x, float a, int n) {
#if GAI_HAVE_AVX2_DISPATCH
    if (n >= 32 && simd::supported()) {
        simd::axpy_avx2(y, x, a, n);
        return;
    }
#endif
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        y[i]     += a * x[i];
        y[i + 1] += a * x[i + 1];
        y[i + 2] += a * x[i + 2];
        y[i + 3] += a * x[i + 3];
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

}  // namespace cpu
}  // namespace gai
