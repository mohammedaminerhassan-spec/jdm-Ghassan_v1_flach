#pragma once

#include "core/dtype.h"
#include <cstdlib>
#include <cstdint>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

// ── x86: AVX2+FMA (baseline) + AVX-512F + F16C dispatch ─────────────────────
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(_MSC_VER)
#define GAI_HAVE_AVX2_DISPATCH 1
#include <immintrin.h>

namespace gai {
namespace cpu {
namespace simd {

// ---- runtime feature detection ----
inline bool avx2_supported() {
    static const bool v = [] {
        const char* e = std::getenv("GAI_NO_SIMD");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return v;
}

inline bool f16c_supported() {
    static const bool v = [] {
        const char* e = std::getenv("GAI_NO_SIMD");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
            && __builtin_cpu_supports("f16c");
    }();
    return v;
}

inline bool avx512_supported() {
    static const bool v = [] {
        const char* e = std::getenv("GAI_NO_SIMD");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") != 0;
    }();
    return v;
}

// legacy alias
inline bool supported() { return avx2_supported(); }

// ── AVX2+FMA: float dot product — 32 elements/iter, 4 accumulators ───────────
__attribute__((target("avx2,fma")))
inline float dot_avx2(const float* a, const float* b, int n) {
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

// ── AVX2+FMA: axpy — y += a*x for 32 floats/iter ────────────────────────────
__attribute__((target("avx2,fma")))
inline void axpy_avx2(float* y, const float* x, float a, int n) {
    const __m256 va = _mm256_set1_ps(a);
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        _mm256_storeu_ps(y + i,      _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),      _mm256_loadu_ps(y + i)));
        _mm256_storeu_ps(y + i + 8,  _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8),  _mm256_loadu_ps(y + i + 8)));
        _mm256_storeu_ps(y + i + 16, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16)));
        _mm256_storeu_ps(y + i + 24, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24)));
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

// ── F16C: dot(f32_input, fp16_weights) — hardware FP16 decode ────────────────
// Critical path for inference with FP16 weights: NO separate dequant buffer!
// Uses F16C instruction set (Intel Ivy Bridge+, AMD Piledriver+)
__attribute__((target("avx2,fma,f16c")))
inline float dot_f16_avx2(const float* x, const uint16_t* w, int K) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    int i = 0;
    const int K16 = K & ~15;
    for (; i < K16; i += 16) {
        // _mm_loadu_si128 loads 128-bit = 8× fp16
        __m256 wf0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
        __m256 wf1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8)));
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),     wf0, s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), wf1, s1);
    }
    __m256 s = _mm256_add_ps(s0, s1);
    const __m128 lo = _mm256_castps256_ps128(s);
    const __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 h = _mm_add_ps(lo, hi);
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float out = _mm_cvtss_f32(h);
    for (; i < K; ++i) out += x[i] * fp16_to_fp32(w[i]);
    return out;
}

// ── AVX-512F: float dot — 32 floats/iter with 512-bit registers ──────────────
#if defined(__GNUC__) && __GNUC__ >= 7
__attribute__((target("avx512f")))
inline float dot_avx512(const float* a, const float* b, int n) {
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    float out = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    for (; i < n; ++i) out += a[i] * b[i];
    return out;
}

__attribute__((target("avx512f")))
inline void axpy_avx512(float* y, const float* x, float a, int n) {
    const __m512 va = _mm512_set1_ps(a);
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        _mm512_storeu_ps(y + i,      _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i),      _mm512_loadu_ps(y + i)));
        _mm512_storeu_ps(y + i + 16, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(y + i + 16)));
    }
    for (; i < n; ++i) y[i] += a * x[i];
}
#endif // __GNUC__ >= 7

} // namespace simd
} // namespace cpu
} // namespace gai

#else
// ── Non-x86 fallback ──────────────────────────────────────────────────────────
#define GAI_HAVE_AVX2_DISPATCH 0
#endif // x86 + GCC

namespace gai {
namespace cpu {

// ── dot_f32: auto-dispatch AVX-512 → AVX2 → scalar ─────────────────────────
inline float dot_f32(const float* a, const float* b, int n) {
#if GAI_HAVE_AVX2_DISPATCH
#if defined(__GNUC__) && __GNUC__ >= 7
    if (n >= 32 && simd::avx512_supported()) return simd::dot_avx512(a, b, n);
#endif
    if (n >= 32 && simd::avx2_supported())   return simd::dot_avx2(a, b, n);
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

// ── dot_f32_f16: dot(f32 input, fp16 weight) — zero-copy, F16C if available ─
// This replaces the old "dequant entire matrix then gemm" pattern.
inline float dot_f32_f16(const float* x, const uint16_t* w, int K) {
#if GAI_HAVE_AVX2_DISPATCH
    if (K >= 16 && simd::f16c_supported()) return simd::dot_f16_avx2(x, w, K);
#endif
    // Scalar fallback
    float s = 0.f;
    for (int i = 0; i < K; ++i) s += x[i] * fp16_to_fp32(w[i]);
    return s;
}

// ── axpy_f32: auto-dispatch AVX-512 → AVX2 → scalar ─────────────────────────
inline void axpy_f32(float* y, const float* x, float a, int n) {
#if GAI_HAVE_AVX2_DISPATCH
#if defined(__GNUC__) && __GNUC__ >= 7
    if (n >= 32 && simd::avx512_supported()) { simd::axpy_avx512(y, x, a, n); return; }
#endif
    if (n >= 32 && simd::avx2_supported())   { simd::axpy_avx2(y, x, a, n);   return; }
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

} // namespace cpu
} // namespace gai
