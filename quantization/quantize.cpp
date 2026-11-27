#include "quantization/quantize.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace quant {

void f32_to_f16(const float* s, u16* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = fp32_to_fp16(s[i]); }
void f16_to_f32(const u16* s, float* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = fp16_to_fp32(s[i]); }
void f32_to_bf16(const float* s, u16* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = fp32_to_bf16(s[i]); }
void bf16_to_f32(const u16* s, float* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = bf16_to_fp32(s[i]); }

void quantize_q8_0(const float* src, void* dst, i64 n) {
    GAI_CHECK(n >= 0, "q8_0 element count must be non-negative");
    BlockQ8_0* out = static_cast<BlockQ8_0*>(dst);
    const i64 full = n / Q8_BLOCK;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if (full > 1024)
#endif
    for (i64 b = 0; b < full; ++b) {
        const float* x = src + b * Q8_BLOCK;
        bool bad = false;
        float amax = 0.0f;
        for (int i = 0; i < Q8_BLOCK; ++i) {
            if (!std::isfinite(x[i])) { bad = true; break; }
            const float a = std::fabs(x[i]);
            if (a > amax) amax = a;
        }
        if (bad) {
            out[b].scale = fp32_to_fp16(0.0f);
            std::memset(out[b].q, 0, sizeof(out[b].q));
            continue;
        }
        float scale = amax / 127.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[b].scale = fp32_to_fp16(scale);
        for (int i = 0; i < Q8_BLOCK; ++i) {
            int q = static_cast<int>(std::lround(x[i] * inv));
            out[b].q[i] = static_cast<i8>(std::clamp(q, -127, 127));
        }
    }
    const i64 rem = n % Q8_BLOCK;
    if (rem > 0) {
        const float* x = src + full * Q8_BLOCK;
        bool bad = false;
        float amax = 0.0f;
        for (i64 i = 0; i < rem; ++i) {
            if (!std::isfinite(x[i])) { bad = true; break; }
            const float a = std::fabs(x[i]);
            if (a > amax) amax = a;
        }
        if (bad) {
            out[full].scale = fp32_to_fp16(0.0f);
            std::memset(out[full].q, 0, sizeof(out[full].q));
            return;
        }
        float scale = amax / 127.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[full].scale = fp32_to_fp16(scale);
        for (i64 i = 0; i < rem; ++i) {
            int q = static_cast<int>(std::lround(x[i] * inv));
            out[full].q[i] = static_cast<i8>(std::clamp(q, -127, 127));
        }
        for (i64 i = rem; i < Q8_BLOCK; ++i) out[full].q[i] = 0;
    }
}

void dequantize_q8_0(const void* src, float* dst, i64 n) {
    GAI_CHECK(n >= 0, "q8_0 element count must be non-negative");
    const BlockQ8_0* in = static_cast<const BlockQ8_0*>(src);
    const i64 full = n / Q8_BLOCK;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if (full > 1024)
#endif
    for (i64 b = 0; b < full; ++b) {
        float s = fp16_to_fp32(in[b].scale);
        float* o = dst + b * Q8_BLOCK;
        for (int i = 0; i < Q8_BLOCK; ++i) o[i] = static_cast<float>(in[b].q[i]) * s;
    }
    const i64 rem = n % Q8_BLOCK;
    if (rem > 0) {
        float tmp[Q8_BLOCK];
        float s = fp16_to_fp32(in[full].scale);
        for (int i = 0; i < Q8_BLOCK; ++i) tmp[i] = static_cast<float>(in[full].q[i]) * s;
        for (i64 i = 0; i < rem; ++i) dst[full * Q8_BLOCK + i] = tmp[i];
    }
}

void quantize_q4_0(const float* src, void* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_0 element count must be non-negative");
    BlockQ4_0* out = static_cast<BlockQ4_0*>(dst);
    const i64 full = n / Q4_BLOCK;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if (full > 1024)
#endif
    for (i64 b = 0; b < full; ++b) {
        const float* x = src + b * Q4_BLOCK;
        bool bad = false;
        float amax = 0.0f;
        float vmax = 0.0f;
        for (int i = 0; i < Q4_BLOCK; ++i) {
            if (!std::isfinite(x[i])) { bad = true; break; }
            const float a = std::fabs(x[i]);
            if (a > amax) {
                amax = a;
                vmax = x[i];
            }
        }
        if (bad || amax == 0.0f) {
            out[b].scale = fp32_to_fp16(0.0f);
            std::memset(out[b].q, 0x88, sizeof(out[b].q));
            continue;
        }
        const float d = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        out[b].scale = fp32_to_fp16(d);
        for (int j = 0; j < Q4_BLOCK / 2; ++j) {
            const int q0 = std::max(0, std::min(15, static_cast<int>(x[j] * id + 8.5f)));
            const int q1 = std::max(0, std::min(15, static_cast<int>(x[j + Q4_BLOCK / 2] * id + 8.5f)));
            out[b].q[j] = static_cast<u8>((q1 << 4) | q0);
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        const float* x = src + full * Q4_BLOCK;
        bool bad = false;
        float amax = 0.0f;
        float vmax = 0.0f;
        for (i64 i = 0; i < rem; ++i) {
            if (!std::isfinite(x[i])) { bad = true; break; }
            const float a = std::fabs(x[i]);
            if (a > amax) {
                amax = a;
                vmax = x[i];
            }
        }
        if (bad || amax == 0.0f) {
            out[full].scale = fp32_to_fp16(0.0f);
            std::memset(out[full].q, 0x88, sizeof(out[full].q));
            return;
        }
        const float d = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        out[full].scale = fp32_to_fp16(d);
        for (int j = 0; j < Q4_BLOCK / 2; ++j) {
            int q0 = 8, q1 = 8;
            if (j < rem)
                q0 = std::max(0, std::min(15, static_cast<int>(x[j] * id + 8.5f)));
            if (j + Q4_BLOCK / 2 < rem)
                q1 = std::max(0, std::min(15, static_cast<int>(x[j + Q4_BLOCK / 2] * id + 8.5f)));
            out[full].q[j] = static_cast<u8>((q1 << 4) | q0);
        }
    }
}

void dequantize_q4_0(const void* src, float* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_0 element count must be non-negative");
    const BlockQ4_0* in = static_cast<const BlockQ4_0*>(src);
    const i64 full = n / Q4_BLOCK;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if (full > 1024)
#endif
    for (i64 b = 0; b < full; ++b) {
        float s = fp16_to_fp32(in[b].scale);
        float* o = dst + b * Q4_BLOCK;
        for (int j = 0; j < Q4_BLOCK / 2; ++j) {
            u8 byte = in[b].q[j];
            o[j]                  = (static_cast<float>(byte & 0x0F) - 8.0f) * s;
            o[j + Q4_BLOCK / 2]   = (static_cast<float>(byte >> 4)   - 8.0f) * s;
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        float tmp[Q4_BLOCK];
        float s = fp16_to_fp32(in[full].scale);
        for (int j = 0; j < Q4_BLOCK / 2; ++j) {
            u8 byte = in[full].q[j];
            tmp[j]                  = (static_cast<float>(byte & 0x0F) - 8.0f) * s;
            tmp[j + Q4_BLOCK / 2]   = (static_cast<float>(byte >> 4)   - 8.0f) * s;
        }
        for (i64 i = 0; i < rem; ++i) dst[full * Q4_BLOCK + i] = tmp[i];
    }
}

void quantize_q4_1(const float* src, void* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_1 element count must be non-negative");
    BlockQ4_1* out = static_cast<BlockQ4_1*>(dst);
    const i64 full = n / Q4_BLOCK;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if (full > 1024)
#endif
    for (i64 b = 0; b < full; ++b) {
        const float* x = src + b * Q4_BLOCK;

        bool bad = false;
        for (int i = 0; i < Q4_BLOCK; ++i)
            if (!std::isfinite(x[i])) { bad = true; break; }
        if (bad) {
            out[b].scale = fp32_to_fp16(0.0f);
            out[b].min = fp32_to_fp16(0.0f);
            std::memset(out[b].q, 0, sizeof(out[b].q));
            continue;
        }
        float mn = x[0], mx = x[0];
        for (int i = 1; i < Q4_BLOCK; ++i) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
        float scale = (mx - mn) / 15.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[b].scale = fp32_to_fp16(scale);
        out[b].min   = fp32_to_fp16(mn);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            int q0 = std::clamp(static_cast<int>(std::lround((x[i]        - mn) * inv)), 0, 15);
            int q1 = std::clamp(static_cast<int>(std::lround((x[i + 16]   - mn) * inv)), 0, 15);
            out[b].q[i] = static_cast<u8>((q1 << 4) | q0);
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        const float* x = src + full * Q4_BLOCK;

        bool bad = false;
        for (i64 i = 0; i < rem; ++i)
            if (!std::isfinite(x[i])) { bad = true; break; }
        if (bad) {
            out[full].scale = fp32_to_fp16(0.0f);
            out[full].min = fp32_to_fp16(0.0f);
            std::memset(out[full].q, 0, sizeof(out[full].q));
            return;
        }
        float mn = x[0], mx = x[0];
        for (i64 i = 1; i < rem; ++i) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
        float scale = (mx - mn) / 15.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[full].scale = fp32_to_fp16(scale);
        out[full].min   = fp32_to_fp16(mn);
        const int qz = std::clamp(static_cast<int>(std::lround((0.0f - mn) * inv)), 0, 15);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            const i64 i0 = static_cast<i64>(i), i1 = i0 + Q4_BLOCK / 2;
            int q0 = (i0 < rem ? std::clamp(static_cast<int>(std::lround((x[i0] - mn) * inv)), 0, 15) : qz);
            int q1 = (i1 < rem ? std::clamp(static_cast<int>(std::lround((x[i1] - mn) * inv)), 0, 15) : qz);
            out[full].q[i] = static_cast<u8>((q1 << 4) | q0);
        }
    }
}

void dequantize_q4_1(const void* src, float* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_1 element count must be non-negative");
    const BlockQ4_1* in = static_cast<const BlockQ4_1*>(src);
    const i64 full = n / Q4_BLOCK;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if (full > 1024)
#endif
    for (i64 b = 0; b < full; ++b) {
        float s = fp16_to_fp32(in[b].scale);
        float m = fp16_to_fp32(in[b].min);
        float* o = dst + b * Q4_BLOCK;
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            u8 byte = in[b].q[i];
            o[i]         = static_cast<float>(byte & 0x0F) * s + m;
            o[i + Q4_BLOCK / 2] = static_cast<float>(byte >> 4) * s + m;
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        float tmp[Q4_BLOCK];
        float s = fp16_to_fp32(in[full].scale);
        float m = fp16_to_fp32(in[full].min);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            u8 byte = in[full].q[i];
            tmp[i]         = static_cast<float>(byte & 0x0F) * s + m;
            tmp[i + Q4_BLOCK / 2] = static_cast<float>(byte >> 4) * s + m;
        }
        for (i64 i = 0; i < rem; ++i) dst[full * Q4_BLOCK + i] = tmp[i];
    }
}

Tensor quantize(const Tensor& src, DType target) {
    GAI_CHECK(src.dtype() == DType::F32, "quantize expects an f32 source");
    GAI_CHECK(src.device() == Device::CPU, "quantize runs on CPU tensors");
    if (target == DType::F32) return src.clone();

    Tensor out(src.shape(), target, Device::CPU);
    const float* s = src.f32();
    const i64 n = src.numel();

    switch (target) {
        case DType::F16:  f32_to_f16(s, out.ptr<u16>(), n); break;
        case DType::BF16: f32_to_bf16(s, out.ptr<u16>(), n); break;
        case DType::Q8_0: quantize_q8_0(s, out.data_ptr(), n); break;
        case DType::Q4_0: quantize_q4_0(s, out.data_ptr(), n); break;
        case DType::Q4_1: quantize_q4_1(s, out.data_ptr(), n); break;
        default: GAI_FAIL(std::string("unsupported quantization target: ") + dtype_name(target));
    }
    out.set_name(src.name());
    return out;
}

Tensor dequantize(const Tensor& src, DType target) {
    GAI_CHECK(target == DType::F32, "dequantize currently targets f32 only");
    GAI_CHECK(src.device() == Device::CPU, "dequantize runs on CPU tensors");
    if (src.dtype() == DType::F32) return src.clone();

    Tensor out(src.shape(), DType::F32, Device::CPU);
    float* d = out.f32();
    const i64 n = src.numel();

    switch (src.dtype()) {
        case DType::F16:  f16_to_f32(src.ptr<u16>(), d, n); break;
        case DType::BF16: bf16_to_f32(src.ptr<u16>(), d, n); break;
        case DType::Q8_0: dequantize_q8_0(src.data_ptr(), d, n); break;
        case DType::Q4_0: dequantize_q4_0(src.data_ptr(), d, n); break;
        case DType::Q4_1: dequantize_q4_1(src.data_ptr(), d, n); break;
        default: GAI_FAIL(std::string("unsupported dequantization source: ") + dtype_name(src.dtype()));
    }
    out.set_name(src.name());
    return out;
}



}
}
