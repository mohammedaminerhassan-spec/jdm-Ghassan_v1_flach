// Regression test for native quantized weight storage.
//
// A packed Q4_0 / Q4_1 / Q8_0 weight is dequantized into the shared FP16
// scratch bank by ops::quant_materialize(), then consumed by the same
// linear_forward_fp16 path used at inference. This test verifies that the
// CPU dequantization (a mirror of the CUDA kernels in cuda/kernels.cu)
// reproduces an FP32 reference GEMM within the expected storage error.

#include "core/quant.h"
#include "core/ops.h"
#include "core/dtype.h"
#include "quantization/quantize.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace gai;

int main() {
    const int M = 4, K = 128, N = 64;
    Tensor wf({static_cast<i64>(K), static_cast<i64>(N)}, DType::F32, Device::CPU);
    Tensor x ({static_cast<i64>(M), static_cast<i64>(K)}, DType::F32, Device::CPU);
    for (i64 i = 0; i < wf.numel(); ++i)
        wf.f32()[i] = 0.9f * std::sin(static_cast<float>(i) * 0.31f) *
                      std::cos(static_cast<float>(i) * 0.07f);
    for (i64 i = 0; i < x.numel(); ++i)
        x.f32()[i] = 0.5f * std::sin(static_cast<float>(i) * 0.17f);

    Tensor y_ref({static_cast<i64>(M), static_cast<i64>(N)}, DType::F32, Device::CPU);
    Tensor y_q  ({static_cast<i64>(M), static_cast<i64>(N)}, DType::F32, Device::CPU);
    ops::linear_forward(Device::CPU, x.f32(), wf.f32(), y_ref.f32(), M, K, N);

    const DType kinds[] = {DType::Q4_0, DType::Q4_1, DType::Q8_0};
    const double limits[] = {0.12, 0.10, 0.02};
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
        Tensor wq = quant::quantize(wf, kinds[k]);
        const u16* hw = ops::quant_materialize(Device::CPU, wq.data_ptr(),
                                               quant_kind(kinds[k]), wq.numel(), 0);

        // Strong check: our dequantizer must match the library reference
        // byte-for-byte up to FP16 rounding (an nibble-order bug would show
        // up here as a large error, not as ordinary quantization noise).
        Tensor wref = quant::dequantize(wq, DType::F32);
        float worst = 0.0f;
        for (i64 i = 0; i < wq.numel(); ++i) {
            const float v = fp16_to_fp32(hw[i]);
            worst = std::max(worst, std::fabs(v - wref.f32()[i]));
        }
        std::printf("[quant] %s dequant vs reference worst |diff| = %.6f\n",
                    dtype_name(kinds[k]), worst);
        if (worst > 1e-3f) ok = false;

        ops::linear_forward_fp16(Device::CPU, x.f32(), hw, y_q.f32(), M, K, N);
        double num = 0.0, den = 0.0;
        for (i64 i = 0; i < y_ref.numel(); ++i) {
            const double d = y_q.f32()[i] - y_ref.f32()[i];
            num += d * d;
            den += static_cast<double>(y_ref.f32()[i]) * y_ref.f32()[i];
        }
        const double rel = std::sqrt(num / std::max(den, 1e-12));
        std::printf("[quant] %s relative GEMM error = %.4f (limit %.2f)\n",
                    dtype_name(kinds[k]), rel, limits[k]);
        if (rel > limits[k]) ok = false;
    }
    ops::quant_scratch_free();
    if (!ok) {
        std::printf("FAIL: native Q4 dequantized GEMM deviates from FP32 reference\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
