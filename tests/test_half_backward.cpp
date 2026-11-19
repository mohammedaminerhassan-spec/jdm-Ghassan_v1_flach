// Regression test for the FP16-weight-storage backward bug.
//
// Before the fix, callers passed a half weight pointer through Tensor::f32()
// (a plain reinterpret_cast), so the GEMM read the u16 bit patterns as floats
// and produced garbage gradients. linear_backward_half() must agree with the
// FP32 reference up to the storage precision of FP16.

#include "core/ops.h"
#include "core/dtype.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace gai;

int main() {
    const int M = 7, K = 11, N = 5;
    std::vector<float> x(static_cast<size_t>(M) * K);
    std::vector<float> dy(static_cast<size_t>(M) * N);
    std::vector<float> w(static_cast<size_t>(K) * N);
    for (size_t i = 0; i < x.size(); ++i)   x[i]  = 0.01f * static_cast<float>((i * 7) % 13) - 0.03f;
    for (size_t i = 0; i < dy.size(); ++i)  dy[i] = 0.02f * static_cast<float>((i * 5) % 11) - 0.05f;
    for (size_t i = 0; i < w.size(); ++i)   w[i]  = 0.015f * static_cast<float>((i * 3) % 17) - 0.12f;

    std::vector<u16> wh(w.size());
    for (size_t i = 0; i < w.size(); ++i) wh[i] = fp32_to_fp16(w[i]);

    std::vector<float> dx_ref(x.size(), 7.0f), dx_half(x.size(), 7.0f);
    std::vector<float> dw_ref(w.size(), 0.0f), dw_half(w.size(), 0.0f);

    ops::linear_backward(Device::CPU, x.data(), w.data(), dy.data(),
                         dx_ref.data(), dw_ref.data(), M, K, N);
    ops::linear_backward_half(Device::CPU, x.data(), wh.data(), dy.data(),
                              dx_half.data(), dw_half.data(), M, K, N);

    float worst_dx = 0.0f, worst_dw = 0.0f;
    for (size_t i = 0; i < dx_ref.size(); ++i)
        worst_dx = std::max(worst_dx, std::fabs(dx_ref[i] - dx_half[i]));
    for (size_t i = 0; i < dw_ref.size(); ++i)
        worst_dw = std::max(worst_dw, std::fabs(dw_ref[i] - dw_half[i]));

    std::printf("[half-backward] worst |dx_ref-dx_half| = %.6f, worst |dw_ref-dw_half| = %.6f\n",
                worst_dx, worst_dw);
    // dw does not depend on the weight values and must be bit-close; dx uses
    // half weights, so a small storage-rounding difference is expected.
    if (worst_dw > 1e-5f || worst_dx > 5e-3f) {
        std::printf("FAIL: half-stored backward does not match the FP32 reference\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
