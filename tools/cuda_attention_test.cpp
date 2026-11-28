#include "core/ops.h"
#include "core/device.h"
#include "core/dtype.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) {
        cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
};

struct AttentionCase {
    int B;
    int Tq;
    int Tk;
    int H;
    int KV;
    int hd;
    int q_pos_offset;
    int window;
};

void run_case(const AttentionCase& test, bool kv_fp16, unsigned seed) {
    const size_t q_count = static_cast<size_t>(test.B) * test.Tq * test.H * test.hd;
    const size_t kv_count = static_cast<size_t>(test.B) * test.Tk * test.KV * test.hd;
    const size_t out_count = q_count;

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.35f);
    std::vector<float> q(q_count);
    std::vector<float> k32(kv_count);
    std::vector<float> v32(kv_count);
    for (float& value : q) value = normal(rng);
    for (float& value : k32) value = normal(rng);
    for (float& value : v32) value = normal(rng);

    std::vector<gai::u16> k16;
    std::vector<gai::u16> v16;
    const void* host_k = k32.data();
    const void* host_v = v32.data();
    size_t kv_bytes = kv_count * sizeof(float);
    if (kv_fp16) {
        k16.resize(kv_count);
        v16.resize(kv_count);
        for (size_t i = 0; i < kv_count; ++i) {
            k16[i] = gai::fp32_to_fp16(k32[i]);
            v16[i] = gai::fp32_to_fp16(v32[i]);
        }
        host_k = k16.data();
        host_v = v16.data();
        kv_bytes = kv_count * sizeof(gai::u16);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(test.hd));
    std::vector<float> reference(out_count, 0.0f);
    std::vector<float> actual(out_count, 0.0f);
    gai::ops::attention_forward_kv_ex(
        gai::Device::CPU, q.data(), host_k, host_v, kv_fp16, reference.data(),
        test.B, test.Tq, test.Tk, test.H, test.KV, test.hd,
        test.q_pos_offset, scale, test.window);

    DeviceBuffer q_dev(q.size() * sizeof(float));
    DeviceBuffer k_dev(kv_bytes);
    DeviceBuffer v_dev(kv_bytes);
    DeviceBuffer out_dev(actual.size() * sizeof(float));
    cuda_check(cudaMemcpy(q_dev.get(), q.data(), q.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy Q to device");
    cuda_check(cudaMemcpy(k_dev.get(), host_k, kv_bytes,
                          cudaMemcpyHostToDevice), "copy K to device");
    cuda_check(cudaMemcpy(v_dev.get(), host_v, kv_bytes,
                          cudaMemcpyHostToDevice), "copy V to device");

    gai::ops::attention_forward_kv_ex(
        gai::Device::CUDA, static_cast<const float*>(q_dev.get()),
        k_dev.get(), v_dev.get(), kv_fp16,
        static_cast<float*>(out_dev.get()), test.B, test.Tq, test.Tk,
        test.H, test.KV, test.hd, test.q_pos_offset, scale, test.window);
    cuda_check(cudaDeviceSynchronize(), "synchronize tiled attention");
    cuda_check(cudaMemcpy(actual.data(), out_dev.get(),
                          actual.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy attention output");

    double squared_error = 0.0;
    float max_abs_error = 0.0f;
    size_t max_error_index = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            throw std::runtime_error("non-finite tiled attention output");
        }
        const float error = std::fabs(actual[i] - reference[i]);
        squared_error += static_cast<double>(error) * error;
        if (error > max_abs_error) {
            max_abs_error = error;
            max_error_index = i;
        }
        const float tolerance = 2.0e-3f + 2.0e-3f * std::fabs(reference[i]);
        if (error > tolerance) {
            throw std::runtime_error(
                "tiled attention differs from CPU online-softmax reference at " +
                std::to_string(i) + " (actual=" + std::to_string(actual[i]) +
                ", reference=" + std::to_string(reference[i]) +
                ", abs_error=" + std::to_string(error) + ")");
        }
    }

    const double rmse = std::sqrt(squared_error / static_cast<double>(out_count));
    std::cout << "[PASS] B=" << test.B
              << " Tq=" << test.Tq
              << " Tk=" << test.Tk
              << " H/KV=" << test.H << "/" << test.KV
              << " hd=" << test.hd
              << " offset=" << test.q_pos_offset
              << " window=" << test.window
              << " KV=" << (kv_fp16 ? "FP16" : "FP32")
              << " max_abs=" << max_abs_error
              << " rmse=" << rmse
              << " max_idx=" << max_error_index << '\n';
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "[SKIP] no CUDA device is available for the attention test\n";
        (void)cudaGetLastError();
        return 0;
    }

    try {
        cuda_check(cudaSetDevice(0), "select CUDA device 0");
        const AttentionCase cases[] = {
            {2, 19, 83, 8, 2, 64, 21, 0},
            {1, 13, 67, 12, 3, 96, 32, 17},
            {1, 1, 130, 4, 1, 128, 129, 64},
            {1, 3, 33, 4, 4, 32, 30, 1}
        };

        unsigned seed = 0xA771u;
        for (const AttentionCase& test : cases) {
            run_case(test, false, seed++);
            run_case(test, true, seed++);
        }
        std::cout << "[PASS] all tiled CUDA attention cases matched the CPU reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
