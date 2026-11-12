#pragma once

#include "core/common.h"
#include "core/ops.h"
#include "core/tensor.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace gai {

// LoRA (Low-Rank Adaptation) adapter for linear projections.
//
// The adapter matrices A and B are real trainable tensors: the model
// registers them as Parameters (so the optimizer, the gradient zeroing and
// the checkpoint format all handle them like any other weight). B starts at
// zero, so the adapter contributes exactly 0 at the beginning of a run and
// the base model is reproduced bit-for-bit until training moves B.
//
//   forward:  y += (alpha/r) * (x @ A^T) @ B^T
//   backward: gB += (alpha/r) * dy^T @ R
//             gA += dR^T @ x
//
struct LoraWeight {
    Tensor A;      // [rank, in_dim]
    Tensor B;      // [out_dim, rank]
    Tensor gA;     // [rank, in_dim]
    Tensor gB;     // [out_dim, rank]
    Tensor r_buf;  // [M, rank] intermediate forward activation buffer
    int rank = 0;
    int in_dim = 0;
    int out_dim = 0;
    float alpha = 16.0f;
    bool enabled = false;

    float scale() const {
        return (rank > 0 && enabled) ? (alpha / static_cast<float>(rank)) : 0.0f;
    }

    void init(int in_d, int out_d, int r, float a, Device dev, std::mt19937_64& rng) {
        if (r <= 0) { enabled = false; return; }
        rank = r;
        in_dim = in_d;
        out_dim = out_d;
        alpha = a;
        enabled = true;

        A = Tensor({rank, in_dim}, DType::F32, dev);
        B = Tensor({out_dim, rank}, DType::F32, dev);
        gA = Tensor({rank, in_dim}, DType::F32, dev);
        gB = Tensor({out_dim, rank}, DType::F32, dev);

        // Standard LoRA initialization: A ~ N(0, 1/sqrt(in_dim)), B = 0
        float stddev = 1.0f / std::sqrt(static_cast<float>(in_dim));
        std::normal_distribution<float> dist(0.0f, stddev);

        std::vector<float> cpu_A(static_cast<size_t>(rank) * in_dim);
        for (float& v : cpu_A) v = dist(rng);
        Tensor tmp_A({rank, in_dim}, DType::F32, Device::CPU);
        std::memcpy(tmp_A.f32(), cpu_A.data(), cpu_A.size() * sizeof(float));
        A.copy_from(tmp_A);

        ops::zero(dev, B.f32(), static_cast<i64>(out_dim) * rank);
        ops::zero(dev, gA.f32(), static_cast<i64>(rank) * in_dim);
        ops::zero(dev, gB.f32(), static_cast<i64>(out_dim) * rank);
    }

    void forward(Device dev, const float* x, float* y, int M) {
        if (!enabled || rank <= 0) return;
        if (r_buf.numel() < static_cast<i64>(M) * rank) {
            r_buf = Tensor({M, rank}, DType::F32, dev);
        }
        float s = scale();
        // R = X * A^T
        ops::linear_forward(dev, x, A.f32(), r_buf.f32(), M, in_dim, rank);
        // Y += s * R * B^T
        ops::gemm(dev, false, true, M, out_dim, rank, s, r_buf.f32(), rank, B.f32(), rank, 1.0f, y, out_dim);
    }

    void backward(Device dev, const float* x, const float* dy, float* dx, int M) {
        if (!enabled || rank <= 0) return;
        float s = scale();
        // gB += s * dY^T * R
        ops::gemm(dev, true, false, out_dim, rank, M, s, dy, out_dim, r_buf.f32(), rank, 1.0f, gB.f32(), rank);
        // dR = s * dY * B
        Tensor dR({M, rank}, DType::F32, dev);
        ops::gemm(dev, false, false, M, rank, out_dim, s, dy, out_dim, B.f32(), rank, 0.0f, dR.f32(), rank);
        // gA += dR^T * X
        ops::gemm(dev, true, false, rank, in_dim, M, 1.0f, dR.f32(), rank, x, in_dim, 1.0f, gA.f32(), in_dim);
        // dx += dR * A (if dx is requested for propagating gradient)
        if (dx) {
            ops::gemm(dev, false, false, M, in_dim, rank, 1.0f, dR.f32(), rank, A.f32(), in_dim, 1.0f, dx, in_dim);
        }
    }

};

} // namespace gai
