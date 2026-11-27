#pragma once

#include "core/common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace gai {
namespace cuda {

constexpr unsigned WARP_FULL_MASK = 0xffffffffU;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val, unsigned mask = WARP_FULL_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val, unsigned mask = WARP_FULL_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_all_reduce_sum(float val, unsigned mask = WARP_FULL_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_all_reduce_max(float val, unsigned mask = WARP_FULL_MASK) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ float4 load_float4_aligned(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4_aligned(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ float4 load_float4_ldg(const float* ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

}
}
#endif
