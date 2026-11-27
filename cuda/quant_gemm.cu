#include "cuda/quant_gemm.cuh"
#include "cuda/warp_primitives.cuh"
#include "cuda/cuda_utils.cuh"

#ifdef __CUDACC__

namespace gai {
namespace cuda {

#pragma pack(push, 1)
struct CudaBlockQ8_0 {
    __half scale;
    int8_t qs[32];
};

struct CudaBlockQ4_0 {
    __half scale;
    uint8_t qs[16];
};
#pragma pack(pop)

__global__ void k_gemv_q8_0(const CudaBlockQ8_0* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int num_blocks_per_row) {
    const int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    const int lane = threadIdx.x;

    if (warp_id >= M) return;

    const CudaBlockQ8_0* row_blocks = W + static_cast<size_t>(warp_id) * num_blocks_per_row;
    float row_sum = 0.0f;

    for (int b = 0; b < num_blocks_per_row; ++b) {
        const float d = __half2float(row_blocks[b].scale);
        const float q_val = static_cast<float>(row_blocks[b].qs[lane]);
        const float x_val = x[b * 32 + lane];
        row_sum = fmaf(d * q_val, x_val, row_sum);
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane == 0) {
        y[warp_id] = row_sum;
    }
}

__global__ void k_gemv_q4_0(const CudaBlockQ4_0* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int num_blocks_per_row) {
    const int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    const int lane = threadIdx.x;

    if (warp_id >= M) return;

    const CudaBlockQ4_0* row_blocks = W + static_cast<size_t>(warp_id) * num_blocks_per_row;
    float row_sum = 0.0f;

    for (int b = 0; b < num_blocks_per_row; ++b) {
        const float d = __half2float(row_blocks[b].scale);
        float q_val;
        float x_val;

        if (lane < 16) {
            const uint8_t byte = row_blocks[b].qs[lane];
            const int v0 = static_cast<int>(byte & 0x0F) - 8;
            q_val = static_cast<float>(v0);
            x_val = x[b * 32 + lane];
        } else {
            const uint8_t byte = row_blocks[b].qs[lane - 16];
            const int v1 = static_cast<int>(byte >> 4) - 8;
            q_val = static_cast<float>(v1);
            x_val = x[b * 32 + lane];
        }

        row_sum = fmaf(d * q_val, x_val, row_sum);
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane == 0) {
        y[warp_id] = row_sum;
    }
}

void gemv_q8_0(void* stream_handle,
               const void* W_q8,
               const float* x,
               float* y,
               int M, int K) {
    if (M <= 0 || K <= 0) return;
    const int num_blocks_per_row = K / 32;
    constexpr int warps_per_block = 4;
    const dim3 block(WARP_SIZE, warps_per_block);
    const int num_blocks = (M + warps_per_block - 1) / warps_per_block;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

    k_gemv_q8_0<<<num_blocks, block, 0, stream>>>(
        static_cast<const CudaBlockQ8_0*>(W_q8), x, y, M, num_blocks_per_row);
}

void gemv_q4_0(void* stream_handle,
               const void* W_q4,
               const float* x,
               float* y,
               int M, int K) {
    if (M <= 0 || K <= 0) return;
    const int num_blocks_per_row = K / 32;
    constexpr int warps_per_block = 4;
    const dim3 block(WARP_SIZE, warps_per_block);
    const int num_blocks = (M + warps_per_block - 1) / warps_per_block;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

    k_gemv_q4_0<<<num_blocks, block, 0, stream>>>(
        static_cast<const CudaBlockQ4_0*>(W_q4), x, y, M, num_blocks_per_row);
}

}
}

#endif
