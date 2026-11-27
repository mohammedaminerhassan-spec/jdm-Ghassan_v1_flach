#include "cuda/flash_decode.cuh"
#include "cuda/warp_primitives.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/cuda_memory_pool.cuh"
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__

namespace gai {
namespace cuda {

constexpr int SPLIT_CHUNK = 128;
constexpr int MAX_SPLITS = 64;

__global__ void k_flash_decode_split(const float* __restrict__ q,
                                     const float* __restrict__ k_cache,
                                     const float* __restrict__ v_cache,
                                     float* __restrict__ partial_out,
                                     float* __restrict__ partial_max,
                                     float* __restrict__ partial_sum,
                                     int num_heads,
                                     int num_kv_heads,
                                     int head_dim,
                                     int seq_len,
                                     float scale,
                                     int num_splits) {
    const int h = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (h >= num_heads || split_idx >= num_splits) return;

    const int gqa_ratio = num_heads / num_kv_heads;
    const int kv_h = h / gqa_ratio;

    const int start_t = split_idx * SPLIT_CHUNK;
    const int end_t = min(start_t + SPLIT_CHUNK, seq_len);
    if (start_t >= end_t) return;

    const float* q_head = q + static_cast<size_t>(h) * head_dim;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_warp_red = smem + head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    const int lane = tid % 32;
    const int wid  = tid / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    float local_max = -1e20f;

    for (int t = start_t + tid; t < end_t; t += blockDim.x) {
        const float* k_tok = k_cache + (static_cast<size_t>(t) * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += s_q[d] * k_tok[d];
        }
        dot *= scale;
        local_max = fmaxf(local_max, dot);
    }

    local_max = warp_all_reduce_max(local_max);
    if (lane == 0) s_warp_red[wid] = local_max;
    __syncthreads();

    if (tid == 0) {
        float m = s_warp_red[0];
        for (int w = 1; w < num_warps; ++w) m = fmaxf(m, s_warp_red[w]);
        s_warp_red[32] = m;
    }
    __syncthreads();
    const float split_max = s_warp_red[32];

    float local_sum = 0.0f;
    for (int t = start_t + tid; t < end_t; t += blockDim.x) {
        const float* k_tok = k_cache + (static_cast<size_t>(t) * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += s_q[d] * k_tok[d];
        }
        dot = expf(dot * scale - split_max);
        local_sum += dot;
    }
    local_sum = warp_all_reduce_sum(local_sum);
    if (lane == 0) s_warp_red[wid] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps; ++w) s += s_warp_red[w];
        s_warp_red[33] = s;
    }
    __syncthreads();
    const float split_sum = s_warp_red[33];

    const size_t split_offset = (static_cast<size_t>(h) * num_splits + split_idx) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = start_t; t < end_t; ++t) {
            const float* k_tok = k_cache + (static_cast<size_t>(t) * num_kv_heads + kv_h) * head_dim;
            const float* v_tok = v_cache + (static_cast<size_t>(t) * num_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                dot += s_q[dd] * k_tok[dd];
            }
            const float weight = expf(dot * scale - split_max);
            acc += weight * v_tok[d];
        }
        partial_out[split_offset + d] = acc;
    }

    if (tid == 0) {
        const size_t meta_offset = static_cast<size_t>(h) * num_splits + split_idx;
        partial_max[meta_offset] = split_max;
        partial_sum[meta_offset] = split_sum;
    }
}

__global__ void k_flash_decode_reduce(const float* __restrict__ partial_out,
                                      const float* __restrict__ partial_max,
                                      const float* __restrict__ partial_sum,
                                      float* __restrict__ out,
                                      int num_heads,
                                      int head_dim,
                                      int num_splits) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;

    if (h >= num_heads || d >= head_dim) return;

    float global_max = -1e20f;
    for (int s = 0; s < num_splits; ++s) {
        const float m = partial_max[h * num_splits + s];
        global_max = fmaxf(global_max, m);
    }

    float total_weight = 0.0f;
    float final_val = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        const float m = partial_max[h * num_splits + s];
        const float l = partial_sum[h * num_splits + s];
        const float alpha = expf(m - global_max);

        total_weight += alpha * l;
        const size_t split_offset = (static_cast<size_t>(h) * num_splits + s) * head_dim;
        final_val += alpha * partial_out[split_offset + d];
    }

    const float inv_total = (total_weight > 0.0f) ? (1.0f / total_weight) : 0.0f;
    out[h * head_dim + d] = final_val * inv_total;
}

void flash_decode_attention(void* stream_handle,
                            const float* q,
                            const float* k_cache,
                            const float* v_cache,
                            float* out,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            int seq_len,
                            float scale) {
    if (seq_len <= 0 || num_heads <= 0 || head_dim <= 0) return;

    const int num_splits = std::min(MAX_SPLITS, (seq_len + SPLIT_CHUNK - 1) / SPLIT_CHUNK);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

    const size_t pout_bytes = static_cast<size_t>(num_heads) * num_splits * head_dim * sizeof(float);
    const size_t meta_bytes = static_cast<size_t>(num_heads) * num_splits * sizeof(float);

    float* d_pout = static_cast<float*>(CudaMemoryPool::instance().allocate(pout_bytes, stream));
    float* d_pmax = static_cast<float*>(CudaMemoryPool::instance().allocate(meta_bytes, stream));
    float* d_psum = static_cast<float*>(CudaMemoryPool::instance().allocate(meta_bytes, stream));

    const dim3 grid_split(num_heads, num_splits);
    const int block_split = std::min(head_dim, 128);
    const size_t smem_bytes = (head_dim + 64) * sizeof(float);

    k_flash_decode_split<<<grid_split, block_split, smem_bytes, stream>>>(
        q, k_cache, v_cache, d_pout, d_pmax, d_psum,
        num_heads, num_kv_heads, head_dim, seq_len, scale, num_splits);

    k_flash_decode_reduce<<<num_heads, head_dim, 0, stream>>>(
        d_pout, d_pmax, d_psum, out, num_heads, head_dim, num_splits);

    CudaMemoryPool::instance().deallocate(d_pout, stream);
    CudaMemoryPool::instance().deallocate(d_pmax, stream);
    CudaMemoryPool::instance().deallocate(d_psum, stream);
}

}
}

#endif
