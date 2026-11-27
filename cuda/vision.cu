#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>

namespace gai {
namespace cuda_ops {

#define VIS_CU_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA vision: ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

namespace {

__global__ void k_layernorm_forward(const float* __restrict__ x,
                                    const float* __restrict__ w,
                                    const float* __restrict__ b,
                                    float* __restrict__ y,
                                    int rows, int dim, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + static_cast<size_t>(r) * dim;
    float* yr       = y + static_cast<size_t>(r) * dim;

    extern __shared__ float smem[];
    float* s_sum  = smem;
    float* s_sum2 = smem + blockDim.x;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += xr[i];
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = s_sum[0] / static_cast<float>(dim);

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = xr[i] - mean;
        local_var += d * d;
    }
    s_sum2[threadIdx.x] = local_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(s_sum2[0] / static_cast<float>(dim) + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (xr[i] - mean) * inv;
        if (w) v *= w[i];
        if (b) v += b[i];
        yr[i] = v;
    }
}

__global__ void k_softmax_rows(float* __restrict__ x, int rows, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + static_cast<size_t>(r) * cols;

    extern __shared__ float smem[];
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row[i];
        if (v > mx) mx = v;
    }
    smem[threadIdx.x] = mx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride] > smem[threadIdx.x]) {
                smem[threadIdx.x] = smem[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    const float row_max = smem[0];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(row[i] - row_max);
        row[i] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = (smem[0] > 0.0f) ? (1.0f / smem[0]) : 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}

__global__ void k_vit_extract_patches(const float* __restrict__ pixels_chw,
                                      float* __restrict__ patches,
                                      int image_size, int patch_size,
                                      int channels, int patches_per_side) {
    const int P  = patches_per_side * patches_per_side;
    const int pd = channels * patch_size * patch_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = P * pd;
    if (idx >= total) return;

    const int p = idx / pd;
    const int k = idx % pd;
    const int ps2 = patch_size * patch_size;
    const int c   = k / ps2;
    const int rem = k % ps2;
    const int y   = rem / patch_size;
    const int x   = rem % patch_size;

    const int py = (p / patches_per_side) * patch_size;
    const int px = (p % patches_per_side) * patch_size;

    const size_t src_idx = (static_cast<size_t>(c) * image_size + (py + y)) * image_size + (px + x);
    patches[idx] = pixels_chw[src_idx];
}

__global__ void k_vit_spatial_pool_2x2(const float* __restrict__ in_patches,
                                       float* __restrict__ out_pooled,
                                       int S, int D) {
    const int S_out = S / 2;
    const int P_out = S_out * S_out;
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P_out * D) return;

    const int out_idx = idx / D;
    const int d       = idx % D;
    const int oy      = out_idx / S_out;
    const int ox      = out_idx % S_out;

    const int p00 = (oy * 2) * S + (ox * 2);
    const int p01 = (oy * 2) * S + (ox * 2 + 1);
    const int p10 = (oy * 2 + 1) * S + (ox * 2);
    const int p11 = (oy * 2 + 1) * S + (ox * 2 + 1);

    const float v = 0.25f * (
        in_patches[static_cast<size_t>(p00) * D + d] +
        in_patches[static_cast<size_t>(p01) * D + d] +
        in_patches[static_cast<size_t>(p10) * D + d] +
        in_patches[static_cast<size_t>(p11) * D + d]);
    out_pooled[idx] = v;
}

}

void layernorm_forward(const float* x, const float* w, const float* b,
                       float* y, int rows, int dim, float eps) {
    if (rows <= 0 || dim <= 0) return;
    const int block = 256;
    const size_t smem = static_cast<size_t>(block) * 2 * sizeof(float);
    k_layernorm_forward<<<rows, block, smem>>>(x, w, b, y, rows, dim, eps);
    VIS_CU_CHECK(cudaGetLastError());
}

void softmax_rows(float* x, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    const int block = 256;
    const size_t smem = static_cast<size_t>(block) * sizeof(float);
    k_softmax_rows<<<rows, block, smem>>>(x, rows, cols);
    VIS_CU_CHECK(cudaGetLastError());
}

void vit_extract_patches(const float* pixels_chw, float* patches,
                         int image_size, int patch_size, int channels) {
    const int pps = image_size / patch_size;
    const int total = pps * pps * channels * patch_size * patch_size;
    if (total <= 0) return;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    k_vit_extract_patches<<<grid, block>>>(pixels_chw, patches, image_size, patch_size, channels, pps);
    VIS_CU_CHECK(cudaGetLastError());
}

void vit_spatial_pool_2x2(const float* in_patches, float* out_pooled, int S, int D) {
    const int S_out = S / 2;
    const int total = S_out * S_out * D;
    if (total <= 0) return;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    k_vit_spatial_pool_2x2<<<grid, block>>>(in_patches, out_pooled, S, D);
    VIS_CU_CHECK(cudaGetLastError());
}

void vit_attention_forward(const float* qkv, float* scores_buf, float* ctx,
                           int P, int H, int hd, float scale) {
    if (P <= 0 || H <= 0 || hd <= 0) return;
    const int D = H * hd;
    cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(gai::cuda::cublas_handle());

    const float* Q0 = qkv;
    const float* K0 = qkv + D;
    const float* V0 = qkv + 2 * D;
    const float zero = 0.0f;
    const float one  = 1.0f;

    cublasStatus_t st = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        P, P, hd,
        &scale,
        K0, 3 * D, static_cast<long long>(hd),
        Q0, 3 * D, static_cast<long long>(hd),
        &zero,
        scores_buf, P, static_cast<long long>(P) * P,
        H);
    if (st != CUBLAS_STATUS_SUCCESS) {
        GAI_FAIL("cublasSgemmStridedBatched QK^T failed in vit_attention_forward");
    }

    softmax_rows(scores_buf, H * P, P);

    st = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hd, P, P,
        &one,
        V0, 3 * D, static_cast<long long>(hd),
        scores_buf, P, static_cast<long long>(P) * P,
        &zero,
        ctx, D, static_cast<long long>(hd),
        H);
    if (st != CUBLAS_STATUS_SUCCESS) {
        GAI_FAIL("cublasSgemmStridedBatched Attn*V failed in vit_attention_forward");
    }
}

}
}
