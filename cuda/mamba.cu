#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <limits>
#include <type_traits>

namespace gai {
namespace cuda_ops {

#define CU_MAMBA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

static thread_local void*   g_mamba_ws = nullptr;
static thread_local size_t   g_mamba_ws_bytes = 0;

static float* mamba_workspace(size_t floats) {
    const size_t bytes = floats * sizeof(float);
    if (bytes <= g_mamba_ws_bytes) return static_cast<float*>(g_mamba_ws);
    const size_t chunk = 64ull << 20;
    size_t want = ((bytes + chunk - 1) / chunk) * chunk;
    if (g_mamba_ws) CU_MAMBA_CHECK(cudaFree(g_mamba_ws));
    gai::cuda::check_free_vram(want, "Mamba workspace");
    CU_MAMBA_CHECK(cudaMalloc(&g_mamba_ws, want));
    g_mamba_ws_bytes = want;
    return static_cast<float*>(g_mamba_ws);
}

void mamba_free_workspace() {
    cudaDeviceSynchronize();
    if (g_mamba_ws) cudaFree(g_mamba_ws);
    g_mamba_ws = nullptr;
    g_mamba_ws_bytes = 0;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem, int lane,
                                                   int nthr) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    if ((lane & 31) == 0) smem[lane >> 5] = v;
    __syncthreads();
    if (lane < 32) {
        const int nwarps = (nthr + 31) >> 5;
        float w = (lane < nwarps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, off);
        if (lane == 0) smem[0] = w;
    }
    __syncthreads();
    const float result = smem[0];
    __syncthreads();
    return result;
}

__global__ void k_mamba_conv1d(const float* __restrict__ in,
                               const float* __restrict__ w,
                               const float* __restrict__ bias,
                               float* __restrict__ out,
                               float* __restrict__ raw_out,
                               int B, int T, int C, int in_stride, int k,
                               float* __restrict__ conv_state,
                               const i32* __restrict__ segment_ids) {
    extern __shared__ float sh[];
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    float* win = sh;
    float* val = sh + (k - 1);

    if (lane == 0) {
        const size_t cbase = (size_t(b) * C + c) * (k - 1 > 0 ? k - 1 : 1);
        for (int j = 0; j < k - 1; ++j) {
            win[j] = conv_state ? conv_state[cbase + j] : 0.0f;
        }
    }
    __syncthreads();

    const float* wc = w + size_t(c) * k;
    const float bc = bias ? bias[c] : 0.0f;
    int prev_seg = -2;

    for (int t = 0; t < T; ++t) {
        if (segment_ids) {
            const int curr = segment_ids[size_t(b) * T + t];
            if (t > 0 && curr >= 0 && prev_seg >= 0 && curr != prev_seg) {
                if (lane == 0) {
                    for (int j = 0; j < k - 1; ++j) win[j] = 0.0f;
                }
                __syncthreads();
            }
            if (curr >= 0) prev_seg = curr;
        }
        const float x = in[((size_t(b) * T + t) * in_stride) + c];
        if (lane == 0) *val = x;
        __syncthreads();

        if (lane == 0) {
            float acc = bc;
            for (int j = 0; j < k; ++j) {
                const float v = (j == k - 1) ? *val : win[j];
                acc += wc[j] * v;
            }
            if (raw_out) raw_out[((size_t(b) * T + t) * C) + c] = acc;
            out[((size_t(b) * T + t) * C) + c] = acc / (1.0f + expf(-acc));
            for (int j = 0; j + 1 < k - 1; ++j) win[j] = win[j + 1];
            if (k - 1 > 0) win[k - 2] = x;
        }
        __syncthreads();
    }
    if (lane == 0 && conv_state && k > 1) {
        const size_t cbase = (size_t(b) * C + c) * (k - 1);
        for (int j = 0; j < k - 1; ++j) conv_state[cbase + j] = win[j];
    }
}

__global__ void k_mamba_scan(const float* __restrict__ proj,
                              const float* __restrict__ bc,
                              const float* __restrict__ dtw,
                              const float* __restrict__ alog,
                              const float* __restrict__ Dv,
                              float* __restrict__ state,
                              float* __restrict__ y,
                              int B, int T, int n_heads, int d_head, int d_state,
                              int d_inner, int conv_dim, int proj_size,
                              const i32* segment_ids = nullptr) {
    extern __shared__ float sh_state[];
    __shared__ float sh_red[32];
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const int inner = n_heads * d_head;
    const int b     = row / inner;
    const int rem   = row - b * inner;
    const int h     = rem / d_head;
    const int i     = rem - h * d_head;

    const int base = (b * n_heads + h) * d_head + i;
    if (state) {
        for (int j = lane; j < d_state; j += nthr) sh_state[j] = state[base * d_state + j];
    } else {
        for (int j = lane; j < d_state; j += nthr) sh_state[j] = 0.0f;
    }
    __syncthreads();

    const float a       = -expf(alog[h]);
    const float dt_bias = dtw[h];
    const float dskip   = Dv[h];
    const int   x_off   = h * d_head + i;
    const int   d_state_tot = (conv_dim - d_inner) / 2;
    const int   groups = (d_state > 0 && d_state_tot >= d_state) ? (d_state_tot / d_state) : 1;
    const int   heads_per_group = (groups > 0 && n_heads >= groups) ? (n_heads / groups) : n_heads;
    const int   g_off   = (heads_per_group > 0) ? (h / heads_per_group) * d_state : 0;
    const int   B_off   = d_inner + g_off;
    const int   C_off   = d_inner + d_state_tot + g_off;

    int prev_seg = -2;
    for (int t = 0; t < T; ++t) {
        if (segment_ids) {
            const int curr_seg = segment_ids[(size_t)b * T + t];
            if (t > 0 && curr_seg >= 0 && prev_seg >= 0 && curr_seg != prev_seg) {
                for (int j = lane; j < d_state; j += nthr) sh_state[j] = 0.0f;
                __syncthreads();
            }
            if (curr_seg >= 0) prev_seg = curr_seg;
        }

        const float* prow = proj + (size_t(b) * T + t) * proj_size;
        const float* brow = bc   + (size_t(b) * T + t) * conv_dim;

        const float dt_in  = prow[proj_size - n_heads + h];
        const float dt_val = (dt_in + dt_bias > 20.0f) ? (dt_in + dt_bias)
                                                      : log1pf(expf(dt_in + dt_bias));
        const float dA = expf(a * dt_val);

        const float xv = brow[x_off];
        float acc = 0.0f;
        for (int j = lane; j < d_state; j += nthr) {
            const float bj = brow[B_off + j];
            float* sj = sh_state + j;
            *sj = dA * (*sj) + (dt_val * bj) * xv;
            acc += brow[C_off + j] * (*sj);
        }
        acc = block_reduce_sum(acc, sh_red, lane, nthr);

        if (lane == 0)
            y[(size_t(b) * T + t) * d_inner + x_off] = acc + dskip * xv;
        __syncthreads();
    }

    if (state) {
        for (int j = lane; j < d_state; j += nthr)
            state[base * d_state + j] = sh_state[j];
    }
}

__global__ void k_mamba_gated_norm(const float* __restrict__ y,
                                   const float* __restrict__ gate,
                                   const float* __restrict__ w,
                                   float* __restrict__ out,
                                   int N, int dim, int gate_stride, float eps) {
    __shared__ float sh_red[32];
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;
    const float* yr = y + size_t(n) * dim;
    const float* gr = gate + size_t(n) * gate_stride;
    float* orow = out + size_t(n) * dim;

    float ss = 0.0f;
    for (int c = lane; c < dim; c += nthr) {
        const float g = gr[c] / (1.0f + expf(-gr[c]));
        const float v = yr[c] * g;
        orow[c] = v;
        ss += v * v;
    }
    ss = block_reduce_sum(ss, sh_red, lane, nthr);
    __shared__ float s_inv;
    if (lane == 0) s_inv = 1.0f / sqrtf(ss / dim + eps);
    __syncthreads();
    for (int c = lane; c < dim; c += nthr)
        orow[c] = orow[c] * s_inv * w[c];
}

template <int K>
__global__ void k_mamba_conv1d_channels(const float* __restrict__ in,
                                        const float* __restrict__ w,
                                        const float* __restrict__ bias,
                                        float* __restrict__ out,
                                        float* __restrict__ raw_out,
                                        int B, int T, int C, int in_stride,
                                        float* __restrict__ conv_state,
                                        const i32* __restrict__ segment_ids) {
    const int channel_batch = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * C;
    if (channel_batch >= total) return;

    const int b = channel_batch / C;
    const int c = channel_batch - b * C;
    const size_t state_base =
        (static_cast<size_t>(b) * C + c) * (K > 1 ? K - 1 : 1);
    float win[K > 1 ? K - 1 : 1];

    if constexpr (K > 1) {
        #pragma unroll
        for (int j = 0; j < K - 1; ++j)
            win[j] = conv_state ? conv_state[state_base + j] : 0.0f;
    }

    const float* wc = w + static_cast<size_t>(c) * K;
    const float bc = bias ? bias[c] : 0.0f;
    int prev_seg = -2;

    for (int t = 0; t < T; ++t) {
        if (segment_ids) {
            const int curr = segment_ids[static_cast<size_t>(b) * T + t];
            if constexpr (K > 1) {
                if (t > 0 && curr >= 0 && prev_seg >= 0 && curr != prev_seg) {
                    #pragma unroll
                    for (int j = 0; j < K - 1; ++j) win[j] = 0.0f;
                }
            }
            if (curr >= 0) prev_seg = curr;
        }

        const float x = in[(static_cast<size_t>(b) * T + t) * in_stride + c];
        float acc = bc;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float value = (j == K - 1) ? x : win[j];
            acc += wc[j] * value;
        }

        const size_t output_idx =
            (static_cast<size_t>(b) * T + t) * C + c;
        if (raw_out) raw_out[output_idx] = acc;
        out[output_idx] = acc / (1.0f + expf(-acc));

        if constexpr (K > 1) {
            #pragma unroll
            for (int j = 0; j + 1 < K - 1; ++j) win[j] = win[j + 1];
            win[K - 2] = x;
        }
    }

    if constexpr (K > 1) {
        if (conv_state) {
            #pragma unroll
            for (int j = 0; j < K - 1; ++j)
                conv_state[state_base + j] = win[j];
        }
    }
}

static void launch_mamba_conv1d(const float* in, const float* w,
                                const float* bias, float* out, float* raw_out,
                                int B, int T, int C, int in_stride, int k,
                                float* conv_state, const i32* segment_ids) {
    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(C);
    GAI_CHECK(total <= static_cast<size_t>((std::numeric_limits<int>::max)()),
              "Mamba convolution channel count exceeds int range");

    if (k >= 1 && k <= 16) {
        const int block = 256;
        const unsigned grid = static_cast<unsigned>((total + block - 1) / block);
        switch (k) {
            case 1: k_mamba_conv1d_channels<1><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 2: k_mamba_conv1d_channels<2><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 3: k_mamba_conv1d_channels<3><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 4: k_mamba_conv1d_channels<4><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 5: k_mamba_conv1d_channels<5><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 6: k_mamba_conv1d_channels<6><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 7: k_mamba_conv1d_channels<7><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 8: k_mamba_conv1d_channels<8><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 9: k_mamba_conv1d_channels<9><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 10: k_mamba_conv1d_channels<10><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 11: k_mamba_conv1d_channels<11><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 12: k_mamba_conv1d_channels<12><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 13: k_mamba_conv1d_channels<13><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 14: k_mamba_conv1d_channels<14><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 15: k_mamba_conv1d_channels<15><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            case 16: k_mamba_conv1d_channels<16><<<grid, block>>>(in,w,bias,out,raw_out,B,T,C,in_stride,conv_state,segment_ids); break;
            default: break;
        }
        return;
    }

    // General fallback for unusually wide convolution kernels.
    const dim3 grid(C, B);
    const int block = 128;
    const size_t shared = sizeof(float) * static_cast<size_t>(k > 1 ? k : 1);
    k_mamba_conv1d<<<grid, block, shared>>>(in,w,bias,out,raw_out,B,T,C,in_stride,
                                            k,conv_state,segment_ids);
}

template <typename W>
static inline void mamba_linear(const float* x, const W* w, float* y, int M, int K, int N,
                                int qkind = -1) {
    if (qkind >= 0) {
        linear_forward_q4(x, reinterpret_cast<const u16*>(w), y, M, K, N, qkind);
        return;
    }
    if constexpr (std::is_same<W, u16>::value) {
        linear_forward_fp16(x, w, y, M, K, N);
    } else {
        linear_forward(x, w, y, M, K, N);
    }
}

template <typename W>
static inline void mamba_linear_bwd(const float* x, const W* w, const float* dy,
                                    float* dx, float* dw, int M, int K, int N) {
    if constexpr (std::is_same<W, u16>::value) {
        linear_backward_half(x, w, dy, dx, dw, M, K, N);
    } else {
        linear_backward(x, w, dy, dx, dw, M, K, N);
    }
}

template <typename W>
static void mamba2_forward_impl(const float* x,
                    const W* w_in,
                    const float* conv_w,
                    const float* conv_b,
                    const float* dt_bias,
                    const float* a_log,
                    const float* D,
                    const float* norm_w,
                    const W* w_out,
                    float* state,
                    float* out,
                    int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                    int d_head, int d_state, int k, float eps, int qkind = -1,
                    const i32* segment_ids = nullptr, float* conv_state = nullptr) {
    if (B <= 0 || T <= 0) return;
    const int proj_size = d_inner + conv_dim + n_heads;
    const size_t n_tok = size_t(B) * T;

    float* ws = mamba_workspace(n_tok * (size_t(proj_size) + conv_dim + d_inner) + 64);
    float* proj = ws;
    float* bc   = proj + n_tok * size_t(proj_size);
    float* y    = bc   + n_tok * size_t(conv_dim);

    mamba_linear(x, w_in, proj, B * T, d, proj_size, qkind);

    {
        launch_mamba_conv1d(proj + d_inner, conv_w, conv_b, bc, nullptr,
                            B, T, conv_dim, proj_size, k, conv_state, segment_ids);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    {
        const int rows = B * n_heads * d_head;
        const int block = 256;
        const size_t sh = sizeof(float) * d_state;
        k_mamba_scan<<<rows, block, sh>>>(proj, bc, dt_bias, a_log, D,
                                          state, y, B, T, n_heads, d_head,
                                          d_state, d_inner, conv_dim, proj_size, segment_ids);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    {
        const int block = 256;
        k_mamba_gated_norm<<<B * T, block>>>(y, proj, norm_w,
                                             y, B * T, d_inner, proj_size, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    mamba_linear(y, w_out, out, B * T, d_inner, d, qkind);
}

void mamba2_forward(const float* x,
                    const float* w_in,
                    const float* conv_w,
                    const float* conv_b,
                    const float* dt_bias,
                    const float* a_log,
                    const float* D,
                    const float* norm_w,
                    const float* w_out,
                    float* state,
                    float* out,
                    int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                    int d_head, int d_state, int k, float eps,
                    const i32* segment_ids, float* conv_state) {
    mamba2_forward_impl<float>(x, w_in, conv_w, conv_b, dt_bias, a_log, D,
                               norm_w, w_out, state, out,
                               B, T, d, d_inner, conv_dim, n_heads,
                               d_head, d_state, k, eps, -1, segment_ids, conv_state);
}

void mamba2_forward_half(const float* x,
                    const u16* w_in,
                    const float* conv_w,
                    const float* conv_b,
                    const float* dt_bias,
                    const float* a_log,
                    const float* D,
                    const float* norm_w,
                    const u16* w_out,
                    float* state,
                    float* out,
                    int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                    int d_head, int d_state, int k, float eps, int qkind,
                    const i32* segment_ids, float* conv_state) {
    mamba2_forward_impl<u16>(x, w_in, conv_w, conv_b, dt_bias, a_log, D,
                             norm_w, w_out, state, out,
                             B, T, d, d_inner, conv_dim, n_heads,
                             d_head, d_state, k, eps, qkind, segment_ids, conv_state);
}

__global__ void k_mamba_gated_norm_bwd(const float* __restrict__ dy_norm,
                                       const float* __restrict__ y_scan,
                                       const float* __restrict__ proj,
                                       const float* __restrict__ norm_w,
                                       float* __restrict__ dy_scan,
                                       float* __restrict__ dgate,
                                       float* __restrict__ dnorm_w,
                                       int N, int dim, int proj_size, float eps) {
    extern __shared__ float sh_red[];
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const float* dyn = dy_norm + size_t(n) * dim;
    const float* yr  = y_scan  + size_t(n) * dim;
    const float* gr  = proj    + size_t(n) * proj_size;
    float* dys       = dy_scan + size_t(n) * dim;
    float* dg        = dgate   + size_t(n) * dim;

    float ss = 0.0f;
    for (int c = lane; c < dim; c += nthr) {
        float sig = 1.0f / (1.0f + expf(-gr[c]));
        float v = yr[c] * (gr[c] * sig);
        ss += v * v;
    }
    ss = block_reduce_sum(ss, sh_red, lane, nthr);
    __shared__ float s_inv;
    if (lane == 0) s_inv = 1.0f / sqrtf(ss / dim + eps);
    __syncthreads();

    float sum_pv = 0.0f;
    for (int c = lane; c < dim; c += nthr) {
        float sig = 1.0f / (1.0f + expf(-gr[c]));
        float v = yr[c] * (gr[c] * sig);
        float p = dyn[c] * norm_w[c];
        sum_pv += p * v;
    }
    sum_pv = block_reduce_sum(sum_pv, sh_red, lane, nthr);
    __shared__ float s_adj;
    if (lane == 0) s_adj = sum_pv * (s_inv * s_inv) / float(dim);
    __syncthreads();

    const float inv = s_inv;
    const float adj = s_adj;
    for (int c = lane; c < dim; c += nthr) {
        float sig = 1.0f / (1.0f + expf(-gr[c]));
        float g = gr[c] * sig;
        float v = yr[c] * g;
        float p = dyn[c] * norm_w[c];
        float dv = inv * (p - v * adj);
        float dsilu = sig * (1.0f + gr[c] * (1.0f - sig));
        dys[c] = dv * g;
        dg[c] = dv * yr[c] * dsilu;
        if (dnorm_w) {
            atomicAdd(&dnorm_w[c], dyn[c] * v * inv);
        }
    }
}

__global__ void k_mamba_scan_traj(const float* __restrict__ proj,
                                  const float* __restrict__ bc,
                                  const float* __restrict__ dtw,
                                  const float* __restrict__ alog,
                                  const float* __restrict__ Dv,
                                  float* __restrict__ traj,
                                  float* __restrict__ y,
                                  int B, int T, int n_heads, int d_head, int d_state,
                                  int d_inner, int conv_dim, int proj_size,
                                  int row_offset,
                                 const i32* __restrict__ segment_ids) {
    extern __shared__ float sh_state[];
    __shared__ float sh_red[32];
    const int lrow = blockIdx.x;
    const int row  = lrow + row_offset;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const int inner = n_heads * d_head;
    const int b     = row / inner;
    const int rem   = row - b * inner;
    const int h     = rem / d_head;
    const int i     = rem - h * d_head;

    const float a       = -expf(alog[h]);
    const float dt_bias = dtw[h];
    const float dskip   = Dv[h];
    const int   x_off   = h * d_head + i;
    const int   d_state_tot = (conv_dim - d_inner) / 2;
    const int   groups = (d_state > 0 && d_state_tot >= d_state) ? (d_state_tot / d_state) : 1;
    const int   heads_per_group = (groups > 0 && n_heads >= groups) ? (n_heads / groups) : n_heads;
    const int   g_off   = (heads_per_group > 0) ? (h / heads_per_group) * d_state : 0;
    const int   B_off   = d_inner + g_off;
    const int   C_off   = d_inner + d_state_tot + g_off;

    float* mytraj = traj + (size_t)lrow * (size_t)(T + 1) * d_state;

    for (int j = lane; j < d_state; j += nthr) { sh_state[j] = 0.0f; mytraj[j] = 0.0f; }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        if (segment_ids && t > 0) {
            const int cs = segment_ids[size_t(b) * T + t];
            const int ps = segment_ids[size_t(b) * T + t - 1];
            if (cs >= 0 && ps >= 0 && cs != ps) {
                for (int j = lane; j < d_state; j += nthr) sh_state[j] = 0.0f;
                __syncthreads();
            }
        }
        const float* prow = proj + (size_t(b) * T + t) * proj_size;
        const float* brow = bc   + (size_t(b) * T + t) * conv_dim;

        const float dt_in  = prow[proj_size - n_heads + h];
        const float dt_val = (dt_in + dt_bias > 20.0f) ? (dt_in + dt_bias)
                                                      : log1pf(expf(dt_in + dt_bias));
        const float dA = expf(a * dt_val);
        const float xv = brow[x_off];

        __syncthreads();
        for (int j = lane; j < d_state; j += nthr)
            mytraj[(size_t)t * d_state + j] = sh_state[j];
        __syncthreads();

        float acc = 0.0f;
        for (int j = lane; j < d_state; j += nthr) {
            const float bj = brow[B_off + j];
            float* sj = sh_state + j;
            *sj = dA * (*sj) + (dt_val * bj) * xv;
            acc += brow[C_off + j] * (*sj);
        }
        __syncthreads();

        acc = block_reduce_sum(acc, sh_red, lane, nthr);
        if (lane == 0)
            y[(size_t(b) * T + t) * d_inner + x_off] = acc + dskip * xv;
        __syncthreads();
    }
}

__global__ void k_mamba_scan_bwd(const float* __restrict__ proj,
                                 const float* __restrict__ bc,
                                 const float* __restrict__ dy_scan,
                                 const float* __restrict__ dtw,
                                 const float* __restrict__ alog,
                                 const float* __restrict__ Dv,
                                 const float* __restrict__ traj,
                                 float* __restrict__ dbc,
                                 float* __restrict__ ddt_raw,
                                 float* __restrict__ dD,
                                 float* __restrict__ da_log,
                                 float* __restrict__ ddt_bias,
                                 int B, int T, int n_heads, int d_head, int d_state,
                                 int d_inner, int conv_dim, int proj_size,
                                 int row_offset,
                                 const i32* __restrict__ segment_ids) {
    extern __shared__ float sh_mem[];
    float* sh_ds  = sh_mem;
    float* sh_red = sh_mem + d_state;

    const int lrow = blockIdx.x;
    const int row  = lrow + row_offset;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const int inner = n_heads * d_head;
    const int b     = row / inner;
    const int rem   = row - b * inner;
    const int h     = rem / d_head;
    const int i     = rem - h * d_head;

    const float a_val     = -expf(alog[h]);
    const float dt_bias_v = dtw[h];
    const float dskip     = Dv[h];
    const int   x_off     = h * d_head + i;
    const int   d_state_tot = (conv_dim - d_inner) / 2;
    const int   groups = (d_state > 0 && d_state_tot >= d_state) ? (d_state_tot / d_state) : 1;
    const int   heads_per_group = (groups > 0 && n_heads >= groups) ? (n_heads / groups) : n_heads;
    const int   g_off     = (heads_per_group > 0) ? (h / heads_per_group) * d_state : 0;
    const int   B_off     = d_inner + g_off;
    const int   C_off     = d_inner + d_state_tot + g_off;

    const float* mytraj = traj + (size_t)lrow * (size_t)(T + 1) * d_state;

    for (int j = lane; j < d_state; j += nthr) sh_ds[j] = 0.0f;
    __syncthreads();

    for (int t = T - 1; t >= 0; --t) {
        if (segment_ids && t + 1 < T) {
            const int cs = segment_ids[size_t(b) * T + t];
            const int ns = segment_ids[size_t(b) * T + t + 1];
            if (cs >= 0 && ns >= 0 && cs != ns) {
                for (int j = lane; j < d_state; j += nthr) sh_ds[j] = 0.0f;
                __syncthreads();
            }
        }
        const size_t tok_idx = size_t(b) * T + t;
        const float* prow = proj + tok_idx * proj_size;
        const float* brow = bc   + tok_idx * conv_dim;
        float* d_brow     = dbc  + tok_idx * conv_dim;
        const float dy_val = dy_scan[tok_idx * d_inner + x_off];

        const float dt_in = prow[proj_size - n_heads + h];
        const float u = dt_in + dt_bias_v;
        const float dt_val = (u > 20.0f) ? u : log1pf(expf(u));
        const float s_softplus = (u > 20.0f) ? 1.0f : (1.0f / (1.0f + expf(-u)));
        const float dA = expf(a_val * dt_val);
        const float xv = brow[x_off];

        const float* s_prev = mytraj + (size_t)t * d_state;

        if (dD && lane == 0) atomicAdd(&dD[h], dy_val * xv);
        float dx_val = (lane == 0) ? dy_val * dskip : 0.0f;

        float local_d_dA = 0.0f;
        float local_d_dt = 0.0f;

        for (int j = lane; j < d_state; j += nthr) {
            const float Cj = brow[C_off + j];
            const float Bj = brow[B_off + j];
            const float s_curr = dA * s_prev[j] + (dt_val * Bj) * xv;

            atomicAdd(&d_brow[C_off + j], dy_val * s_curr);

            const float total_ds = dy_val * Cj + sh_ds[j];
            local_d_dA += total_ds * s_prev[j];
            local_d_dt += total_ds * Bj * xv;

            atomicAdd(&d_brow[B_off + j], total_ds * (dt_val * xv));
            dx_val += total_ds * (dt_val * Bj);
            sh_ds[j] = total_ds * dA;
        }
        __syncthreads();

        dx_val = block_reduce_sum(dx_val, sh_red, lane, nthr);
        if (lane == 0) atomicAdd(&d_brow[x_off], dx_val);

        local_d_dA = block_reduce_sum(local_d_dA, sh_red, lane, nthr);
        local_d_dt = block_reduce_sum(local_d_dt, sh_red, lane, nthr);

        if (lane == 0) {
            const float d_adt = local_d_dA * dA;
            if (da_log) atomicAdd(&da_log[h], d_adt * dt_val * (-expf(alog[h])));
            float d_dt_total = local_d_dt + d_adt * a_val;
            float ddt_raw_val = d_dt_total * s_softplus;
            if (ddt_bias) atomicAdd(&ddt_bias[h], ddt_raw_val);
            atomicAdd(&ddt_raw[tok_idx * n_heads + h], ddt_raw_val);
        }
        __syncthreads();
    }
}

__global__ void k_mamba_conv1d_bwd(const float* __restrict__ proj,
                                   const float* __restrict__ conv_w,
                                   const float* __restrict__ conv_b,
                                   const float* __restrict__ dbc,
                                   const float* __restrict__ conv_raw,
                                   float* __restrict__ dconv_w,
                                   float* __restrict__ dconv_b,
                                   float* __restrict__ dproj_bc,
                                   int B, int T, int conv_dim, int proj_size, int k,
                                   const i32* __restrict__ segment_ids) {
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const float* wc = conv_w + size_t(c) * k;

    for (int t = lane; t < T; t += nthr) {
        const size_t idx = (size_t(b) * T + t) * conv_dim + c;
        const float raw_acc = conv_raw[idx];
        const float sig = 1.0f / (1.0f + expf(-raw_acc));
        const float dsilu = sig * (1.0f + raw_acc * (1.0f - sig));
        const float dacc = dbc[idx] * dsilu;

        if (dconv_b) atomicAdd(&dconv_b[c], dacc);

        for (int j = 0; j < k; ++j) {
            const int tin = t - (k - 1 - j);
            bool same_seg = true;
            if (segment_ids && tin >= 0) {
                const int ct = segment_ids[size_t(b) * T + t];
                const int cn = segment_ids[size_t(b) * T + tin];
                same_seg = !(ct >= 0 && cn >= 0 && ct != cn);
            }
            if (tin >= 0 && same_seg) {
                const size_t pidx = (size_t(b) * T + tin) * proj_size + c;
                if (dconv_w) atomicAdd(&dconv_w[size_t(c) * k + j], dacc * proj[pidx]);
                const size_t out_idx = (size_t(b) * T + tin) * conv_dim + c;
                atomicAdd(&dproj_bc[out_idx], dacc * wc[j]);
            }
        }
    }
}

__global__ void k_mamba_assemble_dproj(const float* __restrict__ dgate,
                                       const float* __restrict__ dproj_bc,
                                       const float* __restrict__ ddt_raw,
                                       float* __restrict__ dproj,
                                       int N, int d_inner, int conv_dim, int n_heads, int proj_size) {
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    float* dp = dproj + size_t(n) * proj_size;
    const float* dg = dgate + size_t(n) * d_inner;
    const float* dbc_p = dproj_bc + size_t(n) * conv_dim;
    const float* ddt_p = ddt_raw + size_t(n) * n_heads;

    for (int i = lane; i < d_inner; i += nthr) dp[i] = dg[i];
    for (int i = lane; i < conv_dim; i += nthr) dp[d_inner + i] = dbc_p[i];
    for (int i = lane; i < n_heads; i += nthr) dp[d_inner + conv_dim + i] = ddt_p[i];
}

template <typename W>
static void mamba2_backward_impl(const float* x,
                     const W* w_in,
                     const float* conv_w,
                     const float* conv_b,
                     const float* dt_bias,
                     const float* a_log,
                     const float* D,
                     const float* norm_w,
                     const W* w_out,
                     const float* dout,
                     float* dx,
                     float* dw_in,
                     float* dconv_w,
                     float* dconv_b,
                     float* ddt_bias,
                     float* da_log,
                     float* dD,
                     float* dnorm_w,
                     float* dw_out,
                     float* dbc_out,
                     int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                     int d_head, int d_state, int k, float eps,
                     const i32* segment_ids = nullptr) {
    if (B <= 0 || T <= 0) return;
    const int proj_size = d_inner + conv_dim + n_heads;
    const size_t n_tok = size_t(B) * T;

    const int rows = B * n_heads * d_head;
    const size_t per_row_traj = (size_t)(T + 1) * d_state;
    const size_t traj_budget = 8u << 20;
    int rows_batch = (int)(traj_budget / (per_row_traj ? per_row_traj : 1));
    if (rows_batch < 1) rows_batch = 1;
    if (rows_batch > rows) rows_batch = rows;
    const size_t traj_floats = (size_t)rows_batch * per_row_traj;

    const size_t total_floats = n_tok * (size_t(proj_size) * 2 + size_t(conv_dim) * 4 + size_t(d_inner) * 5 + n_heads) + traj_floats + 256;
    float* ws = mamba_workspace(total_floats);

    float* proj     = ws;
    float* bc       = proj     + n_tok * size_t(proj_size);
    float* conv_raw = bc       + n_tok * size_t(conv_dim);
    float* y_scan   = conv_raw + n_tok * size_t(conv_dim);
    float* y_norm   = y_scan   + n_tok * size_t(d_inner);
    float* dy_norm  = y_norm   + n_tok * size_t(d_inner);
    float* dy_scan  = dy_norm  + n_tok * size_t(d_inner);
    float* dgate    = dy_scan  + n_tok * size_t(d_inner);
    float* dbc      = dgate    + n_tok * size_t(d_inner);
    float* ddt_raw  = dbc      + n_tok * size_t(conv_dim);
    float* dproj_bc = ddt_raw  + n_tok * size_t(n_heads);
    float* dproj    = dproj_bc + n_tok * size_t(conv_dim);
    float* traj     = dproj    + n_tok * size_t(proj_size);

    mamba_linear(x, w_in, proj, B * T, d, proj_size);
    {
        dim3 grid(conv_dim, B);
        const int block = 128;
        const size_t sh = sizeof(float) * (k > 1 ? k : 1);
        k_mamba_conv1d<<<grid, block, sh>>>(proj + d_inner, conv_w, conv_b,
                                            bc, conv_raw, B, T, conv_dim, proj_size, k,
                                            nullptr, segment_ids);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    {
        const int block = 256;
        const size_t sh = sizeof(float) * d_state;
        for (int r0 = 0; r0 < rows; r0 += rows_batch) {
            const int nb = (rows - r0 < rows_batch) ? (rows - r0) : rows_batch;
            k_mamba_scan_traj<<<nb, block, sh>>>(proj, bc, dt_bias, a_log, D,
                                                 traj, y_scan, B, T, n_heads, d_head,
                                                 d_state, d_inner, conv_dim, proj_size,
                                                 r0, segment_ids);
            CU_MAMBA_CHECK(cudaGetLastError());
        }
    }

    {
        const int block = 256;
        k_mamba_gated_norm<<<B * T, block>>>(y_scan, proj, norm_w,
                                             y_norm, B * T, d_inner, proj_size, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    CU_MAMBA_CHECK(cudaMemset(dy_norm, 0, sizeof(float) * n_tok * d_inner));
    mamba_linear_bwd(y_norm, w_out, dout, dy_norm, dw_out, B * T, d_inner, d);

    {
        const int block = 256;
        const size_t sh = sizeof(float) * 32;
        k_mamba_gated_norm_bwd<<<B * T, block, sh>>>(dy_norm, y_scan, proj, norm_w,
                                                     dy_scan, dgate, dnorm_w,
                                                     B * T, d_inner, proj_size, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    CU_MAMBA_CHECK(cudaMemset(dbc, 0, sizeof(float) * n_tok * conv_dim));
    CU_MAMBA_CHECK(cudaMemset(ddt_raw, 0, sizeof(float) * n_tok * n_heads));
    {
        const int block = 256;
        const size_t sh = sizeof(float) * (d_state + 32);
        const size_t sh_traj = sizeof(float) * d_state;
        for (int r0 = 0; r0 < rows; r0 += rows_batch) {
            const int nb = (rows - r0 < rows_batch) ? (rows - r0) : rows_batch;
            k_mamba_scan_traj<<<nb, block, sh_traj>>>(proj, bc, dt_bias, a_log, D,
                                                      traj, y_scan, B, T, n_heads, d_head,
                                                      d_state, d_inner, conv_dim, proj_size,
                                                      r0, segment_ids);
            CU_MAMBA_CHECK(cudaGetLastError());
            k_mamba_scan_bwd<<<nb, block, sh>>>(proj, bc, dy_scan, dt_bias, a_log, D,
                                                traj, dbc, ddt_raw, dD, da_log, ddt_bias,
                                                B, T, n_heads, d_head, d_state,
                                                d_inner, conv_dim, proj_size, r0, segment_ids);
            CU_MAMBA_CHECK(cudaGetLastError());
        }
    }

    CU_MAMBA_CHECK(cudaMemset(dproj_bc, 0, sizeof(float) * n_tok * conv_dim));
    {
        dim3 grid(conv_dim, B);
        const int block = 128;
        k_mamba_conv1d_bwd<<<grid, block>>>(proj + d_inner, conv_w, conv_b,
                                            dbc, conv_raw, dconv_w, dconv_b, dproj_bc,
                                            B, T, conv_dim, proj_size, k, segment_ids);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    {
        const int block = 256;
        k_mamba_assemble_dproj<<<B * T, block>>>(dgate, dproj_bc, ddt_raw, dproj,
                                                B * T, d_inner, conv_dim, n_heads, proj_size);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    mamba_linear_bwd(x, w_in, dproj, dx, dw_in, B * T, d, proj_size);

    if (dbc_out)
        CU_MAMBA_CHECK(cudaMemcpy(dbc_out, dbc, sizeof(float) * n_tok * conv_dim,
                                  cudaMemcpyDeviceToHost));
}

void mamba2_backward(const float* x,
                     const float* w_in,
                     const float* conv_w,
                     const float* conv_b,
                     const float* dt_bias,
                     const float* a_log,
                     const float* D,
                     const float* norm_w,
                     const float* w_out,
                     const float* dout,
                     float* dx,
                     float* dw_in,
                     float* dconv_w,
                     float* dconv_b,
                     float* ddt_bias,
                     float* da_log,
                     float* dD,
                     float* dnorm_w,
                     float* dw_out,
                     float* dbc_out,
                     int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                     int d_head, int d_state, int k, float eps,
                     const i32* segment_ids) {
    mamba2_backward_impl<float>(x, w_in, conv_w, conv_b, dt_bias, a_log, D,
                                norm_w, w_out, dout, dx, dw_in, dconv_w, dconv_b,
                                ddt_bias, da_log, dD, dnorm_w, dw_out, dbc_out,
                                B, T, d, d_inner, conv_dim, n_heads, d_head,
                                d_state, k, eps, segment_ids);
}

__global__ void k_mamba_step_conv(const float* __restrict__ proj,
                                  const float* __restrict__ w,
                                  const float* __restrict__ b,
                                  float* __restrict__ conv_state,
                                  float* __restrict__ bc_out,
                                  int d_inner, int conv_dim, int k) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    const int win = k - 1;
    float acc = b ? b[ch] : 0.0f;
    for (int j = 0; j < win; ++j)
        acc += w[ch * k + j] * conv_state[(size_t)ch * win + j];
    acc += w[ch * k + win] * proj[d_inner + ch];
    bc_out[ch] = acc / (1.0f + expf(-acc));
    for (int j = 0; j + 1 < win; ++j)
        conv_state[(size_t)ch * win + j] = conv_state[(size_t)ch * win + j + 1];
    if (win > 0) conv_state[(size_t)ch * win + win - 1] = proj[d_inner + ch];
}

__global__ void k_mamba_step_scan(const float* __restrict__ proj,
                                  const float* __restrict__ bc_out,
                                  const float* __restrict__ dt_bias,
                                  const float* __restrict__ a_log,
                                  const float* __restrict__ Dv,
                                  float* __restrict__ ssm_state,
                                  float* __restrict__ y,
                                  int d_inner, int n_heads, int d_head,
                                  int d_state, int proj_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_heads * d_head;
    if (row >= total) return;
    const int h = row / d_head;
    const float u = proj[proj_size - n_heads + h] + dt_bias[h];
    const float dt = (u > 20.0f) ? u : log1pf(expf(u));
    const float A = -expf(a_log[h]);
    const float dA = expf(A * dt);
    const float xv = bc_out[row];
    const float* Bv = bc_out + d_inner;
    const float* Cv = Bv + d_state;
    float* s = ssm_state + (size_t)row * d_state;
    float acc = 0.0f;
    for (int j = 0; j < d_state; ++j) {
        s[j] = dA * s[j] + (dt * Bv[j]) * xv;
        acc += Cv[j] * s[j];
    }
    y[row] = acc + Dv[h] * xv;
}

__global__ void k_mamba_step_norm(const float* __restrict__ y,
                                  const float* __restrict__ gate,
                                  const float* __restrict__ w,
                                  float* __restrict__ out,
                                  int d_inner, float eps) {
    extern __shared__ float sh_red[];
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;
    float ss = 0.0f;
    for (int i = lane; i < d_inner; i += nthr) {
        const float g = gate[i] / (1.0f + expf(-gate[i]));
        const float v = y[i] * g;
        out[i] = v;
        ss += v * v;
    }
    ss = block_reduce_sum(ss, sh_red, lane, nthr);
    __shared__ float s_inv;
    if (lane == 0) s_inv = rsqrtf(ss / static_cast<float>(d_inner) + eps);
    __syncthreads();
    for (int i = lane; i < d_inner; i += nthr) out[i] = out[i] * s_inv * w[i];
}

template <typename W>
static void mamba_step_impl(const float* x,
                            const W* w_in, const float* conv_w, const float* conv_b,
                            const float* dt_bias, const float* a_log, const float* D,
                            const float* norm_w, const W* w_out,
                            float* conv_state, float* ssm_state,
                            float* out,
                            int d, int d_inner, int conv_dim, int n_heads,
                            int d_head, int d_state, int k, float eps, int qkind = -1) {
    const int proj_size = d_inner + conv_dim + n_heads;
    float* ws = mamba_workspace((size_t)proj_size + (size_t)conv_dim +
                                (size_t)d_inner * 2 + 64);
    float* proj = ws;
    float* bc = proj + proj_size;
    float* y = bc + conv_dim;
    float* normed = y + d_inner;

    mamba_linear(x, w_in, proj, 1, d, proj_size, qkind);

    {
        const int n = conv_dim;
        k_mamba_step_conv<<<(n + 127) / 128, 128>>>(proj, conv_w, conv_b,
                                                    conv_state, bc,
                                                    d_inner, conv_dim, k);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    {
        const int n = n_heads * d_head;
        k_mamba_step_scan<<<(n + 255) / 256, 256>>>(proj, bc, dt_bias, a_log, D,
                                                    ssm_state, y,
                                                    d_inner, n_heads, d_head,
                                                    d_state, proj_size);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    {
        k_mamba_step_norm<<<1, 256, sizeof(float) * 32>>>(y, proj, norm_w, normed,
                                                          d_inner, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    mamba_linear(normed, w_out, out, 1, d_inner, d, qkind);
}

void mamba2_step(const float* x,
                 const float* w_in, const float* conv_w, const float* conv_b,
                 const float* dt_bias, const float* a_log, const float* D,
                 const float* norm_w, const float* w_out,
                 float* conv_state, float* ssm_state, float* out,
                 int d, int d_inner, int conv_dim, int n_heads,
                 int d_head, int d_state, int k, float eps) {
    mamba_step_impl<float>(x, w_in, conv_w, conv_b, dt_bias, a_log, D, norm_w, w_out,
                           conv_state, ssm_state, out, d, d_inner, conv_dim, n_heads,
                           d_head, d_state, k, eps);
}

void mamba2_step_half(const float* x,
                 const u16* w_in, const float* conv_w, const float* conv_b,
                 const float* dt_bias, const float* a_log, const float* D,
                 const float* norm_w, const u16* w_out,
                 float* conv_state, float* ssm_state, float* out,
                 int d, int d_inner, int conv_dim, int n_heads,
                 int d_head, int d_state, int k, float eps, int qkind) {
    mamba_step_impl<u16>(x, w_in, conv_w, conv_b, dt_bias, a_log, D, norm_w, w_out,
                         conv_state, ssm_state, out, d, d_inner, conv_dim, n_heads,
                         d_head, d_state, k, eps, qkind);
}

void mamba2_backward_half(const float* x,
                     const u16* w_in,
                     const float* conv_w,
                     const float* conv_b,
                     const float* dt_bias,
                     const float* a_log,
                     const float* D,
                     const float* norm_w,
                     const u16* w_out,
                     const float* dout,
                     float* dx,
                     float* dw_in,
                     float* dconv_w,
                     float* dconv_b,
                     float* ddt_bias,
                     float* da_log,
                     float* dD,
                     float* dnorm_w,
                     float* dw_out,
                     float* dbc_out,
                     int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                     int d_head, int d_state, int k, float eps,
                     const i32* segment_ids) {
    mamba2_backward_impl<u16>(x, w_in, conv_w, conv_b, dt_bias, a_log, D,
                              norm_w, w_out, dout, dx, dw_in, dconv_w, dconv_b,
                              ddt_bias, da_log, dD, dnorm_w, dw_out, dbc_out,
                              B, T, d, d_inner, conv_dim, n_heads, d_head,
                              d_state, k, eps, segment_ids);
}

}
}
