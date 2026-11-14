// IBM Granite-4.0 Mamba-2 layers on the GPU.
//
// The block (from the reference modeling code):
//
//   projected = in_proj(x)                 -> [gate | bc | dt]
//   bc        = causal depthwise conv1d(k) + SiLU
//   x, B, C   = split(bc)
//   dt        = softplus(dt + dt_bias)
//   state     = exp(A*dt) * state + (dt*B) (x)
//   y         = C . state + D * x
//   out       = rmsnorm(y * silu(gate)) * norm_weight -> out_proj
//
// Parallelisation: one CUDA block per (batch, head, head_dim) row. Each row
// owns a private state row of d_state floats (kept in shared memory) and
// walks the sequence; the update is a rank-1 outer product, so the work per
// step is O(d_state) and fully parallel inside the block.
//
// Layout (all contiguous, row-major):
//   x      [B, T, d]                    the layer input
//   proj   [B, T, proj_size]            in_proj output (gate | bc | dt)
//   bc     [B, T, conv_dim]             post-conv values (x | B | C)
//   state  [B, n_heads, d_head, d_state] the recurrent state (in/out)
//   y      [B, T, d_inner]              the scan output
//   out    [B, T, d]                    the layer output

#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <type_traits>

namespace gai {
namespace cuda_ops {

#define CU_MAMBA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

// ---------------------------------------------------------------- workspace
// The projection, the conv output and the scan output are transient: they
// live in a cached device buffer (never in host memory: the kernels below
// read and write device pointers).
static void*   g_mamba_ws = nullptr;
static size_t   g_mamba_ws_bytes = 0;

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
    if (g_mamba_ws) cudaFree(g_mamba_ws);
    g_mamba_ws = nullptr;
    g_mamba_ws_bytes = 0;
}

// ---------------------------------------------------------------- reductions
// A block-wide sum of one value per thread. The warp-level shuffle is not
// enough (a 256-thread block has 8 warps): the per-warp sums go through
// shared memory and the first warp reduces them.
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
    return smem[0];
}

// ---------------------------------------------------------------- conv1d
// Causal depthwise conv1d over the sequence + SiLU.
//   in  [B, T, C]
//   out [B, T, C]
// One block per (b, channel); the window (k-1 previous values) lives in
// shared memory. The window holds the RAW inputs: the activation is applied
// to the OUTPUT (that is what causal_conv1d_fn(activation="silu") does).
__global__ void k_mamba_conv1d(const float* __restrict__ in,
                               const float* __restrict__ w,     // [C, k]
                               const float* __restrict__ bias,  // [C] or null
                               float* __restrict__ out,
                               int B, int T, int C, int in_stride, int k) {
    extern __shared__ float sh[];
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    float* win = sh;                 // [k-1] the previous values
    float* val = sh + (k - 1);       // [1]   the current value

    if (lane < k - 1) win[lane] = 0.0f;
    __syncthreads();

    const float* wc = w + size_t(c) * k;
    const float bc = bias ? bias[c] : 0.0f;

    for (int t = 0; t < T; ++t) {
        const float x = in[((size_t(b) * T + t) * in_stride) + c];
        if (lane == 0) *val = x;
        __syncthreads();

        float acc = bc;
        for (int j = 0; j < k; ++j) {
            const float v = (j == k - 1) ? *val : win[j];
            acc += wc[j] * v;
        }
        out[((size_t(b) * T + t) * C) + c] = acc / (1.0f + expf(-acc));
        __syncthreads();
        if (lane == 0) {
            for (int j = 0; j + 1 < k - 1; ++j) win[j] = win[j + 1];
            if (k - 1 > 0) win[k - 2] = x;
        }
        __syncthreads();
        (void)nthr;
    }
}

// ---------------------------------------------------------------- scan
// One block per (b, head, head_dim) row; the state row is in shared memory.
//   proj  [B, T, proj_size]   gate | bc | dt
//   bc    [B, T, conv_dim]    x | B | C
//   state [B, n_heads, d_head, d_state]
//   y     [B, T, d_inner]
__global__ void k_mamba_scan(const float* __restrict__ proj,
                             const float* __restrict__ bc,
                             const float* __restrict__ dtw,   // [n_heads]
                             const float* __restrict__ alog,  // [n_heads]
                             const float* __restrict__ Dv,    // [n_heads]
                             float* __restrict__ state,
                             float* __restrict__ y,
                             int B, int T, int n_heads, int d_head, int d_state,
                             int d_inner, int conv_dim, int proj_size) {
    extern __shared__ float sh_state[];   // [d_state]
    __shared__ float sh_red[32];
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const int inner = n_heads * d_head;
    const int b     = row / inner;
    const int rem   = row - b * inner;
    const int h     = rem / d_head;
    const int i     = rem - h * d_head;

    // The state is CARRIED between calls (the prefill continues where the
    // previous chunk stopped): it must be read, not zeroed. A null state
    // means "no state to carry" (the backward's recompute): the scan then
    // starts from zero.
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
    // The channel of this row inside d_inner: the head offset matters.
    // Writing to `i` instead of `h * d_head + i` made every head overwrite
    // the same channels (a race between the blocks) and left the rest of
    // d_inner at zero.
    const int   x_off   = h * d_head + i;         // inside d_inner
    const int   B_off   = d_inner;                 // inside conv_dim
    const int   C_off   = d_inner + d_state;       // inside conv_dim

    for (int t = 0; t < T; ++t) {
        const float* prow = proj + (size_t(b) * T + t) * proj_size;
        const float* brow = bc   + (size_t(b) * T + t) * conv_dim;

        // dt sits at the END of the projection: [gate | bc | dt]
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
        // Reduce the partial dot products across the whole block.
        acc = block_reduce_sum(acc, sh_red, lane, nthr);

        if (lane == 0)
            y[(size_t(b) * T + t) * d_inner + x_off] = acc + dskip * xv;
        __syncthreads();
    }

    // Write the final state back.
    if (state) {
        for (int j = lane; j < d_state; j += nthr)
            state[base * d_state + j] = sh_state[j];
    }
}

// ---------------------------------------------------------------- gated norm
// y = rmsnorm(y * silu(gate)) * weight, per token.
//   y    [N, d_inner]
//   gate [N, gate_stride]   the gate is the first d_inner columns of the
//                            projection, whose rows are gate_stride wide
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

// ---------------------------------------------------------------- public API
// The two projections run with the weights in the caller's precision: fp32
// for the reference path, half for the fp16 storage mode (the tensor cores
// multiply, the accumulator stays fp32).
template <typename W>
static inline void mamba_linear(const float* x, const W* w, float* y, int M, int K, int N) {
    if constexpr (std::is_same<W, u16>::value) {
        linear_forward_fp16(x, w, y, M, K, N);
    } else {
        linear_forward(x, w, y, M, K, N);
    }
}

template <typename W>
static void mamba2_forward_impl(const float* x,          // [B, T, d]
                    const W* w_in,       // [proj_size, d]
                    const float* conv_w,     // [conv_dim, k]
                    const float* conv_b,     // [conv_dim] or null
                    const float* dt_bias,    // [n_heads]
                    const float* a_log,      // [n_heads]
                    const float* D,          // [n_heads]
                    const float* norm_w,     // [d_inner]
                    const W* w_out,      // [d, d_inner]
                    float* state,            // [B, n_heads, d_head, d_state]
                    float* out,              // [B, T, d]
                    int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                    int d_head, int d_state, int k, float eps) {
    if (B <= 0 || T <= 0) return;
    const int proj_size = d_inner + conv_dim + n_heads;
    const size_t n_tok = size_t(B) * T;

    // The transient buffers live on the DEVICE (the kernels below take
    // device pointers).
    float* ws = mamba_workspace(n_tok * (size_t(proj_size) + conv_dim + d_inner) + 64);
    float* proj = ws;
    float* bc   = proj + n_tok * size_t(proj_size);
    float* y    = bc   + n_tok * size_t(conv_dim);

    // 1) in_proj
    mamba_linear(x, w_in, proj, B * T, d, proj_size);

    // 2) causal depthwise conv1d + SiLU over the bc part
    {
        dim3 grid(conv_dim, B);
        const int block = 128;
        const size_t sh = sizeof(float) * (k > 1 ? k : 1);
        k_mamba_conv1d<<<grid, block, sh>>>(proj + d_inner, conv_w, conv_b,
                                           bc, B, T, conv_dim, proj_size, k);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    // 3) the selective scan
    {
        const int rows = B * n_heads * d_head;
        const int block = 256;
        const size_t sh = sizeof(float) * d_state;
        k_mamba_scan<<<rows, block, sh>>>(proj, bc, dt_bias, a_log, D,
                                          state, y, B, T, n_heads, d_head,
                                          d_state, d_inner, conv_dim, proj_size);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    // 4) gated RMSNorm + out_proj
    {
        const int block = 256;
        k_mamba_gated_norm<<<B * T, block>>>(y, proj, norm_w,
                                             y, B * T, d_inner, proj_size, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    mamba_linear(y, w_out, out, B * T, d_inner, d);
}

void mamba2_forward(const float* x,          // [B, T, d]
                    const float* w_in,       // [proj_size, d]
                    const float* conv_w,     // [conv_dim, k]
                    const float* conv_b,     // [conv_dim] or null
                    const float* dt_bias,    // [n_heads]
                    const float* a_log,      // [n_heads]
                    const float* D,          // [n_heads]
                    const float* norm_w,     // [d_inner]
                    const float* w_out,      // [d, d_inner]
                    float* state,            // [B, n_heads, d_head, d_state]
                    float* out,              // [B, T, d]
                    int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                    int d_head, int d_state, int k, float eps) {
    mamba2_forward_impl<float>(x, w_in, conv_w, conv_b, dt_bias, a_log, D,
                               norm_w, w_out, state, out,
                               B, T, d, d_inner, conv_dim, n_heads,
                               d_head, d_state, k, eps);
}

void mamba2_forward_half(const float* x,          // [B, T, d]
                    const u16* w_in,       // [proj_size, d]
                    const float* conv_w,     // [conv_dim, k]
                    const float* conv_b,     // [conv_dim] or null
                    const float* dt_bias,    // [n_heads]
                    const float* a_log,      // [n_heads]
                    const float* D,          // [n_heads]
                    const float* norm_w,     // [d_inner]
                    const u16* w_out,      // [d, d_inner]
                    float* state,            // [B, n_heads, d_head, d_state]
                    float* out,              // [B, T, d]
                    int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                    int d_head, int d_state, int k, float eps) {
    mamba2_forward_impl<u16>(x, w_in, conv_w, conv_b, dt_bias, a_log, D,
                             norm_w, w_out, state, out,
                             B, T, d, d_inner, conv_dim, n_heads,
                             d_head, d_state, k, eps);
}

// ---------------------------------------------------------------- backward kernels

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

// The forward scan that RECORDS the state trajectory.
//
//   traj [rows_batch, T+1, d_state]   traj[l][0] = 0 (the state before the
//   first token), traj[l][t+1] = the state after the token t.
//
// The backward needs two states of every token: s_t (the gradient of the
// decay dA) and s_{t+1} (the gradient of the C channels, because
// y_t = C_t . s_{t+1}). The old kernel replayed the whole prefix for
// every token - O(T^2) per row, which is 3.2e11 state updates for one
// Granite layer at T=1024 (hours per step). Recording the trajectory once
// makes the backward O(T) and lets it use s_{t+1} exactly.
//
// The state starts at zero: the training sequence starts from zero (the
// model resets the recurrent states before every sequence) and the
// backward assumes the same.
__global__ void k_mamba_scan_traj(const float* __restrict__ proj,
                                  const float* __restrict__ bc,
                                  const float* __restrict__ dtw,   // [n_heads]
                                  const float* __restrict__ alog,  // [n_heads]
                                  const float* __restrict__ Dv,    // [n_heads]
                                  float* __restrict__ traj,        // [rows_batch, T+1, d_state]
                                  float* __restrict__ y,           // [B, T, d_inner]
                                  int B, int T, int n_heads, int d_head, int d_state,
                                  int d_inner, int conv_dim, int proj_size,
                                  int row_offset) {
    extern __shared__ float sh_state[];   // [d_state]
    __shared__ float sh_red[32];
    const int lrow = blockIdx.x;          // the row inside the batch
    const int row  = lrow + row_offset;   // the global row
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
    const int   B_off   = d_inner;
    const int   C_off   = d_inner + d_state;

    float* mytraj = traj + (size_t)lrow * (size_t)(T + 1) * d_state;

    for (int j = lane; j < d_state; j += nthr) { sh_state[j] = 0.0f; mytraj[j] = 0.0f; }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
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
        __syncthreads();
        // The state after this token: s_{t+1}.
        for (int j = lane; j < d_state; j += nthr)
            mytraj[(size_t)(t + 1) * d_state + j] = sh_state[j];
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
                                 int row_offset) {
    extern __shared__ float sh_mem[];
    float* sh_ds  = sh_mem;              // [d_state]
    float* sh_red = sh_mem + d_state;    // [32]

    const int lrow = blockIdx.x;         // the row inside the batch
    const int row  = lrow + row_offset;  // the global row
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
    const int   B_off     = d_inner;
    const int   C_off     = d_inner + d_state;

    const float* mytraj = traj + (size_t)lrow * (size_t)(T + 1) * d_state;

    // sh_ds holds the part of the state gradient that comes from the
    // future (the y contribution of the previous token is added at every
    // step). It starts at zero: the last token has no future.
    for (int j = lane; j < d_state; j += nthr) sh_ds[j] = 0.0f;
    __syncthreads();

    for (int t = T - 1; t >= 0; --t) {
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

        // s_t (the state BEFORE this token) and s_{t+1} (the state after
        // it). The C channels multiply s_{t+1} in the forward, so their
        // gradient uses s_{t+1}; the decay dA multiplies s_t, so its
        // gradient uses s_t. The old kernel used s_t for both.
        const float* s_prev = mytraj + (size_t)t * d_state;
        const float* s_curr = mytraj + (size_t)(t + 1) * d_state;

        if (dD && lane == 0) atomicAdd(&dD[h], dy_val * xv);
        float dx_val = dy_val * dskip;

        float local_d_dA = 0.0f;
        float local_d_dt = 0.0f;

        for (int j = lane; j < d_state; j += nthr) {
            const float Cj = brow[C_off + j];
            const float Bj = brow[B_off + j];

            atomicAdd(&d_brow[C_off + j], dy_val * s_curr[j]);

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
                                   float* __restrict__ dconv_w,
                                   float* __restrict__ dconv_b,
                                   float* __restrict__ dproj_bc,
                                   int B, int T, int conv_dim, int proj_size, int k) {
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const int nthr = blockDim.x;

    const float* wc = conv_w + size_t(c) * k;
    const float bcv = conv_b ? conv_b[c] : 0.0f;

    for (int t = lane; t < T; t += nthr) {
        float raw_acc = bcv;
        for (int j = 0; j < k; ++j) {
            const int tin = t - (k - 1 - j);
            if (tin >= 0) {
                const size_t pidx = (size_t(b) * T + tin) * proj_size + c;
                raw_acc += wc[j] * proj[pidx];
            }
        }
        const size_t idx = (size_t(b) * T + t) * conv_dim + c;
        const float sig = 1.0f / (1.0f + expf(-raw_acc));
        const float dsilu = sig * (1.0f + raw_acc * (1.0f - sig));
        const float dacc = dbc[idx] * dsilu;

        if (dconv_b) atomicAdd(&dconv_b[c], dacc);

        for (int j = 0; j < k; ++j) {
            const int tin = t - (k - 1 - j);
            if (tin >= 0) {
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
                     int B, int T, int d, int d_inner, int conv_dim, int n_heads,
                     int d_head, int d_state, int k, float eps) {
    if (B <= 0 || T <= 0) return;
    const int proj_size = d_inner + conv_dim + n_heads;
    const size_t n_tok = size_t(B) * T;

    // Transient workspace for backward:
    // proj [N, proj_size] + bc [N, conv_dim] + y_scan [N, d_inner] +
    // dy_norm [N, d_inner] + dy_scan [N, d_inner] + dgate [N, d_inner] +
    // dbc [N, conv_dim] + ddt_raw [N, n_heads] + dproj_bc [N, conv_dim] +
    // dproj [N, proj_size]
    //
    // Plus the TRAJECTORY of the recurrent state: [rows_batch, T+1, d_state].
    // The whole trajectory of Granite (B=1, T=1024) would be 1.6 GB, so the
    // rows are processed in batches and the trajectory buffer is bounded
    // (32 MB). The batch is big enough to keep the GPU busy (a few dozen
    // blocks per launch).
    const int rows = B * n_heads * d_head;
    const size_t per_row_traj = (size_t)(T + 1) * d_state;
    const size_t traj_budget = 8u << 20;          // 8M floats = 32 MB
    int rows_batch = (int)(traj_budget / (per_row_traj ? per_row_traj : 1));
    if (rows_batch < 1) rows_batch = 1;
    if (rows_batch > rows) rows_batch = rows;
    const size_t traj_floats = (size_t)rows_batch * per_row_traj;

    const size_t total_floats = n_tok * (size_t(proj_size) * 2 + size_t(conv_dim) * 3 + size_t(d_inner) * 5 + n_heads) + traj_floats + 256;
    float* ws = mamba_workspace(total_floats);

    float* proj     = ws;
    float* bc       = proj     + n_tok * size_t(proj_size);
    float* y_scan   = bc       + n_tok * size_t(conv_dim);
    float* y_norm   = y_scan   + n_tok * size_t(d_inner);
    float* dy_norm  = y_norm   + n_tok * size_t(d_inner);
    float* dy_scan  = dy_norm  + n_tok * size_t(d_inner);
    float* dgate    = dy_scan  + n_tok * size_t(d_inner);
    float* dbc      = dgate    + n_tok * size_t(d_inner);
    float* ddt_raw  = dbc      + n_tok * size_t(conv_dim);
    float* dproj_bc = ddt_raw  + n_tok * size_t(n_heads);
    float* dproj    = dproj_bc + n_tok * size_t(conv_dim);
    float* traj     = dproj    + n_tok * size_t(proj_size);

    // Forward recompute
    linear_forward(x, w_in, proj, B * T, d, proj_size);
    {
        dim3 grid(conv_dim, B);
        const int block = 128;
        const size_t sh = sizeof(float) * (k > 1 ? k : 1);
        k_mamba_conv1d<<<grid, block, sh>>>(proj + d_inner, conv_w, conv_b,
                                            bc, B, T, conv_dim, proj_size, k);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    {
        // The forward scan, recording the state trajectory (and the scan
        // output the gated norm needs).
        const int block = 256;
        const size_t sh = sizeof(float) * d_state;
        for (int r0 = 0; r0 < rows; r0 += rows_batch) {
            const int nb = (rows - r0 < rows_batch) ? (rows - r0) : rows_batch;
            k_mamba_scan_traj<<<nb, block, sh>>>(proj, bc, dt_bias, a_log, D,
                                                 traj, y_scan, B, T, n_heads, d_head,
                                                 d_state, d_inner, conv_dim, proj_size,
                                                 r0);
            CU_MAMBA_CHECK(cudaGetLastError());
        }
    }

    // B1) Out projection backward.
    //
    // The out projection reads the gated norm OUTPUT, so its weight
    // gradient needs that value. The old code passed a null pointer for it
    // ("x"): the GEMM of the weight gradient would have read the address
    // zero (an illegal memory access). The gated norm output is recomputed
    // here (the same kernel the forward uses).
    {
        const int block = 256;
        k_mamba_gated_norm<<<B * T, block>>>(y_scan, proj, norm_w,
                                             y_norm, B * T, d_inner, proj_size, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    linear_backward(y_norm, w_out, dout, dy_norm, dw_out, B * T, d_inner, d);

    // B2) Gated RMSNorm backward
    {
        const int block = 256;
        const size_t sh = sizeof(float) * 32;
        k_mamba_gated_norm_bwd<<<B * T, block, sh>>>(dy_norm, y_scan, proj, norm_w,
                                                     dy_scan, dgate, dnorm_w,
                                                     B * T, d_inner, proj_size, eps);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    // B3) Scan backward
    CU_MAMBA_CHECK(cudaMemset(dbc, 0, sizeof(float) * n_tok * conv_dim));
    CU_MAMBA_CHECK(cudaMemset(ddt_raw, 0, sizeof(float) * n_tok * n_heads));
    {
        const int block = 256;
        const size_t sh = sizeof(float) * (d_state + 32);
        for (int r0 = 0; r0 < rows; r0 += rows_batch) {
            const int nb = (rows - r0 < rows_batch) ? (rows - r0) : rows_batch;
            k_mamba_scan_bwd<<<nb, block, sh>>>(proj, bc, dy_scan, dt_bias, a_log, D,
                                                traj, dbc, ddt_raw, dD, da_log, ddt_bias,
                                                B, T, n_heads, d_head, d_state,
                                                d_inner, conv_dim, proj_size, r0);
            CU_MAMBA_CHECK(cudaGetLastError());
        }
    }

    // B4) Conv1d backward
    CU_MAMBA_CHECK(cudaMemset(dproj_bc, 0, sizeof(float) * n_tok * conv_dim));
    {
        dim3 grid(conv_dim, B);
        const int block = 128;
        k_mamba_conv1d_bwd<<<grid, block>>>(proj + d_inner, conv_w, conv_b,
                                            dbc, dconv_w, dconv_b, dproj_bc,
                                            B, T, conv_dim, proj_size, k);
        CU_MAMBA_CHECK(cudaGetLastError());
    }

    // B5) Assemble dproj and in_proj backward
    {
        const int block = 256;
        k_mamba_assemble_dproj<<<B * T, block>>>(dgate, dproj_bc, ddt_raw, dproj,
                                                B * T, d_inner, conv_dim, n_heads, proj_size);
        CU_MAMBA_CHECK(cudaGetLastError());
    }
    linear_backward(x, w_in, dproj, dx, dw_in, B * T, d, proj_size);
}

}  // namespace cuda_ops
}  // namespace gai
