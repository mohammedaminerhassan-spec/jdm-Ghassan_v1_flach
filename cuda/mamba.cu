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
                               int B, int T, int C, int k) {
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
        const float x = in[((size_t(b) * T + t) * C) + c];
        if (lane == 0) *val = x;
        __syncthreads();

        float acc = bc;
        for (int j = 0; j < k; ++j) {
            const float v = (j == k - 1) ? *val : win[j];
            acc += wc[j] * v;
        }
        out[((size_t(b) * T + t) * C) + c] = acc / (1.0f + expf(-acc));
        __syncthreads();
        if (lane < k - 1) {
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
    // previous chunk stopped): it must be read, not zeroed.
    const int base = (b * n_heads + h) * d_head + i;
    for (int j = lane; j < d_state; j += nthr) sh_state[j] = state[base * d_state + j];
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
    for (int j = lane; j < d_state; j += nthr)
        state[base * d_state + j] = sh_state[j];
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
    linear_forward(x, w_in, proj, B * T, d, proj_size);

    // 2) causal depthwise conv1d + SiLU over the bc part
    {
        dim3 grid(conv_dim, B);
        const int block = 128;
        const size_t sh = sizeof(float) * (k > 1 ? k : 1);
        k_mamba_conv1d<<<grid, block, sh>>>(proj + d_inner, conv_w, conv_b,
                                           bc, B, T, conv_dim, k);
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
    linear_forward(y, w_out, out, B * T, d_inner, d);
}

}  // namespace cuda_ops
}  // namespace gai
