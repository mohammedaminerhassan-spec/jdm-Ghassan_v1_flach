// IBM Granite-4.0 Mamba-2 layer (CPU).
//
// The block (reference modeling code):
//
//   projected = in_proj(x)                  -> [gate | bc | dt]
//   bc        = causal depthwise conv1d(k) + SiLU
//   x, B, C   = split(bc)
//   dt        = softplus(dt + dt_bias)
//   state     = exp(A*dt) * state + (dt*B) (x)
//   y         = C . state + D * x
//   out       = rmsnorm(y * silu(gate)) * norm_weight -> out_proj
//
// The state is [B, n_heads, d_head, d_state] and is carried between calls:
// a prefill continues the state of the previous chunk, a decode step
// advances it by one token.

#include "core/ops_cpu.h"
#include "core/dtype.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

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
    const int N = B * T;

    // ---- 1) in_proj
    std::vector<float> proj(static_cast<size_t>(N) * proj_size);
    linear_forward(x, w_in, proj.data(), N, d, proj_size);

    // ---- 2) causal depthwise conv1d + SiLU over the bc part
    //
    // The reference (mamba_ssm / Mamba2Mixer) applies the activation to the
    // OUTPUT of the causal depthwise convolution:
    //
    //     x = causal_conv1d_fn(x, w, b, activation="silu")
    //
    // so the convolution window holds the RAW inputs, not the post-SiLU
    // values. The old code (a) never read the current input from the
    // projection at all (it convolved zeros) and (b) fed the post-SiLU
    // values back into the window: the whole B/C/x block was garbage.
    std::vector<float> bc(static_cast<size_t>(N) * conv_dim);
    {
        const int win_len = (k > 1) ? k - 1 : 1;
#ifdef GAI_OPENMP
        #pragma omp parallel
#endif
        {
            // The convolution window is private to the thread (one channel
            // per iteration, so a shared window would be a data race).
            std::vector<float> win(static_cast<size_t>(win_len), 0.0f);
#ifdef GAI_OPENMP
            #pragma omp for schedule(static)
#endif
            for (int c = 0; c < conv_dim; ++c) {
                const float* wc = conv_w + static_cast<size_t>(c) * k;
                const float bcv = conv_b ? conv_b[c] : 0.0f;
                for (int b = 0; b < B; ++b) {
                    // The convolution window is per sequence.
                    std::fill(win.begin(), win.end(), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const size_t idx = (static_cast<size_t>(b) * T + t) * conv_dim + c;
                        // The raw input: the bc part of the projection, which
                        // starts at d_inner (after the gate).
                        const size_t pidx = (static_cast<size_t>(b) * T + t) * proj_size + d_inner + c;
                        const float raw = proj[pidx];
                        float acc = bcv;
                        for (int j = 0; j < k; ++j) {
                            const float v = (j == k - 1) ? raw : win[static_cast<size_t>(j)];
                            acc += wc[j] * v;
                        }
                        bc[idx] = acc / (1.0f + std::exp(-acc));
                        for (int j = 0; j + 1 < win_len; ++j) win[static_cast<size_t>(j)] = win[static_cast<size_t>(j + 1)];
                        win[static_cast<size_t>(win_len - 1)] = raw;
                    }
                }
            }
        }
    }

    // ---- 3) the selective scan
    std::vector<float> y(static_cast<size_t>(N) * d_inner);
    {
        const int inner = n_heads * d_head;
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static) if (B * inner > 4)
#endif
        for (int row = 0; row < B * inner; ++row) {
            const int b  = row / inner;
            const int rem = row - b * inner;
            const int h  = rem / d_head;
            const int i  = rem - h * d_head;
            const float a       = -std::exp(a_log[h]);
            const float dt_bias_v = dt_bias[h];
            const float dskip   = D[h];
            float* srow = state + (static_cast<size_t>(row) * d_state);

            for (int t = 0; t < T; ++t) {
                const float* prow = proj.data() + (static_cast<size_t>(b) * T + t) * proj_size;
                const float* brow = bc.data()   + (static_cast<size_t>(b) * T + t) * conv_dim;
                const float dt_in = prow[proj_size - n_heads + h];
                const float dt_val = (dt_in + dt_bias_v > 20.0f) ? (dt_in + dt_bias_v)
                                                                : std::log1p(std::exp(dt_in + dt_bias_v));
                const float dA = std::exp(a * dt_val);
                // The x channel of this row inside d_inner: the head offset matters
                // (reading `i` instead of `h * d_head + i` gave every head
                // the same input, i.e. head 1 read head 0's channels).
                const float xv = brow[h * d_head + i];
                float acc = 0.0f;
                for (int j = 0; j < d_state; ++j) {
                    srow[j] = dA * srow[j] + (dt_val * brow[d_inner + j]) * xv;
                    acc += brow[d_inner + d_state + j] * srow[j];
                }
                // The channel of this row inside d_inner: the head offset
                // matters. Writing to `i` instead of `h * d_head + i` made
                // every head overwrite the same channels (a data race) and
                // left the rest of d_inner at zero.
                y[(static_cast<size_t>(b) * T + t) * d_inner + h * d_head + i] = acc + dskip * xv;
            }
        }
    }

    // ---- 4) gated RMSNorm
    {
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int n = 0; n < N; ++n) {
            float* yr = y.data() + static_cast<size_t>(n) * d_inner;
            // The gate lives in the PROJECTION, whose rows are proj_size
            // wide (gate | bc | dt). Indexing it with d_inner read the wrong
            // elements for every token but the first.
            const float* gr = proj.data() + static_cast<size_t>(n) * proj_size;
            double var = 0.0;
            for (int c = 0; c < d_inner; ++c) {
                const float g = gr[c] / (1.0f + std::exp(-gr[c]));
                const float v = yr[c] * g;
                yr[c] = v;
                var += static_cast<double>(v) * v;
            }
            var /= static_cast<double>(d_inner);
            const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
            for (int c = 0; c < d_inner; ++c) yr[c] = yr[c] * inv * norm_w[c];
        }
    }

    // ---- 5) out_proj
    linear_forward(y.data(), w_out, out, N, d_inner, d);
}

}  // namespace cpu
}  // namespace gai


