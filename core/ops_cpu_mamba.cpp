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
#include <type_traits>
#include <vector>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

namespace {
// The Mamba-2 layer with the weights in the caller's precision: fp32 for
// the reference path, half for the fp16 storage mode. Only the two
// projections differ; the conv1d, dt_bias, A_log, D and the gated norm stay
// fp32 (they are numerically fragile).
template <typename W>
inline void mamba_linear(const float* x, const W* w, float* y, int M, int K, int N) {
    if constexpr (std::is_same<W, u16>::value) {
        linear_forward_fp16(x, w, y, M, K, N);
    } else {
        linear_forward(x, w, y, M, K, N);
    }
}
}  // namespace

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
    const int N = B * T;

    // ---- 1) in_proj
    std::vector<float> proj(static_cast<size_t>(N) * proj_size);
    mamba_linear(x, w_in, proj.data(), N, d, proj_size);

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
    mamba_linear(y.data(), w_out, out, N, d_inner, d);
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
    mamba2_forward_impl<float>(x, w_in, conv_w, conv_b, dt_bias, a_log, D, norm_w,
                               w_out, state, out, B, T, d, d_inner, conv_dim,
                               n_heads, d_head, d_state, k, eps);
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
    mamba2_forward_impl<u16>(x, w_in, conv_w, conv_b, dt_bias, a_log, D, norm_w,
                             w_out, state, out, B, T, d, d_inner, conv_dim,
                             n_heads, d_head, d_state, k, eps);
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
    const int N = B * T;

    // 1) Forward recomputation: proj = x * w_in^T
    std::vector<float> proj(static_cast<size_t>(N) * proj_size);
    mamba_linear(x, w_in, proj.data(), N, d, proj_size);

    // 2) Forward conv1d + SiLU: keep both raw convolution output and post-SiLU bc
    std::vector<float> conv_raw(static_cast<size_t>(N) * conv_dim, 0.0f);
    std::vector<float> bc(static_cast<size_t>(N) * conv_dim, 0.0f);
    const int win_len = (k > 1) ? k - 1 : 1;
    {
#ifdef GAI_OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<float> win(static_cast<size_t>(win_len), 0.0f);
#ifdef GAI_OPENMP
            #pragma omp for schedule(static)
#endif
            for (int c = 0; c < conv_dim; ++c) {
                const float* wc = conv_w + static_cast<size_t>(c) * k;
                const float bcv = conv_b ? conv_b[c] : 0.0f;
                for (int b = 0; b < B; ++b) {
                    std::fill(win.begin(), win.end(), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const size_t idx = (static_cast<size_t>(b) * T + t) * conv_dim + c;
                        const size_t pidx = (static_cast<size_t>(b) * T + t) * proj_size + d_inner + c;
                        const float raw = proj[pidx];
                        float acc = bcv;
                        for (int j = 0; j < k; ++j) {
                            const float v = (j == k - 1) ? raw : win[static_cast<size_t>(j)];
                            acc += wc[j] * v;
                        }
                        conv_raw[idx] = acc;
                        bc[idx] = acc / (1.0f + std::exp(-acc));
                        for (int j = 0; j + 1 < win_len; ++j) win[static_cast<size_t>(j)] = win[static_cast<size_t>(j + 1)];
                        win[static_cast<size_t>(win_len - 1)] = raw;
                    }
                }
            }
        }
    }

    // 3) Selective scan forward: compute y_scan
    std::vector<float> y_scan(static_cast<size_t>(N) * d_inner, 0.0f);
    const int inner = n_heads * d_head;
    {
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static) if (B * inner > 4)
#endif
        for (int row = 0; row < B * inner; ++row) {
            const int b   = row / inner;
            const int rem = row - b * inner;
            const int h   = rem / d_head;
            const int i   = rem - h * d_head;
            const float a_val     = -std::exp(a_log[h]);
            const float dt_bias_v = dt_bias[h];
            const float dskip     = D[h];
            std::vector<float> s(static_cast<size_t>(d_state), 0.0f);

            for (int t = 0; t < T; ++t) {
                const float* prow = proj.data() + (static_cast<size_t>(b) * T + t) * proj_size;
                const float* brow = bc.data()   + (static_cast<size_t>(b) * T + t) * conv_dim;
                const float dt_in = prow[proj_size - n_heads + h];
                const float dt_val = (dt_in + dt_bias_v > 20.0f) ? (dt_in + dt_bias_v)
                                                                : std::log1p(std::exp(dt_in + dt_bias_v));
                const float dA = std::exp(a_val * dt_val);
                const float xv = brow[h * d_head + i];
                float acc = 0.0f;
                for (int j = 0; j < d_state; ++j) {
                    s[j] = dA * s[j] + (dt_val * brow[d_inner + j]) * xv;
                    acc += brow[d_inner + d_state + j] * s[j];
                }
                y_scan[(static_cast<size_t>(b) * T + t) * d_inner + h * d_head + i] = acc + dskip * xv;
            }
        }
    }

    // 4) Gated RMSNorm forward: compute y_norm, v, s_inv
    std::vector<float> y_norm(static_cast<size_t>(N) * d_inner, 0.0f);
    std::vector<float> v_norm(static_cast<size_t>(N) * d_inner, 0.0f);
    std::vector<float> s_inv(static_cast<size_t>(N), 0.0f);
    {
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int n = 0; n < N; ++n) {
            const float* yr = y_scan.data() + static_cast<size_t>(n) * d_inner;
            const float* gr = proj.data() + static_cast<size_t>(n) * proj_size;
            float* yn = y_norm.data() + static_cast<size_t>(n) * d_inner;
            float* vn = v_norm.data() + static_cast<size_t>(n) * d_inner;
            double var = 0.0;
            for (int c = 0; c < d_inner; ++c) {
                const float g = gr[c] / (1.0f + std::exp(-gr[c]));
                const float val = yr[c] * g;
                vn[c] = val;
                var += static_cast<double>(val) * val;
            }
            var /= static_cast<double>(d_inner);
            const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
            s_inv[n] = inv;
            for (int c = 0; c < d_inner; ++c) yn[c] = vn[c] * inv * norm_w[c];
        }
    }

    // ---- BACKWARD PASS ----

    // B1) Out projection backward:
    std::vector<float> dy_norm(static_cast<size_t>(N) * d_inner, 0.0f);
    linear_backward(y_norm.data(), w_out, dout, dy_norm.data(), dw_out, N, d_inner, d);

    // B2) Gated RMSNorm backward:
    std::vector<float> dy_scan(static_cast<size_t>(N) * d_inner, 0.0f);
    std::vector<float> dgate(static_cast<size_t>(N) * d_inner, 0.0f);
    {
        if (dnorm_w) {
            std::vector<double> dnorm_accum(static_cast<size_t>(d_inner), 0.0);
            for (int n = 0; n < N; ++n) {
                const float* dyn = dy_norm.data() + static_cast<size_t>(n) * d_inner;
                const float* vn = v_norm.data() + static_cast<size_t>(n) * d_inner;
                const float inv = s_inv[n];
                for (int c = 0; c < d_inner; ++c) {
                    dnorm_accum[c] += static_cast<double>(dyn[c]) * vn[c] * inv;
                }
            }
            for (int c = 0; c < d_inner; ++c) dnorm_w[c] += static_cast<float>(dnorm_accum[c]);
        }

#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int n = 0; n < N; ++n) {
            const float* dyn = dy_norm.data() + static_cast<size_t>(n) * d_inner;
            const float* vn = v_norm.data() + static_cast<size_t>(n) * d_inner;
            const float* yr = y_scan.data() + static_cast<size_t>(n) * d_inner;
            const float* gr = proj.data() + static_cast<size_t>(n) * proj_size;
            float* dys = dy_scan.data() + static_cast<size_t>(n) * d_inner;
            float* dg = dgate.data() + static_cast<size_t>(n) * d_inner;
            const float inv = s_inv[n];

            double sum_pv = 0.0;
            for (int c = 0; c < d_inner; ++c) {
                const float p = dyn[c] * norm_w[c];
                sum_pv += static_cast<double>(p) * vn[c];
            }
            const double scale_adj = sum_pv * (inv * inv) / static_cast<double>(d_inner);

            for (int c = 0; c < d_inner; ++c) {
                const float p = dyn[c] * norm_w[c];
                const float dv = inv * static_cast<float>(static_cast<double>(p) - vn[c] * scale_adj);
                const float sig = 1.0f / (1.0f + std::exp(-gr[c]));
                const float g = gr[c] * sig;
                const float dsilu = sig * (1.0f + gr[c] * (1.0f - sig));
                dys[c] = dv * g;
                dg[c] = dv * yr[c] * dsilu;
            }
        }
    }

    // B3) Scan backward:
    std::vector<float> dbc(static_cast<size_t>(N) * conv_dim, 0.0f);
    std::vector<float> ddt_raw(static_cast<size_t>(N) * n_heads, 0.0f);

    std::vector<double> dD_accum(static_cast<size_t>(n_heads), 0.0);
    std::vector<double> da_log_accum(static_cast<size_t>(n_heads), 0.0);
    std::vector<double> ddt_bias_accum(static_cast<size_t>(n_heads), 0.0);

    {
#ifdef GAI_OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> local_dD(static_cast<size_t>(n_heads), 0.0);
            std::vector<double> local_da(static_cast<size_t>(n_heads), 0.0);
            std::vector<double> local_ddt(static_cast<size_t>(n_heads), 0.0);

            std::vector<float> s_traj(static_cast<size_t>(T + 1) * d_state, 0.0f);
            std::vector<float> ds(static_cast<size_t>(d_state), 0.0f);

#ifdef GAI_OPENMP
            #pragma omp for schedule(static)
#endif
            for (int row = 0; row < B * inner; ++row) {
                const int b   = row / inner;
                const int rem = row - b * inner;
                const int h   = rem / d_head;
                const int i   = rem - h * d_head;
                const float a_val     = -std::exp(a_log[h]);
                const float dt_bias_v = dt_bias[h];
                const float dskip     = D[h];

                // Forward pass for this row: record s_t in s_traj
                std::fill(s_traj.begin(), s_traj.begin() + d_state, 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* prow = proj.data() + (static_cast<size_t>(b) * T + t) * proj_size;
                    const float* brow = bc.data()   + (static_cast<size_t>(b) * T + t) * conv_dim;
                    const float dt_in = prow[proj_size - n_heads + h];
                    const float dt_val = (dt_in + dt_bias_v > 20.0f) ? (dt_in + dt_bias_v)
                                                                    : std::log1p(std::exp(dt_in + dt_bias_v));
                    const float dA = std::exp(a_val * dt_val);
                    const float xv = brow[h * d_head + i];
                    const float* s_prev = s_traj.data() + static_cast<size_t>(t) * d_state;
                    float* s_curr = s_traj.data() + static_cast<size_t>(t + 1) * d_state;
                    for (int j = 0; j < d_state; ++j) {
                        s_curr[j] = dA * s_prev[j] + (dt_val * brow[d_inner + j]) * xv;
                    }
                }

                // Backward pass for this row: step t from T-1 down to 0
                std::fill(ds.begin(), ds.end(), 0.0f);
                for (int t = T - 1; t >= 0; --t) {
                    const size_t tok_idx = static_cast<size_t>(b) * T + t;
                    const float* prow = proj.data() + tok_idx * proj_size;
                    const float* brow = bc.data()   + tok_idx * conv_dim;
                    float* d_brow     = dbc.data()  + tok_idx * conv_dim;
                    const float dy_val = dy_scan[tok_idx * d_inner + h * d_head + i];

                    const float dt_in = prow[proj_size - n_heads + h];
                    const float u = dt_in + dt_bias_v;
                    const float dt_val = (u > 20.0f) ? u : std::log1p(std::exp(u));
                    const float s_softplus = (u > 20.0f) ? 1.0f : (1.0f / (1.0f + std::exp(-u)));
                    const float dA = std::exp(a_val * dt_val);
                    const float xv = brow[h * d_head + i];

                    local_dD[h] += static_cast<double>(dy_val) * xv;
                    float dx_val = dy_val * dskip;

                    const float* s_prev = s_traj.data() + static_cast<size_t>(t) * d_state;
                    const float* s_curr = s_traj.data() + static_cast<size_t>(t + 1) * d_state;

                    float d_dA = 0.0f;
                    float d_dt_val = 0.0f;

                    for (int j = 0; j < d_state; ++j) {
                        const float Cj = brow[d_inner + d_state + j];
                        const float Bj = brow[d_inner + j];

                        #pragma omp atomic
                        d_brow[d_inner + d_state + j] += dy_val * s_curr[j];

                        const float total_ds = dy_val * Cj + ds[j];

                        d_dA += total_ds * s_prev[j];
                        d_dt_val += total_ds * Bj * xv;

                        #pragma omp atomic
                        d_brow[d_inner + j] += total_ds * (dt_val * xv);

                        dx_val += total_ds * (dt_val * Bj);
                        ds[j] = total_ds * dA;
                    }

                    #pragma omp atomic
                    d_brow[h * d_head + i] += dx_val;

                    const float d_adt = d_dA * dA;
                    local_da[h] += static_cast<double>(d_adt) * dt_val;
                    d_dt_val += d_adt * a_val;

                    const float ddt_raw_val = d_dt_val * s_softplus;
                    local_ddt[h] += static_cast<double>(ddt_raw_val);
                    #pragma omp atomic
                    ddt_raw[tok_idx * n_heads + h] += ddt_raw_val;
                }
            }

#ifdef GAI_OPENMP
            #pragma omp critical
#endif
            {
                for (int h = 0; h < n_heads; ++h) {
                    dD_accum[h] += local_dD[h];
                    da_log_accum[h] += local_da[h];
                    ddt_bias_accum[h] += local_ddt[h];
                }
            }
        }

        if (dD) {
            for (int h = 0; h < n_heads; ++h) dD[h] += static_cast<float>(dD_accum[h]);
        }
        if (da_log) {
            for (int h = 0; h < n_heads; ++h) {
                da_log[h] += static_cast<float>(da_log_accum[h] * (-std::exp(a_log[h])));
            }
        }
        if (ddt_bias) {
            for (int h = 0; h < n_heads; ++h) ddt_bias[h] += static_cast<float>(ddt_bias_accum[h]);
        }
    }

    // B4) Conv1d backward:
    std::vector<float> dproj_bc(static_cast<size_t>(N) * conv_dim, 0.0f);
    {
        std::vector<double> dconv_b_accum(static_cast<size_t>(conv_dim), 0.0);
        std::vector<double> dconv_w_accum(static_cast<size_t>(conv_dim) * k, 0.0);

#ifdef GAI_OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> local_dconv_b(static_cast<size_t>(conv_dim), 0.0);
            std::vector<double> local_dconv_w(static_cast<size_t>(conv_dim) * k, 0.0);

#ifdef GAI_OPENMP
            #pragma omp for schedule(static)
#endif
            for (int c = 0; c < conv_dim; ++c) {
                const float* wc = conv_w + static_cast<size_t>(c) * k;
                for (int b = 0; b < B; ++b) {
                    for (int t = 0; t < T; ++t) {
                        const size_t idx = (static_cast<size_t>(b) * T + t) * conv_dim + c;
                        const float raw_acc = conv_raw[idx];
                        const float sig = 1.0f / (1.0f + std::exp(-raw_acc));
                        const float dsilu = sig * (1.0f + raw_acc * (1.0f - sig));
                        const float dacc = dbc[idx] * dsilu;

                        local_dconv_b[c] += static_cast<double>(dacc);

                        for (int j = 0; j < k; ++j) {
                            const int tin = t - (k - 1 - j);
                            if (tin >= 0) {
                                const size_t pidx = (static_cast<size_t>(b) * T + tin) * proj_size + d_inner + c;
                                local_dconv_w[static_cast<size_t>(c) * k + j] += static_cast<double>(dacc) * proj[pidx];
                                const size_t bc_out_idx = (static_cast<size_t>(b) * T + tin) * conv_dim + c;
                                #pragma omp atomic
                                dproj_bc[bc_out_idx] += dacc * wc[j];
                            }
                        }
                    }
                }
            }

#ifdef GAI_OPENMP
            #pragma omp critical
#endif
            {
                if (dconv_b) {
                    for (int c = 0; c < conv_dim; ++c) dconv_b_accum[c] += local_dconv_b[c];
                }
                if (dconv_w) {
                    for (size_t idx = 0; idx < static_cast<size_t>(conv_dim) * k; ++idx)
                        dconv_w_accum[idx] += local_dconv_w[idx];
                }
            }
        }

        if (dconv_b) {
            for (int c = 0; c < conv_dim; ++c) dconv_b[c] += static_cast<float>(dconv_b_accum[c]);
        }
        if (dconv_w) {
            for (size_t idx = 0; idx < static_cast<size_t>(conv_dim) * k; ++idx)
                dconv_w[idx] += static_cast<float>(dconv_w_accum[idx]);
        }
    }

    // B5) Assemble dproj [N, proj_size] and in_proj backward:
    std::vector<float> dproj(static_cast<size_t>(N) * proj_size, 0.0f);
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        float* dp = dproj.data() + static_cast<size_t>(n) * proj_size;
        const float* dg = dgate.data() + static_cast<size_t>(n) * d_inner;
        const float* dbc_p = dproj_bc.data() + static_cast<size_t>(n) * conv_dim;
        const float* ddt_p = ddt_raw.data() + static_cast<size_t>(n) * n_heads;

        std::memcpy(dp, dg, sizeof(float) * d_inner);
        std::memcpy(dp + d_inner, dbc_p, sizeof(float) * conv_dim);
        std::memcpy(dp + d_inner + conv_dim, ddt_p, sizeof(float) * n_heads);
    }

    // dx = dproj * w_in (shape [N, d])
    // dw_in += dproj^T * x (shape [proj_size, d])
    linear_backward(x, w_in, dproj.data(), dx, dw_in, N, d, proj_size);
}

}  // namespace cpu
}  // namespace gai


