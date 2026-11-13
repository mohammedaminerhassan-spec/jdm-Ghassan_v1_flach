#include "core/ops_cpu.h"
#include "core/dtype.h"
#include "core/ops_cpu_simd.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>
#include <algorithm>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;

    if (beta == 0.0f) {
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 8) schedule(static)
#endif
        for (int m = 0; m < M; ++m) std::memset(C + static_cast<size_t>(m) * ldc, 0, sizeof(float) * static_cast<size_t>(N));
    } else if (beta != 1.0f) {
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 8) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            float* c = C + static_cast<size_t>(m) * ldc;
            for (int n = 0; n < N; ++n) c[n] *= beta;
        }
    }
    if (K <= 0 || alpha == 0.0f) return;

    if (M == 1 && !trans_a && trans_b) {
        const float* a = A;
        float*       c = C;
#ifdef GAI_OPENMP
        const bool nested = omp_in_parallel();
        #pragma omp parallel for schedule(static) if(!nested && N > 2048)
#endif
        for (int n = 0; n < N; ++n)
            c[n] += alpha * dot_f32(a, B + static_cast<size_t>(n) * ldb, K);
        return;
    }

    if (!trans_a && trans_b) {

        // C[m][n] += alpha * dot(A[m], B[n])
        //
        // B is the weight matrix: the largest operand by far. The previous
        // order (m outer) re-read the whole weight matrix once per row of A,
        // so a prefill of 32 tokens streamed the weights 32 times and the
        // GEMM became purely memory bound on the weights.
        //
        // With n outer, every row of B is read from memory exactly once and
        // reused across all M rows of A (A stays resident in cache). This is
        // the single biggest CPU win in the engine.
#ifdef GAI_OPENMP
        #pragma omp parallel for if(N > 64) schedule(static)
#endif
        for (int n = 0; n < N; ++n) {
            const float* b = B + static_cast<size_t>(n) * ldb;
            for (int m = 0; m < M; ++m) {
                const float* a = A + static_cast<size_t>(m) * lda;
                C[static_cast<size_t>(m) * ldc + n] += alpha * dot_f32(a, b, K);
            }
        }
    } else if (!trans_a && !trans_b) {

#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 2) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            const float* a = A + static_cast<size_t>(m) * lda;
            float*       c = C + static_cast<size_t>(m) * ldc;
            for (int k = 0; k < K; ++k) {
                float av = a[k];
                if (av != 0.0f) axpy_f32(c, B + static_cast<size_t>(k) * ldb, alpha * av, N);
            }
        }
    } else if (trans_a && !trans_b) {

#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 2) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            float* c = C + static_cast<size_t>(m) * ldc;
            for (int k = 0; k < K; ++k) {
                float av = A[static_cast<size_t>(k) * lda + m];
                if (av != 0.0f) axpy_f32(c, B + static_cast<size_t>(k) * ldb, alpha * av, N);
            }
        }
    } else {

#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 2) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            float* c = C + static_cast<size_t>(m) * ldc;
            for (int n = 0; n < N; ++n) {
                float s = 0.f;
                const float* b = B + static_cast<size_t>(n) * ldb;
                for (int k = 0; k < K; ++k) s += A[static_cast<size_t>(k) * lda + m] * b[k];
                c[n] += alpha * s;
            }
        }
    }
}

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N) {
    gemm(false, true, M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
}

void linear_forward_fp16(const float* x, const u16* w, float* y, int M, int K, int N) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(M * N > 256) schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const u16* row = w + static_cast<size_t>(n) * K;
            const float* xrow = x + static_cast<size_t>(m) * K;
            for (int k = 0; k < K; ++k) sum += xrow[k] * fp16_to_fp32(row[k]);
            y[static_cast<size_t>(m) * N + n] = sum;
        }
    }
}

void split_qkv(const float* qkv, float* q, float* k, float* v, i64 n, int qd, int kvd) {
    if (n <= 0) return;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < n; ++t) {
        const float* row = qkv + t * (qd + 2 * kvd);
        std::memcpy(q + t * qd, row, sizeof(float) * static_cast<size_t>(qd));
        std::memcpy(k + t * kvd, row + qd, sizeof(float) * static_cast<size_t>(kvd));
        std::memcpy(v + t * kvd, row + qd + kvd, sizeof(float) * static_cast<size_t>(kvd));
    }
}

void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N) {
    if (dx) gemm(false, false, M, K, N, 1.0f, dy, N, w, K, 1.0f, dx, K);
    if (dw) gemm(true,  false, N, K, M, 1.0f, dy, N, x, K, 1.0f, dw, K);
}

void add(const float* a, const float* b, float* out, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void add_inplace(float* a, const float* b, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) a[i] += b[i];
}

void add_bias(const float* b, float* y, i64 rows, int cols) {
    if (!b || rows <= 0 || cols <= 0) return;
#ifdef GAI_OPENMP
    #pragma omp parallel for if(rows > 16) schedule(static)
#endif
    for (i64 r = 0; r < rows; ++r) {
        float* row = y + r * cols;
        for (int c = 0; c < cols; ++c) {
            row[c] += b[c];
        }
    }
}

void bias_backward(const float* dy, float* db, i64 rows, int cols) {
    if (!db || rows <= 0 || cols <= 0) return;
#ifdef GAI_OPENMP
    #pragma omp parallel for if(cols > 128) schedule(static)
#endif
    for (int c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (i64 r = 0; r < rows; ++r) {
            sum += static_cast<double>(dy[r * cols + c]);
        }
        db[c] += static_cast<float>(sum);
    }
}

void scale_inplace(float* a, float s, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) a[i] *= s;
}

void embedding_forward(const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab) {
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        i32 id = ids[t];
        float* o = out + t * dim;
        if (id < 0 || id >= vocab) { std::memset(o, 0, sizeof(float) * static_cast<size_t>(dim)); continue; }
        std::memcpy(o, table + static_cast<size_t>(id) * dim, sizeof(float) * static_cast<size_t>(dim));
    }
}

void embedding_forward_half(const i32* ids, const u16* table, float* out,
                            i64 ntok, int dim, int vocab) {
    // The reference half-precision embedding: the row is widened on the fly.
    // (The GPU version reads the half table directly.)
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        i32 id = ids[t];
        float* o = out + t * dim;
        if (id < 0 || id >= vocab) { std::memset(o, 0, sizeof(float) * static_cast<size_t>(dim)); continue; }
        const u16* row = table + static_cast<size_t>(id) * dim;
        for (i64 i = 0; i < dim; ++i) o[i] = fp16_to_fp32(row[i]);
    }
}

void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab) {

    if (ntok <= 0) return;
    if (ntok < 256) {
        for (i64 t = 0; t < ntok; ++t) {
            i32 id = ids[t];
            if (id < 0 || id >= vocab) continue;
            float* d = dtable + static_cast<size_t>(id) * dim;
            const float* g = dout + t * dim;
            for (int i = 0; i < dim; ++i) d[i] += g[i];
        }
        return;
    }

    std::vector<std::pair<i32, i64>> order;
    std::vector<size_t> bounds;
    order.reserve(static_cast<size_t>(ntok));
    for (i64 t = 0; t < ntok; ++t) {
        const i32 id = ids[t];
        if (id < 0 || id >= vocab) continue;
        order.emplace_back(id, t);
    }
    if (order.empty()) return;

    std::stable_sort(order.begin(), order.end(),
              [](const std::pair<i32, i64>& a, const std::pair<i32, i64>& b) {
                  return a.first < b.first;
              });
    bounds.clear();
    bounds.reserve(order.size() + 1);
    bounds.push_back(0);
    for (size_t i = 1; i < order.size(); ++i) {
        if (order[i].first != order[i - 1].first) bounds.push_back(i);
    }
    bounds.push_back(order.size());
    const size_t ngroups = bounds.size() - 1;

#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if(ngroups > 4)
#endif
    for (long long g = 0; g < static_cast<long long>(ngroups); ++g) {
        const size_t gs = static_cast<size_t>(g);
        const i32 id = order[bounds[gs]].first;
        float* d = dtable + static_cast<size_t>(id) * dim;
        for (size_t k = bounds[gs]; k < bounds[gs + 1]; ++k) {
            const float* gg = dout + order[k].second * dim;
            for (int i = 0; i < dim; ++i) d[i] += gg[i];
        }
    }
}

void rmsnorm_forward(const float* x, const float* weight, float* out, float* rrms,
                     i64 rows, int dim, float eps) {
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;

        double ss = 0.0;
        for (int i = 0; i < dim; ++i) ss += static_cast<double>(xr[i]) * xr[i];
        double inv_d = 1.0 / std::sqrt(ss / static_cast<double>(dim) + static_cast<double>(eps));

        float inv = std::isfinite(inv_d) ? static_cast<float>(inv_d) : 0.0f;
        if (rrms) rrms[r] = inv;
        float* o = out + r * dim;
        for (int i = 0; i < dim; ++i) o[i] = xr[i] * inv * weight[i];
    }
}

void rmsnorm_backward(const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim) {

    int nthreads = 1;
#ifdef GAI_OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<float> dw_buf(static_cast<size_t>(nthreads) * static_cast<size_t>(dim), 0.0f);

#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 r = 0; r < rows; ++r) {
        int tid = 0;
#ifdef GAI_OPENMP
        tid = omp_get_thread_num();
#endif
        const float* xr = x + r * dim;
        const float* gr = dout + r * dim;
        float inv = rrms[r];
        float* dwl = dw_buf.data() + static_cast<size_t>(tid) * static_cast<size_t>(dim);

        float dot = 0.f;
        for (int i = 0; i < dim; ++i) dot += gr[i] * weight[i] * xr[i];
        float coef = inv * inv * inv / static_cast<float>(dim) * dot;

        float* dxr = dx + r * dim;
        for (int i = 0; i < dim; ++i) {
            dwl[i] += gr[i] * xr[i] * inv;
            dxr[i] += gr[i] * weight[i] * inv - xr[i] * coef;
        }
    }

    if (dweight) {
        for (int t = 0; t < nthreads; ++t) {
            const float* src = dw_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(dim);
            for (int i = 0; i < dim; ++i) dweight[i] += src[i];
        }
    }
}

static inline void rope_pair(float& a, float& b, float c, float s) {
    float na = a * c - b * s;
    float nb = a * s + b * c;
    a = na;
    b = nb;
}

static const float* rope_freqs(int half, int head_dim, float theta,
                               std::vector<float>& buf) {
    thread_local int cached_half = -1;
    thread_local int cached_hd = -1;
    thread_local float cached_theta = 0.0f;
    thread_local std::vector<float> cached;
    if (cached_half == half && cached_hd == head_dim && cached_theta == theta &&
        !cached.empty()) {
        return cached.data();
    }
    buf.resize(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i)
        buf[static_cast<size_t>(i)] =
            1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
    cached = buf;         // copy into thread_local cache
    cached_half = half;
    cached_hd = head_dim;
    cached_theta = theta;
    return cached.data(); // [FIX CRITICAL-3] return pointer to cached, not to buf (buf may be re-used)
}

static inline float yarn_freq_single(int i, int head_dim, float theta,
                                     float yarn_scale, float yarn_low, float yarn_high) {
    float base = 1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
    if (yarn_scale <= 1.0f) return base;
    const float wl = 2.0f * 3.14159265358979f / base;
    if (wl < yarn_low) return base;
    if (wl > yarn_high) return base / yarn_scale;
    float span = yarn_high - yarn_low;
    float t = (span > 0.0f) ? (wl - yarn_low) / span : 1.0f;
    return base * (1.0f - t + t / yarn_scale);
}

static void rope_apply(float* q, float* k, const i32* pos,
                       i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                       int rope_type, float yarn_low, float yarn_high, float yarn_scale,
                       float sign) {
    if (!q && !k) return;
    int half = head_dim / 2;
    thread_local std::vector<float> fbuf;
    fbuf.resize(static_cast<size_t>(half));
    if (yarn_scale > 1.0f) {
        for (int i = 0; i < half; ++i)
            fbuf[static_cast<size_t>(i)] = yarn_freq_single(i, head_dim, theta, yarn_scale, yarn_low, yarn_high);
    } else {
        std::vector<float> tmp;
        const float* f = rope_freqs(half, head_dim, theta, tmp);
        for (int i = 0; i < half; ++i) fbuf[static_cast<size_t>(i)] = f[i];
    }
    const float* freqs = fbuf.data();
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {

        const double p = static_cast<double>(pos[t]);
        for (int i = 0; i < half; ++i) {
            float ang  = static_cast<float>(p * static_cast<double>(freqs[i]));
            float c = std::cos(ang), s = std::sin(ang) * sign;
            if (rope_type == 1) {

                if (q) {
                    for (int h = 0; h < n_heads; ++h) {
                        float* qh = q + (t * n_heads + h) * head_dim;
                        rope_pair(qh[i], qh[i + half], c, s);
                    }
                }
                if (k) {
                    for (int h = 0; h < n_kv; ++h) {
                        float* kh = k + (t * n_kv + h) * head_dim;
                        rope_pair(kh[i], kh[i + half], c, s);
                    }
                }
            } else {
                if (q) {
                    for (int h = 0; h < n_heads; ++h) {
                        float* qh = q + (t * n_heads + h) * head_dim;
                        rope_pair(qh[2 * i], qh[2 * i + 1], c, s);
                    }
                }
                if (k) {
                    for (int h = 0; h < n_kv; ++h) {
                        float* kh = k + (t * n_kv + h) * head_dim;
                        rope_pair(kh[2 * i], kh[2 * i + 1], c, s);
                    }
                }
            }
        }
    }
}

void rope_forward(float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    rope_apply(q, k, pos, ntok, n_heads, n_kv, head_dim, theta, 0, 1.0f, 32.0f, 1.0f, 1.0f);
}

void rope_backward(float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {

    rope_apply(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta, 0, 1.0f, 32.0f, 1.0f, -1.0f);
}

static void rope_apply_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                              i64 ntok, int n_heads, int n_kv, int head_dim,
                              int rope_type, float sign) {
    const int half = head_dim / 2;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {

        const double p = static_cast<double>(pos[t]);
        for (int i = 0; i < half; ++i) {
            const float ang = static_cast<float>(p * static_cast<double>(inv_freq[i]));
            float c = std::cos(ang);
            float s = std::sin(ang) * sign;
            if (q) for (int h = 0; h < n_heads; ++h) {
                float* row = q + (t * n_heads + h) * head_dim;
                if (rope_type == 1) {
                    rope_pair(row[i], row[i + half], c, s);
                } else {
                    rope_pair(row[2 * i], row[2 * i + 1], c, s);
                }
            }
            if (k) for (int h = 0; h < n_kv; ++h) {
                float* row = k + (t * n_kv + h) * head_dim;
                if (rope_type == 1) {
                    rope_pair(row[i], row[i + half], c, s);
                } else {
                    rope_pair(row[2 * i], row[2 * i + 1], c, s);
                }
            }
        }
    }
}

void rope_forward_ex(float* q, float* k, const i32* pos,
                     i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    rope_apply(q, k, pos, ntok, n_heads, n_kv, head_dim, theta,
               rope_type, yarn_low, yarn_high, yarn_scale, 1.0f);
}

void rope_forward_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                         i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type) {
    rope_apply_cached(q, k, pos, inv_freq, ntok, n_heads, n_kv, head_dim, rope_type, 1.0f);
}

void rope_backward_ex(float* dq, float* dk, const i32* pos,
                      i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                      int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    rope_apply(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta,
               rope_type, yarn_low, yarn_high, yarn_scale, -1.0f);
}

void rope_backward_cached(float* dq, float* dk, const i32* pos, const float* inv_freq,
                          i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type) {
    rope_apply_cached(dq, dk, pos, inv_freq, ntok, n_heads, n_kv, head_dim, rope_type, -1.0f);
}

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void swiglu_forward(const float* g, const float* u, float* out, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) out[i] = silu(g[i]) * u[i];
}

void swiglu_backward(const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) {
        float gv = g[i];
        float sg = 1.0f / (1.0f + std::exp(-gv));
        float sl = gv * sg;
        float dsilu = sg * (1.0f + gv * (1.0f - sg));
        dg[i] += dout[i] * u[i] * dsilu;
        du[i] += dout[i] * sl;
    }
}

void softmax_row(float* x, int n) {

    if (n <= 0 || !x) return;
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale) {
    double total = 0.0;
    i64 count = 0;

    for (i64 i = 0; i < n; ++i) if (targets[i] >= 0 && targets[i] < V) ++count;

    for (i64 i = 0; i < n; ++i) {
        GAI_CHECK(targets[i] < V,
                  strfmt("softmax_cross_entropy: target out of range targets[%lld]=%d (vocab=%d; "
                         "tokenizer/dataloader mismatch?)", (long long)i, (int)targets[i], V));
    }

    [[maybe_unused]] const bool deterministic = [] {
        const char* e = std::getenv("GAI_DETERMINISTIC");
        return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
    }();
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:total) if(!deterministic)
#endif
    for (i64 i = 0; i < n; ++i) {
        const float* row = logits + i * V;
        i32 tgt = targets[i];
        float* d = dlogits ? dlogits + i * V : nullptr;

        if (tgt < 0) {
            if (d) std::memset(d, 0, sizeof(float) * static_cast<size_t>(V));
            continue;
        }
        float mx = row[0];
        for (int j = 1; j < V; ++j) mx = std::max(mx, row[j]);
        double sum = 0.0;
        for (int j = 0; j < V; ++j) sum += std::exp(static_cast<double>(row[j] - mx));
        double logZ = std::log(sum) + static_cast<double>(mx);
        total += logZ - static_cast<double>(row[tgt]);
        if (z_scale != 0.0f) total += (double)z_scale * logZ * logZ;

        if (d) {
            float invsum = static_cast<float>(1.0 / sum);
            float scale = 1.0f + ((z_scale != 0.0f) ? (2.0f * z_scale * (float)logZ) : 0.0f);
            for (int j = 0; j < V; ++j) {
                float p = std::exp(row[j] - mx) * invsum;
                d[j] = p * scale;
            }
            d[tgt] -= 1.0f;
        }
    }
    if (out_loss_sum) *out_loss_sum = total;
    if (out_count)    *out_count = count;
}

namespace {
// [FIX CRITICAL-2] Protect SCE accumulator from data races (DDP / future multi-thread)
std::mutex g_sce_acc_mutex;
double g_sce_acc_loss = 0.0;
i64 g_sce_acc_count = 0;
}

void sce_acc_begin() {
    std::lock_guard<std::mutex> lk(g_sce_acc_mutex);
    g_sce_acc_loss = 0.0;
    g_sce_acc_count = 0;
}

void sce_accumulate(const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale) {
    double csum = 0.0;
    i64 cn = 0;
    softmax_cross_entropy(logits, targets, dlogits, n, V, &csum, &cn, z_scale);
    std::lock_guard<std::mutex> lk(g_sce_acc_mutex);
    g_sce_acc_loss += csum;
    g_sce_acc_count += cn;
}

void sce_acc_end(double* out_loss_sum, i64* out_count) {
    std::lock_guard<std::mutex> lk(g_sce_acc_mutex);
    if (out_loss_sum) *out_loss_sum = g_sce_acc_loss;
    if (out_count) *out_count = g_sce_acc_count;
}

void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                float bc1, float bc2, float grad_scale) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) {
        float gi = g[i] * grad_scale;
        float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float mh = mi / bc1;
        float vh = vi / bc2;
        float upd = mh / (std::sqrt(vh) + eps);
        if (weight_decay != 0.0f) upd += weight_decay * w[i];
        w[i] -= lr * upd;
    }
}

void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float beta1, float beta2, float weight_decay,
               float grad_scale) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) {
        float gi = g[i] * grad_scale;
        float mi = m[i];
        float c = beta1 * mi + (1.0f - beta1) * gi;
        float upd = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);
        if (weight_decay != 0.0f) upd += weight_decay * w[i];
        w[i] -= lr * upd;
        m[i] = beta2 * mi + (1.0f - beta2) * gi;
    }
}

double global_sq_norm(const float* g, i64 n) {
    double s = 0.0;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:s)
#endif
    for (i64 i = 0; i < n; ++i) s += static_cast<double>(g[i]) * static_cast<double>(g[i]);
    return s;
}

double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts) {
    double s = 0.0;
    for (auto& pr : parts) s += global_sq_norm(pr.first, pr.second);
    return s;
}

void topk_select(const float* logits, int V, int K, float* out_vals, i32* out_ids) {
    GAI_CHECK(V > 0 && K >= 1 && K <= V, "topk_select: K out of range");
    std::vector<int> idx(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) idx[static_cast<size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) {
                          if (logits[a] != logits[b]) return logits[a] > logits[b];
                          return a < b;
                      });
    for (int k = 0; k < K; ++k) {
        out_vals[k] = logits[idx[static_cast<size_t>(k)]];
        out_ids[k] = static_cast<i32>(idx[static_cast<size_t>(k)]);
    }
}

i32 argmax_token(const float* logits, int V) {
    GAI_CHECK(V > 0, "argmax_token: empty vocabulary");
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return static_cast<i32>(best);
}

void apply_rep_penalties(float* logits, int V, const i32* hist, int hist_n,
                         float rep, float freq, float pres) {
    if (hist_n <= 0 || V <= 0) return;
    const bool use_rep = rep != 1.0f;
    const bool use_freq = freq != 0.0f;
    const bool use_pres = pres != 0.0f;
    if (!use_rep && !use_freq && !use_pres) return;
    thread_local std::vector<int> counts;
    if (counts.size() < static_cast<size_t>(V)) counts.resize(static_cast<size_t>(V));
    std::fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < hist_n; ++i) {
        const i32 tok = hist[i];
        if (tok >= 0 && tok < V) ++counts[static_cast<size_t>(tok)];
    }
    for (int tok = 0; tok < V; ++tok) {
        const int n = counts[static_cast<size_t>(tok)];
        if (n == 0) continue;
        float& l = logits[tok];
        if (use_rep) l = (l > 0.0f) ? l / rep : l * rep;
        if (use_freq) l -= freq * static_cast<float>(n);
        if (use_pres) l -= pres;
    }
}

}
}
