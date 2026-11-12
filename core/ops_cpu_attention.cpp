// The attention kernels (CPU): the prefill (causal, over the whole
// sequence), the backward pass and the incremental decode (the ring KV cache,
// in fp32 and in the packed fp16 form).
//
// They are shared by every model: the dense FFN models, the MoE models and
// the Granite hybrid (whose attention layers are exactly these).

#include "core/ops_cpu.h"
#include "core/ops_cpu_simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

void attention_forward(const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale) {
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

#ifdef GAI_OPENMP
    #pragma omp parallel
#endif
    {
        thread_local std::vector<float> tls;
        if (tls.size() < static_cast<size_t>(T)) tls.resize(static_cast<size_t>(T));
        float* s = tls.data();
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
    for (int bh = 0; bh < B * H; ++bh) {
        {
            int b = bh / H;
            int h = bh % H;
            int kvh = h / group;
            for (int t = 0; t < T; ++t) {
                const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                int len = t + 1;
                float mx = -3.4e38f;
                for (int j = 0; j < len; ++j) {
                    const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                    float d = dot_f32(qh, kh, hd) * scale;
                    s[static_cast<size_t>(j)] = d;
                    mx = std::max(mx, d);
                }
                float sum = 0.f;
                for (int j = 0; j < len; ++j) { s[static_cast<size_t>(j)] = std::exp(s[static_cast<size_t>(j)] - mx); sum += s[static_cast<size_t>(j)]; }
                float inv = 1.0f / sum;
                float* o = out + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
                for (int j = 0; j < len; ++j) {
                    float p = s[static_cast<size_t>(j)] * inv;
                    s[static_cast<size_t>(j)] = p;
                    const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                    axpy_f32(o, vh, p, hd);
                }
                if (probs) {
                    float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    std::memcpy(pr, s, sizeof(float) * static_cast<size_t>(len));
                    if (len < T) std::memset(pr + len, 0, sizeof(float) * static_cast<size_t>(T - len));
                }
            }
        }
    }
    }
}

void attention_backward(const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale) {

    GAI_CHECK(probs != nullptr, "attention_backward requires cached probs (training path)");
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

#ifdef GAI_OPENMP
    #pragma omp parallel
#endif
    {
        thread_local std::vector<float> tld;
        if (tld.size() < static_cast<size_t>(T)) tld.resize(static_cast<size_t>(T));
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
    for (int bkv = 0; bkv < B * KV; ++bkv) {
        {
            int b = bkv / KV;
            int kvh = bkv % KV;
            float* dsv = tld.data();
            for (int hg = 0; hg < group; ++hg) {
                int h = kvh * group + hg;
                for (int t = 0; t < T; ++t) {
                    int len = t + 1;
                    const float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    const float* go = dout + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    float* dqh = dq + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;

                    float dot_pg = 0.f;
                    for (int j = 0; j < len; ++j) {
                        const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float dpj = dot_f32(go, vh, hd);
                        dsv[static_cast<size_t>(j)] = dpj;
                        dot_pg += pr[j] * dpj;
                        float* dvh = dv + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dvh, go, pr[j], hd);
                    }

                    for (int j = 0; j < len; ++j) {
                        float ds = pr[j] * (dsv[static_cast<size_t>(j)] - dot_pg) * scale;
                        if (ds == 0.0f) continue;
                        const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float* dkh = dk + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dqh, kh, ds, hd);
                        axpy_f32(dkh, qh, ds, hd);
                    }
                }
            }
        }
    }
    }
}

void attention_decode(const float* q, const float* kcache, const float* vcache,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch) {
    attention_decode_ex(q, kcache, vcache, out, H, KV, hd, cur_len, max_len, scale, scratch, 0);
}

void attention_forward_ex(const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids) {
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

#ifdef GAI_OPENMP
    #pragma omp parallel
#endif
    {
        thread_local std::vector<float> tls;
        if (tls.size() < static_cast<size_t>(T)) tls.resize(static_cast<size_t>(T));
        float* s = tls.data();
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
    for (int bh = 0; bh < B * H; ++bh) {
        {
            int b = bh / H;
            int h = bh % H;
            int kvh = h / group;
            for (int t = 0; t < T; ++t) {
                const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                int j0 = (window > 0) ? (t - window + 1 > 0 ? t - window + 1 : 0) : 0;
                int len = t + 1 - j0;
                const i32 seg_t = segment_ids ? segment_ids[static_cast<size_t>(b) * T + t] : 0;
                float mx = -3.4e38f;
                for (int j = j0; j <= t; ++j) {
                    i32 seg_j = segment_ids ? segment_ids[static_cast<size_t>(b) * T + j] : 0;
                    if (segment_ids && seg_t >= 0 && seg_j != seg_t) {
                        s[static_cast<size_t>(j - j0)] = -3.4e38f;
                        continue;
                    }
                    const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                    float d = dot_f32(qh, kh, hd) * scale;
                    s[static_cast<size_t>(j - j0)] = d;
                    mx = std::max(mx, d);
                }
                float sum = 0.f;
                for (int j = 0; j < len; ++j) {
                    float value = s[static_cast<size_t>(j)];
                    value = value < -3.0e38f ? 0.0f : std::exp(value - mx);
                    s[static_cast<size_t>(j)] = value;
                    sum += s[static_cast<size_t>(j)];
                }
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                float* o = out + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
                for (int j = 0; j < len; ++j) {
                    float p = s[static_cast<size_t>(j)] * inv;
                    s[static_cast<size_t>(j)] = p;
                    const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j + j0) * KV + kvh) * hd;
                    axpy_f32(o, vh, p, hd);
                }
                if (probs) {
                    float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    std::memset(pr, 0, sizeof(float) * static_cast<size_t>(T));
                    std::memcpy(pr + j0, s, sizeof(float) * static_cast<size_t>(len));
                }
            }
        }
    }
    }
}

void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window) {
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

#ifdef GAI_OPENMP
    #pragma omp parallel
#endif
    {
        thread_local std::vector<float> tld;
        if (tld.size() < static_cast<size_t>(T)) tld.resize(static_cast<size_t>(T));
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
    for (int bkv = 0; bkv < B * KV; ++bkv) {
        {
            int b = bkv / KV;
            int kvh = bkv % KV;
            float* dsv = tld.data();
            for (int hg = 0; hg < group; ++hg) {
                int h = kvh * group + hg;
                for (int t = 0; t < T; ++t) {
                    int j0 = (window > 0) ? (t - window + 1 > 0 ? t - window + 1 : 0) : 0;
                    const float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    const float* go = dout + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    float* dqh = dq + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;

                    float dot_pg = 0.f;
                    for (int j = j0; j <= t; ++j) {
                        const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float dpj = dot_f32(go, vh, hd);
                        dsv[static_cast<size_t>(j - j0)] = dpj;
                        dot_pg += pr[j] * dpj;
                        float* dvh = dv + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dvh, go, pr[j], hd);
                    }
                    for (int j = j0; j <= t; ++j) {
                        float ds = pr[j] * (dsv[static_cast<size_t>(j - j0)] - dot_pg) * scale;
                        if (ds == 0.0f) continue;
                        const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float* dkh = dk + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dqh, kh, ds, hd);
                        axpy_f32(dkh, qh, ds, hd);
                    }
                }
            }
        }
    }
    }
}

void attention_decode_ex(const float* q, const float* kcache, const float* vcache,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "attention_decode_ex: empty shape");
    GAI_CHECK(H % KV == 0, "attention_decode_ex: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && max_len >= 0 && cur_len <= max_len,
              "attention_decode_ex: cur_len exceeds KV cache max_len (increase max_context)");
    GAI_CHECK(q != nullptr && kcache != nullptr && vcache != nullptr && out != nullptr,
              "attention_decode_ex: null pointer");
    GAI_CHECK(scratch != nullptr, "attention_decode_ex: null scratch (need H*cur_len floats)");
    const int group = H / KV;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; ++h) {
        int kvh = h / group;
        float* s = scratch + static_cast<size_t>(h) * static_cast<size_t>(cur_len);
        const float* qh = q + static_cast<size_t>(h) * hd;
        int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
        float mx = -3.4e38f;
        for (int j = j0; j < cur_len; ++j) {
            const float* kh = kcache + (static_cast<size_t>(j) * KV + kvh) * hd;
            float d = dot_f32(qh, kh, hd) * scale;
            s[j] = d;
            mx = std::max(mx, d);
        }
        float sum = 0.f;
        for (int j = j0; j < cur_len; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float inv = sum > 0.f ? 1.0f / sum : 0.0f; // [FIX MEDIUM-1] guard against sum=0 (all-masked window)
        float* o = out + static_cast<size_t>(h) * hd;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
        for (int j = j0; j < cur_len; ++j) {
            const float* vh = vcache + (static_cast<size_t>(j) * KV + kvh) * hd;
            axpy_f32(o, vh, s[j] * inv, hd);
        }
    }
}

static inline size_t decode_ring_slot(int j, int pinned_prefix, int ring_start, int ring_capacity) {
    if (j < pinned_prefix) return static_cast<size_t>(j);
    const int cap = ring_capacity > 0 ? ring_capacity : 1;
    return static_cast<size_t>(pinned_prefix) +
           static_cast<size_t>((ring_start + (j - pinned_prefix)) % cap);
}

void attention_decode_ring(const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "attention_decode_ring: empty shape");
    GAI_CHECK(H % KV == 0, "attention_decode_ring: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && cache_max >= 0 && cur_len <= cache_max,
              "attention_decode_ring: length exceeds cache capacity");
    GAI_CHECK(pinned_prefix >= 0 && pinned_prefix <= cur_len,
              "attention_decode_ring: pinned prefix out of range");
    const int ring_capacity = cache_max - pinned_prefix;
    if (cur_len > pinned_prefix) {
        GAI_CHECK(ring_capacity > 0, "attention_decode_ring: ring has no rolling slots");
        GAI_CHECK(ring_start >= 0 && ring_start < ring_capacity,
                  "attention_decode_ring: ring start out of range");
    }
    const int group = H / KV;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; ++h) {
        int kvh = h / group;
        float* s = scratch + static_cast<size_t>(h) * static_cast<size_t>(cur_len);
        const float* qh = q + static_cast<size_t>(h) * hd;
        int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
        float mx = -3.4e38f;
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            const float* kh = kcache + (slot * static_cast<size_t>(KV) + kvh) * hd;
            float d = dot_f32(qh, kh, hd) * scale;
            s[j] = d;
            mx = std::max(mx, d);
        }
        float sum = 0.f;
        for (int j = j0; j < cur_len; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float inv = sum > 0.f ? 1.0f / sum : 0.0f; // [FIX MEDIUM-1] guard against sum=0 (all-masked ring window)
        float* o = out + static_cast<size_t>(h) * hd;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            const float* vh = vcache + (slot * static_cast<size_t>(KV) + kvh) * hd;
            axpy_f32(o, vh, s[j] * inv, hd);
        }
    }
}

// fp16 KV cache variant of attention_decode_ring. The values are converted
// on the fly: the cache stays at half the size and the decode step reads
// half the bytes.
void attention_decode_ring_fp16(const float* q, const u16* kcache, const u16* vcache,
                                float* out, int H, int KV, int hd, int ring_start,
                                int pinned_prefix, int cur_len, int cache_max,
                                float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "attention_decode_ring_fp16: empty shape");
    GAI_CHECK(H % KV == 0, "attention_decode_ring_fp16: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && cache_max >= 0 && cur_len <= cache_max,
              "attention_decode_ring_fp16: length exceeds cache capacity");
    GAI_CHECK(pinned_prefix >= 0 && pinned_prefix <= cur_len,
              "attention_decode_ring_fp16: pinned prefix out of range");
    const int ring_capacity = cache_max - pinned_prefix;
    if (cur_len > pinned_prefix) {
        GAI_CHECK(ring_capacity > 0, "attention_decode_ring_fp16: ring has no rolling slots");
        GAI_CHECK(ring_start >= 0 && ring_start < ring_capacity,
                  "attention_decode_ring_fp16: ring start out of range");
    }
    const int group = H / KV;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; ++h) {
        int kvh = h / group;
        float* s = scratch + static_cast<size_t>(h) * static_cast<size_t>(cur_len);
        const float* qh = q + static_cast<size_t>(h) * hd;
        int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
        float mx = -3.4e38f;
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            const u16* kh = kcache + (slot * static_cast<size_t>(KV) + kvh) * hd;
            float d = 0.0f;
            for (int c = 0; c < hd; ++c) d += qh[c] * fp16_to_fp32(kh[c]);
            d *= scale;
            s[j] = d;
            mx = std::max(mx, d);
        }
        float sum = 0.f;
        for (int j = j0; j < cur_len; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float inv = sum > 0.f ? 1.0f / sum : 0.0f;
        float* o = out + static_cast<size_t>(h) * hd;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            const u16* vh = vcache + (slot * static_cast<size_t>(KV) + kvh) * hd;
            const float w = s[j] * inv;
            for (int c = 0; c < hd; ++c) o[c] += w * fp16_to_fp32(vh[c]);
        }
    }
}
}  // namespace cpu
}  // namespace gai
