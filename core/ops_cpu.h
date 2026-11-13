#pragma once

// The CPU kernels. They are split per family, and every declaration below
// names the file that implements it:
//
//   core/ops_cpu_dense.cpp    the shared primitives (the GEMM, the linear
//                             ops, the elementwise ops, the embedding, the
//                             norms, the RoPE, the SwiGLU), the loss, the
//                             optimizer steps and the sampler helpers
//   core/ops_cpu_attention.cpp the attention family (the prefill, the
//                             backward, the incremental decode)
//   core/ops_cpu_moe.cpp      the mixture of experts (the routing, the expert
//                             GEMMs, the shared expert)
//   core/ops_cpu_mamba.cpp    the IBM Granite Mamba-2 layer
//   core/ops_cpu_simd.h       the AVX2/FMA helpers shared by the above
//
// The dense and the MoE models share the primitives: a "dense" model is the
// same code path with the MoE block replaced by a plain MLP.

#include "core/common.h"
#include "core/dtype.h"
#include <vector>
#include <utility>

namespace gai {
namespace cpu {

// ---- the GEMM and the linear ops (ops_cpu_dense.cpp)

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc);

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N);
void linear_forward_fp16(const float* x, const u16* w, float* y, int M, int K, int N);
void split_qkv(const float* qkv, float* q, float* k, float* v, i64 n, int qd, int kvd);
void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N);

// ---- the elementwise ops (ops_cpu_dense.cpp)

void add(const float* a, const float* b, float* out, i64 n);
void add_inplace(float* a, const float* b, i64 n);
void add_bias(const float* b, float* y, i64 rows, int cols);
void bias_backward(const float* dy, float* db, i64 rows, int cols);
void scale_inplace(float* a, float s, i64 n);

// ---- the embedding (ops_cpu_dense.cpp)

void embedding_forward(const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab);
void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab);

// The fp16 storage mode on the CPU: the weights are half, the reference
// implementation converts them on the fly (correct, not fast). It exists so
// the half-precision path can be verified on a machine without a GPU.
void embedding_forward_half(const i32* ids, const u16* table, float* out,
                            i64 ntok, int dim, int vocab);

// ---- the norms (ops_cpu_dense.cpp)

void rmsnorm_forward(const float* x, const float* weight, float* out, float* rrms,
                     i64 rows, int dim, float eps);
void rmsnorm_backward(const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim);

// ---- the RoPE (ops_cpu_dense.cpp)

void rope_forward(float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta);
void rope_backward(float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta);

void rope_forward_ex(float* q, float* k, const i32* pos,
                     i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale);
void rope_backward_ex(float* dq, float* dk, const i32* pos,
                      i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                      int rope_type, float yarn_low, float yarn_high, float yarn_scale);
void rope_forward_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                         i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type);
void rope_backward_cached(float* dq, float* dk, const i32* pos, const float* inv_freq,
                          i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type);

// ---- the SwiGLU (the dense FFN and the expert FFN) (ops_cpu_dense.cpp)

void swiglu_forward(const float* g, const float* u, float* out, i64 n);
void swiglu_backward(const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n);

// ---- the mixture of experts (ops_cpu_moe.cpp)

void moe_forward(const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K, int SH);

void moe_forward_bias(const float* x, const float* router_w, const float* router_bias,
                      const float* gates, const float* ups, const float* downs,
                      const float* sh_g, const float* sh_u, const float* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K, int SH);

// The fp16 storage mode (reference implementation: the weights are
// converted to fp32 on the fly).
void moe_forward_half(const float* x, const float* router_w,
                      const u16* gates, const u16* ups, const u16* downs,
                      const u16* sh_g, const u16* sh_u, const u16* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K, int SH);
void moe_forward_bias_half(const float* x, const float* router_w, const float* router_bias,
                      const u16* gates, const u16* ups, const u16* downs,
                      const u16* sh_g, const u16* sh_u, const u16* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K, int SH);
void moe_backward(const float* x, const float* router_w,
                  const float* gates, const float* ups, const float* downs,
                  const float* sh_g, const float* sh_u, const float* sh_d,
                  const float* probs, const i32* idx, const float* tw,
                  const float* s_gate, const float* s_up, const float* s_act,
                  const float* aux_frac, float aux_scale,
                  const float* dout,
                  float* dx,
                  float* drouter_w, float* dgates, float* dups, float* ddowns,
                  float* dsh_g, float* dsh_u, float* dsh_d,
                  float* s_dact,
                  i64 N, int d, int E, int ne, int K, int SH);





void moe_count_slots(const i32* idx, float* acc, i64 NK, int ne);

void set_moe_jitter_cpu(float j);
void set_moe_jitter_seed_cpu(u64 seed);

void softmax_row(float* x, int n);

// ---- the IBM Granite Mamba-2 layer (ops_cpu_mamba.cpp)
//
//   x      [B, T, d]        the layer input
//   w_in   [proj_size, d]   in_proj weight
//   conv_w [conv_dim, k]    depthwise conv weight
//   conv_b [conv_dim]       depthwise conv bias (may be null)
//   dt_bias/alog/D [n_heads]
//   norm_w [d_inner]        gated RMSNorm weight
//   w_out  [d, d_inner]     out_proj weight
//   state  [B, n_heads, d_head, d_state]  recurrent state (in/out)
//   out    [B, T, d]        the layer output
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
                    int d_head, int d_state, int k, float eps);

// The fp16 storage mode (reference implementation: the two projections are
// converted to fp32 on the fly).
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
                    int d_head, int d_state, int k, float eps);

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
                     int d_head, int d_state, int k, float eps);

// ---- the attention (ops_cpu_attention.cpp)

void attention_forward(const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale);
void attention_backward(const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale);
void attention_decode(const float* q, const float* kcache, const float* vcache,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch);

void attention_forward_ex(const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids = nullptr);
void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window);
void attention_decode_ex(const float* q, const float* kcache, const float* vcache,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window);
void attention_decode_ring(const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window);

void attention_decode_ring_fp16(const float* q, const u16* kcache, const u16* vcache,
                                float* out, int H, int KV, int hd, int ring_start,
                                int pinned_prefix, int cur_len, int cache_max,
                                float scale, float* scratch, int window);

// ---- the loss (ops_cpu_dense.cpp)

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);

void sce_acc_begin();
void sce_accumulate(const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale = 0.0f);
void sce_acc_end(double* out_loss_sum, i64* out_count);

// ---- the optimizer steps (ops_cpu_dense.cpp)

void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                float bc1, float bc2, float grad_scale);

void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float beta1, float beta2, float weight_decay,
               float grad_scale);

double global_sq_norm(const float* g, i64 n);

double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts);

// ---- the sampler helpers (ops_cpu_dense.cpp)

void topk_select(const float* logits, int V, int K, float* out_vals, i32* out_ids);

i32 argmax_token(const float* logits, int V);

void apply_rep_penalties(float* logits, int V, const i32* hist, int hist_n,
                         float rep, float freq, float pres);

}
}
