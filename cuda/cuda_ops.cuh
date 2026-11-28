#pragma once

#include "core/common.h"
#include "cuda/warp_primitives.cuh"
#include "cuda/cuda_graphs.cuh"
#include "cuda/cuda_memory_pool.cuh"
#include "cuda/quant_gemm.cuh"
#include <utility>
#include <vector>

namespace gai {
namespace cuda_ops {

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc, bool force_fp32 = false);

void set_fp16_gemm(bool on);
bool fp16_gemm_enabled();

void set_bf16_gemm(bool on);
bool bf16_gemm_enabled();

void set_moe_jitter(float j);
void set_moe_jitter_seed(u64 seed);

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N);
void linear_forward_f32(const float* x, const float* w, float* y, int M, int K, int N);
void linear_forward_fp16(const float* x, const u16* w, float* y, int M, int K, int N);
void split_qkv(const float* qkv, float* q, float* k, float* v, i64 n, int qd, int kvd);
void convert_f32_to_f16(const float* src, u16* dst, i64 n);
void convert_f16_to_f32(const u16* src, float* dst, i64 n);
void dequantize_quant(const u16* src, u16* dst, i64 n, int kind);
void linear_forward_q4(const float* x, const u16* w, float* y,
                       int M, int K, int N, int kind);
u16* quant_scratch(int slot, i64 n);
void free_quant_scratch();
void moe_backward_half(const float* x, const float* router_w,
                   const u16* gates, const u16* ups, const u16* downs,
                   const u16* sh_g, const u16* sh_u, const u16* sh_d,
                   const float* probs, const i32* idx, const float* tw,
                   const float* s_gate, const float* s_up, const float* s_act,
                   const float* aux_frac, float aux_scale,
                   const float* dout, float* dx,
                   float* drouter_w, float* dgates, float* dups, float* ddowns,
                   float* dsh_g, float* dsh_u, float* dsh_d,
                   float* s_dact,
                   i64 N, int d, int E, int ne, int K, int SH);
void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N);
void gemm_nn_half(const float* A, const u16* B, float* C, int M, int N, int K);
void linear_backward_half(const float* x, const u16* w, const float* dy,
                          float* dx, float* dw, int M, int K, int N);

void add(const float* a, const float* b, float* out, i64 n);
void add_inplace(float* a, const float* b, i64 n);
void add_bias(const float* b, float* y, i64 rows, int cols);
void bias_backward(const float* dy, float* db, i64 rows, int cols);
void scale_inplace(float* a, float s, i64 n);
void zero(float* a, i64 n);
void copy(float* dst, const float* src, i64 n);

void embedding_forward(const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab);
void embedding_forward_half(const i32* ids, const u16* table, float* out,
                            i64 ntok, int dim, int vocab);
void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab);

void rmsnorm_forward(const float* x, const float* weight, float* out, float* rrms,
                     i64 rows, int dim, float eps);
void rmsnorm_backward(const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim);

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

void swiglu_forward(const float* g, const float* u, float* out, i64 n);
void gelu_forward(const float* in, float* out, i64 n);
void gelu_backward(const float* in, const float* dout, float* din, i64 n);
void swiglu_backward(const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n);

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

void moe_forward_half(const float* x, const float* router_w,
                      const u16* gates, const u16* ups, const u16* downs,
                      const u16* sh_g, const u16* sh_u, const u16* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K, int SH, int qkind = -1);
void moe_forward_bias_half(const float* x, const float* router_w, const float* router_bias,
                      const u16* gates, const u16* ups, const u16* downs,
                      const u16* sh_g, const u16* sh_u, const u16* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K, int SH, int qkind = -1);

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
                    const i32* segment_ids = nullptr,
                    float* conv_state = nullptr);
void mamba_free_workspace();

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
                    int d_head, int d_state, int k, float eps, int qkind = -1,
                    const i32* segment_ids = nullptr,
                    float* conv_state = nullptr);

void mamba2_step(const float* x,
                 const float* w_in, const float* conv_w, const float* conv_b,
                 const float* dt_bias, const float* a_log, const float* D,
                 const float* norm_w, const float* w_out,
                 float* conv_state, float* ssm_state, float* out,
                 int d, int d_inner, int conv_dim, int n_heads,
                 int d_head, int d_state, int k, float eps);
void mamba2_step_half(const float* x,
                 const u16* w_in, const float* conv_w, const float* conv_b,
                 const float* dt_bias, const float* a_log, const float* D,
                 const float* norm_w, const u16* w_out,
                 float* conv_state, float* ssm_state, float* out,
                 int d, int d_inner, int conv_dim, int n_heads,
                 int d_head, int d_state, int k, float eps, int qkind = -1);

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
                     const i32* segment_ids = nullptr);

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
                     const i32* segment_ids = nullptr);

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

void attention_forward_ex(const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids = nullptr);
void attention_forward_kv_ex(const float* q, const void* k, const void* v, bool kv_fp16,
                             float* out, int B, int Tq, int Tk, int H, int KV, int hd,
                             int q_pos_offset, float scale, int window);
void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window);
void attention_decode_ring(const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window);

void attention_decode_ring_fp16(const float* q, const u16* kc, const u16* vc,
                                float* out, int H, int KV, int hd, int ring_start,
                                int pinned_prefix, int cur_len, int cache_max,
                                float scale, float* scratch, int window);

void moe_aux_frac_gpu(const float* probs, const i32* idx, float* frac,
                      float* h_frac, float* h_psum,
                      i64 N, int K, int ne, double* d_raw_accum = nullptr);
double* moe_aux_begin();
double moe_aux_end();

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);

void sce_acc_begin();
void sce_accumulate(const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale = 0.0f);
void sce_acc_end(double* out_loss_sum, i64* out_count);

void moe_count_slots(const i32* idx, float* acc, i64 NK, int ne);

void topk_select(const float* logits, int V, int K, float* out_vals, i32* out_ids);
i32 argmax_token(const float* logits, int V);
void apply_rep_penalties(float* logits, int V, const i32* hist, int hist_n,
                         float rep, float freq, float pres);
void free_sampling_workspace();

void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                float bc1, float bc2, float grad_scale);

void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float beta1, float beta2, float weight_decay,
               float grad_scale);

void scale_grad(float* g, i64 n, float scale);

double global_sq_norm(const float* g, i64 n);

double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts);

void  free_workspace();
void  moe_free_workspace();

void phase_start(int ph);
void phase_stop(int ph);
void phase_reset();
void phase_shutdown();
std::string phase_report();
void  reserve_workspaces(size_t gemm_bytes, size_t moe_bytes);

void  free_workspaces();
void  moe_reserve_workspace(size_t bytes);

size_t pool_bytes();
size_t moe_pool_bytes();

void layernorm_forward(const float* x, const float* w, const float* b,
                       float* y, int rows, int dim, float eps);
void softmax_rows(float* x, int rows, int cols);
void vit_extract_patches(const float* pixels_chw, float* patches,
                         int image_size, int patch_size, int channels);
void vit_spatial_pool_2x2(const float* in_patches, float* out_pooled, int S, int D);
void vit_attention_forward(const float* qkv, float* scores_buf, float* ctx,
                           int P, int H, int hd, float scale);

}
}
