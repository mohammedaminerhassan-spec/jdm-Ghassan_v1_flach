#pragma once

#include "core/tensor.h"
#include "core/config.h"
#include "core/rng.h"
#include <vector>
#include <string>
#include <memory>
#include <random>

namespace gai {

struct ModelConfig {
    int   vocab_size        = 16000;
    int   hidden_size       = 768;
    int   num_layers        = 26;
    int   num_heads         = 12;
    int   num_kv_heads      = 4;
    int   intermediate_size = 2240;
    int   max_seq_len       = 4096;
    float rope_theta        = 10000.0f;
    float rms_eps           = 1e-5f;
    bool  tie_embeddings    = true;
    float init_std          = 0.02f;

    bool  use_moe           = false;
    int   num_experts       = 8;
    int   moe_top_k         = 2;
    int   moe_expert_dim    = 768;
    bool  moe_shared        = true;
    float moe_aux_scale     = 0.01f;
    float moe_jitter        = 0.0f;

    bool  moe_allow_dense   = false;

    int shared_expert_intermediate_size = 0;

    bool  use_qk_norm       = false;
    bool  use_qkv_bias      = false;
    float z_loss_scale      = 0.0f;
    float rope_scale        = 1.0f;
    float rope_yarn_mscale  = 0.0f;

    bool  moe_aux_free      = false;

    float rope_yarn_low     = 1.0f;
    float rope_yarn_high    = 32.0f;
    int   sliding_window    = 0;

    int   rope_type         = 1;

    int   position_embedding = 0;

    int   lora_rank         = 0;
    float lora_alpha        = 16.0f;
    bool  lora_target_q     = true;
    bool  lora_target_k     = false;
    bool  lora_target_v     = true;
    bool  lora_target_o     = true;
    bool  lora_target_gate  = false;
    bool  lora_target_up    = false;
    bool  lora_target_down  = false;

    int head_dim_override = 0;

    enum Arch : u8 { ArchQwen = 0, ArchGranite = 1 } arch = ArchQwen;

    std::string base_model;
    std::string base_org;

    std::vector<int> layer_types;

    float embedding_multiplier = 1.0f;
    float residual_multiplier  = 1.0f;
    float attention_multiplier = 0.0f;
    float logits_scaling       = 1.0f;

    int mamba_n_heads   = 48;
    int mamba_d_head    = 64;
    int mamba_d_state   = 128;
    int mamba_n_groups  = 1;
    int mamba_d_conv    = 4;
    int mamba_expand    = 2;
    int mamba_chunk_size = 256;
    bool mamba_conv_bias = true;
    bool mamba_proj_bias = false;

    int head_dim() const {
        if (head_dim_override > 0) return head_dim_override;
        return hidden_size / num_heads;
    }

    int shared_width() const {
        return shared_expert_intermediate_size > 0 ? shared_expert_intermediate_size
                                                  : moe_expert_dim;
    }
    int kv_dim()   const { return num_kv_heads * head_dim(); }
    int q_dim()    const { return num_heads * head_dim(); }

    void validate() const;
    static ModelConfig from_config(const Config& c, const std::string& prefix = "model");
    std::string summary() const;
    std::string hybrid_summary() const;

    std::string arch_identity() const;
    bool same_architecture_as(const ModelConfig& o, std::string* reason = nullptr) const;

    void take_adapter_settings_from(const ModelConfig& recipe);
};

float rope_theta_eff(const ModelConfig& m);

float rope_mscale(const ModelConfig& m);

struct Parameter {
    std::string      name;
    std::vector<i64> shape;
    Tensor           w;
    Tensor           g;
    Tensor           fp16_cache;
    bool             decay = true;
    bool             frozen = false;

    i64 numel() const { return w.numel(); }
    bool is_half() const { return w.defined() && w.dtype() == DType::F16; }
};

struct LoraAdapter {
    Parameter A;
    Parameter B;
    Tensor   r_buf;
    Tensor   dR_buf;
    int   rank = 0;
    int   in_dim = 0;
    int   out_dim = 0;
    float alpha = 16.0f;
    bool  enabled = false;

    float scale() const {
        return (rank > 0 && enabled) ? (alpha / static_cast<float>(rank)) : 0.0f;
    }
};

struct LayerParams {
    Parameter attn_norm;
    Parameter wq;
    Parameter wk;
    Parameter wv;
    Parameter bq;
    Parameter bk;
    Parameter bv;
    Tensor    wqkv_fp16;
    Parameter wo;
    Parameter ffn_norm;
    Parameter qk_qnorm;
    Parameter qk_knorm;

    bool has_qkv_bias() const { return bq.w.defined(); }

    Parameter router;
    Parameter moe_gate;
    Parameter moe_up;
    Parameter moe_down;
    Parameter sh_gate;
    Parameter sh_up;
    Parameter sh_down;

    Parameter w_gate;
    Parameter w_up;
    Parameter w_down;

    Parameter mamba_in_proj;
    Parameter mamba_conv_w;
    Parameter mamba_conv_b;
    Parameter mamba_dt_bias;
    Parameter mamba_A_log;
    Parameter mamba_D;
    Parameter mamba_norm;
    Parameter mamba_out_proj;

    bool has_mamba() const { return mamba_in_proj.w.defined(); }

    LoraAdapter lora_q;
    LoraAdapter lora_k;
    LoraAdapter lora_v;
    LoraAdapter lora_o;
    LoraAdapter lora_gate;
    LoraAdapter lora_up;
    LoraAdapter lora_down;

    bool has_moe() const { return moe_gate.w.defined(); }
};

struct ParamGroupCount {
    std::string name;
    i64         count = 0;
    i64         per_unit = 0;
    int         units = 1;
};

struct ParamReport {
    std::vector<ParamGroupCount> groups;
    i64 total = 0;
    i64 embedding = 0;
    i64 non_embedding = 0;
    std::string to_string(const ModelConfig& cfg) const;
};

struct Activations {
    int B = 0, T = 0;
    bool with_grad = false;

    Tensor x;
    Tensor xb;
    Tensor xb2;
    Tensor q, k, v;
    Tensor qkv;
    Tensor att_out;
    Tensor proj;
    Tensor gate, up, act;

    std::vector<Tensor> mamba_states;
    std::vector<Tensor> mamba_conv_states;

    Tensor moe_gate, moe_up, moe_act;
    Tensor moe_probs;
    Tensor moe_idx;
    Tensor moe_w;
    Tensor moe_dact;
    Tensor moe_auxfrac;
    Tensor ffn_out;

    Tensor logits;
    Tensor pos;

    int pos_cached_B = -1;
    int pos_cached_T = -1;
    int pos_offset = 0;
    int pos_cached_offset = -1;
    int ce_chunks = 1;
    i64 ce_rows = 0;

    std::vector<Tensor> saved_x;
    std::vector<Tensor> saved_xb;
    std::vector<Tensor> saved_rrms1;
    std::vector<Tensor> saved_q, saved_k, saved_v;
    std::vector<Tensor> saved_qk_rms_q;
    std::vector<Tensor> saved_qk_rms_k;
    std::vector<Tensor> saved_qk_raw_q;
    std::vector<Tensor> saved_qk_raw_k;
    Tensor attn_probs_tmp;
    std::vector<Tensor> saved_attout;
    std::vector<Tensor> saved_xmid;
    std::vector<Tensor> saved_xb2;
    std::vector<Tensor> saved_rrms2;
    std::vector<Tensor> saved_gate, saved_up, saved_act;
    std::vector<Tensor> saved_moe_probs;
    std::vector<Tensor> saved_moe_idx;
    std::vector<Tensor> saved_moe_w;
    Tensor saved_xfinal;
    Tensor saved_rrms_final;
    Tensor saved_hnorm;

    Tensor dx, dxb, dq, dk, dv, dattout, dproj, dgate, dup, dact, dffn, dlogits, dtmp;

    size_t bytes = 0;
};

class Model {
public:
    explicit Model(ModelConfig cfg, Device dev = Device::CPU, bool fp16_weights = false,
                   int quant_kind = -1);
    ~Model();

    const ModelConfig& config() const { return cfg_; }
    Device device() const { return device_; }

    std::vector<Parameter*>&       parameters()       { return params_; }
    const std::vector<Parameter*>& parameters() const { return params_; }
    Parameter* find_parameter(const std::string& name);
    const Parameter* find_parameter(const std::string& name) const;
    i64  num_parameters() const;
    i64  num_parameters_non_embedding() const;
    ParamReport parameter_report() const;
    void print_parameter_report() const;

    void init_weights(u64 seed = 42);
    void enable_fp16_weight_cache(bool on);
    void mark_weights_dirty();
    void invalidate_fp16_cache();
    bool fp16_weight_cache_enabled() const { return fp16_weight_cache_; }
    size_t fp16_weight_cache_bytes() const;

    bool quant_weight_storage_enabled() const { return weight_q4_; }
    bool fp16_weight_storage_enabled() const { return weight_f16_; }
    int  quant_kind() const { return q4_kind_; }
    size_t weight_storage_bytes() const;
    void enable_grad(bool on);
    bool grad_enabled() const { return grad_enabled_; }
    void zero_grad();
    void to(Device dev);
    void set_rope_runtime(float scale, float yarn_mscale);
    const float* rope_inv_freq_ptr() const;

    Tensor& forward(const i32* ids, int B, int T, Activations& act,
                    const i32* segment_ids = nullptr,
                    bool reset_mamba_states = true);

    double  forward_backward(const i32* ids, const i32* targets, int B, int T,
                              Activations& act, i64* out_ntok = nullptr,
                              float dout_scale = 1.0f, bool want_aux_stats = false,
                              const i32* segment_ids = nullptr,
                              const i32* host_targets = nullptr,
                              const float* embed_override = nullptr,
                              int override_pos = 0,
                              int n_override = 0,
                              float* out_override_dx = nullptr);

    Activations make_activations(int B, int T, bool with_grad, int ce_chunks = 1) const;
    size_t estimate_activation_bytes(int B, int T, bool with_grad, int ce_chunks = 1) const;

    struct MemoryPlan {
        u64    params = 0;
        u64    params_no_embedding = 0;
        size_t weights = 0;
        size_t grads = 0;
        size_t fp16_cache = 0;
        size_t activations = 0;
        size_t optimizer = 0;
        size_t static_total = 0;
        size_t total = 0;
    };
    static u64    count_parameters(const ModelConfig& cfg);

    static size_t count_fp16_cache_bytes(const ModelConfig& cfg);
    static MemoryPlan plan_memory(const ModelConfig& cfg, int B, int T,
                                  bool with_grad, int ce_chunks,
                                  bool fp16_weight_cache,
                                  bool fp16_weights = false,
                                  int quant_kind = -1);
    static size_t estimate_activation_bytes_for(const ModelConfig& cfg,
                                                bool fp16_weight_cache_on,
                                                int B, int T, bool with_grad,
                                                int ce_chunks);

    static size_t mamba_state_bytes(const ModelConfig& cfg, int batch = 1);
    static size_t kv_cache_bytes(const ModelConfig& cfg, int max_len);

    struct WorkspacePlan {
        size_t gemm_bytes = 0;
        size_t moe_bytes = 0;
        size_t mamba_bytes = 0;
    };
    static WorkspacePlan workspace_plan(const ModelConfig& cfg, int B, int T);

    double moe_aux_loss(Activations& act, int B, int T);

    std::string moe_balance_report() const;
    void moe_balance_reset();

    void merge_lora();
    bool lora_delta_for(const std::string& name, std::vector<float>& out) const;
    bool has_lora() const { return cfg_.lora_rank > 0; }
    i64  trainable_parameters() const;

    void ensure_moe_bias();
    const float* moe_bias_ptr(int layer) const;
    void update_moe_bias(int layer, const float* frac_host, int ne);
    void set_moe_bias(int layer, const float* values, size_t count);
    const std::vector<std::vector<float>>& moe_bias_all() const { return moe_bias_; }

    void accumulate_moe_bias_fracs(Activations& act, int B, int T);

    void apply_moe_bias_step(const float* global_count_sum, bool apply);

    std::vector<float>& moe_bias_acc_host() { return moe_bias_acc_; }

    Tensor& moe_bias_acc_dev() { return moe_bias_acc_dev_; }

    void save_raw(const std::string& path) const;
    bool load_raw(const std::string& path);

    LayerParams&       layer(int i)       { return layers_[static_cast<size_t>(i)]; }
    const LayerParams& layer(int i) const { return layers_[static_cast<size_t>(i)]; }
    const u16* fused_qkv_ptr(int i) const;
    Parameter& tok_embeddings() { return tok_emb_; }
    Parameter& final_norm()     { return final_norm_; }
    Parameter& lm_head()        { return cfg_.tie_embeddings ? tok_emb_ : lm_head_; }
    const Parameter& lm_head() const { return cfg_.tie_embeddings ? tok_emb_ : lm_head_; }

private:
    void alloc_param(Parameter& p, const std::string& name, std::vector<i64> shape, bool decay);
    DType storage_dtype_for(const std::string& name) const;
    void init_lora(LoraAdapter& ad, const std::string& name,
                   int in_d, int out_d, int r, float alpha,
                   Device dev, std::mt19937_64& rng);
    void rebuild_rope_cache();

    void move_to_device(Device dev, bool announce);

    double moe_layer_aux(Device dev, const float* probs, const i32* idx,
                         float* auxfrac_dev, int layer, i64 N, int K, int ne,
                         bool want_stats = true, double* d_raw_accum = nullptr);

    void forward_mamba_layer(Device dev, int l, Activations& act, i64 N, const int32_t* segment_ids = nullptr);

    void forward_body(const i32* ids, int B, int T, Activations& act,
                      const i32* segment_ids, bool reset_mamba_states,
                      const float* embed_override = nullptr,
                      int override_pos = 0,
                      int n_override = 0);

    ModelConfig cfg_;

    mutable std::vector<std::vector<double>> moe_tok_acc_;
    mutable u64 moe_aux_batches_ = 0;

    std::vector<std::vector<float>> moe_bias_;
    std::vector<Tensor>             moe_bias_dev_;
    float                           moe_bias_lr_ = 0.001f;

    std::vector<float>              moe_bias_acc_;

    Tensor                          moe_bias_acc_dev_;
    Device      device_ = Device::CPU;
    bool        grad_enabled_ = false;
    bool        fp16_weight_cache_ = false;
    bool        fp16_weights_dirty_ = false;
    bool        weight_f16_ = false;
    bool        weight_q4_ = false;
    int         q4_kind_ = 0;
    Tensor      rope_inv_freq_;

    Parameter                tok_emb_;
    Parameter                lm_head_;
    Parameter                final_norm_;
    std::vector<LayerParams> layers_;
    std::vector<Parameter*>  params_;
};

}
