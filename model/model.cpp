#include "model/model.h"
#include "core/ops.h"
#include "core/quant.h"
#include "quantization/quantize.h"
#include "core/device.h"
#include "core/common.h"

#ifdef GAI_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>
#include <mutex>
#include <unordered_set>

namespace gai {

namespace {

inline void linear_param(Device dev, const Tensor& w, const float* x, float* y,
                         int M, int K, int N) {
    if (w.dtype() == DType::F16)
        ops::linear_forward_fp16(dev, x, w.ptr<u16>(), y, M, K, N);
    else if (is_quant(w)) {
        if (ops::fused_q4_enabled())
            ops::linear_forward_q4(dev, x, w.data_ptr(), y, M, K, N, quant_kind(w.dtype()));
        else
            ops::linear_forward_fp16(dev, x, weight16(dev, w, 0), y, M, K, N);
    }
    else
        ops::linear_forward(dev, x, w.f32(), y, M, K, N);
}

inline float* gptr(Tensor& g) {
    return g.defined() ? g.ptr<float>() : nullptr;
}

}

double Model::moe_layer_aux(Device dev, const float* probs, const i32* idx,
                             float* auxfrac_dev, int layer, i64 N, int K, int ne,
                             bool want_stats, double* d_raw_accum) {
    if (dev == Device::CUDA) {
        std::vector<float> h_frac, h_psum;
        float* hf = nullptr;
        float* hp = nullptr;
        if (want_stats || !d_raw_accum) {
            h_frac.assign(static_cast<size_t>(ne), 0.0f);
            h_psum.assign(static_cast<size_t>(ne), 0.0f);
            hf = h_frac.data();
            hp = h_psum.data();
        }
        if (ops::moe_aux_gpu(dev, probs, idx, auxfrac_dev,
                             hf, hp, N, K, ne, d_raw_accum)) {
            if (!hf) return 0.0;
            std::vector<double> cnt(ne, 0.0);
            const double denom = static_cast<double>(N) * K;
            for (int e = 0; e < ne; ++e) cnt[e] = (double)h_frac[(size_t)e] * denom;
            double raw = 0.0;
            for (int e = 0; e < ne; ++e) {
                double f = denom > 0 ? cnt[e] / denom : 0.0;
                double m = N > 0 ? (double)h_psum[(size_t)e] / (double)N : 0.0;
                raw += m * f;
            }
            raw *= ne;
            if (moe_tok_acc_.size() != layers_.size())
                moe_tok_acc_.assign(layers_.size(), std::vector<double>(ne, 0.0));
            if (layer >= 0 && static_cast<size_t>(layer) < moe_tok_acc_.size()) {
                for (int e = 0; e < ne && e < static_cast<int>(moe_tok_acc_[static_cast<size_t>(layer)].size()); ++e)
                    moe_tok_acc_[static_cast<size_t>(layer)][static_cast<size_t>(e)] += cnt[e];
                ++moe_aux_batches_;
            }
            return raw;
        }
    }
    std::vector<i32> h_idx;
    std::vector<float> h_probs;
    const i32* idx_h = idx;
    const float* probs_h = probs;
    if (dev != Device::CPU) {
        h_idx.resize(static_cast<size_t>(N) * K);
        device_copy(h_idx.data(), Device::CPU, idx, dev,
                    sizeof(i32) * h_idx.size());
        h_probs.resize(static_cast<size_t>(N) * ne);
        device_copy(h_probs.data(), Device::CPU, probs, dev,
                    sizeof(float) * h_probs.size());
        idx_h = h_idx.data();
        probs_h = h_probs.data();
    }

    std::vector<double> cnt(ne, 0.0), psum(ne, 0.0);
    for (i64 t = 0; t < N; ++t)
        for (int k = 0; k < K; ++k) {
            int e = idx_h[static_cast<size_t>(t) * K + k];
            if (e >= 0 && e < ne) cnt[e] += 1.0;
        }
    for (i64 t = 0; t < N; ++t)
        for (int e = 0; e < ne; ++e)
            psum[e] += probs_h[static_cast<size_t>(t) * ne + e];

    std::vector<float> frac(ne);
    double raw = 0.0;
    const double denom = static_cast<double>(N) * K;
    for (int e = 0; e < ne; ++e) {
        double f = denom > 0 ? cnt[e] / denom : 0.0;
        double m = N > 0 ? psum[e] / static_cast<double>(N) : 0.0;
        frac[e] = static_cast<float>(f);
        raw += m * f;
    }
    raw *= ne;
    device_copy(auxfrac_dev, dev, frac.data(), Device::CPU,
                sizeof(float) * frac.size());

    if (moe_tok_acc_.size() != layers_.size())
        moe_tok_acc_.assign(layers_.size(), std::vector<double>(ne, 0.0));
    if (layer >= 0 && static_cast<size_t>(layer) < moe_tok_acc_.size()) {
        for (int e = 0; e < ne && e < static_cast<int>(moe_tok_acc_[static_cast<size_t>(layer)].size()); ++e)
            moe_tok_acc_[static_cast<size_t>(layer)][static_cast<size_t>(e)] += cnt[e];
        ++moe_aux_batches_;
    }
    return raw;
}

std::string Model::moe_balance_report() const {
    if (!cfg_.use_moe || moe_tok_acc_.empty() || moe_aux_batches_ == 0) return "";
    std::ostringstream ss;
    ss << "expert load (share of routed slots):";
    for (size_t l = 0; l < moe_tok_acc_.size(); ++l) {
        double tot = 0;
        for (double c : moe_tok_acc_[l]) tot += c;
        if (tot <= 0) continue;
        double mn = 1e9, mx = 0;
        for (double c : moe_tok_acc_[l]) {
            double s = c / tot;
            mn = std::min(mn, s);
            mx = std::max(mx, s);
        }
        ss << strfmt(" L%zu[%.2f..%.2f]%s", l, mn, mx,
                     (mn < 0.25 / cfg_.num_experts || mx > 4.0 / cfg_.num_experts)
                     ? "!" : "");
    }
    return ss.str();
}

void Model::moe_balance_reset() {
    moe_tok_acc_.clear();
    moe_aux_batches_ = 0;
}

void Model::ensure_moe_bias() {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return;
    const int L = cfg_.num_layers, ne = cfg_.num_experts;
    if (static_cast<int>(moe_bias_.size()) != L) {
        moe_bias_.assign(static_cast<size_t>(L), std::vector<float>(static_cast<size_t>(ne), 0.0f));
        moe_bias_dev_.clear();
        moe_bias_dev_.reserve(static_cast<size_t>(L));
        for (int l = 0; l < L; ++l) {
            Tensor t({(i64)ne}, DType::F32, device_);
            t.zero_();
            moe_bias_dev_.push_back(std::move(t));
        }
    }
}

const float* Model::moe_bias_ptr(int layer) const {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return nullptr;
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size()) return nullptr;
    if (device_ == Device::CPU) return moe_bias_[static_cast<size_t>(layer)].data();
    if (static_cast<size_t>(layer) >= moe_bias_dev_.size()) return nullptr;
    return moe_bias_dev_[static_cast<size_t>(layer)].f32();
}

void Model::update_moe_bias(int layer, const float* frac_host, int ne) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free || !frac_host || ne <= 0) return;
    ensure_moe_bias();
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size()) return;
    const float target = static_cast<float>(cfg_.moe_top_k) / static_cast<float>(ne);
    auto& b = moe_bias_[static_cast<size_t>(layer)];
    for (int e = 0; e < ne && e < static_cast<int>(b.size()); ++e) {
        float err = frac_host[e] - target;
        b[static_cast<size_t>(e)] -= moe_bias_lr_ * err;
        if (b[static_cast<size_t>(e)] > 0.5f) b[static_cast<size_t>(e)] = 0.5f;
        if (b[static_cast<size_t>(e)] < -0.5f) b[static_cast<size_t>(e)] = -0.5f;
    }
    if (device_ == Device::CUDA && static_cast<size_t>(layer) < moe_bias_dev_.size()) {
        device_copy(moe_bias_dev_[static_cast<size_t>(layer)].data_ptr(), device_,
                    b.data(), Device::CPU, sizeof(float) * b.size());
    }
}

void Model::accumulate_moe_bias_fracs(Activations& act, int B, int T) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return;
    const int L  = cfg_.num_layers;
    const int ne = cfg_.num_experts;
    const int K  = cfg_.moe_top_k;
    const i64 N  = static_cast<i64>(B) * T;
    ensure_moe_bias();
    const size_t total = static_cast<size_t>(L) * static_cast<size_t>(ne);
    if (static_cast<int>(act.saved_moe_probs.size()) != L ||
        static_cast<int>(act.saved_moe_idx.size())   != L) {
        return;
    }
    if (device_ == Device::CUDA) {
        if (!moe_bias_acc_dev_.defined() ||
            static_cast<size_t>(moe_bias_acc_dev_.numel()) != total) {
            moe_bias_acc_dev_ = Tensor::zeros({static_cast<i64>(total)}, DType::F32, device_);
        }
        float* base = moe_bias_acc_dev_.f32();
        for (int l = 0; l < L; ++l) {
            const size_t sl = static_cast<size_t>(l);
            if (!act.saved_moe_idx[sl].defined()) continue;
            ops::moe_count_slots(device_, act.saved_moe_idx[sl].i32p(),
                                 base + sl * static_cast<size_t>(ne),
                                 N * K, ne);
        }
        return;
    }
    if (moe_bias_acc_.size() != total) moe_bias_acc_.assign(total, 0.0f);
    for (int l = 0; l < L; ++l) {
        const size_t sl = static_cast<size_t>(l);
        if (!act.saved_moe_idx[sl].defined()) continue;
        std::vector<i32> h_idx(static_cast<size_t>(N) * static_cast<size_t>(K));
        device_copy(h_idx.data(), Device::CPU, act.saved_moe_idx[sl].i32p(), device_,
                    sizeof(i32) * h_idx.size());
        float* acc = moe_bias_acc_.data() + sl * static_cast<size_t>(ne);
        for (i64 t = 0; t < N; ++t) {
            for (int k = 0; k < K; ++k) {
                const i32 e = h_idx[static_cast<size_t>(t) * static_cast<size_t>(K) + k];
                if (e >= 0 && e < ne) acc[e] += 1.0f;
            }
        }
    }
}

void Model::apply_moe_bias_step(const float* global_count_sum, bool apply) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) {
        moe_bias_acc_.clear();
        moe_bias_acc_dev_ = Tensor();
        return;
    }
    const int L  = cfg_.num_layers;
    const int ne = cfg_.num_experts;
    if (!apply || !global_count_sum || (moe_bias_acc_.empty() && !moe_bias_acc_dev_.defined())) {
        moe_bias_acc_.clear();
        moe_bias_acc_dev_ = Tensor();
        return;
    }
    ensure_moe_bias();
    std::vector<float> frac(static_cast<size_t>(ne));
    for (int l = 0; l < L; ++l) {
        const size_t base = static_cast<size_t>(l) * static_cast<size_t>(ne);
        double row_sum = 0.0;
        for (int e = 0; e < ne; ++e) row_sum += static_cast<double>(global_count_sum[base + e]);
        const double routed_tokens = row_sum /
            static_cast<double>(cfg_.moe_top_k > 0 ? cfg_.moe_top_k : 1);
        const double inv = (routed_tokens > 0.0) ? 1.0 / routed_tokens : 0.0;
        if (inv <= 0.0) continue;
        for (int e = 0; e < ne; ++e)
            frac[static_cast<size_t>(e)] =
                static_cast<float>(static_cast<double>(global_count_sum[base + e]) * inv);
        update_moe_bias(l, frac.data(), ne);
    }
    moe_bias_acc_.clear();
    moe_bias_acc_dev_ = Tensor();
}

void Model::set_moe_bias(int layer, const float* values, size_t count) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free || !values) return;
    ensure_moe_bias();
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size() ||
        count != moe_bias_[static_cast<size_t>(layer)].size()) {
        GAI_FAIL("checkpoint moe bias shape mismatch");
    }
    auto& bias = moe_bias_[static_cast<size_t>(layer)];
    std::copy(values, values + count, bias.begin());
    if (device_ == Device::CUDA && static_cast<size_t>(layer) < moe_bias_dev_.size()) {
        device_copy(moe_bias_dev_[static_cast<size_t>(layer)].data_ptr(), device_,
                    bias.data(), Device::CPU, sizeof(float) * bias.size());
    }
}

void ModelConfig::validate() const {
    GAI_CHECK(vocab_size > 0, "vocab_size must be > 0");
    GAI_CHECK(hidden_size > 0, "hidden_size must be > 0");
    GAI_CHECK(num_layers > 0, "num_layers must be > 0");
    GAI_CHECK(num_heads > 0, "num_heads must be > 0");
    GAI_CHECK(num_kv_heads > 0, "num_kv_heads must be > 0");
    GAI_CHECK(hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads");
    GAI_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    GAI_CHECK(head_dim() % 2 == 0, "head_dim must be even (RoPE pairs)");
    GAI_CHECK(intermediate_size > 0, "intermediate_size must be > 0");
    GAI_CHECK(max_seq_len > 0, "max_seq_len must be > 0");
    GAI_CHECK(std::isfinite(rope_theta) && rope_theta > 0.0f, "rope_theta must be finite and > 0");
    GAI_CHECK(std::isfinite(rms_eps) && rms_eps >= 1e-12f && rms_eps <= 1e-2f,
              "rms_eps must be finite and in [1e-12,1e-2]");
    GAI_CHECK(std::isfinite(init_std) && init_std > 0.0f && init_std <= 0.1f,
              "init_std must be finite and in (0,0.1]");
    GAI_CHECK(std::isfinite(rope_scale) && rope_scale >= 1.0f, "rope_scale must be finite and >= 1.0 (1.0 = off)");
    GAI_CHECK(rope_scale <= 8.0f, "rope_scale must be <= 8.0 (larger is unvalidated extrapolation)");
    GAI_CHECK(std::isfinite(z_loss_scale) && z_loss_scale >= 0.0f && z_loss_scale <= 0.01f,
              "z_loss_scale must be finite and in [0,0.01]");
    GAI_CHECK(std::isfinite(logits_scaling) && logits_scaling > 0.0f,
              "logits_scaling must be finite and > 0");
    GAI_CHECK(std::isfinite(embedding_multiplier) && std::isfinite(residual_multiplier) &&
              std::isfinite(attention_multiplier),
              "multipliers must be finite");
    GAI_CHECK(std::isfinite(moe_jitter) && moe_jitter >= 0.0f && moe_jitter <= 0.5f,
              "moe_jitter must be finite and in [0,0.5]");
    GAI_CHECK(std::isfinite(rope_yarn_mscale) && rope_yarn_mscale >= 0.0f && rope_yarn_mscale <= 2.0f,
              "rope_yarn_mscale must be finite and in [0,2]");
    GAI_CHECK(std::isfinite(rope_yarn_low) && rope_yarn_low >= 1.0f && rope_yarn_low <= 128.0f,
              "rope_yarn_low must be finite and in [1,128]");
    GAI_CHECK(std::isfinite(rope_yarn_high) && rope_yarn_high >= 1.0f && rope_yarn_high <= 128.0f,
              "rope_yarn_high must be finite and in [1,128]");
    GAI_CHECK(rope_yarn_high >= rope_yarn_low, "rope_yarn_high must be >= rope_yarn_low");
    GAI_CHECK(sliding_window >= 0 && sliding_window <= 16384, "sliding_window must be in [0,16384]");
    GAI_CHECK(!mamba_proj_bias,
              "mamba_proj_bias=true is not implemented (no in_proj/out_proj bias tensors)");
    GAI_CHECK(rope_type == 0 || rope_type == 1, "rope_type must be 0 (interleaved) or 1 (neox)");
    if (moe_aux_free && moe_aux_scale != 0.0f) {
        GAI_CHECK(false, "moe_aux_free=true requires moe_aux_scale: 0.0 (aux-loss-free has no aux grad)");
    }
    if (use_moe) {
        GAI_CHECK(num_experts > 0 && num_experts <= 64, "num_experts must be in [1,64]");
        GAI_CHECK(moe_top_k > 0 && moe_top_k <= num_experts, "moe_top_k must be in [1,num_experts]");
        GAI_CHECK(moe_top_k <= 8, "moe_top_k must be <= 8 (router kernel limit)");
        GAI_CHECK(moe_allow_dense || moe_top_k * 2 <= num_experts,
                  "moe_top_k must be <= num_experts/2 (sparse MoE; dense routing OOMs T4 and removes the MoE advantage; "
                  "set model.moe_allow_dense=true only for short research runs)");
        GAI_CHECK(moe_expert_dim > 0, "moe_expert_dim must be > 0");
        GAI_CHECK(std::isfinite(moe_aux_scale) && moe_aux_scale >= 0.0f && moe_aux_scale <= 1.0f,
                  "moe_aux_scale must be finite and in [0,1] (larger drowns CE loss)");
    }
    if (arch == ArchGranite) {
        GAI_CHECK(static_cast<int>(layer_types.size()) == num_layers,
                  strfmt("granite layer_types must have exactly num_layers (%d) entries, got %zu",
                         num_layers, layer_types.size()));
        int n_mamba = 0, n_attn = 0;
        for (int k : layer_types) {
            GAI_CHECK(k == 0 || k == 1,
                      "granite layer_types entries must be 0 (mamba) or 1 (attention)");
            (k == 0 ? n_mamba : n_attn)++;
        }
        GAI_CHECK(n_mamba > 0, "granite layer_types declares no mamba layers");
        GAI_CHECK(n_attn > 0, "granite layer_types declares no attention layers");
        GAI_CHECK(mamba_n_groups == 1,
                  strfmt("granite: mamba_n_groups must be 1 (the conv output layout is "
                         "[x | B | C] with one B and one C block), got %d",
                         mamba_n_groups));
        GAI_CHECK(mamba_n_heads > 0 && mamba_d_head > 0 && mamba_d_state > 0 &&
                  mamba_d_conv > 0 && mamba_expand > 0 && mamba_chunk_size > 0,
                  strfmt("granite: mamba geometry must be positive (heads=%d d_head=%d "
                         "d_state=%d d_conv=%d expand=%d chunk=%d)",
                         mamba_n_heads, mamba_d_head, mamba_d_state,
                         mamba_d_conv, mamba_expand, mamba_chunk_size));
        const i64 d_inner = static_cast<i64>(mamba_expand) *
                            static_cast<i64>(hidden_size);
        const i64 scan_inner = static_cast<i64>(mamba_n_heads) *
                               static_cast<i64>(mamba_d_head);
        GAI_CHECK(scan_inner == d_inner,
                  "granite: mamba_n_heads*mamba_d_head (" +
                      std::to_string(scan_inner) +
                      ") must equal expand*hidden_size (" +
                      std::to_string(d_inner) +
                      "); a smaller scan width leaves output elements unwritten");
    }
    {
        const int KV_TILE = head_dim() <= 64 ? 64 : (head_dim() <= 96 ? 48 : 32);
        const int group = num_heads / num_kv_heads;
        size_t shmem_fwd = sizeof(float) * (static_cast<size_t>(head_dim()) + static_cast<size_t>(KV_TILE) * head_dim() * 2 + KV_TILE + static_cast<size_t>(head_dim()));
        size_t shmem_bwd = sizeof(float) * (static_cast<size_t>(2) * KV_TILE * head_dim() + group);
        size_t shmem_max = std::max(shmem_fwd, shmem_bwd);
        GAI_CHECK(shmem_max <= 48 * 1024,
                  strfmt("head_dim=%d too large for attention kernel shared memory (need %s, T4 limit 48KB). "
                         "Reduce num_heads or hidden_size, or use a GPU with more shared memory.",
                         head_dim(), human_bytes(shmem_max).c_str()));
    }
}

ModelConfig ModelConfig::from_config(const Config& c, const std::string& p) {
    ModelConfig m;
    auto key = [&](const char* k) { return p.empty() ? std::string(k) : p + "." + k; };
    auto get_int_checked = [&](const char* k, int def) {
        i64 v = c.get_int(key(k), static_cast<i64>(def));
        GAI_CHECK(v > 0 && v <= static_cast<i64>(std::numeric_limits<int>::max()),
                  std::string("model.") + k + " out of int range");
        return static_cast<int>(v);
    };
    auto get_int_checked0 = [&](const char* k, int def) {
        i64 v = c.get_int(key(k), static_cast<i64>(def));
        GAI_CHECK(v >= 0 && v <= static_cast<i64>(std::numeric_limits<int>::max()),
                  std::string("model.") + k + " out of int range");
        return static_cast<int>(v);
    };
    m.vocab_size        = get_int_checked("vocab_size", m.vocab_size);
    m.hidden_size       = get_int_checked("hidden_size", m.hidden_size);
    m.num_layers        = get_int_checked("num_layers", m.num_layers);
    m.num_heads         = get_int_checked("num_heads", m.num_heads);
    m.num_kv_heads      = get_int_checked("num_kv_heads", m.num_kv_heads);
    m.intermediate_size = get_int_checked("intermediate_size", m.intermediate_size);
    m.max_seq_len       = get_int_checked("max_seq_len", m.max_seq_len);
    m.rope_theta        = c.get_f32(key("rope_theta"), m.rope_theta);
    m.rms_eps           = c.get_f32(key("rms_eps"), m.rms_eps);
    m.tie_embeddings    = c.get_bool(key("tie_embeddings"), m.tie_embeddings);
    m.init_std          = c.get_f32(key("init_std"), m.init_std);
    m.use_moe           = c.get_bool(key("use_moe"), m.use_moe);
    m.num_experts       = get_int_checked("num_experts", m.num_experts);
    m.moe_top_k         = get_int_checked("moe_top_k", m.moe_top_k);
    m.moe_expert_dim    = get_int_checked("moe_expert_dim", m.moe_expert_dim);
    m.moe_shared        = c.get_bool(key("moe_shared"), m.moe_shared);
    m.shared_expert_intermediate_size =
        get_int_checked0("shared_expert_intermediate_size",
                         m.shared_expert_intermediate_size);
    if (m.shared_expert_intermediate_size > 0) m.moe_shared = true;
    m.moe_aux_scale     = c.get_f32(key("moe_aux_scale"), m.moe_aux_scale);
    m.moe_jitter        = c.get_f32(key("moe_jitter"), m.moe_jitter);
    m.moe_allow_dense   = c.get_bool(key("moe_allow_dense"), m.moe_allow_dense);
    m.moe_aux_free      = c.get_bool(key("moe_aux_free"), m.moe_aux_free);
    m.rope_yarn_low     = c.get_f32(key("rope_yarn_low"), m.rope_yarn_low);
    m.rope_yarn_high    = c.get_f32(key("rope_yarn_high"), m.rope_yarn_high);
    m.sliding_window    = get_int_checked0("sliding_window", m.sliding_window);
    m.rope_type         = get_int_checked0("rope_type", m.rope_type);
    m.head_dim_override = get_int_checked0("head_dim", m.head_dim_override);
    m.position_embedding = get_int_checked0("position_embedding", m.position_embedding);

    m.arch = (get_int_checked0("arch", 0) == 1) ? ModelConfig::ArchGranite : ModelConfig::ArchQwen;
    m.embedding_multiplier = c.get_f32(key("embedding_multiplier"), m.embedding_multiplier);
    m.residual_multiplier  = c.get_f32(key("residual_multiplier"),  m.residual_multiplier);
    m.attention_multiplier = c.get_f32(key("attention_multiplier"), m.attention_multiplier);
    m.logits_scaling       = c.get_f32(key("logits_scaling"),       m.logits_scaling);
    m.mamba_n_heads   = get_int_checked0("mamba_n_heads", m.mamba_n_heads);
    m.mamba_d_head    = get_int_checked0("mamba_d_head", m.mamba_d_head);
    m.mamba_d_state   = get_int_checked0("mamba_d_state", m.mamba_d_state);
    m.mamba_n_groups  = get_int_checked0("mamba_n_groups", m.mamba_n_groups);
    m.mamba_d_conv    = get_int_checked0("mamba_d_conv", m.mamba_d_conv);
    m.mamba_expand    = get_int_checked0("mamba_expand", m.mamba_expand);
    m.mamba_chunk_size = get_int_checked0("mamba_chunk_size", m.mamba_chunk_size);
    m.mamba_conv_bias = c.get_bool(key("mamba_conv_bias"), m.mamba_conv_bias);
    m.mamba_proj_bias = c.get_bool(key("mamba_proj_bias"), m.mamba_proj_bias);
    {
        const std::string lt = c.get_str(key("layer_types"), "");
        if (!lt.empty()) {
            m.layer_types.clear();
            size_t i = 0;
            while (i < lt.size()) {
                size_t cpos = lt.find(',', i);
                std::string tok = lt.substr(i, cpos == std::string::npos ? cpos : cpos - i);
                size_t b = tok.find_first_not_of(" \t");
                size_t e = tok.find_last_not_of(" \t");
                if (b != std::string::npos) tok = tok.substr(b, e - b + 1);
                if (tok == "attention") m.layer_types.push_back(1);
                else if (tok == "mamba") m.layer_types.push_back(0);
                else GAI_CHECK(false, "model.layer_types: unknown layer kind '" + tok +
                                      "' (expected 'attention' or 'mamba')");
                if (cpos == std::string::npos) break;
                i = cpos + 1;
            }
        }
    }
    m.use_qk_norm       = c.get_bool(key("use_qk_norm"), m.use_qk_norm);
    m.use_qkv_bias      = c.get_bool(key("use_qkv_bias"), m.use_qkv_bias);
    m.z_loss_scale      = c.get_f32(key("z_loss_scale"), m.z_loss_scale);
    m.rope_scale        = c.get_f32(key("rope_scale"), m.rope_scale);
    m.rope_yarn_mscale  = c.get_f32(key("rope_yarn_mscale"), m.rope_yarn_mscale);

    m.lora_rank        = get_int_checked0("lora_rank", m.lora_rank);
    m.lora_alpha       = c.get_f32(key("lora_alpha"), m.lora_alpha);
    m.lora_target_q    = c.get_bool(key("lora_target_q"), m.lora_target_q);
    m.lora_target_k    = c.get_bool(key("lora_target_k"), m.lora_target_k);
    m.lora_target_v    = c.get_bool(key("lora_target_v"), m.lora_target_v);
    m.lora_target_o    = c.get_bool(key("lora_target_o"), m.lora_target_o);
    m.lora_target_gate = c.get_bool(key("lora_target_gate"), m.lora_target_gate);
    m.lora_target_up   = c.get_bool(key("lora_target_up"), m.lora_target_up);
    m.lora_target_down = c.get_bool(key("lora_target_down"), m.lora_target_down);

    m.validate();
    return m;
}

static float rope_theta_eff_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return m.rope_theta;
    float hd = static_cast<float>(m.head_dim());
    float e = (hd > 2.0f) ? hd / (hd - 2.0f) : 1.0f;
    return m.rope_theta * std::pow(m.rope_scale, e);
}

static float rope_mscale_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return 1.0f;
    if (m.rope_yarn_mscale > 0.0f) return m.rope_yarn_mscale;
    float base = 0.1f * std::log(m.rope_scale) + 1.0f;
    if (m.rope_yarn_low <= 1.0f && m.rope_yarn_high >= 32.0f) return base;
    float span = m.rope_yarn_high - m.rope_yarn_low;
    float ramp = (span > 0.0f) ? (32.0f - m.rope_yarn_low) / span : 1.0f;
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    return 1.0f + (base - 1.0f) * (0.5f + 0.5f * ramp);
}


std::string ModelConfig::arch_identity() const {
    std::ostringstream ss;
    ss << "v1|voc=" << vocab_size << "|d=" << hidden_size << "|L=" << num_layers
       << "|H=" << num_heads << "|KV=" << num_kv_heads << "|ffn=" << intermediate_size
       << "|ctx=" << max_seq_len << "|theta=" << strfmt("%.6f", (double)rope_theta).c_str()
       << "|eps=" << strfmt("%.9f", (double)rms_eps).c_str()
       << "|tie=" << (tie_embeddings ? 1 : 0)
       << "|moe=" << (use_moe ? 1 : 0) << "|ne=" << num_experts << "|k=" << moe_top_k
       << "|E=" << moe_expert_dim << "|sh=" << (moe_shared ? 1 : 0)
       << "|aux=" << strfmt("%.6f", (double)moe_aux_scale).c_str()
       << "|auxfree=" << (moe_aux_free ? 1 : 0)
       << "|denseok=" << (moe_allow_dense ? 1 : 0)
       << "|jit=" << strfmt("%.6f", (double)moe_jitter).c_str()
       << "|qkn=" << (use_qk_norm ? 1 : 0)
       << "|z=" << strfmt("%.7f", (double)z_loss_scale).c_str()
       << "|rs=" << strfmt("%.4f", (double)rope_scale).c_str()
       << "|ym=" << strfmt("%.4f", (double)rope_yarn_mscale).c_str()
       << "|yl=" << strfmt("%.4f", (double)rope_yarn_low).c_str()
       << "|yh=" << strfmt("%.4f", (double)rope_yarn_high).c_str()
       << "|swa=" << sliding_window << "|rt=" << rope_type
       << "|pos=" << position_embedding
       << "|ahd=" << head_dim_override
       << "|arch=" << (arch == ArchGranite ? "hybrid" : "standard")
       << "|mult=" << strfmt("%.4f,%.4f,%.6f,%.4f",
                              (double)embedding_multiplier,
                              (double)residual_multiplier,
                              (double)attention_multiplier,
                              (double)logits_scaling).c_str()
       << "|sst=" << shared_expert_intermediate_size
       << "|qkvb=" << (use_qkv_bias ? 1 : 0) << "|lt=";
    for (int lt : layer_types) ss << lt;
    if (arch == ArchGranite)
        ss << "|mb=" << mamba_n_heads << "," << mamba_d_head << "," << mamba_d_state
           << "," << mamba_n_groups << "," << mamba_d_conv << "," << mamba_expand
           << "," << mamba_chunk_size << "," << (mamba_conv_bias ? 1 : 0)
           << "," << (mamba_proj_bias ? 1 : 0);
    ss << "|lora=" << lora_rank << "," << strfmt("%.6f", (double)lora_alpha).c_str()
       << "," << (lora_target_q ? 1 : 0) << (lora_target_k ? 1 : 0)
       << (lora_target_v ? 1 : 0) << (lora_target_o ? 1 : 0)
       << (lora_target_gate ? 1 : 0) << (lora_target_up ? 1 : 0)
       << (lora_target_down ? 1 : 0);
    return ss.str();
}

bool ModelConfig::same_architecture_as(const ModelConfig& o, std::string* reason) const {
    auto fail = [&](const std::string& why) {
        if (reason) *reason = why;
        return false;
    };
    if (vocab_size != o.vocab_size) return fail("vocab_size differs");
    if (hidden_size != o.hidden_size) return fail("hidden_size differs");
    if (num_layers != o.num_layers) return fail("num_layers differs");
    if (num_heads != o.num_heads) return fail("num_heads differs");
    if (num_kv_heads != o.num_kv_heads) return fail("num_kv_heads differs");
    if (intermediate_size != o.intermediate_size) return fail("intermediate_size differs");
    if ((use_moe ? 1 : 0) != (o.use_moe ? 1 : 0)) return fail("moe flag differs");
    if (use_moe) {
        if (num_experts != o.num_experts) return fail("num_experts differs");
        if (moe_top_k != o.moe_top_k) return fail("moe_top_k differs");
        if (moe_expert_dim != o.moe_expert_dim) return fail("moe_expert_dim differs");
        if ((moe_shared ? 1 : 0) != (o.moe_shared ? 1 : 0)) return fail("moe_shared differs");
        if (moe_aux_free != o.moe_aux_free) return fail("moe_aux_free differs");
        if (shared_expert_intermediate_size != o.shared_expert_intermediate_size)
            return fail("shared_expert_intermediate_size differs");
        if (moe_aux_scale != o.moe_aux_scale) return fail("moe_aux_scale differs");
        if (moe_allow_dense != o.moe_allow_dense) return fail("moe_allow_dense differs");
    }
    if (tie_embeddings != o.tie_embeddings) return fail("tie_embeddings differs");
    if (use_qk_norm != o.use_qk_norm) return fail("use_qk_norm differs");
    if (use_qkv_bias != o.use_qkv_bias) return fail("use_qkv_bias differs");
    if (rope_type != o.rope_type) return fail("rope_type differs");
    if (position_embedding != o.position_embedding) return fail("position_embedding differs");
    if (head_dim_override != o.head_dim_override) return fail("head_dim override differs");
    if (z_loss_scale != o.z_loss_scale) return fail("z_loss_scale differs");
    if (rope_theta != o.rope_theta) return fail("rope_theta differs");
    if (rms_eps != o.rms_eps) return fail("rms_eps differs");
    if (rope_scale != o.rope_scale) return fail("rope_scale differs");
    if (rope_yarn_mscale != o.rope_yarn_mscale) return fail("rope_yarn_mscale differs");
    if (sliding_window != o.sliding_window) return fail("sliding_window differs");
    if (arch != o.arch) return fail("architecture family differs (hybrid vs standard)");
    if (layer_types.size() != o.layer_types.size()) return fail("layer_types length differs");
    for (size_t i = 0; i < layer_types.size(); ++i)
        if (layer_types[i] != o.layer_types[i]) return fail("layer kind differs at layer " + std::to_string(i));
    if (embedding_multiplier != o.embedding_multiplier) return fail("embedding_multiplier differs");
    if (residual_multiplier != o.residual_multiplier) return fail("residual_multiplier differs");
    if (attention_multiplier != o.attention_multiplier) return fail("attention_multiplier differs");
    if (logits_scaling != o.logits_scaling) return fail("logits_scaling differs");
    if (mamba_n_heads != o.mamba_n_heads) return fail("mamba_n_heads differs");
    if (mamba_d_head != o.mamba_d_head) return fail("mamba_d_head differs");
    if (mamba_d_state != o.mamba_d_state) return fail("mamba_d_state differs");
    if (mamba_n_groups != o.mamba_n_groups) return fail("mamba_n_groups differs");
    if (mamba_d_conv != o.mamba_d_conv) return fail("mamba_d_conv differs");
    if (mamba_expand != o.mamba_expand) return fail("mamba_expand differs");
    if (mamba_chunk_size != o.mamba_chunk_size) return fail("mamba_chunk_size differs");
    if (mamba_conv_bias != o.mamba_conv_bias) return fail("mamba_conv_bias differs");
    if (mamba_proj_bias != o.mamba_proj_bias) return fail("mamba_proj_bias differs");
    if (rope_yarn_low != o.rope_yarn_low) return fail("rope_yarn_low differs");
    if (rope_yarn_high != o.rope_yarn_high) return fail("rope_yarn_high differs");
    if (moe_jitter != o.moe_jitter) return fail("moe_jitter differs");
    return true;
}

void ModelConfig::take_adapter_settings_from(const ModelConfig& recipe) {
    lora_rank        = recipe.lora_rank;
    lora_alpha       = recipe.lora_alpha;
    lora_target_q    = recipe.lora_target_q;
    lora_target_k    = recipe.lora_target_k;
    lora_target_v    = recipe.lora_target_v;
    lora_target_o    = recipe.lora_target_o;
    lora_target_gate = recipe.lora_target_gate;
    lora_target_up   = recipe.lora_target_up;
    lora_target_down = recipe.lora_target_down;
}

std::string ModelConfig::summary() const {
    std::ostringstream ss;
    ss << "vocab=" << vocab_size << " d=" << hidden_size << " L=" << num_layers
       << " H=" << num_heads << " KV=" << num_kv_heads << " hd=" << head_dim()
       << " ctx=" << max_seq_len << " rope=" << rope_theta;
    if (rope_scale > 1.0f) ss << strfmt("(x%.1f NTK->%.0f)", (double)rope_scale,
                                        (double)rope_theta_eff_local(*this)).c_str();
    ss << (tie_embeddings ? " tied" : " untied");
    if (use_qk_norm) ss << " QK-Norm";
    if (z_loss_scale > 0) ss << strfmt(" Z(%.0e)", (double)z_loss_scale).c_str();
    if (rope_scale > 1.0f) ss << strfmt(" YaRN-m%.2f", (double)rope_mscale_local(*this)).c_str();
    if (moe_jitter > 0) ss << strfmt(" jit=%.2f", (double)moe_jitter).c_str();
    if (use_moe)
        ss << " MoE(ne=" << num_experts << ",k=" << moe_top_k
           << ",E=" << moe_expert_dim << (moe_shared ? ",shared" : "")
           << (moe_aux_free ? ",aux-free" : "")
           << (moe_allow_dense ? ",dense-ok" : "") << ")";
    else
        ss << " ffn=" << intermediate_size << " (dense)";
    if (sliding_window > 0) ss << " SWA(" << sliding_window << ")";
    if (rope_type == 1) ss << " RoPE-neox";
    return ss.str();
}

std::string ModelConfig::hybrid_summary() const {
    if (arch != ArchGranite) return {};
    std::ostringstream ss;
    ss << "  position    : " << (position_embedding == 1 ? "NoPE (no RoPE)" : "RoPE")
       << "\n  multipliers : embeddings x" << strfmt("%.6g", (double)embedding_multiplier)
       << " | residual x" << strfmt("%.6g", (double)residual_multiplier)
       << " | attention x" << strfmt("%.6g", (double)attention_multiplier)
       << " | logits /" << strfmt("%.6g", (double)logits_scaling)
       << "\n  mamba-2     : n_heads=" << mamba_n_heads << " d_head=" << mamba_d_head
       << " d_state=" << mamba_d_state << " groups=" << mamba_n_groups
       << " conv_k=" << mamba_d_conv << " expand=" << mamba_expand
       << " chunk=" << mamba_chunk_size
       << " | d_inner=" << (mamba_expand * hidden_size)
       << " conv_dim=" << (mamba_expand * hidden_size + 2 * mamba_n_groups * mamba_d_state)
       << "\n  layer types : ";
    int n_mamba = 0, n_attn = 0;
    for (int k : layer_types) (k == 0 ? n_mamba : n_attn)++;
    ss << n_mamba << " mamba + " << n_attn << " attention";
    if (n_attn > 0) {
        ss << " (attention at layers";
        bool first = true;
        for (size_t i = 0; i < layer_types.size(); ++i) {
            if (layer_types[i] != 1) continue;
            if (!first) ss << ",";
            first = false;
            ss << " " << i;
        }
        ss << ")";
    }
    ss << "\n  shared expert: " << (moe_shared ? strfmt("width %d", shared_width()) : "none")
       << "\n  routed expert: width " << moe_expert_dim << " | " << num_experts
       << " experts | top-" << moe_top_k;
    return ss.str();
}

float rope_theta_eff(const ModelConfig& m) { return rope_theta_eff_local(m); }
float rope_mscale(const ModelConfig& m) { return rope_mscale_local(m); }

void Model::rebuild_rope_cache() {
    const int half = cfg_.head_dim() / 2;
    std::vector<float> host(static_cast<size_t>(half));
    const float theta = rope_theta_eff_local(cfg_);
    for (int i = 0; i < half; ++i) {
        float base = 1.0f / std::pow(theta,
            (2.0f * static_cast<float>(i)) / static_cast<float>(cfg_.head_dim()));
        if (cfg_.rope_scale > 1.0f) {
            const float wavelength = 2.0f * 3.14159265358979f / base;
            if (wavelength > cfg_.rope_yarn_high) {
                base /= cfg_.rope_scale;
            } else if (wavelength > cfg_.rope_yarn_low) {
                const float span = cfg_.rope_yarn_high - cfg_.rope_yarn_low;
                const float t = span > 0.0f ? (wavelength - cfg_.rope_yarn_low) / span : 1.0f;
                base *= 1.0f - t + t / cfg_.rope_scale;
            }
        }
        host[static_cast<size_t>(i)] = base;
    }
    rope_inv_freq_ = Tensor({half}, DType::F32, device_);
    device_copy(rope_inv_freq_.data_ptr(), device_, host.data(), Device::CPU,
                host.size() * sizeof(float));
}

const float* Model::rope_inv_freq_ptr() const {
    return rope_inv_freq_.defined() ? rope_inv_freq_.f32() : nullptr;
}

static void lora_forward(LoraAdapter& ad, Device dev, const float* x, float* y, int M) {
    if (!ad.enabled || ad.rank <= 0) return;
    const int r = ad.rank;
    if (ad.r_buf.numel() < static_cast<i64>(M) * r) {
        ad.r_buf = Tensor({M, r}, DType::F32, dev);
    }
    const float s = ad.scale();
    ops::linear_forward(dev, x, ad.A.w.f32(), ad.r_buf.f32(), M, ad.in_dim, r);
    ops::gemm(dev, false, true, M, ad.out_dim, r, s, ad.r_buf.f32(), r, ad.B.w.f32(), r, 1.0f, y, ad.out_dim);
}

static void lora_backward(LoraAdapter& ad, Device dev, const float* x, const float* dy,
                          float* dx, int M) {
    if (!ad.enabled || ad.rank <= 0) return;
    const int r = ad.rank;
    const float s = ad.scale();
    ops::gemm(dev, true, false, ad.out_dim, r, M, s, dy, ad.out_dim,
              ad.r_buf.f32(), r, 1.0f, gptr(ad.B.g), r);
    if (ad.dR_buf.numel() < static_cast<i64>(M) * r)
        ad.dR_buf = Tensor({M, r}, DType::F32, dev);
    float* dR = ad.dR_buf.f32();
    ops::gemm(dev, false, false, M, r, ad.out_dim, s, dy, ad.out_dim,
              ad.B.w.f32(), r, 0.0f, dR, r);
    ops::gemm(dev, true, false, r, ad.in_dim, M, 1.0f, dR, r, x, ad.in_dim,
              1.0f, gptr(ad.A.g), ad.in_dim);
    if (dx) {
        ops::gemm(dev, false, false, M, ad.in_dim, r, 1.0f, dR, r, ad.A.w.f32(), ad.in_dim,
                  1.0f, dx, ad.in_dim);
    }
}

void Model::init_lora(LoraAdapter& ad, const std::string& name,
                      int in_d, int out_d, int r, float alpha,
                      Device dev, std::mt19937_64& rng) {
    ad.rank    = r;
    ad.in_dim  = in_d;
    ad.out_dim = out_d;
    ad.alpha   = alpha;
    ad.enabled = true;

    alloc_param(ad.A, name + ".A", {r, in_d}, true);
    alloc_param(ad.B, name + ".B", {out_d, r}, false);

    const float stddev = 1.0f / std::sqrt(static_cast<float>(in_d));
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> cpu_A(static_cast<size_t>(r) * in_d);
    for (float& v : cpu_A) v = dist(rng);
    Tensor tmp_A({r, in_d}, DType::F32, Device::CPU);
    std::memcpy(tmp_A.f32(), cpu_A.data(), cpu_A.size() * sizeof(float));
    ad.A.w.copy_from(tmp_A);

    ad.B.w.zero_();
}

void Model::forward_mamba_layer(Device dev, int l, Activations& act, i64 N, const int32_t* segment_ids) {
    LayerParams& L = layers_[static_cast<size_t>(l)];
    const int d        = cfg_.hidden_size;
    const int d_inner  = cfg_.mamba_expand * d;
    const int n_heads  = cfg_.mamba_n_heads;
    const int d_head   = cfg_.mamba_d_head;
    const int d_state  = cfg_.mamba_d_state;
    const int n_groups = cfg_.mamba_n_groups;
    const int T        = act.T;
    const int B        = act.B;
    const int d_state_tot = n_groups * d_state;
    const int conv_dim    = d_inner + 2 * d_state_tot;

    GAI_CHECK(L.mamba_in_proj.w.defined() && L.mamba_out_proj.w.defined() &&
                  L.mamba_conv_w.w.defined() && L.mamba_norm.w.defined() &&
                  L.mamba_A_log.w.defined() && L.mamba_D.w.defined() &&
                  L.mamba_dt_bias.w.defined(),
              "granite mamba: the layer weights are missing (load the checkpoint first)");
    const int state_rows = B * n_heads * d_head;
    size_t si = 0;
    for (int k = 0; k < l && k < static_cast<int>(cfg_.layer_types.size()); ++k)
        if (cfg_.layer_types[static_cast<size_t>(k)] == 0) ++si;
    GAI_CHECK(si < act.mamba_states.size(),
              "granite mamba: the recurrent state is missing for this layer");
    Tensor& st = act.mamba_states[si];
    if (!st.defined() || st.numel() < static_cast<i64>(state_rows) * d_state) {
        st = Tensor::zeros({static_cast<i64>(state_rows) * d_state}, DType::F32, dev);
    }
    float* conv_state = nullptr;
    if (si < act.mamba_conv_states.size()) {
        Tensor& cs = act.mamba_conv_states[si];
        const i64 win = cfg_.mamba_d_conv > 1 ? cfg_.mamba_d_conv - 1 : 0;
        const i64 need = static_cast<i64>(B) * conv_dim * win;
        if (!cs.defined() || cs.numel() < need) {
            cs = Tensor::zeros({need}, DType::F32, dev);
        }
        conv_state = cs.numel() > 0 ? cs.f32() : nullptr;
    }

    if (is_half_or_quant(L.mamba_in_proj.w)) {
        const WeightRef wi = weight_ref(dev, L.mamba_in_proj.w, 0);
        const WeightRef wo = weight_ref(dev, L.mamba_out_proj.w, 1);
        ops::mamba2_forward_half(dev,
                                 act.xb.f32(),
                                 wi.ptr,
                                 L.mamba_conv_w.w.f32(),
                                 L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                                 L.mamba_dt_bias.w.f32(),
                                 L.mamba_A_log.w.f32(),
                                 L.mamba_D.w.f32(),
                                 L.mamba_norm.w.f32(),
                                 wo.ptr,
                                 st.f32(),
                                 act.proj.f32(),
                                 B, T, d, d_inner, conv_dim, n_heads, d_head, d_state,
                                 cfg_.mamba_d_conv, cfg_.rms_eps, wi.qkind, segment_ids, conv_state);
    } else {
    ops::mamba2_forward(dev,
                        act.xb.f32(),
                        L.mamba_in_proj.w.f32(),
                        L.mamba_conv_w.w.f32(),
                        L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                        L.mamba_dt_bias.w.f32(),
                        L.mamba_A_log.w.f32(),
                        L.mamba_D.w.f32(),
                        L.mamba_norm.w.f32(),
                        L.mamba_out_proj.w.f32(),
                        st.f32(),
                        act.proj.f32(),
                        B, T, d, d_inner, conv_dim, n_heads, d_head, d_state,
                        cfg_.mamba_d_conv, cfg_.rms_eps, segment_ids, conv_state);
    }
}

DType Model::storage_dtype_for(const std::string& name) const {
    if (weight_q4_) {
        if (name.find("lora_") != std::string::npos) return DType::F32;
        if (name == "tok_embeddings" || name == "lm_head") return DType::F16;
        const bool fragile =
            name.find("moe_router") != std::string::npos ||
            name.find("mamba.conv1d") != std::string::npos ||
            name.find("mamba.dt_bias") != std::string::npos ||
            name.find("mamba.A_log") != std::string::npos ||
            name.find("mamba.D") != std::string::npos ||
            name.find("norm") != std::string::npos ||
            name.find("qk_qnorm") != std::string::npos ||
            name.find("qk_knorm") != std::string::npos ||
            name.find("bias") != std::string::npos ||
            name.find(".bq") != std::string::npos ||
            name.find(".bk") != std::string::npos ||
            name.find(".bv") != std::string::npos;
        if (fragile) return DType::F32;
        return q4_kind_ == 0 ? DType::Q4_0 : q4_kind_ == 1 ? DType::Q4_1 : DType::Q8_0;
    }

    if (!weight_f16_) return DType::F32;

    if (name.find("lora_") != std::string::npos) return DType::F32;

    const bool fragile =
        name.find("moe_router") != std::string::npos ||
        name.find("mamba.conv1d") != std::string::npos ||
        name.find("mamba.dt_bias") != std::string::npos ||
        name.find("mamba.A_log") != std::string::npos ||
        name.find("mamba.D") != std::string::npos ||
        name.find("norm") != std::string::npos ||
        name.find("qk_qnorm") != std::string::npos ||
        name.find("qk_knorm") != std::string::npos ||
        name.find("bias") != std::string::npos ||
        name.find(".bq") != std::string::npos ||
        name.find(".bk") != std::string::npos ||
        name.find(".bv") != std::string::npos;
    return fragile ? DType::F32 : DType::F16;
}

void Model::alloc_param(Parameter& p, const std::string& name, std::vector<i64> shape, bool decay) {
    p.name  = name;
    p.shape = shape;
    p.decay = decay;
    p.w = Tensor::zeros(shape, storage_dtype_for(name), device_);
    p.w.set_name(name);
    params_.push_back(&p);
}

Model::~Model() {
    enable_fp16_weight_cache(false);
}

Model::Model(ModelConfig cfg, Device dev, bool fp16_weights, int quant_kind)
    : cfg_(std::move(cfg)), device_(dev) {
    cfg_.validate();
    GAI_CHECK(!(fp16_weights && quant_kind >= 0),
              "fp16 weight storage and native quantized storage are mutually exclusive");
    weight_f16_ = fp16_weights;
    weight_q4_ = quant_kind >= 0;
    q4_kind_ = weight_q4_ ? quant_kind : 0;
    if (weight_q4_) {
        log_info(strfmt("[prec] native quantized weight storage: %s packed on device, "
                        "dequantized to FP16 scratch per op (~%.1f GiB weights instead of "
                        "12.9 GiB of FP16)",
                        q4_kind_ == 0 ? "Q4_0" : q4_kind_ == 1 ? "Q4_1" : "Q8_0",
                        (q4_kind_ == 2 ? 6.5 : 4.7)));
    }
    if (weight_f16_ && cfg_.lora_rank > 0) {
        log_info("[prec] the half weight storage is on: the base weights are "
                 "frozen (12.9 GB instead of 27.7 GB for the Granite) and the "
                 "LoRA adapters stay fp32");
    }

    if (cfg_.rope_scale > 1.0f) {
        log_warn(strfmt("[rope] rope_scale=%.1f is approximate NTK/YaRN extrapolation "
                        "(theta->%.0f, mscale=%.2f): validate ppl at %d ctx before long runs",
                        (double)cfg_.rope_scale, (double)rope_theta_eff_local(cfg_),
                        (double)rope_mscale_local(cfg_), cfg_.max_seq_len));
    }
    if (cfg_.use_moe && cfg_.moe_allow_dense) {
        log_warn("[moe ] moe_allow_dense=true: dense routing enabled for research "
                 "(expect T4 OOM at 1B; keep runs short)");
    }
    ops::set_moe_jitter(cfg_.moe_jitter);

    const i64 d   = cfg_.hidden_size;
    const i64 V   = cfg_.vocab_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.intermediate_size;
    const i64 E   = cfg_.moe_expert_dim;
    const i64 ne  = cfg_.num_experts;

    alloc_param(tok_emb_, "tok_embeddings", {V, d}, false);

    layers_.resize(static_cast<size_t>(cfg_.num_layers));
    for (int i = 0; i < cfg_.num_layers; ++i) {
        LayerParams& L = layers_[static_cast<size_t>(i)];
        std::string pre = "layers." + std::to_string(i) + ".";
        alloc_param(L.attn_norm, pre + "attn_norm", {d},      false);
        alloc_param(L.ffn_norm,  pre + "ffn_norm",  {d},      false);
        const bool is_granite = (cfg_.arch == ModelConfig::ArchGranite);
        const int  gk = (is_granite && i < static_cast<int>(cfg_.layer_types.size()))
                            ? cfg_.layer_types[static_cast<size_t>(i)]
                            : 1;
        if (is_granite && gk == 0) {
            const i64 d_inner = static_cast<i64>(cfg_.mamba_expand) * d;
            const i64 conv_dim = d_inner + 2LL * cfg_.mamba_n_groups * cfg_.mamba_d_state;
            const i64 proj = d_inner + conv_dim + cfg_.mamba_n_heads;
            std::string mp = pre + "mamba.";
            alloc_param(L.mamba_in_proj, mp + "in_proj",  {proj, d},   true);
            alloc_param(L.mamba_conv_w,  mp + "conv1d.weight",
                        {conv_dim, 1, cfg_.mamba_d_conv}, true);
            alloc_param(L.mamba_conv_b,  mp + "conv1d.bias", {conv_dim}, false);
            alloc_param(L.mamba_dt_bias, mp + "dt_bias",   {cfg_.mamba_n_heads}, true);
            alloc_param(L.mamba_A_log,   mp + "A_log",     {cfg_.mamba_n_heads}, true);
            alloc_param(L.mamba_D,       mp + "D",         {cfg_.mamba_n_heads}, true);
            alloc_param(L.mamba_norm,    mp + "norm.weight", {d_inner}, false);
            alloc_param(L.mamba_out_proj, mp + "out_proj", {d, d_inner}, true);
        } else {
            alloc_param(L.wq,        pre + "wq",        {qd, d},  true);
            alloc_param(L.wk,        pre + "wk",        {kvd, d}, true);
            alloc_param(L.wv,        pre + "wv",        {kvd, d}, true);
            alloc_param(L.wo,        pre + "wo",        {d, qd},  true);
            if (cfg_.use_qkv_bias) {
                alloc_param(L.bq, pre + "bq", {qd},  false);
                alloc_param(L.bk, pre + "bk", {kvd}, false);
                alloc_param(L.bv, pre + "bv", {kvd}, false);
            }
        }
        if (cfg_.use_qk_norm && !(is_granite && gk == 0)) {
            const i64 hd = cfg_.head_dim();
            alloc_param(L.qk_qnorm, pre + "qk_qnorm", {hd}, false);
            alloc_param(L.qk_knorm, pre + "qk_knorm", {hd}, false);
        }
        if (cfg_.use_moe) {
            alloc_param(L.router,   pre + "moe_router", {ne, d},    true);
            alloc_param(L.moe_gate, pre + "moe_gate",   {ne * E, d}, true);
            alloc_param(L.moe_up,   pre + "moe_up",     {ne * E, d}, true);
            alloc_param(L.moe_down, pre + "moe_down",   {ne * d, E}, true);
            if (cfg_.moe_shared) {
                const i64 SH = cfg_.shared_width();
                alloc_param(L.sh_gate, pre + "moe_sh_gate", {SH, d}, true);
                alloc_param(L.sh_up,   pre + "moe_sh_up",   {SH, d}, true);
                alloc_param(L.sh_down, pre + "moe_sh_down", {d, SH}, true);
            }
        } else {
            alloc_param(L.w_gate,    pre + "w_gate",    {F, d},   true);
            alloc_param(L.w_up,      pre + "w_up",      {F, d},   true);
            alloc_param(L.w_down,    pre + "w_down",    {d, F},   true);
        }
    }

    alloc_param(final_norm_, "final_norm", {d}, false);
    if (!cfg_.tie_embeddings) alloc_param(lm_head_, "lm_head", {V, d}, true);
    if (cfg_.use_moe && cfg_.moe_aux_free) {
        ensure_moe_bias();
        log_info("[moe ] aux-loss-free bias enabled (DeepSeek-V3 sec. 3.2): EMA steering, no aux grad");
    }
    if (cfg_.lora_rank > 0) {
        for (Parameter* p : params_) p->frozen = true;

        std::mt19937_64 rng(42);
        const int r = cfg_.lora_rank;
        for (int i = 0; i < cfg_.num_layers; ++i) {
            LayerParams& L = layers_[static_cast<size_t>(i)];
            if (L.has_mamba()) continue;
            const std::string pre = "layers." + std::to_string(i) + ".lora_";
            if (cfg_.lora_target_q)
                init_lora(L.lora_q, pre + "q", d, qd, r, cfg_.lora_alpha, device_, rng);
            if (cfg_.lora_target_k)
                init_lora(L.lora_k, pre + "k", d, kvd, r, cfg_.lora_alpha, device_, rng);
            if (cfg_.lora_target_v)
                init_lora(L.lora_v, pre + "v", d, kvd, r, cfg_.lora_alpha, device_, rng);
            if (cfg_.lora_target_o)
                init_lora(L.lora_o, pre + "o", qd, d, r, cfg_.lora_alpha, device_, rng);
            if (!cfg_.use_moe) {
                if (cfg_.lora_target_gate)
                    init_lora(L.lora_gate, pre + "gate", d, F, r, cfg_.lora_alpha, device_, rng);
                if (cfg_.lora_target_up)
                    init_lora(L.lora_up, pre + "up", d, F, r, cfg_.lora_alpha, device_, rng);
                if (cfg_.lora_target_down)
                    init_lora(L.lora_down, pre + "down", F, d, r, cfg_.lora_alpha, device_, rng);
            }
        }
        log_info(strfmt("[lora] initialized LoRA rank=%d alpha=%.1f on targets "
                        "(base weights frozen, %lld trainable adapter params)",
                        cfg_.lora_rank, (double)cfg_.lora_alpha,
                        (long long)trainable_parameters()));
    }
    rebuild_rope_cache();
}

const u16* Model::fused_qkv_ptr(int i) const {
    if (!fp16_weight_cache_ || i < 0 || static_cast<size_t>(i) >= layers_.size()) return nullptr;
    const Tensor& fused = layers_[static_cast<size_t>(i)].wqkv_fp16;
    return fused.defined() ? fused.ptr<u16>() : nullptr;
}

Parameter* Model::find_parameter(const std::string& name) {
    for (Parameter* p : params_) if (p->name == name) return p;
    return nullptr;
}

const Parameter* Model::find_parameter(const std::string& name) const {
    for (const Parameter* p : params_) if (p->name == name) return p;
    return nullptr;
}

i64 Model::num_parameters() const {
    i64 n = 0;
    for (const Parameter* p : params_) n += p->numel();
    return n;
}

i64 Model::num_parameters_non_embedding() const {
    i64 n = 0;
    for (const Parameter* p : params_) {
        if (p->name == "tok_embeddings" || p->name == "lm_head") continue;
        n += p->numel();
    }
    return n;
}

ParamReport Model::parameter_report() const {
    ParamReport r;
    const i64 d   = cfg_.hidden_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.intermediate_size;
    const i64 E   = cfg_.moe_expert_dim;
    const i64 ne  = cfg_.num_experts;
    const int L   = cfg_.num_layers;

    auto add = [&](const std::string& n, i64 per, int units) {
        r.groups.push_back(ParamGroupCount{n, per * units, per, units});
    };

    add(cfg_.tie_embeddings ? "tok_embeddings (tied with lm_head)" : "tok_embeddings",
        cfg_.vocab_size * d, 1);
    int n_attn = L, n_mamba = 0;
    bool hybrid = (cfg_.arch == ModelConfig::ArchGranite) &&
                  static_cast<int>(cfg_.layer_types.size()) == L;
    if (hybrid) {
        n_attn = 0;
        for (int lt : cfg_.layer_types) if (lt == 1) ++n_attn;
        n_mamba = L - n_attn;
    }
    add("attn.wq   [q_dim, d]",  qd * d,  n_attn);
    add("attn.wk   [kv_dim, d]", kvd * d, n_attn);
    add("attn.wv   [kv_dim, d]", kvd * d, n_attn);
    add("attn.wo   [d, q_dim]",  d * qd,  n_attn);
    if (cfg_.use_qkv_bias) {
        add("attn.bq   [q_dim]",  qd,  n_attn);
        add("attn.bk   [kv_dim]", kvd, n_attn);
        add("attn.bv   [kv_dim]", kvd, n_attn);
    }
    if (n_mamba > 0) {
        const i64 proj = static_cast<i64>(cfg_.mamba_expand) * d + 2 * cfg_.mamba_n_groups * cfg_.mamba_d_state + cfg_.mamba_n_heads;
        add("mamba.in_proj [proj, d]", proj * d, n_mamba);
        add("mamba.out_proj [d, inner]", d * (static_cast<i64>(cfg_.mamba_expand) * d), n_mamba);
        const i64 conv_dim = static_cast<i64>(cfg_.mamba_expand) * d +
                             2 * static_cast<i64>(cfg_.mamba_n_groups) * cfg_.mamba_d_state;
        add("mamba.conv1d", conv_dim * cfg_.mamba_d_conv, n_mamba);
        add("mamba.conv1d.bias", conv_dim, n_mamba);
        add("mamba.A_log/dt_bias/D", 3 * cfg_.mamba_n_heads, n_mamba);
        add("mamba.norm", static_cast<i64>(cfg_.mamba_expand) * d, n_mamba);
    }
    if (cfg_.use_moe) {
        const i64 SH = cfg_.shared_width();
        add("moe.router [ne, d]",      ne * d,     L);
        add("moe.gate   [ne*E, d]",    ne * E * d, L);
        add("moe.up     [ne*E, d]",    ne * E * d, L);
        add("moe.down   [ne*d, E]",    ne * d * E, L);
        if (cfg_.moe_shared) {
            add("moe.shared.gate [SH, d]", SH * d, L);
            add("moe.shared.up   [SH, d]", SH * d, L);
            add("moe.shared.down [d, SH]", d * SH, L);
        }
    } else {
        add("ffn.gate  [ffn, d]",    F * d,   L);
        add("ffn.up    [ffn, d]",    F * d,   L);
        add("ffn.down  [d, ffn]",    d * F,   L);
    }
    add("attn_norm [d]",         d,       L);
    add("ffn_norm  [d]",         d,       L);
    if (cfg_.use_qk_norm) {
        const i64 hd = cfg_.head_dim();
        add("qk_qnorm  [hd]", hd, n_attn);
        add("qk_knorm  [hd]", hd, n_attn);
    }
    add("final_norm [d]",        d,       1);
    if (!cfg_.tie_embeddings) add("lm_head [V, d]", cfg_.vocab_size * d, 1);

    for (const auto& g : r.groups) r.total += g.count;
    r.embedding = cfg_.vocab_size * d * (cfg_.tie_embeddings ? 1 : 2);
    r.non_embedding = r.total - r.embedding;
    return r;
}

void Model::to(Device dev) { move_to_device(dev, true); }

void Model::move_to_device(Device dev, bool announce) {
    if (device_ == dev) return;
    const bool restore_fp16_cache = fp16_weight_cache_;
    invalidate_fp16_cache();
    if (announce)
        log_warn("[model] Model::to() moved weights; REBUILD Trainer activations and "
                 "Generator cache/scratch on the new device before next step "
                 "(cross-device use-after-move crashes T4).");
    for (Parameter* p : params_) {
        if (p->w.defined()) p->w = p->w.to(dev);
        if (p->g.defined()) p->g = p->g.to(dev);
    }
    device_ = dev;
    if (moe_bias_acc_dev_.defined()) moe_bias_acc_dev_ = Tensor();
    if (rope_inv_freq_.defined()) rope_inv_freq_ = rope_inv_freq_.to(dev);
    if (cfg_.use_moe && cfg_.moe_aux_free) {
        const size_t L = static_cast<size_t>(cfg_.num_layers);
        if (moe_bias_.size() != L)
            moe_bias_.assign(L, std::vector<float>(static_cast<size_t>(cfg_.num_experts), 0.0f));
        moe_bias_dev_.clear();
        if (dev == Device::CUDA) {
            moe_bias_dev_.reserve(L);
            for (size_t l = 0; l < L; ++l) {
                Tensor t({cfg_.num_experts}, DType::F32, dev);
                device_copy(t.data_ptr(), dev, moe_bias_[l].data(), Device::CPU,
                            sizeof(float) * moe_bias_[l].size());
                moe_bias_dev_.push_back(std::move(t));
            }
        }
    }
    if (restore_fp16_cache) enable_fp16_weight_cache(true);
}

void Model::set_rope_runtime(float scale, float yarn_mscale) {
    cfg_.rope_scale = scale;
    cfg_.rope_yarn_mscale = yarn_mscale;
    cfg_.validate();
    rebuild_rope_cache();
    log_info(strfmt("[rope] runtime scale=%.3f mscale=%.3f theta=%.1f",
                    (double)scale, (double)rope_mscale_local(cfg_),
                    (double)rope_theta_eff_local(cfg_)));
}

std::string ParamReport::to_string(const ModelConfig& cfg) const {
    std::ostringstream ss;
    ss << "\n";
    ss << "+--------------------------------------+-------------+-------+---------------+\n";
    ss << "| group                                |    per unit | units |         total |\n";
    ss << "+--------------------------------------+-------------+-------+---------------+\n";
    for (const auto& g : groups) {
        ss << "| " << g.name;
        for (size_t i = g.name.size(); i < 36; ++i) ss << ' ';
        ss << " | " << std::string(11 - std::min<size_t>(11, std::to_string(g.per_unit).size()), ' ')
           << g.per_unit
           << " | " << std::string(5 - std::min<size_t>(5, std::to_string(g.units).size()), ' ')
           << g.units
           << " | " << std::string(13 - std::min<size_t>(13, std::to_string(g.count).size()), ' ')
           << g.count << " |\n";
    }
    ss << "+--------------------------------------+-------------+-------+---------------+\n";
    ss << strfmt("  total parameters      : %lld  (%.2f M)\n",
                 static_cast<long long>(total), double(total) / 1e6);
    ss << strfmt("  embedding parameters  : %lld  (%.1f%%)\n",
                 static_cast<long long>(embedding),
                 total ? 100.0 * double(embedding) / double(total) : 0.0);
    ss << strfmt("  non-embedding params  : %lld  (%.2f M)\n",
                 static_cast<long long>(non_embedding), double(non_embedding) / 1e6);
    ss << strfmt("  weight memory  fp32   : %s\n", human_bytes(static_cast<u64>(total) * 4).c_str());
    ss << strfmt("  weight memory  fp16   : %s\n", human_bytes(static_cast<u64>(total) * 2).c_str());
    ss << strfmt("  weight memory  int8   : %s\n", human_bytes(static_cast<u64>(total)).c_str());
    ss << strfmt("  weight memory  int4   : %s\n", human_bytes(static_cast<u64>(total) / 2).c_str());
    {
        u64 kv = static_cast<u64>(cfg.num_layers) * 2ull *
                 static_cast<u64>(cfg.kv_dim()) * static_cast<u64>(cfg.max_seq_len) * 2ull;
        ss << strfmt("  kv cache @%d ctx fp16 : %s\n", cfg.max_seq_len, human_bytes(kv).c_str());
    }
    return ss.str();
}

void Model::print_parameter_report() const {
    log_info("---------------- Ghassan v1 Flash ----------------");
    log_info("  " + cfg_.summary());
    log_raw(LogLevel::Info, parameter_report().to_string(cfg_));
}

i64 Model::trainable_parameters() const {
    i64 n = 0;
    for (const Parameter* p : params_) {
        if (p->frozen) continue;
        n += p->numel();
    }
    return n;
}

void Model::merge_lora() {
    if (cfg_.lora_rank <= 0) return;

    struct Job { LoraAdapter* ad; Parameter* base; };
    std::vector<Job> jobs;

    for (int i = 0; i < cfg_.num_layers; ++i) {
        LayerParams& L = layers_[static_cast<size_t>(i)];
        if (L.lora_q.enabled && L.wq.w.defined()) jobs.push_back({&L.lora_q, &L.wq});
        if (L.lora_k.enabled && L.wk.w.defined()) jobs.push_back({&L.lora_k, &L.wk});
        if (L.lora_v.enabled && L.wv.w.defined()) jobs.push_back({&L.lora_v, &L.wv});
        if (L.lora_o.enabled && L.wo.w.defined()) jobs.push_back({&L.lora_o, &L.wo});
        if (!cfg_.use_moe) {
            if (L.lora_gate.enabled && L.w_gate.w.defined()) jobs.push_back({&L.lora_gate, &L.w_gate});
            if (L.lora_up.enabled   && L.w_up.w.defined())   jobs.push_back({&L.lora_up, &L.w_up});
            if (L.lora_down.enabled && L.w_down.w.defined()) jobs.push_back({&L.lora_down, &L.w_down});
        }
    }
    if (jobs.empty()) return;

    for (const Job& j : jobs) {
        const int r = j.ad->rank;
        const int in_d = j.ad->in_dim;
        const int out_d = j.ad->out_dim;
        const float sc = j.ad->scale();

        Parameter& base = *j.base;
        if (base.w.dtype() == DType::F16) {
            Tensor tmp(base.w.shape(), DType::F32, device_);
            ops::convert_f16_to_f32(device_, base.w.ptr<u16>(), tmp.f32(), base.numel());
            ops::gemm(device_, false, false, out_d, in_d, r, sc,
                      j.ad->B.w.f32(), r,
                      j.ad->A.w.f32(), in_d,
                      1.0f, tmp.f32(), in_d);
            ops::convert_f32_to_f16(device_, tmp.f32(), base.w.ptr<u16>(), base.numel());
        } else if (is_quant(base.w)) {
            Tensor tmp(base.w.shape(), DType::F32, device_);
            ops::convert_f16_to_f32(device_,
                                    weight16(device_, base.w, 0),
                                    tmp.f32(), base.numel());
            ops::gemm(device_, false, false, out_d, in_d, r, sc,
                      j.ad->B.w.f32(), r,
                      j.ad->A.w.f32(), in_d,
                      1.0f, tmp.f32(), in_d);
            Tensor host(base.w.shape(), DType::F32, Device::CPU);
            host.copy_from(tmp);
            Tensor packed = quant::quantize(host, base.w.dtype());
            base.w.copy_from(packed);
        } else {
            GAI_CHECK(base.w.dtype() == DType::F32,
                      "merge_lora: unsupported base weight dtype for " + base.name);
            ops::gemm(device_, false, false, out_d, in_d, r, sc,
                      j.ad->B.w.f32(), r,
                      j.ad->A.w.f32(), in_d,
                      1.0f, base.w.f32(), in_d);
        }
    }

    for (const Job& j : jobs) {
        j.ad->B.w.zero_();
        j.ad->A.w.zero_();
    }

    mark_weights_dirty();
    log_info(strfmt("[lora] merged %d adapters into the base weights", (int)jobs.size()));
}

bool Model::lora_delta_for(const std::string& name, std::vector<float>& out) const {
    if (cfg_.lora_rank <= 0) return false;

    struct Pair { const Parameter* base; const LoraAdapter* adapter; };
    for (const LayerParams& L : layers_) {
        const Pair pairs[] = {
            {&L.wq, &L.lora_q},
            {&L.wk, &L.lora_k},
            {&L.wv, &L.lora_v},
            {&L.wo, &L.lora_o},
            {&L.w_gate, &L.lora_gate},
            {&L.w_up, &L.lora_up},
            {&L.w_down, &L.lora_down},
        };
        for (const Pair& pr : pairs) {
            const LoraAdapter& ad = *pr.adapter;
            if (!pr.base->name.empty() && pr.base->name == name && ad.enabled) {
                const int r = ad.rank;
                const int in_d = ad.in_dim;
                const int out_d = ad.out_dim;
                GAI_CHECK(r > 0 && in_d > 0 && out_d > 0,
                          "lora_delta_for: adapter with non-positive geometry");
                GAI_CHECK(ad.A.w.numel() == static_cast<i64>(r) * in_d &&
                              ad.B.w.numel() == static_cast<i64>(out_d) * r,
                          "lora_delta_for: adapter shape does not match the base weight");
                const float sc = ad.scale();
                const float* a = ad.A.w.f32();
                const float* b = ad.B.w.f32();
                out.assign(static_cast<size_t>(out_d) * static_cast<size_t>(in_d), 0.0f);
                for (int o = 0; o < out_d; ++o) {
                    float* drow = out.data() + static_cast<size_t>(o) * static_cast<size_t>(in_d);
                    for (int k = 0; k < r; ++k) {
                        const float bk = sc * b[static_cast<size_t>(o) * static_cast<size_t>(r) + k];
                        const float* arow = a + static_cast<size_t>(k) * static_cast<size_t>(in_d);
                        for (int i = 0; i < in_d; ++i) drow[i] += bk * arow[i];
                    }
                }
                return true;
            }
        }
    }
    return false;
}

void Model::init_weights(u64 seed) {
    GAI_CHECK(!weight_f16_ && !weight_q4_,
              "init_weights fills fp32 masters: turn the fp16/quantized weight "
              "storage off (--weight-f16 0 / omit --weight-q4)");
    const bool restore_fp16_cache = fp16_weight_cache_;
    invalidate_fp16_cache();
    Rng rng(seed);
    if (cfg_.use_moe && cfg_.moe_aux_free) {
        for (auto& bias : moe_bias_) std::fill(bias.begin(), bias.end(), 0.0f);
    }
    const float std_base = cfg_.init_std;
    const float std_res = std_base / std::sqrt(2.0f * static_cast<float>(cfg_.num_layers));

    Device orig_dev = device_;
    if (device_ != Device::CPU) {
        move_to_device(Device::CPU, false);
    }

    auto fill_normal = [&](Parameter& p, float sd) {
        float* w = p.w.f32();
        for (i64 i = 0; i < p.w.numel(); ++i) w[i] = rng.truncated_normal(sd);
    };
    auto fill_ones = [&](Parameter& p) {
        float* w = p.w.f32();
        for (i64 i = 0; i < p.w.numel(); ++i) w[i] = 1.0f;
    };

    fill_normal(tok_emb_, std_base);
    if (!cfg_.tie_embeddings) fill_normal(lm_head_, std_base);
    fill_ones(final_norm_);

    for (auto& L : layers_) {
        fill_ones(L.attn_norm);
        fill_ones(L.ffn_norm);
        if (L.qk_qnorm.w.defined()) fill_ones(L.qk_qnorm);
        if (L.qk_knorm.w.defined()) fill_ones(L.qk_knorm);
        if (L.has_mamba()) {
            fill_normal(L.mamba_in_proj, std_base);
            fill_normal(L.mamba_out_proj, std_res);
            fill_normal(L.mamba_conv_w, std_base);
            if (L.mamba_conv_b.w.defined()) fill_ones(L.mamba_conv_b);
            if (L.mamba_norm.w.defined()) fill_ones(L.mamba_norm);
            if (L.mamba_A_log.w.defined()) {
                for (i64 i = 0; i < L.mamba_A_log.w.numel(); ++i)
                    L.mamba_A_log.w.f32()[i] = std::log(static_cast<float>(i + 1));
            }
            if (L.mamba_D.w.defined())
                for (i64 i = 0; i < L.mamba_D.w.numel(); ++i) L.mamba_D.w.f32()[i] = 1.0f;
            if (L.mamba_dt_bias.w.defined())
                for (i64 i = 0; i < L.mamba_dt_bias.w.numel(); ++i) L.mamba_dt_bias.w.f32()[i] = 1.0f;
        } else {
        fill_normal(L.wq, std_base);
        fill_normal(L.wk, std_base);
        fill_normal(L.wv, std_base);
        fill_normal(L.wo, std_res);
        }
        if (L.has_moe()) {
            fill_normal(L.router, std::min(std_base, 0.01f));
            fill_normal(L.moe_gate, std_base);
            fill_normal(L.moe_up,   std_base);
            fill_normal(L.moe_down, std_res);
            if (L.sh_gate.w.defined()) {
                fill_normal(L.sh_gate, std_base);
                fill_normal(L.sh_up,   std_base);
                fill_normal(L.sh_down, std_res);
            }
        } else {
            fill_normal(L.w_gate, std_base);
            fill_normal(L.w_up,   std_base);
            fill_normal(L.w_down, std_res);
        }
    }

    if (orig_dev != Device::CPU) {
        move_to_device(orig_dev, false);
    }
    if (restore_fp16_cache) enable_fp16_weight_cache(true);
}

void Model::enable_fp16_weight_cache(bool on) {
    if (!on) {
        for (Parameter* p : params_) {
            if (p->fp16_cache.defined()) {
                ops::unregister_fp16_weight(p->w.f32());
                p->fp16_cache = Tensor();
            }
        }
        for (LayerParams& layer : layers_) layer.wqkv_fp16 = Tensor();
        fp16_weight_cache_ = false;
        fp16_weights_dirty_ = false;
        return;
    }
    if (weight_q4_) {
        log_info("[prec] the fp16 weight cache is not used in native quantized mode "
                 "(weights are dequantized to scratch per op)");
        return;
    }
    if (weight_f16_) {
        log_info("[prec] the fp16 weight cache is not needed: the weights are "
                 "already stored in half precision");
        return;
    }
    if (!fp16_weight_cache_) {
        for (Parameter* p : params_) {
            if (p->shape.size() != 2) continue;
            p->fp16_cache = Tensor::empty(p->shape, DType::F16, device_);
        }
        fp16_weight_cache_ = true;
    }
    for (Parameter* p : params_) {
        if (!p->fp16_cache.defined()) continue;
        ops::convert_f32_to_f16(device_, p->w.f32(), p->fp16_cache.ptr<u16>(), p->numel());
        ops::register_fp16_weight(p->w.f32(), p->fp16_cache.ptr<u16>(), p->numel());
    }
    const i64 d = cfg_.hidden_size;
    const i64 qd = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    for (LayerParams& layer : layers_) {
        if (!layer.wq.w.defined()) {
            layer.wqkv_fp16 = Tensor();
            continue;
        }
        layer.wqkv_fp16 = Tensor::empty({qd + 2 * kvd, d}, DType::F16, device_);
        u16* fused = layer.wqkv_fp16.ptr<u16>();
        ops::convert_f32_to_f16(device_, layer.wq.w.f32(), fused, qd * d);
        ops::convert_f32_to_f16(device_, layer.wk.w.f32(), fused + qd * d, kvd * d);
        ops::convert_f32_to_f16(device_, layer.wv.w.f32(), fused + (qd + kvd) * d, kvd * d);
    }
    fp16_weights_dirty_ = false;
}

void Model::mark_weights_dirty() {
    if (fp16_weight_cache_) fp16_weights_dirty_ = true;
}

void Model::invalidate_fp16_cache() {
    if (!fp16_weight_cache_) return;
    {
        std::lock_guard<std::mutex> lock(ops::get_fp16_weights_mutex());
        auto& registry = ops::get_fp16_weights_map();
        for (Parameter* p : params_) {
            if (p->w.defined()) registry.erase(p->w.f32());
        }
    }
    for (Parameter* p : params_) {
        if (p->fp16_cache.defined()) {
            p->fp16_cache = Tensor();
        }
    }
    for (LayerParams& layer : layers_) {
        if (layer.wqkv_fp16.defined()) {
            layer.wqkv_fp16 = Tensor();
        }
    }
    fp16_weight_cache_ = false;
    fp16_weights_dirty_ = false;
}

size_t Model::fp16_weight_cache_bytes() const {
    size_t bytes = 0;
    for (const Parameter* p : params_) bytes += p->fp16_cache.nbytes();
    for (const LayerParams& layer : layers_) bytes += layer.wqkv_fp16.nbytes();
    return bytes;
}

size_t Model::weight_storage_bytes() const {
    size_t bytes = 0;
    for (const Parameter* p : params_)
        if (p->w.defined()) bytes += p->w.nbytes();
    return bytes;
}

void Model::enable_grad(bool on) {
    grad_enabled_ = on;
    if (on) {
        for (const Parameter* p : params_) {
            if (p->frozen) continue;
            GAI_CHECK(!p->w.is_external(),
                      "enable_grad on mmap-backed trainable weights is forbidden (read-only file pages): " +
                      p->name + " - freeze this parameter or reload without --mmap for full fine-tuning");
        }
    }
    for (Parameter* p : params_) {
        if (on) {
            if (p->frozen) {
                p->g = Tensor();
                continue;
            }
            if (!p->g.defined()) {
                p->g = Tensor::zeros(p->shape, DType::F32, device_);
                p->g.set_name(p->name + ".grad");
            }
        } else {
            p->g = Tensor();
        }
    }
}

void Model::zero_grad() {
    for (Parameter* p : params_)
        if (p->g.defined()) p->g.zero_();
}

static Tensor mk(std::vector<i64> shape, Device dev, size_t& acc) {
    Tensor t = Tensor::zeros(std::move(shape), DType::F32, dev);
    acc += t.nbytes();
    return t;
}

Activations Model::make_activations(int B, int T, bool with_grad, int ce_chunks) const {
    Activations a;
    a.B = B;
    a.T = T;
    a.with_grad = with_grad;

    const i64 N   = static_cast<i64>(B) * T;
    if (ce_chunks < 1) ce_chunks = 1;
    const i64 Cc = (with_grad && ce_chunks > 1) ? (N + ce_chunks - 1) / ce_chunks : N;
    a.ce_chunks = (with_grad && ce_chunks > 1) ? ce_chunks : 1;
    a.ce_rows = Cc;
    const i64 d   = cfg_.hidden_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.intermediate_size;
    const i64 E   = cfg_.moe_expert_dim;
    const i64 ne  = cfg_.num_experts;
    const i64 K   = cfg_.moe_top_k;
    const i64 V   = cfg_.vocab_size;
    const int L   = cfg_.num_layers;
    const bool moe = cfg_.use_moe;
    size_t& acc = a.bytes;

    auto mk_dt = [&](std::vector<i64> shape, DType dt) {
        Tensor t = Tensor::zeros(std::move(shape), dt, device_);
        acc += t.nbytes();
        return t;
    };

    a.x       = mk({N, d},  device_, acc);
    a.xb      = mk({N, d},  device_, acc);
    a.xb2     = mk({N, d},  device_, acc);
    if (fp16_weight_cache_) a.qkv = mk({N, qd + 2 * kvd}, device_, acc);
    a.q = mk({N, qd}, device_, acc);
    a.k = mk({N, kvd}, device_, acc);
    a.v = mk({N, kvd}, device_, acc);
    a.att_out = mk({N, qd}, device_, acc);
    a.proj    = mk({N, d},  device_, acc);
    if (moe) {
        a.moe_gate  = mk({N, K, E}, device_, acc);
        a.moe_up    = mk({N, K, E}, device_, acc);
        a.moe_act   = mk({N, K, E}, device_, acc);
        a.moe_probs = mk({N, ne},   device_, acc);
        a.moe_idx   = mk_dt({N, K}, DType::I32);
        a.moe_w     = mk({N, K},     device_, acc);
    } else {
        a.gate    = mk({N, F},  device_, acc);
        a.up      = mk({N, F},  device_, acc);
        a.act     = mk({N, F},  device_, acc);
    }
    {
        int mamba_layers = 0;
        if (cfg_.arch == ModelConfig::ArchGranite) {
            for (int k : cfg_.layer_types) if (k == 0) ++mamba_layers;
        }
        if (mamba_layers > 0) {
            const i64 state_rows = static_cast<i64>(B) * cfg_.mamba_n_heads * cfg_.mamba_d_head;
            const i64 per = state_rows * cfg_.mamba_d_state;
            a.mamba_states.reserve(static_cast<size_t>(mamba_layers));
            for (int i = 0; i < mamba_layers; ++i) {
                Tensor st = Tensor::zeros({per}, DType::F32, device_);
                acc += st.nbytes();
                a.mamba_states.push_back(std::move(st));
            }
            const int conv_dim = cfg_.mamba_expand * cfg_.hidden_size +
                                 2 * cfg_.mamba_n_groups * cfg_.mamba_d_state;
            const int win = cfg_.mamba_d_conv > 1 ? cfg_.mamba_d_conv - 1 : 0;
            a.mamba_conv_states.reserve(static_cast<size_t>(mamba_layers));
            for (int i = 0; i < mamba_layers; ++i) {
                Tensor cs = Tensor::zeros({static_cast<i64>(B) * conv_dim * win},
                                          DType::F32, device_);
                acc += cs.nbytes();
                a.mamba_conv_states.push_back(std::move(cs));
            }
        }
    }
    a.ffn_out = mk({N, d},  device_, acc);
    a.logits  = mk({Cc, V}, device_, acc);

    if (with_grad) {
        a.saved_x.resize(static_cast<size_t>(L));
        a.saved_xb.resize(static_cast<size_t>(L));
        a.saved_rrms1.resize(static_cast<size_t>(L));
        a.saved_q.resize(static_cast<size_t>(L));
        a.saved_k.resize(static_cast<size_t>(L));
        a.saved_v.resize(static_cast<size_t>(L));
        if (cfg_.use_qk_norm) {
            a.saved_qk_rms_q.resize(static_cast<size_t>(L));
            a.saved_qk_rms_k.resize(static_cast<size_t>(L));
            a.saved_qk_raw_q.resize(static_cast<size_t>(L));
            a.saved_qk_raw_k.resize(static_cast<size_t>(L));
        }
        {
            const size_t need_tmp =
                static_cast<size_t>(B) * static_cast<size_t>(cfg_.num_heads) *
                static_cast<size_t>(T) * static_cast<size_t>(T) * sizeof(float);
            if (need_tmp > (800ULL << 20)) {
                GAI_FAIL(strfmt("attention tmp needs %s (B=%d H=%d T=%d). "
                                "Lower batch_size/seq_len or enable ce_chunks "
                                "(T=4096 needs 1.6GB transient).",
                                human_bytes(need_tmp).c_str(), B, cfg_.num_heads, T));
            } else if (need_tmp > (400ULL << 20)) {
                log_warn(strfmt("[model] attention tmp large (%s); T4 frag risk",
                                human_bytes(need_tmp).c_str()));
            }
        }
        a.attn_probs_tmp = mk({static_cast<i64>(B) * cfg_.num_heads * T * T}, device_, acc);
        a.saved_attout.resize(static_cast<size_t>(L));
        a.saved_xmid.resize(static_cast<size_t>(L));
        a.saved_xb2.resize(static_cast<size_t>(L));
        a.saved_rrms2.resize(static_cast<size_t>(L));
        a.saved_gate.resize(static_cast<size_t>(L));
        a.saved_up.resize(static_cast<size_t>(L));
        a.saved_act.resize(static_cast<size_t>(L));
        if (moe) {
            a.saved_moe_probs.resize(static_cast<size_t>(L));
            a.saved_moe_idx.resize(static_cast<size_t>(L));
            a.saved_moe_w.resize(static_cast<size_t>(L));
            a.moe_dact    = mk({N, K, E}, device_, acc);
            a.moe_auxfrac = mk({ne},      device_, acc);
        }

        for (int i = 0; i < L; ++i) {
            size_t s = static_cast<size_t>(i);
            const bool layer_is_mamba =
                (cfg_.arch == ModelConfig::ArchGranite &&
                 s < cfg_.layer_types.size() &&
                 cfg_.layer_types[s] == 0);

            a.saved_x[s]      = mk({N, d},   device_, acc);
            a.saved_xb[s]     = mk({N, d},   device_, acc);
            a.saved_rrms1[s]  = mk({N},      device_, acc);
            if (!layer_is_mamba) {
                a.saved_q[s]      = mk({N, qd},  device_, acc);
                a.saved_k[s]      = mk({N, kvd}, device_, acc);
                a.saved_v[s]      = mk({N, kvd}, device_, acc);
                if (cfg_.use_qk_norm) {
                    const i64 Hh = cfg_.num_heads;
                    const i64 KVh = cfg_.num_kv_heads;
                    a.saved_qk_rms_q[s] = mk({N * Hh}, device_, acc);
                    a.saved_qk_rms_k[s] = mk({N * KVh}, device_, acc);
                    a.saved_qk_raw_q[s] = mk({N, qd},  device_, acc);
                    a.saved_qk_raw_k[s] = mk({N, kvd}, device_, acc);
                }
                a.saved_attout[s] = mk({N, qd},  device_, acc);
            }
            a.saved_xmid[s]   = mk({N, d},   device_, acc);
            a.saved_xb2[s]    = mk({N, d},   device_, acc);
            a.saved_rrms2[s]  = mk({N},      device_, acc);
            if (moe) {
                a.saved_gate[s]      = mk({N, K, E}, device_, acc);
                a.saved_up[s]        = mk({N, K, E}, device_, acc);
                a.saved_act[s]       = mk({N, K, E}, device_, acc);
                a.saved_moe_probs[s] = mk({N, ne},   device_, acc);
                a.saved_moe_idx[s]   = mk_dt({N, K}, DType::I32);
                a.saved_moe_w[s]     = mk({N, K},     device_, acc);
            } else {
                a.saved_gate[s]   = mk({N, F},   device_, acc);
                a.saved_up[s]     = mk({N, F},   device_, acc);
                a.saved_act[s]    = mk({N, F},   device_, acc);
            }
        }
        a.saved_xfinal    = mk({N, d}, device_, acc);
        a.saved_rrms_final= mk({N},    device_, acc);
        a.saved_hnorm     = mk({N, d}, device_, acc);

        a.dx      = mk({N, d},   device_, acc);
        a.dxb     = mk({N, d},   device_, acc);
        a.dq      = mk({N, qd},  device_, acc);
        a.dk      = mk({N, kvd}, device_, acc);
        a.dv      = mk({N, kvd}, device_, acc);
        a.dattout = mk({N, qd},  device_, acc);
        a.dproj   = mk({N, d},   device_, acc);
        if (!moe) {
            a.dgate   = mk({N, F},   device_, acc);
            a.dup     = mk({N, F},   device_, acc);
            a.dact    = mk({N, F},   device_, acc);
            a.dffn    = mk({N, d},   device_, acc);
        }
        a.dlogits = mk({Cc, V},  device_, acc);
        a.dtmp    = mk({N, d},   device_, acc);
    }
    return a;
}

size_t Model::estimate_activation_bytes(int B, int T, bool with_grad, int ce_chunks) const {
    return estimate_activation_bytes_for(cfg_, fp16_weight_cache_, B, T, with_grad, ce_chunks);
}

size_t Model::estimate_activation_bytes_for(const ModelConfig& cfg, bool fp16_on,
                                            int B, int T, bool with_grad, int ce_chunks) {
    auto sat_add = [](u64 a, u64 b) -> u64 {
        const u64 cap = (1ull << 40);
        if (a > cap || b > cap) return cap;
        u64 s = a + b;
        return (s < a || s > cap) ? cap : s;
    };
    auto sat_mul = [&](u64 a, u64 b) -> u64 {
        if (a == 0 || b == 0) return 0;
        const u64 cap = (1ull << 40);
        if (a > cap / b) return cap;
        u64 p = a * b;
        return p > cap ? cap : p;
    };
    const u64 N   = static_cast<u64>(B) * static_cast<u64>(T);
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 qd  = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F   = cfg.use_moe ? static_cast<u64>(cfg.moe_top_k) * static_cast<u64>(cfg.moe_expert_dim)
                                 : static_cast<u64>(cfg.intermediate_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 L   = static_cast<u64>(cfg.num_layers);

    u64 e = sat_mul(N, (4 * d + qd * 2 + kvd * 2 + F * 3 + V));
    if (fp16_on) e = sat_add(e, sat_mul(N, (qd + 2 * kvd)));
    if (cfg.use_moe) e = sat_add(e, sat_mul(N, (static_cast<u64>(cfg.num_experts) + 2 * static_cast<u64>(cfg.moe_top_k))));
    e = sat_add(e, N);
    if (with_grad) {
        u64 per_layer = sat_mul(N, (4 * d + qd * 2 + kvd * 2 + F * 3 + 2));
        if (cfg.use_moe) per_layer = sat_add(per_layer, sat_mul(N, (static_cast<u64>(cfg.num_experts) + 3 * static_cast<u64>(cfg.moe_top_k))));
        if (cfg.use_qk_norm) per_layer = sat_add(per_layer, sat_mul(N, (static_cast<u64>(cfg.num_heads) + static_cast<u64>(cfg.num_kv_heads) + qd + kvd)));
        e = sat_add(e, sat_mul(L, per_layer));
        e = sat_add(e, sat_mul(sat_mul(static_cast<u64>(B), static_cast<u64>(cfg.num_heads)), sat_mul(static_cast<u64>(T), static_cast<u64>(T))));
        if (!cfg.use_moe) {
            e = sat_add(e, sat_mul(N, (5 * d + qd * 2 + kvd * 2 + F * 3 + V + 2)));
        } else {
            e = sat_add(e, sat_mul(N, (5 * d + qd * 2 + kvd * 2 + V + 2)));
            e = sat_add(e, sat_add(sat_mul(N, sat_mul(static_cast<u64>(cfg.moe_top_k), static_cast<u64>(cfg.moe_expert_dim))), static_cast<u64>(cfg.num_experts)));
        }
        e = sat_add(e, sat_mul(N, (4 * d + qd * 2 + kvd * 2 + V + d * 2)));
    }
    if (with_grad && ce_chunks > 1 && N > 0) {
        const u64 cc = static_cast<u64>(ce_chunks);
        const u64 crow = N / cc + (N % cc ? 1ULL : 0ULL);
        const u64 sub = sat_mul(sat_mul(2ULL, V), N - crow);
        e = (e >= sub) ? e - sub : 0;
    }
    const u64 mamba_bytes = mamba_state_bytes(cfg, B);
    e = sat_add(e, (mamba_bytes + 3ull) / 4ull);
    if (e > ((1ull << 40) / 4)) return (1ull << 40);
    return static_cast<size_t>(e) * sizeof(float);
}

static u64 mamba_layer_parameters(const ModelConfig& cfg) {
    const u64 d      = static_cast<u64>(cfg.hidden_size);
    const u64 d_inner = static_cast<u64>(cfg.mamba_expand) * d;
    const u64 conv_dim = d_inner + 2ull * static_cast<u64>(cfg.mamba_n_groups) *
                                      static_cast<u64>(cfg.mamba_d_state);
    const u64 n_heads = static_cast<u64>(cfg.mamba_n_heads);
    const u64 proj    = d_inner + conv_dim + n_heads;
    const u64 k       = static_cast<u64>(cfg.mamba_d_conv);

    u64 n = 0;
    n += proj * d;
    n += conv_dim * k;
    n += conv_dim;
    n += n_heads;
    n += n_heads;
    n += n_heads;
    n += d_inner;
    n += d * d_inner;
    return n;
}

namespace {
u64 lora_trainable_count(const ModelConfig& cfg) {
    if (cfg.lora_rank <= 0) return 0;
    const u64 r  = static_cast<u64>(cfg.lora_rank);
    const u64 d  = static_cast<u64>(cfg.hidden_size);
    const u64 qd = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F  = static_cast<u64>(cfg.use_moe ? cfg.moe_expert_dim
                                                 : cfg.intermediate_size);
    u64 per = 0;
    auto add = [&](bool on, u64 in, u64 out) {
        if (on) per += r * (in + out);
    };
    add(cfg.lora_target_q, d, qd);
    add(cfg.lora_target_k, d, kvd);
    add(cfg.lora_target_v, d, kvd);
    add(cfg.lora_target_o, qd, d);
    if (!cfg.use_moe) {
        add(cfg.lora_target_gate, d, F);
        add(cfg.lora_target_up, d, F);
        add(cfg.lora_target_down, F, d);
    }
    return per * static_cast<u64>(cfg.num_layers);
}
}

u64 Model::count_parameters(const ModelConfig& cfg) {
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 qd  = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F   = static_cast<u64>(cfg.intermediate_size);
    const u64 E   = static_cast<u64>(cfg.moe_expert_dim);
    const u64 SH  = static_cast<u64>(cfg.shared_width());
    const u64 ne  = static_cast<u64>(cfg.num_experts);
    const u64 L   = static_cast<u64>(cfg.num_layers);
    const u64 hd  = static_cast<u64>(cfg.head_dim());

    auto layer_is_mamba = [&](u64 i) {
        if (cfg.arch != ModelConfig::ArchGranite) return false;
        if (i >= static_cast<u64>(cfg.layer_types.size())) return false;
        return cfg.layer_types[static_cast<size_t>(i)] == 0;
    };

    u64 n = V * d;
    for (u64 i = 0; i < L; ++i) {
        n += d;
        n += d;
        if (layer_is_mamba(i)) {
            n += mamba_layer_parameters(cfg);
        } else {
            n += qd * d + kvd * d + kvd * d + d * qd;
            if (cfg.use_qkv_bias) n += qd + 2 * kvd;
            if (cfg.use_qk_norm) n += 2 * hd;
        }
        if (cfg.use_moe) {
            n += ne * d;
            n += ne * E * d + ne * E * d;
            n += ne * d * E;
            if (cfg.moe_shared) n += SH * d + SH * d + d * SH;
        } else {
            n += F * d + F * d + d * F;
        }
    }
    n += d;
    if (!cfg.tie_embeddings) n += V * d;
    return n;
}

size_t Model::count_fp16_cache_bytes(const ModelConfig& cfg) {
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 qd  = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F   = static_cast<u64>(cfg.intermediate_size);
    const u64 E   = static_cast<u64>(cfg.moe_expert_dim);
    const u64 SH  = static_cast<u64>(cfg.shared_width());
    const u64 ne  = static_cast<u64>(cfg.num_experts);
    const u64 L   = static_cast<u64>(cfg.num_layers);

    auto layer_is_mamba = [&](u64 i) {
        if (cfg.arch != ModelConfig::ArchGranite) return false;
        if (i >= static_cast<u64>(cfg.layer_types.size())) return false;
        return cfg.layer_types[static_cast<size_t>(i)] == 0;
    };

    u64 n2 = 0;
    u64 fused = 0;
    n2 += V * d;
    for (u64 i = 0; i < L; ++i) {
        if (layer_is_mamba(i)) {
            const u64 d_inner = static_cast<u64>(cfg.mamba_expand) * d;
            const u64 conv_dim = d_inner + 2ull * static_cast<u64>(cfg.mamba_n_groups) *
                                          static_cast<u64>(cfg.mamba_d_state);
            const u64 proj = d_inner + conv_dim + static_cast<u64>(cfg.mamba_n_heads);
            n2 += proj * d;
            n2 += d * d_inner;
        } else {
            n2 += qd * d + kvd * d + kvd * d + d * qd;
            fused += (qd + 2 * kvd) * d;
        }
        if (cfg.use_moe) {
            n2 += ne * d;
            n2 += ne * E * d + ne * E * d;
            n2 += ne * d * E;
            if (cfg.moe_shared) n2 += SH * d + SH * d + d * SH;
        } else {
            n2 += F * d + F * d + d * F;
        }
    }
    if (!cfg.tie_embeddings) n2 += V * d;
    return static_cast<size_t>((n2 + fused) * sizeof(u16));
}

static u64 fragile_param_count(const ModelConfig& cfg) {
    u64 n = 0;
    const u64 d = static_cast<u64>(cfg.hidden_size);
    const u64 hd = static_cast<u64>(cfg.head_dim());
    const u64 d_inner = static_cast<u64>(cfg.mamba_expand) * d;
    const u64 conv_dim = d_inner +
                         2ull * static_cast<u64>(cfg.mamba_n_groups) *
                         static_cast<u64>(cfg.mamba_d_state);
    for (int l = 0; l < cfg.num_layers; ++l) {
        n += 2ull * d;
        if (cfg.use_moe)
            n += static_cast<u64>(cfg.num_experts) * d;
        if (cfg.use_qk_norm)
            n += 2ull * hd;
        const bool attn_layer = !((cfg.arch == ModelConfig::ArchGranite) &&
                                  l < static_cast<int>(cfg.layer_types.size()) &&
                                  cfg.layer_types[static_cast<size_t>(l)] == 0);
        if (cfg.use_qkv_bias && attn_layer)
            n += static_cast<u64>(cfg.q_dim()) + 2ull * static_cast<u64>(cfg.kv_dim());
        const bool is_mamba = (cfg.arch == ModelConfig::ArchGranite) &&
                              l < static_cast<int>(cfg.layer_types.size()) &&
                              cfg.layer_types[static_cast<size_t>(l)] == 0;
        if (is_mamba) {
            n += conv_dim * static_cast<u64>(cfg.mamba_d_conv);
            n += conv_dim;
            n += 3ull * static_cast<u64>(cfg.mamba_n_heads);
            n += d_inner;
        }
    }
    n += d;
    return n;
}

Model::MemoryPlan Model::plan_memory(const ModelConfig& cfg, int B, int T,
                                      bool with_grad, int ce_chunks,
                                      bool fp16_weight_cache,
                                      bool fp16_weights,
                                      int quant_kind) {
    MemoryPlan p;
    p.params = count_parameters(cfg);
    p.params_no_embedding = p.params - static_cast<u64>(cfg.vocab_size) * static_cast<u64>(cfg.hidden_size);
    if (!cfg.tie_embeddings)
        p.params_no_embedding -= static_cast<u64>(cfg.vocab_size) * static_cast<u64>(cfg.hidden_size);
    if (quant_kind >= 0) {
        const u64 fragile = fragile_param_count(cfg);
        const u64 emb = static_cast<u64>(cfg.vocab_size) * static_cast<u64>(cfg.hidden_size);
        const u64 emb_all = cfg.tie_embeddings ? emb : 2 * emb;
        const u64 big = p.params > fragile + emb_all ? p.params - fragile - emb_all : 0;
        const u64 blk = quant_kind == 2 ? 34 : quant_kind == 1 ? 20 : 18;
        p.weights = static_cast<size_t>((big * blk + 31) / 32) +
                    static_cast<size_t>(emb_all) * 2 + static_cast<size_t>(fragile) * 4;
    } else if (fp16_weights) {
        const u64 fragile = fragile_param_count(cfg);
        const u64 half = p.params > fragile ? p.params - fragile : 0;
        p.weights = static_cast<size_t>(half) * 2 + static_cast<size_t>(fragile) * 4;
    } else {
        p.weights = static_cast<size_t>(p.params) * sizeof(float);
    }
    const u64 trainable = cfg.lora_rank > 0 ? lora_trainable_count(cfg) : p.params;
    p.grads = with_grad ? static_cast<size_t>(trainable) * sizeof(float) : 0;
    p.fp16_cache = fp16_weight_cache ? count_fp16_cache_bytes(cfg) : 0;
    p.activations = estimate_activation_bytes_for(cfg, fp16_weight_cache, B, T,
                                                 with_grad, ce_chunks);

    size_t dequant_scratch = 0;
    if (quant_kind >= 0) {
        const u64 d = static_cast<u64>(cfg.hidden_size);
        const u64 max_mat = std::max({
            static_cast<u64>(cfg.intermediate_size) * d,
            static_cast<u64>(cfg.moe_expert_dim) * d,
            static_cast<u64>(cfg.vocab_size) * d
        });
        dequant_scratch = static_cast<size_t>(max_mat * sizeof(u16));
    }
    const WorkspacePlan ws = workspace_plan(cfg, B, T);
    p.optimizer = with_grad ? static_cast<size_t>(trainable) * 8 : 0;
    p.static_total = p.weights + p.grads + p.fp16_cache + dequant_scratch;
    p.total = p.static_total + p.activations + p.optimizer + ws.gemm_bytes + ws.moe_bytes + ws.mamba_bytes;
    return p;
}

Model::WorkspacePlan Model::workspace_plan(const ModelConfig& cfg, int B, int T) {
    const u64 cap = (1ull << 40);
    auto sat = [&](u64 v) -> u64 { return v > cap ? cap : v; };
    const u64 N   = static_cast<u64>(B) * static_cast<u64>(T);
    const u64 NK  = N * static_cast<u64>(cfg.moe_top_k > 0 ? cfg.moe_top_k : 1);
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 E   = static_cast<u64>(cfg.moe_expert_dim);
    const u64 ne  = static_cast<u64>(cfg.num_experts);

    WorkspacePlan p;
    const u64 gemm_conv = sat(2 * sat(N * d + d * V));
    const u64 sce = sat(N + 260);
    p.gemm_bytes = static_cast<size_t>(sat(2 * sat(gemm_conv + sce)));
    if (cfg.use_moe) {
        const u64 fwd = sat(N * ne + 2 * NK + 3 * NK * E + NK * d);
        const u64 bwd = sat(6 * N * E + NK + 4 * NK * E + 2 * NK * d + NK + N * ne);
        const u64 peak_floats = fwd > bwd ? fwd : bwd;
        p.moe_bytes = static_cast<size_t>(sat(4 * peak_floats + (16ull << 20)));
    }
    if (cfg.arch == ModelConfig::ArchGranite) {
        const u64 d_inner = static_cast<u64>(cfg.mamba_expand) * d;
        const u64 conv_dim = d_inner + 2ull * static_cast<u64>(cfg.mamba_n_groups) * static_cast<u64>(cfg.mamba_d_state);
        const u64 proj_size = d_inner + conv_dim + static_cast<u64>(cfg.mamba_n_heads);
        const u64 floats = sat(N * sat(2 * proj_size + 4 * conv_dim + 5 * d_inner + cfg.mamba_n_heads));
        p.mamba_bytes = static_cast<size_t>(sat(4 * floats + (32ull << 20)));
    }
    return p;
}

size_t Model::mamba_state_bytes(const ModelConfig& cfg, int batch) {
    if (cfg.arch != ModelConfig::ArchGranite || batch <= 0) return 0;
    int mamba_layers = 0;
    for (int k : cfg.layer_types) if (k == 0) ++mamba_layers;
    if (mamba_layers == 0) return 0;
    const u64 ssm = static_cast<u64>(batch) *
                    static_cast<u64>(cfg.mamba_n_heads) *
                    static_cast<u64>(cfg.mamba_d_head) *
                    static_cast<u64>(cfg.mamba_d_state);
    const u64 conv_dim = static_cast<u64>(cfg.mamba_expand) *
                         static_cast<u64>(cfg.hidden_size) +
                         2ull * static_cast<u64>(cfg.mamba_n_groups) *
                         static_cast<u64>(cfg.mamba_d_state);
    const u64 win = cfg.mamba_d_conv > 1 ? static_cast<u64>(cfg.mamba_d_conv) - 1 : 0;
    const u64 conv = static_cast<u64>(batch) * conv_dim * win;
    return static_cast<size_t>(static_cast<u64>(mamba_layers) * (ssm + conv) * 4ull);
}

size_t Model::kv_cache_bytes(const ModelConfig& cfg, int max_len) {
    if (max_len <= 0) return 0;
    const u64 kv_dim = static_cast<u64>(cfg.kv_dim());
    u64 layers = static_cast<u64>(cfg.num_layers);
    if (cfg.arch == ModelConfig::ArchGranite && !cfg.layer_types.empty()) {
        layers = 0;
        for (int k : cfg.layer_types) if (k == 1) ++layers;
    }
    return static_cast<size_t>(2ull * layers * static_cast<u64>(max_len) * kv_dim * 2ull);
}

static void check_act_device(Device dev, const char* name, const Tensor& t) {
    if (t.defined() && t.device() != dev) {
        GAI_FAIL(std::string("stale activations: ") + name +
                 " is on " + device_name(t.device()) + ", model is on " +
                 device_name(dev) + " (rebuild activations after Model::to)");
    }
}

void Model::forward_body(const i32* ids, int B, int T, Activations& act,
                         const i32* segment_ids, bool reset_mamba_states,
                         const float* embed_override,
                         int override_pos,
                         int n_override) {
    GAI_CHECK(ids != nullptr, "forward: null ids");
    GAI_CHECK(B > 0 && T > 0, "forward: B and T must be > 0");
    const int d   = cfg_.hidden_size;
    const int qd  = cfg_.q_dim();
    const int kvd = cfg_.kv_dim();
    const int F   = cfg_.intermediate_size;
    const int V   = cfg_.vocab_size;
    const int H   = cfg_.num_heads;
    const int KV  = cfg_.num_kv_heads;
    const int hd  = cfg_.head_dim();
    const i64 N   = static_cast<i64>(B) * T;
    const float scale = (cfg_.attention_multiplier > 0.0f)
                            ? cfg_.attention_multiplier
                            : (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale_local(cfg_);
    const bool  train = act.with_grad;
    Device dev = device_;
    if (reset_mamba_states) {
        for (Tensor& st : act.mamba_states) {
            if (st.defined() && st.numel() > 0) ops::zero(dev, st.f32(), st.numel());
        }
        for (Tensor& cw : act.mamba_conv_states) {
            if (cw.defined() && cw.numel() > 0) ops::zero(dev, cw.f32(), cw.numel());
        }
    }
    if (fp16_weight_cache_ && fp16_weights_dirty_) {
        for (Parameter* p : params_) {
            if (!p->fp16_cache.defined()) continue;
            ops::convert_f32_to_f16(dev, p->w.f32(), p->fp16_cache.ptr<u16>(), p->numel());
        }
        const i64 d = cfg_.hidden_size;
        const i64 qd = cfg_.q_dim();
        const i64 kvd = cfg_.kv_dim();
        for (LayerParams& layer : layers_) {
            if (!layer.wq.w.defined() || !layer.wqkv_fp16.defined()) continue;
            u16* fused = layer.wqkv_fp16.ptr<u16>();
            ops::convert_f32_to_f16(dev, layer.wq.w.f32(), fused, qd * d);
            ops::convert_f32_to_f16(dev, layer.wk.w.f32(), fused + qd * d, kvd * d);
            ops::convert_f32_to_f16(dev, layer.wv.w.f32(), fused + (qd + kvd) * d, kvd * d);
        }
        fp16_weights_dirty_ = false;
    }
    if (dev == Device::CPU) {
        for (i64 i = 0; i < N; ++i) {
            i32 id = ids[i];
            GAI_CHECK(id >= 0 && id < V,
                      strfmt("forward: token id out of range ids[%lld]=%d (vocab=%d; tokenizer/dataloader mismatch?)",
                             (long long)i, (int)id, V));
        }
    }

    GAI_CHECK(act.B == B && act.T == T, "activation buffer shape mismatch");
    GAI_CHECK(T <= cfg_.max_seq_len, "sequence longer than max_seq_len");
    GAI_CHECK(N <= static_cast<i64>(std::numeric_limits<int>::max()),
              "forward: B*T exceeds int range (reduce batch/seq_len)");
    if (train) {
        GAI_CHECK(static_cast<int>(act.saved_x.size()) == cfg_.num_layers,
                  "forward: stale Activations (layer count mismatch, rebuild)");
    }
    if (train && cfg_.use_moe) {
        GAI_CHECK(static_cast<int>(act.saved_moe_probs.size()) == cfg_.num_layers,
                  "forward: MoE training needs with_grad Activations");
    }
    check_act_device(dev, "act.x", act.x);
    check_act_device(dev, "act.xb", act.xb);
    check_act_device(dev, "act.logits", act.logits);
    check_act_device(dev, "act.pos", act.pos);
    if (train) {
        check_act_device(dev, "act.saved_hnorm", act.saved_hnorm);
        check_act_device(dev, "act.dx", act.dx);
    }

    if (!act.pos.defined() || act.pos.numel() < N) {
        act.pos = Tensor::empty({N}, DType::I32, device_);
        act.bytes += act.pos.nbytes();
        act.pos_cached_B = -1;
        act.pos_cached_T = -1;
        act.pos_cached_offset = -1;
    }
    if (act.pos_cached_B != B || act.pos_cached_T != T || act.pos_cached_offset != act.pos_offset) {
        thread_local std::vector<i32> pos_staging;
        if (pos_staging.size() < static_cast<size_t>(N))
            pos_staging.resize(static_cast<size_t>(N));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                pos_staging[static_cast<size_t>(b) * T + t] = act.pos_offset + t;
        device_copy(act.pos.data_ptr(), device_, pos_staging.data(), Device::CPU,
                    static_cast<size_t>(N) * sizeof(i32));
        act.pos_cached_B = B;
        act.pos_cached_T = T;
        act.pos_cached_offset = act.pos_offset;
    }

    if (is_half_or_quant(tok_emb_.w))
        ops::embedding_forward_half(dev, ids, weight16(dev, tok_emb_.w, 0), act.x.f32(), N, d, V);
    else
        ops::embedding_forward(dev, ids, tok_emb_.w.f32(), act.x.f32(), N, d, V);
    if (cfg_.embedding_multiplier != 1.0f) {
        ops::scale_inplace(dev, act.x.f32(), cfg_.embedding_multiplier, N * d);
    }
    if (embed_override && n_override > 0 && override_pos >= 0 &&
        static_cast<i64>(override_pos) + n_override <= N) {
        device_copy(act.x.f32() + static_cast<size_t>(override_pos) * d, dev,
                    embed_override, dev,
                    sizeof(float) * static_cast<size_t>(n_override) * d);
    }

    const float* rope_freq = rope_inv_freq_ptr();
    GAI_CHECK(rope_freq != nullptr, "forward: RoPE frequency cache is missing");

    for (int l = 0; l < cfg_.num_layers; ++l) {
        LayerParams& L = layers_[static_cast<size_t>(l)];
        size_t sl = static_cast<size_t>(l);
        const bool is_granite    = (cfg_.arch == ModelConfig::ArchGranite);
        const int  granite_kind  = (is_granite && sl < cfg_.layer_types.size())
                                       ? cfg_.layer_types[sl] : 1;
        const bool granite_mamba = is_granite && L.has_mamba() && granite_kind == 0;

        if (train) ops::copy(dev, act.saved_x[sl].f32(), act.x.f32(), N * d);

        float* rrms1 = train ? act.saved_rrms1[sl].f32() : nullptr;
        ops::rmsnorm_forward(dev, act.x.f32(), L.attn_norm.w.f32(), act.xb.f32(),
                             rrms1, N, d, cfg_.rms_eps);
        if (train) ops::copy(dev, act.saved_xb[sl].f32(), act.xb.f32(), N * d);

        if (granite_mamba) {
            forward_mamba_layer(dev, l, act, N, segment_ids);
        } else {

        float* qp = nullptr;
        float* kp = nullptr;
        float* vp = nullptr;
        if (act.qkv.defined() && L.wqkv_fp16.defined()) {
            ops::linear_forward_fp16(dev, act.xb.f32(), L.wqkv_fp16.ptr<u16>(),
                                     act.qkv.f32(), static_cast<int>(N), d, qd + 2 * kvd);
            ops::split_qkv(dev, act.qkv.f32(), act.q.f32(), act.k.f32(), act.v.f32(),
                            N, qd, kvd);
            qp = act.q.f32();
            kp = act.k.f32();
            vp = act.v.f32();
        } else {
            qp = act.q.f32();
            kp = act.k.f32();
            vp = act.v.f32();
            linear_param(dev, L.wq.w, act.xb.f32(), qp, static_cast<int>(N), d, qd);
            linear_param(dev, L.wk.w, act.xb.f32(), kp, static_cast<int>(N), d, kvd);
            linear_param(dev, L.wv.w, act.xb.f32(), vp, static_cast<int>(N), d, kvd);
        }

        if (L.has_qkv_bias()) {
            ops::add_bias(dev, L.bq.w.f32(), qp, static_cast<int>(N), qd);
            ops::add_bias(dev, L.bk.w.f32(), kp, static_cast<int>(N), kvd);
            ops::add_bias(dev, L.bv.w.f32(), vp, static_cast<int>(N), kvd);
        }
        lora_forward(L.lora_q, dev, act.xb.f32(), qp, static_cast<int>(N));
        lora_forward(L.lora_k, dev, act.xb.f32(), kp, static_cast<int>(N));
        lora_forward(L.lora_v, dev, act.xb.f32(), vp, static_cast<int>(N));

        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined() && L.qk_knorm.w.defined()) {
            if (train) {
                ops::copy(dev, act.saved_qk_raw_q[sl].f32(), qp, N * qd);
                ops::copy(dev, act.saved_qk_raw_k[sl].f32(), kp, N * kvd);
            }
            float* rrms_q = train ? act.saved_qk_rms_q[sl].f32() : nullptr;
            float* rrms_k = train ? act.saved_qk_rms_k[sl].f32() : nullptr;
            ops::rmsnorm_forward(dev, qp, L.qk_qnorm.w.f32(), qp,
                                 rrms_q, N * H, hd, cfg_.rms_eps);
            ops::rmsnorm_forward(dev, kp, L.qk_knorm.w.f32(), kp,
                                 rrms_k, N * KV, hd, cfg_.rms_eps);
        }

        if (cfg_.position_embedding != 1) {
            ops::rope_forward_cached(dev, qp, kp, act.pos.i32p(), rope_freq, N, H, KV, hd,
                                     cfg_.rope_type);
        }

        if (train) {
            ops::copy(dev, act.saved_q[sl].f32(), qp, N * qd);
            ops::copy(dev, act.saved_k[sl].f32(), kp, N * kvd);
            ops::copy(dev, act.saved_v[sl].f32(), vp, N * kvd);
        }

        ops::attention_forward_ex(dev, qp, kp, vp,
                                  act.att_out.f32(), nullptr,
                                  B, T, H, KV, hd, scale, cfg_.sliding_window,
                                  segment_ids);
        if (train) ops::copy(dev, act.saved_attout[sl].f32(), act.att_out.f32(), N * qd);

        linear_param(dev, L.wo.w, act.att_out.f32(), act.proj.f32(), static_cast<int>(N), qd, d);
        lora_forward(L.lora_o, dev, act.att_out.f32(), act.proj.f32(), static_cast<int>(N));
        }

        if (cfg_.residual_multiplier != 1.0f) {
            ops::scale_inplace(dev, act.proj.f32(), cfg_.residual_multiplier, N * d);
        }
        ops::add_inplace(dev, act.x.f32(), act.proj.f32(), N * d);

        if (train) ops::copy(dev, act.saved_xmid[sl].f32(), act.x.f32(), N * d);

        float* rrms2 = train ? act.saved_rrms2[sl].f32() : nullptr;
        ops::rmsnorm_forward(dev, act.x.f32(), L.ffn_norm.w.f32(), act.xb2.f32(),
                             rrms2, N, d, cfg_.rms_eps);
        if (train) ops::copy(dev, act.saved_xb2[sl].f32(), act.xb2.f32(), N * d);

        if (L.has_moe()) {
            const int E  = cfg_.moe_expert_dim;
            const int ne = cfg_.num_experts;
            const int K  = cfg_.moe_top_k;
            float* probs = train ? act.saved_moe_probs[sl].f32() : nullptr;
            i32*   ridx  = train ? act.saved_moe_idx[sl].i32p()  : nullptr;
            float* rw    = train ? act.saved_moe_w[sl].f32()     : nullptr;
            const float* bias = (cfg_.moe_aux_free && !moe_bias_.empty())
                ? moe_bias_ptr(l) : nullptr;
            if (is_half_or_quant(L.moe_gate.w)) {
                const WeightRef mg = weight_ref(dev, L.moe_gate.w, 0);
                const WeightRef mu = weight_ref(dev, L.moe_up.w, 1);
                const WeightRef md = weight_ref(dev, L.moe_down.w, 2);
                const WeightRef msg = L.sh_gate.w.defined() ? weight_ref(dev, L.sh_gate.w, 3) : WeightRef{};
                const WeightRef msu = L.sh_up.w.defined()   ? weight_ref(dev, L.sh_up.w, 4)   : WeightRef{};
                const WeightRef msd = L.sh_down.w.defined() ? weight_ref(dev, L.sh_down.w, 5) : WeightRef{};
                ops::moe_forward_bias_half(dev, act.xb2.f32(), L.router.w.f32(), bias,
                                           mg.ptr, mu.ptr, md.ptr, msg.ptr, msu.ptr, msd.ptr,
                                           act.ffn_out.f32(), probs, ridx, rw,
                                           act.moe_gate.f32(), act.moe_up.f32(), act.moe_act.f32(),
                                           N, d, E, ne, K, cfg_.shared_width(), mg.qkind);
            } else {
                ops::moe_forward_bias(dev, act.xb2.f32(), L.router.w.f32(), bias,
                                      L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                      L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                      L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                      L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                      act.ffn_out.f32(), probs, ridx, rw,
                                      act.moe_gate.f32(), act.moe_up.f32(), act.moe_act.f32(),
                                      N, d, E, ne, K, cfg_.shared_width());
            }
            if (train) {
                ops::copy(dev, act.saved_gate[sl].f32(), act.moe_gate.f32(), N * K * E);
                ops::copy(dev, act.saved_up[sl].f32(),   act.moe_up.f32(),   N * K * E);
                ops::copy(dev, act.saved_act[sl].f32(),  act.moe_act.f32(),  N * K * E);
            }
        } else {
            linear_param(dev, L.w_gate.w, act.xb2.f32(), act.gate.f32(), static_cast<int>(N), d, F);
            linear_param(dev, L.w_up.w,   act.xb2.f32(), act.up.f32(),   static_cast<int>(N), d, F);
            lora_forward(L.lora_gate, dev, act.xb2.f32(), act.gate.f32(), static_cast<int>(N));
            lora_forward(L.lora_up, dev, act.xb2.f32(), act.up.f32(), static_cast<int>(N));
            ops::swiglu_forward(dev, act.gate.f32(), act.up.f32(), act.act.f32(), N * F);

            if (train) {
                ops::copy(dev, act.saved_gate[sl].f32(), act.gate.f32(), N * F);
                ops::copy(dev, act.saved_up[sl].f32(),   act.up.f32(),   N * F);
                ops::copy(dev, act.saved_act[sl].f32(),  act.act.f32(),  N * F);
            }

            linear_param(dev, L.w_down.w, act.act.f32(), act.ffn_out.f32(), static_cast<int>(N), F, d);
            lora_forward(L.lora_down, dev, act.act.f32(), act.ffn_out.f32(), static_cast<int>(N));
        }
        if (cfg_.residual_multiplier != 1.0f) {
            ops::scale_inplace(dev, act.ffn_out.f32(), cfg_.residual_multiplier, N * d);
        }
        ops::add_inplace(dev, act.x.f32(), act.ffn_out.f32(), N * d);
    }

    if (train) ops::copy(dev, act.saved_xfinal.f32(), act.x.f32(), N * d);
    float* rrmsf = train ? act.saved_rrms_final.f32() : nullptr;
    ops::rmsnorm_forward(dev, act.x.f32(), final_norm_.w.f32(), act.xb.f32(), rrmsf, N, d, cfg_.rms_eps);
    if (train) ops::copy(dev, act.saved_hnorm.f32(), act.xb.f32(), N * d);
}

Tensor& Model::forward(const i32* ids, int B, int T, Activations& act,
                       const i32* segment_ids, bool reset_mamba_states) {
    forward_body(ids, B, T, act, segment_ids, reset_mamba_states);
    const int d = cfg_.hidden_size;
    const int V = cfg_.vocab_size;
    const i64 N = static_cast<i64>(B) * T;
    {
        const i64 need = N * static_cast<i64>(V);
        GAI_CHECK(act.logits.numel() >= need,
                  "forward: logits buffer too small (chunked training activations "
                  "need forward_backward, not forward)");
    }
    linear_param(device_, lm_head().w, act.xb.f32(), act.logits.f32(),
                 static_cast<int>(N), d, V);
    if (cfg_.logits_scaling != 1.0f) {
        ops::scale_inplace(device_, act.logits.f32(), 1.0f / cfg_.logits_scaling, N * V);
    }
    return act.logits;
}

double Model::forward_backward(const i32* ids, const i32* targets, int B, int T,
                               Activations& act, i64* out_ntok, float dout_scale,
                               bool want_aux_stats, const i32* segment_ids,
                               const i32* host_targets,
                               const float* embed_override,
                               int override_pos,
                               int n_override,
                               float* out_override_dx) {
    if (weight_f16_ || weight_q4_) {
        for (const Parameter* p : params_) {
            if (p->frozen) continue;
            GAI_CHECK(p->w.dtype() == DType::F32,
                      "the training path needs an fp32 master for the trainable "
                      "parameter " + p->name +
                      " (turn the half/quantized weight storage off, or freeze it "
                      "and train LoRA adapters only)");
        }
    }
    GAI_CHECK(act.with_grad, "forward_backward requires gradient activations");
    GAI_CHECK(grad_enabled_, "call enable_grad(true) before forward_backward");

    const int d   = cfg_.hidden_size;
    const int qd  = cfg_.q_dim();
    const int kvd = cfg_.kv_dim();
    const int F   = cfg_.intermediate_size;
    const int V   = cfg_.vocab_size;
    const int H   = cfg_.num_heads;
    const int KV  = cfg_.num_kv_heads;
    const int hd  = cfg_.head_dim();
    const i64 N   = static_cast<i64>(B) * T;
    const float scale = (cfg_.attention_multiplier > 0.0f)
                            ? cfg_.attention_multiplier
                            : (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale_local(cfg_);
    Device dev = device_;

    forward_body(ids, B, T, act, segment_ids, true,
                 embed_override, override_pos, n_override);

    float eff_scale = dout_scale;
    const i64 Cc = act.ce_rows > 0 ? act.ce_rows : N;
    const float* hnorm_full = act.saved_hnorm.f32();
    float* logits_c = act.logits.f32();
    float* dlogits_c = act.dlogits.f32();
    ops::zero(dev, act.dxb.f32(), N * d);
    ops::sce_acc_begin(dev);
    for (i64 r0 = 0; r0 < N; r0 += Cc) {
        const i64 Cr = std::min(Cc, N - r0);
        const int Ci = static_cast<int>(Cr);
        if (host_targets) {
            bool any_supervised = false;
            for (i64 i = 0; i < Cr; ++i) {
                if (host_targets[r0 + i] >= 0) { any_supervised = true; break; }
            }
            if (!any_supervised) continue;
        }
        linear_param(dev, lm_head().w, hnorm_full + r0 * d, logits_c, Ci, d, V);
        if (cfg_.logits_scaling != 1.0f) {
            ops::scale_inplace(dev, logits_c, 1.0f / cfg_.logits_scaling, Cr * V);
        }
        ops::sce_accumulate(dev, logits_c, targets + r0, dlogits_c,
                            Cr, V, cfg_.z_loss_scale);
        float chunk_dscale = eff_scale;
        if (cfg_.logits_scaling != 1.0f) chunk_dscale /= cfg_.logits_scaling;
        if (chunk_dscale != 1.0f) ops::scale_inplace(dev, dlogits_c, chunk_dscale, Cr * V);
        if (is_half_or_quant(lm_head().w))
            ops::linear_backward_half(dev, hnorm_full + r0 * d, weight16(dev, lm_head().w, 0), dlogits_c,
                                      act.dxb.f32() + r0 * d, gptr(lm_head().g), Ci, d, V);
        else
            ops::linear_backward(dev, hnorm_full + r0 * d, lm_head().w.f32(), dlogits_c,
                                 act.dxb.f32() + r0 * d, gptr(lm_head().g), Ci, d, V);
    }
    double loss_sum = 0.0;
    i64 ntok = 0;
    ops::sce_acc_end(dev, &loss_sum, &ntok);
    if (out_ntok) *out_ntok = ntok;
    if (ntok == 0) return 0.0;

    double aux_total = 0.0;
    const bool use_dev_aux = (dev == Device::CUDA) && cfg_.use_moe &&
                             cfg_.moe_aux_scale > 0.0f;
    double* aux_accum = use_dev_aux ? ops::moe_aux_begin(dev) : nullptr;

    const float* rope_freq_bwd = rope_inv_freq_ptr();
    GAI_CHECK(rope_freq_bwd != nullptr, "backward: RoPE frequency cache is missing");

    ops::zero(dev, act.dx.f32(), N * d);
    ops::rmsnorm_backward(dev, act.saved_xfinal.f32(), final_norm_.w.f32(), act.dxb.f32(),
                          act.saved_rrms_final.f32(), act.dx.f32(), gptr(final_norm_.g), N, d);

    for (int l = cfg_.num_layers - 1; l >= 0; --l) {
        LayerParams& L = layers_[static_cast<size_t>(l)];
        size_t sl = static_cast<size_t>(l);

        const float* dy_ffn = act.dx.f32();
        if (cfg_.residual_multiplier != 1.0f) {
            ops::copy(dev, act.dtmp.f32(), act.dx.f32(), N * d);
            ops::scale_inplace(dev, act.dtmp.f32(), cfg_.residual_multiplier, N * d);
            dy_ffn = act.dtmp.f32();
        }
        if (L.has_moe()) {
            const int E  = cfg_.moe_expert_dim;
            const int ne = cfg_.num_experts;
            const int K  = cfg_.moe_top_k;
            const float* aux_frac = nullptr;
            float aux_scale_grad = 0.0f;
            if (cfg_.moe_aux_scale > 0.0f) {
                aux_total += moe_layer_aux(dev, act.saved_moe_probs[sl].f32(),
                                           act.saved_moe_idx[sl].i32p(),
                                           act.moe_auxfrac.f32(),
                                           l, N, K, ne,
                                           want_aux_stats, aux_accum);
                aux_frac = act.moe_auxfrac.f32();
                aux_scale_grad = cfg_.moe_aux_scale * eff_scale * static_cast<float>(ntok) /
                                 static_cast<float>(cfg_.num_layers > 0 ? cfg_.num_layers : 1);
            }
            ops::zero(dev, act.dxb.f32(), N * d);
            if (is_half_or_quant(L.moe_gate.w)) {
                ops::moe_backward_half(dev, act.saved_xb2[sl].f32(), L.router.w.f32(),
                              weight16(dev, L.moe_gate.w, 0), weight16(dev, L.moe_up.w, 1),
                              weight16(dev, L.moe_down.w, 2),
                              L.sh_gate.w.defined() ? weight16(dev, L.sh_gate.w, 3) : nullptr,
                              L.sh_up.w.defined()   ? weight16(dev, L.sh_up.w, 4)   : nullptr,
                              L.sh_down.w.defined() ? weight16(dev, L.sh_down.w, 5) : nullptr,
                              act.saved_moe_probs[sl].f32(),
                              act.saved_moe_idx[sl].i32p(),
                              act.saved_moe_w[sl].f32(),
                              act.saved_gate[sl].f32(), act.saved_up[sl].f32(),
                              act.saved_act[sl].f32(),
                              aux_frac, aux_scale_grad,
                              dy_ffn, act.dxb.f32(),
                              gptr(L.router.g), gptr(L.moe_gate.g),
                              gptr(L.moe_up.g), gptr(L.moe_down.g),
                              L.sh_gate.g.defined() ? L.sh_gate.g.f32() : nullptr,
                              L.sh_up.g.defined()   ? L.sh_up.g.f32()   : nullptr,
                              L.sh_down.g.defined() ? L.sh_down.g.f32() : nullptr,
                              act.moe_dact.f32(),
                              N, d, E, ne, K, cfg_.shared_width());
            } else {
                ops::moe_backward(dev, act.saved_xb2[sl].f32(), L.router.w.f32(),
                              L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                              L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                              L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                              L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                              act.saved_moe_probs[sl].f32(),
                              act.saved_moe_idx[sl].i32p(),
                              act.saved_moe_w[sl].f32(),
                              act.saved_gate[sl].f32(), act.saved_up[sl].f32(),
                              act.saved_act[sl].f32(),
                              aux_frac, aux_scale_grad,
                              dy_ffn, act.dxb.f32(),
                              gptr(L.router.g), gptr(L.moe_gate.g),
                              gptr(L.moe_up.g), gptr(L.moe_down.g),
                              L.sh_gate.g.defined() ? L.sh_gate.g.f32() : nullptr,
                              L.sh_up.g.defined()   ? L.sh_up.g.f32()   : nullptr,
                              L.sh_down.g.defined() ? L.sh_down.g.f32() : nullptr,
                              act.moe_dact.f32(),
                              N, d, E, ne, K, cfg_.shared_width());
            }
        } else {
            ops::zero(dev, act.dact.f32(), N * F);
            if (is_half_or_quant(L.w_down.w))
                ops::linear_backward_half(dev, act.saved_act[sl].f32(), weight16(dev, L.w_down.w, 0), dy_ffn,
                                          act.dact.f32(), gptr(L.w_down.g), static_cast<int>(N), F, d);
            else
                ops::linear_backward(dev, act.saved_act[sl].f32(), L.w_down.w.f32(), dy_ffn,
                                     act.dact.f32(), gptr(L.w_down.g), static_cast<int>(N), F, d);
            lora_backward(L.lora_down, dev, act.saved_act[sl].f32(), dy_ffn, act.dact.f32(), static_cast<int>(N));

            ops::zero(dev, act.dgate.f32(), N * F);
            ops::zero(dev, act.dup.f32(),   N * F);
            ops::swiglu_backward(dev, act.saved_gate[sl].f32(), act.saved_up[sl].f32(),
                                 act.dact.f32(), act.dgate.f32(), act.dup.f32(), N * F);

            ops::zero(dev, act.dxb.f32(), N * d);
            if (is_half_or_quant(L.w_gate.w)) {
                ops::linear_backward_half(dev, act.saved_xb2[sl].f32(), weight16(dev, L.w_gate.w, 0), act.dgate.f32(),
                                          act.dxb.f32(), gptr(L.w_gate.g), static_cast<int>(N), d, F);
                ops::linear_backward_half(dev, act.saved_xb2[sl].f32(), weight16(dev, L.w_up.w, 1), act.dup.f32(),
                                          act.dxb.f32(), gptr(L.w_up.g), static_cast<int>(N), d, F);
            } else {
                ops::linear_backward(dev, act.saved_xb2[sl].f32(), L.w_gate.w.f32(), act.dgate.f32(),
                                     act.dxb.f32(), gptr(L.w_gate.g), static_cast<int>(N), d, F);
                ops::linear_backward(dev, act.saved_xb2[sl].f32(), L.w_up.w.f32(), act.dup.f32(),
                                     act.dxb.f32(), gptr(L.w_up.g), static_cast<int>(N), d, F);
            }
            lora_backward(L.lora_gate, dev, act.saved_xb2[sl].f32(), act.dgate.f32(), act.dxb.f32(), static_cast<int>(N));
            lora_backward(L.lora_up, dev, act.saved_xb2[sl].f32(), act.dup.f32(), act.dxb.f32(), static_cast<int>(N));
        }

        ops::rmsnorm_backward(dev, act.saved_xmid[sl].f32(), L.ffn_norm.w.f32(), act.dxb.f32(),
                              act.saved_rrms2[sl].f32(), act.dx.f32(), gptr(L.ffn_norm.g), N, d);

        const bool is_granite    = (cfg_.arch == ModelConfig::ArchGranite);
        const int  granite_kind  = (is_granite && sl < cfg_.layer_types.size())
                                       ? cfg_.layer_types[sl] : 1;
        const bool granite_mamba = is_granite && L.has_mamba() && granite_kind == 0;

        if (granite_mamba) {
            const float* dy_mamba = act.dx.f32();
            if (cfg_.residual_multiplier != 1.0f) {
                ops::copy(dev, act.dtmp.f32(), act.dx.f32(), N * d);
                ops::scale_inplace(dev, act.dtmp.f32(), cfg_.residual_multiplier, N * d);
                dy_mamba = act.dtmp.f32();
            }

            ops::zero(dev, act.dxb.f32(), N * d);
            {
                const int d_inner     = cfg_.mamba_expand * d;
                const int n_heads     = cfg_.mamba_n_heads;
                const int d_head      = cfg_.mamba_d_head;
                const int d_state     = cfg_.mamba_d_state;
                const int n_groups    = cfg_.mamba_n_groups;
                const int d_state_tot = n_groups * d_state;
                const int conv_dim    = d_inner + 2 * d_state_tot;
                if (is_half_or_quant(L.mamba_in_proj.w)) {
                    ops::mamba2_backward_half(dev,
                                     act.saved_xb[sl].f32(),
                                     weight16(dev, L.mamba_in_proj.w, 0),
                                     L.mamba_conv_w.w.f32(),
                                     L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                                     L.mamba_dt_bias.w.f32(),
                                     L.mamba_A_log.w.f32(),
                                     L.mamba_D.w.f32(),
                                     L.mamba_norm.w.f32(),
                                     weight16(dev, L.mamba_out_proj.w, 1),
                                     dy_mamba,
                                     act.dxb.f32(),
                                     L.mamba_in_proj.g.defined() ? L.mamba_in_proj.g.f32() : nullptr,
                                     L.mamba_conv_w.g.defined() ? L.mamba_conv_w.g.f32() : nullptr,
                                     L.mamba_conv_b.w.defined() && L.mamba_conv_b.g.defined() ? L.mamba_conv_b.g.f32() : nullptr,
                                     L.mamba_dt_bias.g.defined() ? L.mamba_dt_bias.g.f32() : nullptr,
                                     L.mamba_A_log.g.defined() ? L.mamba_A_log.g.f32() : nullptr,
                                     L.mamba_D.g.defined() ? L.mamba_D.g.f32() : nullptr,
                                     L.mamba_norm.g.defined() ? L.mamba_norm.g.f32() : nullptr,
                                     L.mamba_out_proj.g.defined() ? L.mamba_out_proj.g.f32() : nullptr,
                                     nullptr,
                                     B, T, d, d_inner, conv_dim, n_heads, d_head, d_state,
                                     cfg_.mamba_d_conv, cfg_.rms_eps, segment_ids);
                } else {
                    ops::mamba2_backward(dev,
                                     act.saved_xb[sl].f32(),
                                     L.mamba_in_proj.w.f32(),
                                     L.mamba_conv_w.w.f32(),
                                     L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                                     L.mamba_dt_bias.w.f32(),
                                     L.mamba_A_log.w.f32(),
                                     L.mamba_D.w.f32(),
                                     L.mamba_norm.w.f32(),
                                     L.mamba_out_proj.w.f32(),
                                     dy_mamba,
                                     act.dxb.f32(),
                                     L.mamba_in_proj.g.defined() ? L.mamba_in_proj.g.f32() : nullptr,
                                     L.mamba_conv_w.g.defined() ? L.mamba_conv_w.g.f32() : nullptr,
                                     L.mamba_conv_b.w.defined() && L.mamba_conv_b.g.defined() ? L.mamba_conv_b.g.f32() : nullptr,
                                     L.mamba_dt_bias.g.defined() ? L.mamba_dt_bias.g.f32() : nullptr,
                                     L.mamba_A_log.g.defined() ? L.mamba_A_log.g.f32() : nullptr,
                                     L.mamba_D.g.defined() ? L.mamba_D.g.f32() : nullptr,
                                     L.mamba_norm.g.defined() ? L.mamba_norm.g.f32() : nullptr,
                                     L.mamba_out_proj.g.defined() ? L.mamba_out_proj.g.f32() : nullptr,
                                     nullptr,
                                     B, T, d, d_inner, conv_dim, n_heads, d_head, d_state,
                                     cfg_.mamba_d_conv, cfg_.rms_eps, segment_ids);
                }
            }

            ops::rmsnorm_backward(dev, act.saved_x[sl].f32(), L.attn_norm.w.f32(), act.dxb.f32(),
                                  act.saved_rrms1[sl].f32(), act.dx.f32(), gptr(L.attn_norm.g), N, d);
        } else {

        ops::zero(dev, act.dattout.f32(), N * qd);
        const float* dy_attn = act.dx.f32();
        if (cfg_.residual_multiplier != 1.0f) {
            ops::copy(dev, act.dtmp.f32(), act.dx.f32(), N * d);
            ops::scale_inplace(dev, act.dtmp.f32(), cfg_.residual_multiplier, N * d);
            dy_attn = act.dtmp.f32();
        }
        if (is_half_or_quant(L.wo.w))
            ops::linear_backward_half(dev, act.saved_attout[sl].f32(), weight16(dev, L.wo.w, 0), dy_attn,
                                      act.dattout.f32(), gptr(L.wo.g), static_cast<int>(N), qd, d);
        else
            ops::linear_backward(dev, act.saved_attout[sl].f32(), L.wo.w.f32(), dy_attn,
                                 act.dattout.f32(), gptr(L.wo.g), static_cast<int>(N), qd, d);
        lora_backward(L.lora_o, dev, act.saved_attout[sl].f32(), dy_attn, act.dattout.f32(), static_cast<int>(N));

        ops::zero(dev, act.dq.f32(), N * qd);
        ops::zero(dev, act.dk.f32(), N * kvd);
        ops::zero(dev, act.dv.f32(), N * kvd);
        ops::attention_forward_ex(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                  act.saved_v[sl].f32(), act.att_out.f32(),
                                  act.attn_probs_tmp.f32(),
                                  B, T, H, KV, hd, scale, cfg_.sliding_window,
                                  segment_ids);
        ops::attention_backward_ex(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                   act.saved_v[sl].f32(), act.attn_probs_tmp.f32(),
                                   act.dattout.f32(),
                                   act.dq.f32(), act.dk.f32(), act.dv.f32(),
                                   B, T, H, KV, hd, scale, cfg_.sliding_window);

        if (cfg_.position_embedding != 1)
            ops::rope_backward_cached(dev, act.dq.f32(), act.dk.f32(), act.pos.i32p(), rope_freq_bwd,
                                      N, H, KV, hd, cfg_.rope_type);

        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined() && L.qk_knorm.w.defined()) {
            ops::copy(dev, act.att_out.f32(), act.dq.f32(), N * qd);
            ops::zero(dev, act.dq.f32(), N * qd);
            ops::rmsnorm_backward(dev, act.saved_qk_raw_q[sl].f32(), L.qk_qnorm.w.f32(), act.att_out.f32(),
                                  act.saved_qk_rms_q[sl].f32(), act.dq.f32(), gptr(L.qk_qnorm.g),
                                  N * H, hd);
            ops::copy(dev, act.dattout.f32(), act.dk.f32(), N * kvd);
            ops::zero(dev, act.dk.f32(), N * kvd);
            ops::rmsnorm_backward(dev, act.saved_qk_raw_k[sl].f32(), L.qk_knorm.w.f32(), act.dattout.f32(),
                                  act.saved_qk_rms_k[sl].f32(), act.dk.f32(), gptr(L.qk_knorm.g),
                                  N * KV, hd);
        }

        if (L.has_qkv_bias()) {
            ops::bias_backward(dev, act.dq.f32(), gptr(L.bq.g), N, qd);
            ops::bias_backward(dev, act.dk.f32(), gptr(L.bk.g), N, kvd);
            ops::bias_backward(dev, act.dv.f32(), gptr(L.bv.g), N, kvd);
        }

        ops::zero(dev, act.dxb.f32(), N * d);
        lora_backward(L.lora_q, dev, act.saved_xb[sl].f32(), act.dq.f32(), act.dxb.f32(), static_cast<int>(N));
        lora_backward(L.lora_k, dev, act.saved_xb[sl].f32(), act.dk.f32(), act.dxb.f32(), static_cast<int>(N));
        lora_backward(L.lora_v, dev, act.saved_xb[sl].f32(), act.dv.f32(), act.dxb.f32(), static_cast<int>(N));
        if (is_half_or_quant(L.wq.w)) {
            ops::linear_backward_half(dev, act.saved_xb[sl].f32(), weight16(dev, L.wq.w, 0), act.dq.f32(),
                                      act.dxb.f32(), gptr(L.wq.g), static_cast<int>(N), d, qd);
            ops::linear_backward_half(dev, act.saved_xb[sl].f32(), weight16(dev, L.wk.w, 1), act.dk.f32(),
                                      act.dxb.f32(), gptr(L.wk.g), static_cast<int>(N), d, kvd);
            ops::linear_backward_half(dev, act.saved_xb[sl].f32(), weight16(dev, L.wv.w, 2), act.dv.f32(),
                                      act.dxb.f32(), gptr(L.wv.g), static_cast<int>(N), d, kvd);
        } else {
            ops::linear_backward(dev, act.saved_xb[sl].f32(), L.wq.w.f32(), act.dq.f32(),
                                 act.dxb.f32(), gptr(L.wq.g), static_cast<int>(N), d, qd);
            ops::linear_backward(dev, act.saved_xb[sl].f32(), L.wk.w.f32(), act.dk.f32(),
                                 act.dxb.f32(), gptr(L.wk.g), static_cast<int>(N), d, kvd);
            ops::linear_backward(dev, act.saved_xb[sl].f32(), L.wv.w.f32(), act.dv.f32(),
                                 act.dxb.f32(), gptr(L.wv.g), static_cast<int>(N), d, kvd);
        }

        ops::rmsnorm_backward(dev, act.saved_x[sl].f32(), L.attn_norm.w.f32(), act.dxb.f32(),
                              act.saved_rrms1[sl].f32(), act.dx.f32(), gptr(L.attn_norm.g), N, d);
        }
    }

    if (embed_override && n_override > 0 && override_pos >= 0 &&
        static_cast<i64>(override_pos) + n_override <= N) {
        if (out_override_dx) {
            device_copy(out_override_dx, dev,
                        act.dx.f32() + static_cast<size_t>(override_pos) * d, dev,
                        sizeof(float) * static_cast<size_t>(n_override) * d);
        }
        ops::zero(dev, act.dx.f32() + static_cast<size_t>(override_pos) * d,
                  static_cast<i64>(n_override) * d);
    }
    if (cfg_.embedding_multiplier != 1.0f) {
        ops::scale_inplace(dev, act.dx.f32(), cfg_.embedding_multiplier, N * d);
    }
    ops::embedding_backward(dev, ids, act.dx.f32(), gptr(tok_emb_.g), N, d, V);

    double loss = loss_sum / static_cast<double>(ntok);
    if (use_dev_aux) aux_total = ops::moe_aux_end(dev);
    if (cfg_.moe_aux_free) moe_aux_loss(act, B, T);
    if (cfg_.use_moe && cfg_.moe_aux_scale > 0.0f && cfg_.num_layers > 0)
        loss += cfg_.moe_aux_scale * aux_total / static_cast<double>(cfg_.num_layers);
    return loss;
}

double Model::moe_aux_loss(Activations& act, int B, int T) {
    if (!cfg_.use_moe) return 0.0;
    if (cfg_.moe_aux_free) {
        accumulate_moe_bias_fracs(act, B, T);
        return 0.0;
    }
    GAI_CHECK(act.with_grad, "moe_aux_loss needs training Activations (with_grad)");
    GAI_CHECK(static_cast<int>(act.saved_moe_probs.size()) == cfg_.num_layers,
              "moe_aux_loss: stale Activations");
    GAI_CHECK(act.moe_auxfrac.defined(), "moe_aux_loss: missing auxfrac buffer");
    const i64 N = static_cast<i64>(B) * T;
    double total = 0.0;
    for (int l = 0; l < cfg_.num_layers; ++l) {
        size_t sl = static_cast<size_t>(l);
        total += moe_layer_aux(device_, act.saved_moe_probs[sl].f32(),
                               act.saved_moe_idx[sl].i32p(),
                               act.moe_auxfrac.f32(), l,
                               N, cfg_.moe_top_k, cfg_.num_experts);
    }
    return total;
}

static constexpr u32 RAW_MAGIC = 0x57415247u;
static constexpr u32 RAW_MAGIC2 = 0x32515247u;
static constexpr u32 dtype_code(DType d) {
    switch (d) {
        case DType::F16:  return 1u;
        case DType::Q8_0: return 2u;
        case DType::Q4_0: return 3u;
        case DType::Q4_1: return 4u;
        default:          return 0u;
    }
}
static DType dtype_from_code(u32 c) {
    switch (c) {
        case 1u: return DType::F16;
        case 2u: return DType::Q8_0;
        case 3u: return DType::Q4_0;
        case 4u: return DType::Q4_1;
        default: return DType::F32;
    }
}

void Model::save_raw(const std::string& path) const {
    namespace fs = std::filesystem;
    const fs::path target(path);
    fs::path parent = target.parent_path();
    if (parent.empty()) parent = ".";
    const fs::path tmp = parent / ("." + target.filename().string() + ".tmp");
    std::error_code ec;
    fs::remove(tmp, ec);
    ec.clear();
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        GAI_CHECK(f.good(), "cannot write model: " + path);
        u32 magic = RAW_MAGIC2;
        f.write(reinterpret_cast<const char*>(&magic), 4);
        u32 n = static_cast<u32>(params_.size());
        f.write(reinterpret_cast<const char*>(&n), 4);
        for (const Parameter* p : params_) {
            u32 len = static_cast<u32>(p->name.size());
            f.write(reinterpret_cast<const char*>(&len), 4);
            f.write(p->name.data(), len);
            u64 ne = static_cast<u64>(p->numel());
            f.write(reinterpret_cast<const char*>(&ne), 8);
            u32 dt = dtype_code(p->w.dtype());
            f.write(reinterpret_cast<const char*>(&dt), 4);
            Tensor cpu = p->w.to(Device::CPU);
            f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()));
        }
        GAI_CHECK(f.good(), "model write failed");
    }
    fs::rename(tmp, target, ec);
    GAI_CHECK(!ec, "cannot finalise model write: " + ec.message());
}

bool Model::load_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    u32 magic = 0, n = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), 4)) return false;
    if (magic != RAW_MAGIC && magic != RAW_MAGIC2) return false;
    const bool has_dtype = (magic == RAW_MAGIC2);
    if (!f.read(reinterpret_cast<char*>(&n), 4)) return false;
    if (n != static_cast<u32>(params_.size())) return false;
    std::unordered_set<std::string> seen;
    for (u32 i = 0; i < n; ++i) {
        u32 len = 0;
        if (!f.read(reinterpret_cast<char*>(&len), 4) || len == 0 || len > 512) return false;
        std::string name(len, '\0');
        if (!f.read(name.data(), static_cast<std::streamsize>(len))) return false;
        if (!seen.insert(name).second) return false;
        u64 ne = 0;
        if (!f.read(reinterpret_cast<char*>(&ne), 8)) return false;
        u32 dt = 0;
        if (has_dtype && !f.read(reinterpret_cast<char*>(&dt), 4)) return false;
        Parameter* p = find_parameter(name);
        if (!p || static_cast<u64>(p->numel()) != ne) return false;
        if (p->shape.empty()) return false;
        if (has_dtype && dt > 4u) return false;
        DType file_dt = has_dtype ? dtype_from_code(dt) : DType::F32;
        Tensor cpu(p->shape, file_dt, Device::CPU);
        if (cpu.nbytes() == 0) return false;
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        const bool target_quant = p->w.dtype() == DType::Q4_0 ||
                                  p->w.dtype() == DType::Q4_1 ||
                                  p->w.dtype() == DType::Q8_0;
        if (file_dt == p->w.dtype()) {
            p->w.copy_from(cpu);
        } else if (file_dt == DType::F32) {
            if (target_quant) p->w.copy_from(quant::quantize(cpu, p->w.dtype()));
            else p->w.copy_from_f32(cpu);
        } else {
            Tensor f32 = quant::dequantize(cpu, DType::F32);
            if (target_quant) p->w.copy_from(quant::quantize(f32, p->w.dtype()));
            else p->w.copy_from_f32(f32);
        }
    }
    if (fp16_weight_cache_) enable_fp16_weight_cache(true);
    return true;
}

}
