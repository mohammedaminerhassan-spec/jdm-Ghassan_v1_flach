#include "inference/generator.h"
#include "core/ops.h"
#include "core/quant.h"
#include "core/device.h"
#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

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

}

std::string GenerationStats::summary() const {
    return strfmt("prompt %d tok in %s (%.1f tok/s) | generated %d tok in %s (%.1f tok/s)",
                  prompt_tokens, human_duration(prefill_sec).c_str(), prefill_tps(),
                  new_tokens, human_duration(decode_sec).c_str(), decode_tps());
}

Generator::Generator(Model& model, const Tokenizer& tok, int max_context)
    : model_(model), tok_(tok) {
    const ModelConfig& c = model.config();
    Device dev = model.device();
    const int default_ctx = std::min(c.max_seq_len, 4096);
    max_context_ = max_context > 0 ? std::min(max_context, c.max_seq_len) : default_ctx;
    expected_device_ = dev;

    {
        int kv_layers = c.num_layers;
        if (c.arch == ModelConfig::ArchGranite && !c.layer_types.empty()) {
            kv_layers = 0;
            for (int lt : c.layer_types) if (lt != 0) ++kv_layers;
        }
        const size_t kv_elem_bytes = sizeof(u16);
        size_t need = 2ULL * static_cast<size_t>(kv_layers) *
                      static_cast<size_t>(max_context_) *
                      static_cast<size_t>(c.kv_dim()) * kv_elem_bytes;
        if (dev == Device::CUDA && cuda_available()) {

#ifdef GAI_CUDA
            size_t free_b = cuda::free_bytes_live();
#else
            size_t free_b = device_info().free_mem;
#endif
            if (free_b == 0) free_b = device_info().free_mem;
            if (free_b > 0 && need > free_b) {
                GAI_FAIL(strfmt("KV cache needs %s but only %s free on device "
                                "(kv_layers=%d ctx=%d kv_dim=%d). "
                                "Pass --ctx <smaller value> to reduce the cache.",
                                human_bytes(need).c_str(), human_bytes(free_b).c_str(),
                                kv_layers, max_context_, c.kv_dim()));
            }
        } else if (need > (1ULL << 30)) {
            log_warn(strfmt("[gen ] KV cache large (%s); pass --ctx to cap it",
                            human_bytes(need).c_str()));
        }
    }
    cache_ = KVCache(c, max_context_, dev);

    if (c.sliding_window == 0)
        cache_.set_pinned_prefix(std::max(0, std::min(8, max_context_ - 1)));
    else
        cache_.set_pinned_prefix(0);

    const int d = c.hidden_size;
    x_          = Tensor::empty({d}, DType::F32, dev);
    xb_         = Tensor::empty({d}, DType::F32, dev);
    q_          = Tensor::empty({c.q_dim()}, DType::F32, dev);
    k_          = Tensor::empty({c.kv_dim()}, DType::F32, dev);
    v_          = Tensor::empty({c.kv_dim()}, DType::F32, dev);
    if (model_.fp16_weight_cache_enabled()) {
        qkv_ = Tensor::empty({c.q_dim() + 2 * c.kv_dim()}, DType::F32, dev);
    }
    attn_       = Tensor::empty({c.q_dim()}, DType::F32, dev);
    proj_       = Tensor::empty({d}, DType::F32, dev);
    gate_       = Tensor::empty({c.intermediate_size}, DType::F32, dev);
    up_         = Tensor::empty({c.intermediate_size}, DType::F32, dev);
    act_        = Tensor::empty({c.intermediate_size}, DType::F32, dev);
    if (c.use_moe) {
        const i64 nke = static_cast<i64>(c.moe_top_k) * c.moe_expert_dim;
        moe_gate_ = Tensor::empty({nke}, DType::F32, dev);
        moe_up_   = Tensor::empty({nke}, DType::F32, dev);
        moe_act_  = Tensor::empty({nke}, DType::F32, dev);
    }
    logits_dev_ = Tensor::empty({c.vocab_size}, DType::F32, dev);
    scores_     = Tensor::empty({static_cast<i64>(c.num_heads) * max_context_}, DType::F32, dev);
    tok_dev_    = Tensor::empty({1}, DType::I32, dev);
    pos_dev_    = Tensor::empty({1}, DType::I32, dev);
    hlast_      = Tensor::empty({d}, DType::F32, dev);
    topk_vals_dev_ = Tensor::empty({(i64)FAST_TOPK_MAX}, DType::F32, dev);
    topk_ids_dev_  = Tensor::empty({(i64)FAST_TOPK_MAX}, DType::I32, dev);
    topk_vals_host_.assign(FAST_TOPK_MAX, 0.0f);
    topk_ids_host_.assign(FAST_TOPK_MAX, 0);
    logits_host_.assign(static_cast<size_t>(c.vocab_size), 0.0f);

    {
        const int d_inner  = c.mamba_expand * c.hidden_size;
        const int conv_dim = d_inner + 2 * c.mamba_n_groups * c.mamba_d_state;
        const int conv_k   = c.mamba_d_conv;
        const int conv_win = conv_k > 1 ? conv_k - 1 : 1;
        const int proj     = d_inner + conv_dim + c.mamba_n_heads;
        auto stage = [](const Tensor& t) -> std::vector<float> {
            std::vector<float> out;
            if (!t.defined()) return out;
            out.resize(static_cast<size_t>(t.numel()));
            if (t.device() == Device::CPU) {
                std::memcpy(out.data(), t.f32(), out.size() * sizeof(float));
            } else {
                device_copy(out.data(), Device::CPU, t.f32(), t.device(),
                            out.size() * sizeof(float));
            }
            return out;
        };
        mamba_states_.clear();
        mamba_host_conv_w_.clear();
        mamba_host_conv_b_.clear();
        mamba_host_dt_bias_.clear();
        mamba_host_a_log_.clear();
        mamba_host_D_.clear();
        mamba_host_norm_.clear();
        const bool has_mamba =
            c.arch == ModelConfig::ArchGranite &&
            std::any_of(c.layer_types.begin(), c.layer_types.end(),
                        [](int k) { return k == 0; });
        if (has_mamba) {
            mamba_proj_dev_   = Tensor::empty({(i64)proj}, DType::F32, dev);
            mamba_norm_dev_   = Tensor::empty({(i64)d_inner}, DType::F32, dev);
            mamba_proj_host_.assign(static_cast<size_t>(proj), 0.0f);
            mamba_normed_host_.assign(static_cast<size_t>(d_inner), 0.0f);
            mamba_bc_out_host_.assign(static_cast<size_t>(conv_dim), 0.0f);
            mamba_y_host_.assign(static_cast<size_t>(d_inner), 0.0f);
        }
        kv_stage_k_.assign(static_cast<size_t>(c.kv_dim()), 0.0f);
        kv_stage_v_.assign(static_cast<size_t>(c.kv_dim()), 0.0f);
        for (int l = 0; l < c.num_layers; ++l) {
            const bool is_mamba = (c.arch == ModelConfig::ArchGranite) &&
                                  l < static_cast<int>(c.layer_types.size()) &&
                                  c.layer_types[static_cast<size_t>(l)] == 0;
            if (!is_mamba) continue;
            const LayerParams& L = model_.layer(l);
            MambaState st;
            const bool cuda_step_native = (dev == Device::CUDA && c.mamba_n_groups == 1);
            if (dev == Device::CPU || !cuda_step_native) {
                // n_groups != 1 currently uses the CPU recurrent-step fallback even
                // after a CUDA prefill, so it needs a synchronized host state mirror.
                st.conv_host.assign(static_cast<size_t>(conv_dim) * conv_win, 0.0f);
                st.ssm_host.assign(static_cast<size_t>(c.mamba_n_heads) *
                                   c.mamba_d_head * c.mamba_d_state, 0.0f);
            }
            if (dev == Device::CUDA) {
                st.conv_dev = Tensor::zeros({static_cast<i64>(conv_dim) * conv_win},
                                            DType::F32, dev);
                st.ssm_dev = Tensor::zeros({static_cast<i64>(c.mamba_n_heads) *
                                            c.mamba_d_head * c.mamba_d_state},
                                           DType::F32, dev);
            }

            // The native CUDA decode-step kernel currently handles n_groups == 1.
            // Preserve host mirrors for CPU and the supported CUDA fallback geometry.
            if (dev == Device::CPU || !cuda_step_native) {
                mamba_host_conv_w_.push_back(stage(L.mamba_conv_w.w));
                mamba_host_conv_b_.push_back(stage(L.mamba_conv_b.w));
                mamba_host_dt_bias_.push_back(stage(L.mamba_dt_bias.w));
                mamba_host_a_log_.push_back(stage(L.mamba_A_log.w));
                mamba_host_D_.push_back(stage(L.mamba_D.w));
                mamba_host_norm_.push_back(stage(L.mamba_norm.w));
            } else {
                mamba_host_conv_w_.emplace_back();
                mamba_host_conv_b_.emplace_back();
                mamba_host_dt_bias_.emplace_back();
                mamba_host_a_log_.emplace_back();
                mamba_host_D_.emplace_back();
                mamba_host_norm_.emplace_back();
            }
            mamba_states_.push_back(std::move(st));
        }
        mamba_states_ready_ = !mamba_states_.empty();
    }

    log_info(strfmt("[gen ] context %d | kv cache %s | device %s%s",
                    max_context_, human_bytes(cache_.bytes()).c_str(), device_name(dev),
                    mamba_states_ready_
                        ? strfmt(" | mamba states %zu layer(s)", mamba_states_.size()).c_str()
                        : ""));
}

void Generator::reset() {
    cache_.reset();
    history_.clear();
    stats_ = GenerationStats{};
    absolute_pos_ = 0;
    for (MambaState& st : mamba_states_) {
        st.conv_host.assign(st.conv_host.size(), 0.0f);
        st.ssm_host.assign(st.ssm_host.size(), 0.0f);
        if (st.conv_dev.defined()) st.conv_dev.zero_();
        if (st.ssm_dev.defined())  st.ssm_dev.zero_();
    }
}

float* Generator::decode_step_logits(i32 token, int position,
                                     const float* token_embed) {
    GAI_CHECK(position >= 0 && (i64)position <= (i64)std::numeric_limits<int>::max(),
              "decode_step_logits: position out of int range");
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");
    const ModelConfig& c = model_.config();
    const int d   = c.hidden_size;
    const int qd  = c.q_dim();
    const int kvd = c.kv_dim();
    const int F   = c.intermediate_size;
    const int V   = c.vocab_size;
    const int H   = c.num_heads;
    const int KV  = c.num_kv_heads;
    const int hd  = c.head_dim();
    const float scale = (c.attention_multiplier > 0.0f)
                            ? c.attention_multiplier
                            : (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale(c);
    Device dev = model_.device();

    device_copy(tok_dev_.data_ptr(), dev, &token, Device::CPU, sizeof(i32));
    if (is_half_or_quant(model_.tok_embeddings().w))
        ops::embedding_forward_half(dev, tok_dev_.i32p(), weight16(dev, model_.tok_embeddings().w, 0),
                                    x_.f32(), 1, d, V);
    else
        ops::embedding_forward(dev, tok_dev_.i32p(), model_.tok_embeddings().w.f32(), x_.f32(), 1, d, V);
    if (token_embed)
        device_copy(x_.data_ptr(), dev, token_embed, Device::CPU,
                    sizeof(float) * static_cast<size_t>(d));
    if (c.embedding_multiplier != 1.0f)
        ops::scale_inplace(dev, x_.f32(), c.embedding_multiplier, d);

    const int slot = cache_.allocate_slot();

    for (int l = 0; l < c.num_layers; ++l) {
        LayerParams& L = model_.layer(l);

        ops::rmsnorm_forward(dev, x_.f32(), L.attn_norm.w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);

        const bool is_mamba_layer =
            (c.arch == ModelConfig::ArchGranite) &&
            l < static_cast<int>(c.layer_types.size()) &&
            c.layer_types[static_cast<size_t>(l)] == 0;
        if (is_mamba_layer) {
            decode_mamba_layer(l, xb_.f32(), proj_.f32());
            if (c.residual_multiplier != 1.0f)
                ops::scale_inplace(dev, proj_.f32(), c.residual_multiplier, d);
            ops::add_inplace(dev, x_.f32(), proj_.f32(), d);

            ops::rmsnorm_forward(dev, x_.f32(), L.ffn_norm.w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);
            if (L.has_moe()) {
                const float* bias = model_.moe_bias_ptr(l);
                if (is_half_or_quant(L.moe_gate.w)) {
                    const WeightRef mg = weight_ref(dev, L.moe_gate.w, 0);
                    const WeightRef mu = weight_ref(dev, L.moe_up.w, 1);
                    const WeightRef md = weight_ref(dev, L.moe_down.w, 2);
                    const WeightRef msg = L.sh_gate.w.defined() ? weight_ref(dev, L.sh_gate.w, 3) : WeightRef{};
                    const WeightRef msu = L.sh_up.w.defined()   ? weight_ref(dev, L.sh_up.w, 4)   : WeightRef{};
                    const WeightRef msd = L.sh_down.w.defined() ? weight_ref(dev, L.sh_down.w, 5) : WeightRef{};
                    ops::moe_forward_bias_half(dev, xb_.f32(), L.router.w.f32(), bias,
                                               mg.ptr, mu.ptr, md.ptr, msg.ptr, msu.ptr, msd.ptr,
                                               proj_.f32(), nullptr, nullptr, nullptr,
                                               moe_gate_.f32(), moe_up_.f32(), moe_act_.f32(),
                                               1, d, c.moe_expert_dim, c.num_experts, c.moe_top_k,
                                               c.shared_width(), mg.qkind);
                } else {
                    ops::moe_forward_bias(dev, xb_.f32(), L.router.w.f32(), bias,
                                          L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                          L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                          L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                          L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                          proj_.f32(), nullptr, nullptr, nullptr,
                                          moe_gate_.f32(), moe_up_.f32(), moe_act_.f32(),
                                          1, d, c.moe_expert_dim, c.num_experts, c.moe_top_k,
                                          c.shared_width());
                }
            } else {
                linear_param(dev, L.w_gate.w, xb_.f32(), gate_.f32(), 1, d, F);
                linear_param(dev, L.w_up.w, xb_.f32(),   up_.f32(),   1, d, F);
                ops::swiglu_forward(dev, gate_.f32(), up_.f32(), act_.f32(), F);
                linear_param(dev, L.w_down.w, act_.f32(), proj_.f32(), 1, F, d);
            }
            if (c.residual_multiplier != 1.0f)
                ops::scale_inplace(dev, proj_.f32(), c.residual_multiplier, d);
            ops::add_inplace(dev, x_.f32(), proj_.f32(), d);
            continue;
        }

        float* qp = q_.f32();
        float* kp = k_.f32();
        float* vp = v_.f32();
        if (qkv_.defined() && model_.fused_qkv_ptr(l) != nullptr) {
            ops::linear_forward_fp16(dev, xb_.f32(), model_.fused_qkv_ptr(l),
                                     qkv_.f32(), 1, d, qd + 2 * kvd);
            qp = qkv_.f32();
            kp = qp + qd;
            vp = kp + kvd;
        } else {
            linear_param(dev, L.wq.w, xb_.f32(), qp, 1, d, qd);
            linear_param(dev, L.wk.w, xb_.f32(), kp, 1, d, kvd);
            linear_param(dev, L.wv.w, xb_.f32(), vp, 1, d, kvd);
        }

        if (L.has_qkv_bias()) {
            ops::add_bias(dev, L.bq.w.f32(), qp, 1, qd);
            ops::add_bias(dev, L.bk.w.f32(), kp, 1, kvd);
            ops::add_bias(dev, L.bv.w.f32(), vp, 1, kvd);
        }

        if (c.use_qk_norm && L.qk_qnorm.w.defined()) {
            ops::rmsnorm_forward(dev, qp, L.qk_qnorm.w.f32(), qp,
                                 nullptr, H, hd, c.rms_eps);
            ops::rmsnorm_forward(dev, kp, L.qk_knorm.w.f32(), kp,
                                 nullptr, KV, hd, c.rms_eps);
        }

        const float* rope_freq = model_.rope_inv_freq_ptr();
        GAI_CHECK(rope_freq != nullptr, "decode: RoPE frequency cache is missing");

        if (c.position_embedding != 1) {
            if (l == 0)
                device_copy(pos_dev_.data_ptr(), dev, &position, Device::CPU, sizeof(i32));
            ops::rope_forward_cached(dev, qp, kp, pos_dev_.i32p(), rope_freq, 1, H, KV, hd,
                                     c.rope_type);
        }

        {
            if (cache_.fp16()) {
                u16* kdst = cache_.k_f16(l) + static_cast<size_t>(slot) * kvd;
                u16* vdst = cache_.v_f16(l) + static_cast<size_t>(slot) * kvd;
                ops::convert_f32_to_f16(dev, kp, kdst, kvd);
                ops::convert_f32_to_f16(dev, vp, vdst, kvd);
            } else {
                size_t bytes = sizeof(float) * static_cast<size_t>(kvd);
                char* kdst = reinterpret_cast<char*>(cache_.k(l)) + static_cast<size_t>(slot) * bytes;
                char* vdst = reinterpret_cast<char*>(cache_.v(l)) + static_cast<size_t>(slot) * bytes;
                device_copy(kdst, dev, kp, dev, bytes);
                device_copy(vdst, dev, vp, dev, bytes);
            }
        }

        {
            const int pinned = std::min(cache_.pinned_prefix(), cache_.length());
            const int win = (pinned > 0) ? 0 : c.sliding_window;
            if (cache_.fp16()) {
                ops::attention_decode_ring_fp16(dev, qp, cache_.k_f16(l), cache_.v_f16(l),
                                                attn_.f32(), H, KV, hd, cache_.ring_start(),
                                                pinned, cache_.length(), cache_.capacity(),
                                                scale, scores_.f32(), win);
            } else {
                ops::attention_decode_ring(dev, qp, cache_.k(l), cache_.v(l), attn_.f32(),
                                           H, KV, hd, cache_.ring_start(), pinned,
                                           cache_.length(), cache_.capacity(), scale, scores_.f32(),
                                           win);
            }
        }

        linear_param(dev, L.wo.w, attn_.f32(), proj_.f32(), 1, qd, d);
        if (c.residual_multiplier != 1.0f)
            ops::scale_inplace(dev, proj_.f32(), c.residual_multiplier, d);
        ops::add_inplace(dev, x_.f32(), proj_.f32(), d);

        ops::rmsnorm_forward(dev, x_.f32(), L.ffn_norm.w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);
        if (L.has_moe()) {
            const float* bias = model_.moe_bias_ptr(l);
            if (is_half_or_quant(L.moe_gate.w)) {
                const WeightRef mg = weight_ref(dev, L.moe_gate.w, 0);
                const WeightRef mu = weight_ref(dev, L.moe_up.w, 1);
                const WeightRef md = weight_ref(dev, L.moe_down.w, 2);
                const WeightRef msg = L.sh_gate.w.defined() ? weight_ref(dev, L.sh_gate.w, 3) : WeightRef{};
                const WeightRef msu = L.sh_up.w.defined()   ? weight_ref(dev, L.sh_up.w, 4)   : WeightRef{};
                const WeightRef msd = L.sh_down.w.defined() ? weight_ref(dev, L.sh_down.w, 5) : WeightRef{};
                ops::moe_forward_bias_half(dev, xb_.f32(), L.router.w.f32(), bias,
                                           mg.ptr, mu.ptr, md.ptr, msg.ptr, msu.ptr, msd.ptr,
                                           proj_.f32(), nullptr, nullptr, nullptr,
                                           moe_gate_.f32(), moe_up_.f32(), moe_act_.f32(),
                                           1, d, c.moe_expert_dim, c.num_experts, c.moe_top_k,
                                           c.shared_width(), mg.qkind);
            } else {
                ops::moe_forward_bias(dev, xb_.f32(), L.router.w.f32(), bias,
                                      L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                      L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                      L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                      L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                      proj_.f32(), nullptr, nullptr, nullptr,
                                      moe_gate_.f32(), moe_up_.f32(), moe_act_.f32(),
                                      1, d, c.moe_expert_dim, c.num_experts, c.moe_top_k,
                                      c.shared_width());
            }
        } else {
            linear_param(dev, L.w_gate.w, xb_.f32(), gate_.f32(), 1, d, F);
            linear_param(dev, L.w_up.w, xb_.f32(),   up_.f32(),   1, d, F);
            ops::swiglu_forward(dev, gate_.f32(), up_.f32(), act_.f32(), F);
            linear_param(dev, L.w_down.w, act_.f32(), proj_.f32(), 1, F, d);
        }
        if (c.residual_multiplier != 1.0f)
            ops::scale_inplace(dev, proj_.f32(), c.residual_multiplier, d);
        ops::add_inplace(dev, x_.f32(), proj_.f32(), d);
    }

    ops::rmsnorm_forward(dev, x_.f32(), model_.final_norm().w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);
    linear_param(dev, model_.lm_head().w, xb_.f32(), logits_dev_.f32(), 1, d, V);
    if (c.logits_scaling != 1.0f)
        ops::scale_inplace(dev, logits_dev_.f32(), 1.0f / c.logits_scaling, V);
    return logits_dev_.f32();
}

void Generator::decode_mamba_layer(int layer_index, const float* x_in, float* out) {
    const ModelConfig& c = model_.config();
    const LayerParams& L = model_.layer(layer_index);
    const Device dev = model_.device();

    const int d        = c.hidden_size;
    const int d_inner  = c.mamba_expand * d;
    const int n_heads  = c.mamba_n_heads;
    const int d_head   = c.mamba_d_head;
    const int d_state  = c.mamba_d_state;
    const int n_groups = c.mamba_n_groups;
    const int conv_k   = c.mamba_d_conv;
    const int conv_dim = d_inner + 2 * n_groups * d_state;
    const int proj     = d_inner + conv_dim + n_heads;
    const int conv_win = conv_k > 1 ? conv_k - 1 : 1;

    size_t si = 0;
    for (int l = 0; l < layer_index && l < static_cast<int>(c.layer_types.size()); ++l)
        if (c.layer_types[static_cast<size_t>(l)] == 0) ++si;
    if (si >= mamba_states_.size()) {
        GAI_FAIL("granite mamba: the recurrent state is missing");
    }
    MambaState& st = mamba_states_[si];

    if (dev == Device::CUDA && st.ssm_dev.defined() && n_groups == 1) {
        if (is_half_or_quant(L.mamba_in_proj.w)) {
            const WeightRef wi = weight_ref(dev, L.mamba_in_proj.w, 0);
            const WeightRef wo = weight_ref(dev, L.mamba_out_proj.w, 1);
            ops::mamba2_step_half(dev, x_in,
                wi.ptr, L.mamba_conv_w.w.f32(),
                L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                L.mamba_dt_bias.w.f32(), L.mamba_A_log.w.f32(), L.mamba_D.w.f32(),
                L.mamba_norm.w.f32(), wo.ptr,
                st.conv_dev.f32(), st.ssm_dev.f32(), out,
                d, d_inner, conv_dim, n_heads, d_head, d_state, conv_k, c.rms_eps, wi.qkind);
        } else {
            ops::mamba2_step(dev, x_in,
                L.mamba_in_proj.w.f32(), L.mamba_conv_w.w.f32(),
                L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                L.mamba_dt_bias.w.f32(), L.mamba_A_log.w.f32(), L.mamba_D.w.f32(),
                L.mamba_norm.w.f32(), L.mamba_out_proj.w.f32(),
                st.conv_dev.f32(), st.ssm_dev.f32(), out,
                d, d_inner, conv_dim, n_heads, d_head, d_state, conv_k, c.rms_eps);
        }
        return;
    }

    const size_t proj_n = static_cast<size_t>(proj);
    linear_param(dev, L.mamba_in_proj.w, x_in, mamba_proj_dev_.f32(), 1, d, proj);
    device_copy(mamba_proj_host_.data(), Device::CPU, mamba_proj_dev_.f32(), dev,
                sizeof(float) * proj_n);
    const float* proj_buf = mamba_proj_host_.data();

    const float* gate = proj_buf;
    const float* bc   = gate + d_inner;
    const float* dt_r = bc + conv_dim;

    const std::vector<float>& conv_w = mamba_host_conv_w_[si];
    const std::vector<float>& conv_b = mamba_host_conv_b_[si];
    float* bc_out = mamba_bc_out_host_.data();
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int ch = 0; ch < conv_dim; ++ch) {
        float acc = conv_b.empty() ? 0.0f : conv_b[static_cast<size_t>(ch)];
        const float* w_ch = conv_w.data() + static_cast<size_t>(ch) * conv_k;
        for (int j = 0; j < conv_k; ++j) {
            const float v = (j == conv_k - 1) ? bc[ch]
                                               : st.conv_host[static_cast<size_t>(ch) * conv_win + j];
            acc += w_ch[j] * v;
        }
        bc_out[static_cast<size_t>(ch)] = acc / (1.0f + std::exp(-acc));
    }
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int ch = 0; ch < conv_dim; ++ch)
        for (int j = 0; j + 1 < conv_win; ++j)
            st.conv_host[static_cast<size_t>(ch) * conv_win + j] =
                st.conv_host[static_cast<size_t>(ch) * conv_win + j + 1];
    if (conv_win > 0)
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int ch = 0; ch < conv_dim; ++ch)
            st.conv_host[static_cast<size_t>(ch) * conv_win + (conv_win - 1)] =
                bc[static_cast<size_t>(ch)];

    const float* xs  = bc_out;
    const float* Bm  = xs + d_inner;
    const float* Cm  = Bm + n_groups * d_state;

    const std::vector<float>& A_log   = mamba_host_a_log_[si];
    const std::vector<float>& dt_bias = mamba_host_dt_bias_[si];
    const std::vector<float>& D       = mamba_host_D_[si];
    float* y = mamba_y_host_.data();
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < n_heads; ++h) {
        const float a = -std::exp(A_log[h]);
        float dtb = dt_r[h] + dt_bias[h];
        const float dt_ = (dtb > 20.0f) ? dtb : std::log1p(std::exp(dtb));
        const float dA = std::exp(a * dt_);
        for (int hh = 0; hh < d_head; ++hh) {
            const int i = h * d_head + hh;
            const float xv = xs[i];
            float* srow = st.ssm_host.data() +
                          ((static_cast<size_t>(h) * d_head + hh) * d_state);
            float acc = 0.0f;
            for (int j = 0; j < d_state; ++j) {
                srow[j] = dA * srow[j] + (Bm[j] * dt_) * xv;
                acc += Cm[j] * srow[j];
            }
            y[static_cast<size_t>(i)] = acc + D[h] * xv;
        }
    }

    const std::vector<float>& norm_w = mamba_host_norm_[si];
    const float* nw = norm_w.data();
    float* normed = mamba_normed_host_.data();
    double var = 0.0;
    for (int i = 0; i < d_inner; ++i) {
        const float g = gate[i] / (1.0f + std::exp(-gate[i]));
        normed[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] * g;
        var += static_cast<double>(normed[static_cast<size_t>(i)]) *
               static_cast<double>(normed[static_cast<size_t>(i)]);
    }
    var /= static_cast<double>(d_inner);
    const float inv = 1.0f / std::sqrt(static_cast<float>(var) + c.rms_eps);
    for (int i = 0; i < d_inner; ++i)
        normed[static_cast<size_t>(i)] = normed[static_cast<size_t>(i)] * inv * nw[i];

    device_copy(mamba_norm_dev_.f32(), dev, normed, Device::CPU,
                sizeof(float) * static_cast<size_t>(d_inner));
    linear_param(dev, L.mamba_out_proj.w, mamba_norm_dev_.f32(), out, 1, d_inner, d);
}

float* Generator::decode_step(i32 token, int position, const float* token_embed) {
    const int V = model_.config().vocab_size;
    float* lp = decode_step_logits(token, position, token_embed);
    device_copy(logits_host_.data(), Device::CPU, lp, model_.device(),
                sizeof(float) * static_cast<size_t>(V));
    return logits_host_.data();
}

void Generator::forward_prefill(const std::vector<i32>& tokens, int start_pos,
                                const float* embed_override,
                                const int* override_pos,
                                int n_override, int position_offset) {
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");

    if (tokens.empty()) return;
    GAI_CHECK(start_pos >= 0 && cache_.length() == start_pos,
              "prefill: start_pos must match the current KV cache length");
    GAI_CHECK(static_cast<i64>(start_pos) + static_cast<i64>(tokens.size()) <=
                  cache_.capacity(),
              "prefill: chunk exceeds the KV cache capacity");
    const ModelConfig& c = model_.config();

    Device dev = model_.device();
    const int P = static_cast<int>(tokens.size());
    const int d = c.hidden_size, qd = c.q_dim(), kvd = c.kv_dim();
    const int H = c.num_heads, KV = c.num_kv_heads, hd = c.head_dim();
    const float scale = (c.attention_multiplier > 0.0f)
                            ? c.attention_multiplier
                            : (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale(c);

    Tensor ids({P}, DType::I32, Device::CPU);
    std::memcpy(ids.data_ptr(), tokens.data(), sizeof(i32) * tokens.size());
    Tensor d_ids({P}, DType::I32, dev);
    d_ids.copy_from(ids);
    Tensor pos({P}, DType::I32, Device::CPU);
    for (int i = 0; i < P; ++i)
        pos.i32p()[i] = position_offset + start_pos + i;
    Tensor d_pos({P}, DType::I32, dev);
    d_pos.copy_from(pos);

    Tensor x({(i64)P * d}, DType::F32, dev);
    Tensor xb({(i64)P * d}, DType::F32, dev);
    Tensor q({(i64)P * qd}, DType::F32, dev);
    Tensor k({(i64)P * kvd}, DType::F32, dev);
    Tensor v({(i64)P * kvd}, DType::F32, dev);
    Tensor qkv;
    if (model_.fp16_weight_cache_enabled()) {
        qkv = Tensor({(i64)P * (qd + 2 * kvd)}, DType::F32, dev);
    }
    Tensor att({(i64)P * qd}, DType::F32, dev);
    Tensor proj({(i64)P * d}, DType::F32, dev);

    const i64 nke = (i64)c.moe_top_k * c.moe_expert_dim;
    Tensor mg({(i64)P * nke}, DType::F32, dev);
    Tensor mu({(i64)P * nke}, DType::F32, dev);
    Tensor ma({(i64)P * nke}, DType::F32, dev);
    Tensor gtmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);
    Tensor utmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);
    Tensor atmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);

    if (is_half_or_quant(model_.tok_embeddings().w))
        ops::embedding_forward_half(dev, d_ids.i32p(), weight16(dev, model_.tok_embeddings().w, 0),
                                    x.f32(), P, d, c.vocab_size);
    else
        ops::embedding_forward(dev, d_ids.i32p(), model_.tok_embeddings().w.f32(),
                               x.f32(), P, d, c.vocab_size);
    if (c.embedding_multiplier != 1.0f)
        ops::scale_inplace(dev, x.f32(), c.embedding_multiplier, (i64)P * d);

    if (embed_override && override_pos && n_override > 0) {
        float* xh = x.f32();
        bool is_contiguous = true;
        for (int i = 0; i < n_override; ++i) {
            if (override_pos[i] != override_pos[0] + i) {
                is_contiguous = false;
                break;
            }
        }
        if (is_contiguous && override_pos[0] >= 0 && override_pos[0] + n_override <= P) {
            device_copy(xh + static_cast<size_t>(override_pos[0]) * d, dev,
                        embed_override, Device::CPU,
                        sizeof(float) * static_cast<size_t>(n_override) * static_cast<size_t>(d));
        } else {
            for (int i = 0; i < n_override; ++i) {
                const int p = override_pos[i];
                if (p < 0 || p >= P) continue;
                device_copy(xh + static_cast<size_t>(p) * d, dev,
                            embed_override + static_cast<size_t>(i) * d, Device::CPU,
                            sizeof(float) * static_cast<size_t>(d));
            }
        }
    }
    const float* rope_freq = model_.rope_inv_freq_ptr();
    GAI_CHECK(rope_freq != nullptr, "prefill: RoPE frequency cache is missing");
    for (int l = 0; l < c.num_layers; ++l) {
        LayerParams& L = model_.layer(l);

        const bool is_mamba_layer =
            (c.arch == ModelConfig::ArchGranite) &&
            l < static_cast<int>(c.layer_types.size()) &&
            c.layer_types[static_cast<size_t>(l)] == 0 &&
            mamba_states_ready_;

        if (is_mamba_layer) {
            ops::rmsnorm_forward(dev, x.f32(), L.attn_norm.w.f32(), xb.f32(),
                                 nullptr, P, d, c.rms_eps);

            size_t si = 0;
            for (int k = 0; k < l && k < static_cast<int>(c.layer_types.size()); ++k)
                if (c.layer_types[static_cast<size_t>(k)] == 0) ++si;
            GAI_CHECK(si < mamba_states_.size(), "granite mamba: recurrent state missing");
            MambaState& st = mamba_states_[si];

            float* ssm_ptr = (dev == Device::CUDA && st.ssm_dev.defined()) ? st.ssm_dev.f32() : st.ssm_host.data();
            float* conv_ptr = (dev == Device::CUDA && st.conv_dev.defined()) ? st.conv_dev.f32() : st.conv_host.data();

            const int d_inner = c.mamba_expand * d;
            const int conv_dim = d_inner + 2 * c.mamba_n_groups * c.mamba_d_state;
            const int conv_k = c.mamba_d_conv;

            if (is_half_or_quant(L.mamba_in_proj.w)) {
                const WeightRef wi = weight_ref(dev, L.mamba_in_proj.w, 0);
                const WeightRef wo = weight_ref(dev, L.mamba_out_proj.w, 1);
                ops::mamba2_forward_half(dev,
                                         xb.f32(),
                                         wi.ptr,
                                         L.mamba_conv_w.w.f32(),
                                         L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                                         L.mamba_dt_bias.w.f32(),
                                         L.mamba_A_log.w.f32(),
                                         L.mamba_D.w.f32(),
                                         L.mamba_norm.w.f32(),
                                         wo.ptr,
                                         ssm_ptr,
                                         proj.f32(),
                                         1, P, d, d_inner, conv_dim,
                                         c.mamba_n_heads, c.mamba_d_head, c.mamba_d_state,
                                         conv_k, c.rms_eps, wi.qkind,
                                         nullptr, conv_ptr);
            } else {
                ops::mamba2_forward(dev,
                                    xb.f32(),
                                    L.mamba_in_proj.w.f32(),
                                    L.mamba_conv_w.w.f32(),
                                    L.mamba_conv_b.w.defined() ? L.mamba_conv_b.w.f32() : nullptr,
                                    L.mamba_dt_bias.w.f32(),
                                    L.mamba_A_log.w.f32(),
                                    L.mamba_D.w.f32(),
                                    L.mamba_norm.w.f32(),
                                    L.mamba_out_proj.w.f32(),
                                    ssm_ptr,
                                    proj.f32(),
                                    1, P, d, d_inner, conv_dim,
                                    c.mamba_n_heads, c.mamba_d_head, c.mamba_d_state,
                                    conv_k, c.rms_eps,
                                    nullptr, conv_ptr);
            }

            // For n_groups != 1 decode_mamba_layer uses the CPU recurrent-step
            // fallback; synchronize the CUDA prefill state once before that fallback.
            // The common n_groups == 1 path keeps all recurrent state device-resident.
            if (dev == Device::CUDA && c.mamba_n_groups != 1 && st.ssm_dev.defined()) {
                device_copy(st.ssm_host.data(), Device::CPU, st.ssm_dev.f32(),
                            dev, sizeof(float) * st.ssm_host.size());
                if (st.conv_dev.defined() && !st.conv_host.empty()) {
                    device_copy(st.conv_host.data(), Device::CPU, st.conv_dev.f32(),
                                dev, sizeof(float) * st.conv_host.size());
                }
            }

            if (c.residual_multiplier != 1.0f)
                ops::scale_inplace(dev, proj.f32(), c.residual_multiplier,
                                   (i64)P * d);
            ops::add_inplace(dev, x.f32(), proj.f32(), (i64)P * d);

            ops::rmsnorm_forward(dev, x.f32(), L.ffn_norm.w.f32(), xb.f32(),
                                 nullptr, P, d, c.rms_eps);
            if (L.has_moe()) {
                const float* bias = model_.moe_bias_ptr(l);
                if (is_half_or_quant(L.moe_gate.w)) {
                    const WeightRef mgg = weight_ref(dev, L.moe_gate.w, 0);
                    const WeightRef muu = weight_ref(dev, L.moe_up.w,   1);
                    const WeightRef mdd = weight_ref(dev, L.moe_down.w, 2);
                    const WeightRef msg = L.sh_gate.w.defined() ? weight_ref(dev, L.sh_gate.w, 3) : WeightRef{};
                    const WeightRef msu = L.sh_up.w.defined()   ? weight_ref(dev, L.sh_up.w,   4) : WeightRef{};
                    const WeightRef msd = L.sh_down.w.defined() ? weight_ref(dev, L.sh_down.w, 5) : WeightRef{};
                    ops::moe_forward_bias_half(dev, xb.f32(), L.router.w.f32(), bias,
                                       mgg.ptr, muu.ptr, mdd.ptr,
                                       msg.ptr, msu.ptr, msd.ptr,
                                       proj.f32(), nullptr, nullptr, nullptr,
                                       mg.f32(), mu.f32(), ma.f32(),
                                       P, d, c.moe_expert_dim, c.num_experts,
                                       c.moe_top_k, c.shared_width(), mgg.qkind);
                } else {
                    ops::moe_forward_bias(dev, xb.f32(), L.router.w.f32(), bias,
                                      L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                      L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                      L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                      L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                      proj.f32(), nullptr, nullptr, nullptr,
                                      mg.f32(), mu.f32(), ma.f32(),
                                      P, d, c.moe_expert_dim, c.num_experts,
                                      c.moe_top_k, c.shared_width());
                }
            } else {
                linear_param(dev, L.w_gate.w, xb.f32(), gtmp.f32(), P, d, c.intermediate_size);
                linear_param(dev, L.w_up.w,   xb.f32(), utmp.f32(), P, d, c.intermediate_size);
                ops::swiglu_forward(dev, gtmp.f32(), utmp.f32(), atmp.f32(),
                                    (i64)P * c.intermediate_size);
                linear_param(dev, L.w_down.w, atmp.f32(), proj.f32(), P, c.intermediate_size, d);
            }
            if (c.residual_multiplier != 1.0f)
                ops::scale_inplace(dev, proj.f32(), c.residual_multiplier, (i64)P * d);
            ops::add_inplace(dev, x.f32(), proj.f32(), (i64)P * d);
            continue;
        }

        ops::rmsnorm_forward(dev, x.f32(), L.attn_norm.w.f32(), xb.f32(), nullptr, P, d, c.rms_eps);
        float* qp = q.f32();
        float* kp = k.f32();
        float* vp = v.f32();
        if (qkv.defined() && model_.fused_qkv_ptr(l) != nullptr) {
            ops::linear_forward_fp16(dev, xb.f32(), model_.fused_qkv_ptr(l),
                                     qkv.f32(), P, d, qd + 2 * kvd);
            ops::split_qkv(dev, qkv.f32(), q.f32(), k.f32(), v.f32(), P, qd, kvd);
            qp = q.f32();
            kp = k.f32();
            vp = v.f32();
        } else {
            linear_param(dev, L.wq.w, xb.f32(), qp, P, d, qd);
            linear_param(dev, L.wk.w, xb.f32(), kp, P, d, kvd);
            linear_param(dev, L.wv.w, xb.f32(), vp, P, d, kvd);
        }

        if (L.has_qkv_bias()) {
            ops::add_bias(dev, L.bq.w.f32(), qp, P, qd);
            ops::add_bias(dev, L.bk.w.f32(), kp, P, kvd);
            ops::add_bias(dev, L.bv.w.f32(), vp, P, kvd);
        }

        if (c.use_qk_norm && L.qk_qnorm.w.defined()) {
            ops::rmsnorm_forward(dev, qp, L.qk_qnorm.w.f32(), qp,
                                 nullptr, (i64)P * H, hd, c.rms_eps);
            ops::rmsnorm_forward(dev, kp, L.qk_knorm.w.f32(), kp,
                                 nullptr, (i64)P * KV, hd, c.rms_eps);
        }
        if (c.position_embedding != 1) {
            ops::rope_forward_cached(dev, qp, kp, d_pos.i32p(), rope_freq, P, H, KV, hd,
                                     c.rope_type);
        }

        if (cache_.fp16()) {
            const size_t n = static_cast<size_t>(P) * static_cast<size_t>(kvd);
            u16* kdst = cache_.k_f16(l) + static_cast<size_t>(start_pos) * kvd;
            u16* vdst = cache_.v_f16(l) + static_cast<size_t>(start_pos) * kvd;
            ops::convert_f32_to_f16(dev, kp, kdst, static_cast<i64>(n));
            ops::convert_f32_to_f16(dev, vp, vdst, static_cast<i64>(n));
            ops::attention_forward_kv_ex(dev, qp, cache_.k_f16(l), cache_.v_f16(l),
                                          true, att.f32(), 1, P, start_pos + P,
                                          H, KV, hd, start_pos, scale, c.sliding_window);
        } else {
            float* kdst = cache_.k(l) + static_cast<size_t>(start_pos) * kvd;
            float* vdst = cache_.v(l) + static_cast<size_t>(start_pos) * kvd;
            device_copy(kdst, dev, kp, dev, sizeof(float) * (size_t)P * kvd);
            device_copy(vdst, dev, vp, dev, sizeof(float) * (size_t)P * kvd);
            ops::attention_forward_kv_ex(dev, qp, cache_.k(l), cache_.v(l),
                                          false, att.f32(), 1, P, start_pos + P,
                                          H, KV, hd, start_pos, scale, c.sliding_window);
        }
        linear_param(dev, L.wo.w, att.f32(), proj.f32(), P, qd, d);
        if (c.residual_multiplier != 1.0f)
            ops::scale_inplace(dev, proj.f32(), c.residual_multiplier, (i64)P * d);
        ops::add_inplace(dev, x.f32(), proj.f32(), (i64)P * d);
        ops::rmsnorm_forward(dev, x.f32(), L.ffn_norm.w.f32(), xb.f32(), nullptr, P, d, c.rms_eps);
        if (L.has_moe()) {
            const float* bias = model_.moe_bias_ptr(l);
            if (is_half_or_quant(L.moe_gate.w)) {
                const WeightRef mgg = weight_ref(dev, L.moe_gate.w, 0);
                const WeightRef muu = weight_ref(dev, L.moe_up.w, 1);
                const WeightRef mdd = weight_ref(dev, L.moe_down.w, 2);
                const WeightRef msg = L.sh_gate.w.defined() ? weight_ref(dev, L.sh_gate.w, 3) : WeightRef{};
                const WeightRef msu = L.sh_up.w.defined()   ? weight_ref(dev, L.sh_up.w, 4)   : WeightRef{};
                const WeightRef msd = L.sh_down.w.defined() ? weight_ref(dev, L.sh_down.w, 5) : WeightRef{};
                ops::moe_forward_bias_half(dev, xb.f32(), L.router.w.f32(), bias,
                                   mgg.ptr, muu.ptr, mdd.ptr, msg.ptr, msu.ptr, msd.ptr,
                                   proj.f32(), nullptr, nullptr, nullptr,
                                   mg.f32(), mu.f32(), ma.f32(),
                                   P, d, c.moe_expert_dim, c.num_experts, c.moe_top_k,
                                   c.shared_width(), mgg.qkind);
            } else {
                ops::moe_forward_bias(dev, xb.f32(), L.router.w.f32(), bias,
                                  L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                  L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                  L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                  L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                  proj.f32(), nullptr, nullptr, nullptr,
                                  mg.f32(), mu.f32(), ma.f32(),
                                  P, d, c.moe_expert_dim, c.num_experts, c.moe_top_k,
                                  c.shared_width());
            }
        } else {
            linear_param(dev, L.w_gate.w, xb.f32(), gtmp.f32(), P, d, c.intermediate_size);
            linear_param(dev, L.w_up.w, xb.f32(), utmp.f32(), P, d, c.intermediate_size);
            ops::swiglu_forward(dev, gtmp.f32(), utmp.f32(), atmp.f32(), (i64)P * c.intermediate_size);
            linear_param(dev, L.w_down.w, atmp.f32(), proj.f32(), P, c.intermediate_size, d);
        }
        if (c.residual_multiplier != 1.0f)
            ops::scale_inplace(dev, proj.f32(), c.residual_multiplier, (i64)P * d);
        ops::add_inplace(dev, x.f32(), proj.f32(), (i64)P * d);
    }
    cache_.set_length(start_pos + P);
    ops::rmsnorm_forward(dev, x.f32() + (size_t)(P - 1) * d,
                         model_.final_norm().w.f32(), hlast_.f32(),
                         nullptr, 1, d, c.rms_eps);
    linear_param(dev, model_.lm_head().w, hlast_.f32(),
                 logits_dev_.f32(), 1, d, c.vocab_size);
    if (c.logits_scaling != 1.0f)
        ops::scale_inplace(dev, logits_dev_.f32(), 1.0f / c.logits_scaling, c.vocab_size);
    device_copy(logits_host_.data(), Device::CPU,
                logits_dev_.f32(), dev,
                sizeof(float) * (size_t)c.vocab_size);

    device_copy(x_.f32(), dev, x.f32() + (size_t)(P - 1) * d, dev, sizeof(float) * (size_t)d);
    for (auto t : tokens) push_history(t);
}

void Generator::prefill(const std::vector<i32>& tokens, int front_drop,
                        const float* embed_override,
                        const int* override_pos,
                        int n_override) {
    if (tokens.empty()) return;
    GAI_CHECK(front_drop >= 0, "prefill: front_drop must be >= 0");
    Timer t;
    int start = cache_.length();

    int processed = static_cast<int>(tokens.size());
    const int emb_d = model_.config().hidden_size;
    auto override_for_pos = [&](int pos) -> const float* {
        if (!embed_override || !override_pos || n_override <= 0) return nullptr;
        for (int i = 0; i < n_override; ++i)
            if (override_pos[i] == pos)
                return embed_override + static_cast<size_t>(i) * static_cast<size_t>(emb_d);
        return nullptr;
    };
    if (start == 0 && tokens.size() > 1) {

        std::vector<i32> work = tokens;
        int drop = 0;
        if (static_cast<int>(work.size()) > max_context_) {
            drop = static_cast<int>(work.size()) - max_context_;
            work.erase(work.begin() + 1, work.begin() + 1 + drop);
            log_warn(strfmt("prefill: prompt truncated by %d tokens", drop));
            processed = static_cast<int>(work.size());
        }

        const int base = drop + front_drop;
        static constexpr size_t PREFILL_CHUNK = 512;
        for (size_t offset = 0; offset < work.size(); offset += PREFILL_CHUNK) {
            const size_t count = std::min(PREFILL_CHUNK, work.size() - offset);
            std::vector<i32> chunk(work.begin() + offset, work.begin() + offset + count);
            const int global_begin = base + static_cast<int>(offset);

            std::vector<int> chunk_override_pos;
            std::vector<float> chunk_override_embeds;
            if (embed_override && override_pos && n_override > 0) {
                chunk_override_pos.reserve(static_cast<size_t>(n_override));
                chunk_override_embeds.reserve(static_cast<size_t>(n_override) *
                                               static_cast<size_t>(emb_d));
                for (int i = 0; i < n_override; ++i) {
                    const int global_pos = override_pos[i];
                    if (global_pos < global_begin ||
                        global_pos >= global_begin + static_cast<int>(count)) continue;
                    chunk_override_pos.push_back(global_pos - global_begin);
                    const float* row = embed_override +
                        static_cast<size_t>(i) * static_cast<size_t>(emb_d);
                    chunk_override_embeds.insert(chunk_override_embeds.end(),
                                                  row, row + emb_d);
                }
            }
            const float* chunk_embeds = chunk_override_embeds.empty()
                ? nullptr : chunk_override_embeds.data();
            const int* chunk_positions = chunk_override_pos.empty()
                ? nullptr : chunk_override_pos.data();
            forward_prefill(chunk, static_cast<int>(offset), chunk_embeds,
                            chunk_positions, static_cast<int>(chunk_override_pos.size()),
                            base);
            absolute_pos_ = static_cast<int64_t>(base) +
                            static_cast<int64_t>(offset + count);
        }
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            const int pos = static_cast<int>(absolute_pos_);
            decode_step(tokens[i], pos, override_for_pos(pos));
            ++absolute_pos_;
            push_history(tokens[i]);
        }
    }
    stats_.prompt_tokens += processed;
    stats_.prefill_sec += t.seconds();
}

std::vector<i32> Generator::generate(const GenerationConfig& cfg, StreamFn on_token) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    std::vector<i32> out;
    Sampler sampler(cfg.sampling);
    Tokenizer::Stream stream(tok_);
    Timer t;

    std::string tail;
    const int limit = cfg.max_new_tokens;
    const int V = model_.config().vocab_size;
    Device dev = model_.device();
    const SamplingConfig& sc = sampler.config();

    const bool want_greedy = sc.greedy || sc.temperature <= 0.0f || sc.top_k == 1;
    const int K = (!want_greedy && sc.top_k > 0 && sc.top_k <= FAST_TOPK_MAX)
                      ? sc.top_k
                      : 0;

    size_t tail_cap = 256;
    for (const auto& s : cfg.stop_strings)
        tail_cap = std::max(tail_cap, s.size());
    auto use_fast_for = [&](int hist_size) {
        const int w = sc.repetition_window > 0 ? sc.repetition_window : hist_size;

        const bool topk_ok = want_greedy || (K > 0 && V >= K && V <= 262144);
        return sc.gpu_fast_sample && dev == Device::CUDA && V >= 512 &&
               w <= FAST_HIST_MAX && sc.no_repeat_ngram == 0 && topk_ok;
    };

    bool saw_fast = false;
    const bool use_fast = sc.repetition_window != 0 &&
                          use_fast_for(static_cast<int>(history_.size()));
    std::vector<i32> stops = cfg.stop_tokens;
    if (stops.empty()) stops.push_back(tok_.eos_id());
    for (int step = 0; step < limit; ++step) {

        const int win_now = sc.repetition_window > 0 ? sc.repetition_window
                                                     : static_cast<int>(history_.size());
        if (use_fast) saw_fast = true;
        i32 next = -1;
        if (use_fast) {
            int wn = std::min({win_now, static_cast<int>(history_.size()), FAST_HIST_MAX});
            const i32* hptr = wn > 0 ? history_.data() + history_.size() - wn : nullptr;
            ops::apply_rep_penalties(dev, logits_dev_.f32(), V, hptr, wn,
                                     sc.repetition_penalty, sc.frequency_penalty,
                                     sc.presence_penalty);
            if (want_greedy) {
                next = ops::argmax_token(dev, logits_dev_.f32(), V);
            } else {
                ops::topk_select(dev, logits_dev_.f32(), V, K,
                                 topk_vals_dev_.f32(), topk_ids_dev_.i32p());
                device_copy(topk_vals_host_.data(), Device::CPU,
                            topk_vals_dev_.f32(), dev, sizeof(float) * (size_t)K);
                device_copy(topk_ids_host_.data(), Device::CPU,
                            topk_ids_dev_.i32p(), dev, sizeof(i32) * (size_t)K);
                next = sampler.sample_candidates(topk_vals_host_.data(),
                                                 topk_ids_host_.data(), K);
            }
        } else {
            next = sampler.sample(logits_host_.data(), V, history_);
        }

        bool is_stop = std::find(stops.begin(), stops.end(), next) != stops.end();
        if (is_stop) break;

        std::string piece = stream.push(next);
        tail += piece;
        if (tail.size() > tail_cap) tail.erase(0, tail.size() - tail_cap);
        bool hit = false;
        for (const auto& s : cfg.stop_strings) {
            if (!s.empty() && tail.size() >= s.size() &&
                tail.compare(tail.size() - s.size(), s.size(), s) == 0) { hit = true; break; }
        }
        if (hit) break;

        out.push_back(next);
        push_history(next);
        ++stats_.new_tokens;
        if (!piece.empty() && on_token && !on_token(piece, next)) break;

        GAI_CHECK(absolute_pos_ >= 0 && absolute_pos_ <= (i64)std::numeric_limits<int>::max(),
                  "generate: absolute_pos overflow (2B tokens); reset the Generator");
        if (use_fast) decode_step_logits(next, static_cast<int>(absolute_pos_));
        else decode_step(next, static_cast<int>(absolute_pos_));
        ++absolute_pos_;
    }

    std::string rest = stream.flush();
    if (!rest.empty() && on_token) on_token(rest, -1);

    if (saw_fast && dev == Device::CUDA && static_cast<int>(logits_host_.size()) == V) {
        device_copy(logits_host_.data(), Device::CPU, logits_dev_.f32(), dev,
                    sizeof(float) * static_cast<size_t>(V));
    }
    stats_.decode_sec += t.seconds();
    return out;
}

bool Generator::has_attention_layers() const {
    const ModelConfig& c = model_.config();
    if (c.arch != ModelConfig::ArchGranite) return true;
    for (int lt : c.layer_types)
        if (lt != 0) return true;
    return false;
}

std::vector<std::pair<i32, float>> Generator::top_candidates(int n) {
    const int V = model_.config().vocab_size;
    std::vector<float> host(static_cast<size_t>(V), 0.0f);
    device_copy(host.data(), Device::CPU, logits_dev_.f32(), model_.device(),
                sizeof(float) * static_cast<size_t>(V));
    std::vector<std::pair<i32, float>> v;
    v.reserve(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) v.emplace_back(static_cast<i32>(i), host[static_cast<size_t>(i)]);
    const int k = std::min<int>(n, static_cast<int>(v.size()));
    if (k > 0 && k < static_cast<int>(v.size())) {
        std::partial_sort(v.begin(), v.begin() + k, v.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        v.resize(static_cast<size_t>(k));
    }
    return v;
}

std::string Generator::complete(const std::string& prompt, const GenerationConfig& cfg,
                                StreamFn on_token,
                                std::function<void()> after_prefill) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    reset();
    std::vector<i32> ids = tok_.encode(prompt, tok_.add_bos(), false);
    prefill(ids);
    if (after_prefill) after_prefill();
    std::vector<i32> gen = generate(cfg, on_token);
    std::string text = tok_.decode(gen, true);
    return cfg.echo_prompt ? prompt + text : text;
}

std::string Generator::chat(const std::vector<Message>& msgs, const GenerationConfig& cfg,
                            StreamFn on_token,
                            std::function<void()> after_prefill,
                            bool thinking,
                            std::string* reasoning_out) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    reset();

    const ChatTemplate tmpl = chat_template_from_vocab(tok_, thinking);

    std::vector<i32> ids = encode_chat(tok_, tmpl, msgs, true);

    int budget = max_context_ - cfg.max_new_tokens - 4;
    if (budget > 0 && static_cast<int>(ids.size()) > budget) {
        std::vector<Message> trimmed = msgs;
        const size_t sys = (!trimmed.empty() && trimmed.front().role == Role::System) ? 1 : 0;
        for (int guard = 0; guard < 1024; ++guard) {
            const size_t turns = trimmed.size() - sys;
            if (turns <= 2) break;
            trimmed.erase(trimmed.begin() + static_cast<std::ptrdiff_t>(sys),
                          trimmed.begin() + static_cast<std::ptrdiff_t>(sys + 2));
            ids = encode_chat(tok_, tmpl, trimmed, true);
            if (static_cast<int>(ids.size()) <= budget) break;
        }
        if (!ids.empty() && static_cast<int>(ids.size()) > budget)
            log_warn(strfmt("chat: history still exceeds the context budget (%d tokens)",
                            budget));
    }
    prefill(ids, 0, nullptr, nullptr, 0);
    if (after_prefill) after_prefill();

    const i32 think_close_id = thinking ? tok_.token_to_id("</think>") : -1;
    bool in_think = thinking && think_close_id >= 0;
    std::string held_back;
    StreamFn stream = on_token;
    if (in_think && on_token) {
        stream = [&](const std::string& piece, i32 id) -> bool {
            if (in_think && id == think_close_id) {
                in_think = false;
                return true;
            }
            if (in_think) {
                held_back += piece;
                return true;
            }
            return on_token(piece, id);
        };
    }

    std::vector<i32> gen = generate(cfg, stream);
    std::vector<i32> reasoning, answer;
    bool split_in_think = thinking && think_close_id >= 0;
    for (i32 id : gen) {
        if (split_in_think && id == think_close_id) {
            split_in_think = false;
            continue;
        }
        if (split_in_think) reasoning.push_back(id);
        else answer.push_back(id);
    }
    const bool closed = (think_close_id >= 0) && !split_in_think;
    if (thinking && !closed) {
        if (!held_back.empty() && on_token) on_token(held_back, -1);
        answer = std::move(gen);
        reasoning.clear();
    }
    if (reasoning_out) *reasoning_out = tok_.decode(reasoning, true);
    return tok_.decode(answer, true);
}

std::string Generator::complete_multimodal(const std::string& prompt,
                                           const float* visual_embeds,
                                           int num_visual_tokens,
                                           const GenerationConfig& cfg,
                                           StreamFn on_token,
                                           std::function<void()> after_prefill) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    reset();
    std::vector<i32> ids = tok_.encode(prompt, tok_.add_bos(), false);
    if (visual_embeds && num_visual_tokens > 0) {
        const i32 pad_id = tok_.bos_id() >= 0 ? tok_.bos_id() : 0;
        std::vector<i32> full_ids;
        full_ids.reserve(num_visual_tokens + ids.size());
        std::vector<int> override_pos(num_visual_tokens);
        for (int i = 0; i < num_visual_tokens; ++i) {
            full_ids.push_back(pad_id);
            override_pos[i] = i;
        }
        full_ids.insert(full_ids.end(), ids.begin(), ids.end());
        prefill(full_ids, 0, visual_embeds, override_pos.data(), num_visual_tokens);
    } else {
        prefill(ids);
    }
    if (after_prefill) after_prefill();
    std::vector<i32> gen = generate(cfg, on_token);
    std::string text = tok_.decode(gen, true);
    return cfg.echo_prompt ? prompt + text : text;
}

std::string Generator::chat_multimodal(const std::vector<Message>& msgs,
                                       const float* visual_embeds,
                                       int num_visual_tokens,
                                       const GenerationConfig& cfg,
                                       StreamFn on_token,
                                       std::function<void()> after_prefill,
                                       bool thinking,
                                       std::string* reasoning_out) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    reset();
    const ChatTemplate tmpl = chat_template_from_vocab(tok_, thinking);
    std::vector<i32> ids = encode_chat(tok_, tmpl, msgs, true);

    if (visual_embeds && num_visual_tokens > 0) {
        const i32 pad_id = tok_.bos_id() >= 0 ? tok_.bos_id() : 0;
        std::vector<i32> full_ids;
        full_ids.reserve(num_visual_tokens + ids.size());
        std::vector<int> override_pos(num_visual_tokens);
        for (int i = 0; i < num_visual_tokens; ++i) {
            full_ids.push_back(pad_id);
            override_pos[i] = i;
        }
        full_ids.insert(full_ids.end(), ids.begin(), ids.end());
        prefill(full_ids, 0, visual_embeds, override_pos.data(), num_visual_tokens);
    } else {
        prefill(ids);
    }

    if (after_prefill) after_prefill();

    const i32 think_close_id = thinking ? tok_.token_to_id("</think>") : -1;
    StreamFn stream = on_token;
    std::string held_back;
    if (thinking && think_close_id >= 0 && on_token) {
        bool in_thinking = true;
        stream = [on_token, think_close_id, &in_thinking, &held_back](const std::string& piece, i32 id) -> bool {
            if (in_thinking) {
                if (id == think_close_id) in_thinking = false;
                held_back += piece;
                return true;
            }
            return on_token(piece, id);
        };
    }

    std::vector<i32> gen = generate(cfg, stream);
    std::vector<i32> reasoning, answer;
    bool split_in_think = thinking && think_close_id >= 0;
    for (i32 id : gen) {
        if (split_in_think && id == think_close_id) {
            split_in_think = false;
            continue;
        }
        if (split_in_think) reasoning.push_back(id);
        else answer.push_back(id);
    }
    const bool closed = (think_close_id >= 0) && !split_in_think;
    if (thinking && !closed) {
        if (!held_back.empty() && on_token) on_token(held_back, -1);
        answer = std::move(gen);
        reasoning.clear();
    }
    if (reasoning_out) *reasoning_out = tok_.decode(reasoning, true);
    return tok_.decode(answer, true);
}

double Generator::score_tokens(const std::vector<i32>& tokens, const std::vector<u8>* mask,
                               i64* out_ntok, const std::vector<i32>* segment_ids) {
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");
    if (tokens.size() < 2) { if (out_ntok) *out_ntok = 0; return 0.0; }
    const int T = static_cast<int>(std::min<size_t>(tokens.size(), static_cast<size_t>(model_.config().max_seq_len)));
    if (static_cast<int>(tokens.size()) > T) {
        log_warn(strfmt("score_tokens: input truncated to max_seq_len (%d -> %d tokens); "
                        "perplexity covers the head window only",
                        static_cast<int>(tokens.size()), T));
    }

    GAI_CHECK(!mask || mask->size() >= static_cast<size_t>(T), "score_tokens: mask shorter than scored tokens");
    GAI_CHECK(!segment_ids || segment_ids->size() >= static_cast<size_t>(T),
              "score_tokens: segment_ids shorter than scored tokens");

    Device dev = model_.device();
    double sum_all = 0.0;
    i64 n_all = 0;
    const bool can_chunk = !has_attention_layers();
    static constexpr int SCORE_CHUNK = 512;
    const int chunk = can_chunk ? SCORE_CHUNK : T;
    const int Tn_max = std::min(chunk + 1, T);
    Activations act = model_.make_activations(1, Tn_max, false);
    for (int off = 0; off < T; off += chunk) {
        const int Tc = std::min(chunk, T - off);
        const int Tend = std::min(T, off + Tc + 1);
        const int start = off;
        const int Tn = Tend - start;
        if (Tn < 2) break;
        act.T = Tn;
        act.pos_offset = start;
        std::vector<i32> ids(tokens.begin() + start, tokens.begin() + start + Tn);
        std::vector<i32> tgt(static_cast<size_t>(Tn), -100);
        for (int i = 0; i + 1 < Tn; ++i) {
            const int pos = start + i;
            bool supervised = !mask || (*mask)[static_cast<size_t>(pos) + 1];
            if (supervised) tgt[static_cast<size_t>(i)] = tokens[static_cast<size_t>(pos) + 1];
        }

        Tensor d_ids({Tn}, DType::I32, Device::CPU);
        std::memcpy(d_ids.data_ptr(), ids.data(), sizeof(i32) * ids.size());
        Tensor dev_ids({Tn}, DType::I32, dev);
        dev_ids.copy_from(d_ids);
        Tensor d_tgt({Tn}, DType::I32, Device::CPU);
        std::memcpy(d_tgt.data_ptr(), tgt.data(), sizeof(i32) * tgt.size());
        Tensor dev_tgt({Tn}, DType::I32, dev);
        dev_tgt.copy_from(d_tgt);
        Tensor dev_segments;
        if (segment_ids) {
            Tensor h_segments({Tn}, DType::I32, Device::CPU);
            for (int i = 0; i < Tn; ++i)
                h_segments.i32p()[i] = (*segment_ids)[static_cast<size_t>(start + i)];
            dev_segments = Tensor({Tn}, DType::I32, dev);
            dev_segments.copy_from(h_segments);
        }

        bool boundary_at_start = (start > 0 && segment_ids &&
            (*segment_ids)[static_cast<size_t>(start)] >= 0 &&
            (*segment_ids)[static_cast<size_t>(start)] !=
            (*segment_ids)[static_cast<size_t>(start - 1)]);
        Tensor& logits = model_.forward(dev_ids.i32p(), 1, Tn, act,
                                        segment_ids ? dev_segments.i32p() : nullptr,
                                        start == 0 || boundary_at_start);
        double sum = 0.0;
        i64 n = 0;

        ops::softmax_cross_entropy(dev, logits.f32(), dev_tgt.i32p(), nullptr,
                                   Tn, model_.config().vocab_size, &sum, &n,
                                   0.0f);
        sum_all += sum;
        n_all += n;
    }    if (out_ntok) *out_ntok = n_all;
    return n_all > 0 ? sum_all / static_cast<double>(n_all) : 0.0;
}

}
