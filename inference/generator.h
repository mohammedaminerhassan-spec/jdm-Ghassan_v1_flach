#pragma once

#include "model/model.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "inference/kv_cache.h"
#include "inference/sampler.h"
#include <functional>

namespace gai {

struct GenerationConfig {
    int  max_new_tokens = 256;
    int  max_context    = 0;
    // Stop-token ids. Empty (the default) = the tokenizer's eos_id().
    // HF checkpoints loaded through HFCheckpoint fill this from
    // generation_config.json (e.g. [<|im_end|>, <|endoftext|>] for Qwen).
    std::vector<i32>         stop_tokens;
    std::vector<std::string> stop_strings;
    bool echo_prompt = false;
    SamplingConfig sampling{};
};

struct GenerationStats {
    int    prompt_tokens = 0;
    int    new_tokens    = 0;
    double prefill_sec   = 0;
    double decode_sec    = 0;
    double prefill_tps() const { return prefill_sec > 0 ? prompt_tokens / prefill_sec : 0; }
    double decode_tps()  const { return decode_sec  > 0 ? new_tokens / decode_sec : 0; }
    std::string summary() const;
};

class Generator {
public:
    Generator(Model& model, const Tokenizer& tok, int max_context = 0);

    using StreamFn = std::function<bool(const std::string& piece, i32 token)>;

    void reset();

    void prefill(const std::vector<i32>& tokens, int front_drop = 0,
                 const float* embed_override = nullptr,
                 const int* override_pos = nullptr,
                 int n_override = 0);

    std::vector<i32> generate(const GenerationConfig& cfg, StreamFn on_token = nullptr);

    std::string complete(const std::string& prompt, const GenerationConfig& cfg,
                         StreamFn on_token = nullptr);

    std::string chat(const std::vector<Message>& msgs, const GenerationConfig& cfg,
                     StreamFn on_token = nullptr);

    double score_tokens(const std::vector<i32>& tokens, const std::vector<u8>* mask = nullptr,
                        i64* out_ntok = nullptr,
                        const std::vector<i32>* segment_ids = nullptr);

    const GenerationStats& stats() const { return stats_; }
    KVCache& cache() { return cache_; }
    int context_used() const { return cache_.length(); }

private:

    float* decode_step(i32 token, int position);

    float* decode_step_logits(i32 token, int position);

    // IBM Granite-4.0: one Mamba-2 layer, one token, using the recurrent
    // state kept between the decoded tokens. Writes the layer output into
    // `out` (d floats).
    void decode_mamba_layer(int layer_index, const float* x_in, float* out);
    void forward_prefill(const std::vector<i32>& tokens, int start_pos,
                           const float* embed_override = nullptr,
                           const int* override_pos = nullptr,
                           int n_override = 0);

    static constexpr int FAST_TOPK_MAX = 128;
    static constexpr int FAST_HIST_MAX = 2048;

    static constexpr size_t HIST_MAX = 8192;
    void push_history(i32 tok) {
        if (history_.size() >= HIST_MAX)
            history_.erase(history_.begin(),
                           history_.begin() + (history_.size() - HIST_MAX + 1));
        history_.push_back(tok);
    }

    Model&           model_;
    const Tokenizer& tok_;
    KVCache          cache_;
    int              max_context_;
    Device           expected_device_ = Device::CPU;

    int64_t          absolute_pos_ = 0;

    Tensor x_, xb_, q_, k_, v_, qkv_, attn_, proj_, gate_, up_, act_, logits_dev_, scores_;
    Tensor tok_dev_, pos_dev_;
    Tensor hlast_;
    Tensor topk_vals_dev_, topk_ids_dev_;
    std::vector<float> topk_vals_host_;
    std::vector<i32>   topk_ids_host_;

    Tensor moe_gate_, moe_up_, moe_act_;

    // IBM Granite-4.0: the Mamba-2 layers are stateful. Two pieces of state
    // per mamba layer:
    //   conv_state : [conv_dim, d_conv-1]  the causal convolution window
    //   ssm_state  : [n_heads, d_head, d_state] the recurrent SSM state
    // They are updated once per decoded token and reset with the cache.
    struct MambaState {
        Tensor conv;   // [conv_dim, d_conv-1]
        Tensor ssm;    // [n_heads, d_head, d_state]
        std::vector<float> conv_host;
        std::vector<float> ssm_host;
    };
    std::vector<MambaState> mamba_states_;
    bool mamba_states_ready_ = false;

    std::vector<float> logits_host_;
    std::vector<i32>   history_;
    GenerationStats    stats_;
};

}
