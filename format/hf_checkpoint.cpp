#include "format/hf_checkpoint.h"

#include "core/json.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace gai {

namespace fs = std::filesystem;

namespace {

std::string read_whole_file(const std::string& path, bool* ok = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        if (ok) *ok = false;
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (ok) *ok = true;
    return ss.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// SafeTensorsFile
// ---------------------------------------------------------------------------

bool SafeTensorsFile::open(const std::string& path) {
    tensors_.clear();
    data_ = nullptr;
    path_ = path;

    if (!mmap_.open(path)) {
        log_error("[safetensors] cannot open or mmap: " + path);
        return false;
    }
    file_size_ = mmap_.size();
    if (file_size_ < 8) {
        log_error("[safetensors] file too small: " + path);
        return false;
    }

    const uint8_t* base = static_cast<const uint8_t*>(mmap_.data());
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    if (header_len == 0 || header_len > file_size_ - 8) {
        log_error("[safetensors] bad header length: " + path);
        return false;
    }

    std::string header(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(header_len));

    Json j;
    std::string err;
    if (!Json::parse(header, j, &err) || !j.is_object()) {
        log_error("[safetensors] bad header JSON: " + path + " (" + err + ")");
        return false;
    }

    for (const auto& [name, info] : j.items()) {
        if (name == "__metadata__") continue;
        if (!info.is_object()) continue;
        SafeTensorInfo ti;
        ti.name  = name;
        ti.dtype = info.at("dtype").as_string();
        const Json& shape = info.at("shape");
        if (shape.is_array()) {
            for (const Json& d : shape.elements()) ti.shape.push_back(d.as_int());
        }
        const Json& offs = info.at("data_offsets");
        if (offs.is_array() && offs.size() == 2) {
            ti.begin = static_cast<uint64_t>(offs[0].as_int());
            ti.end   = static_cast<uint64_t>(offs[1].as_int());
        } else {
            log_error("[safetensors] bad data_offsets for " + name);
            return false;
        }
        if (ti.end > file_size_ - 8 - header_len) {
            log_error("[safetensors] " + name + " points past the end of the file");
            return false;
        }
        tensors_.emplace(name, std::move(ti));
    }

    // Direct pointer to data buffer via mmap: zero-copy
    data_ = base + 8 + header_len;
    return true;
}

const SafeTensorInfo* SafeTensorsFile::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

Tensor SafeTensorsFile::read_f32(const std::string& name) const {
    const SafeTensorInfo* ti = find(name);
    if (!ti || !data_) return Tensor();

    i64 numel = 1;
    for (i64 d : ti->shape) numel *= d;

    Tensor out(ti->shape, DType::F32, Device::CPU);
    if (!out.defined()) return Tensor();

    const uint64_t nbytes = ti->end - ti->begin;
    const uint8_t* src = data_ + ti->begin;
    float* dst = out.f32();

    if (ti->dtype == "F32") {
        if (nbytes != static_cast<uint64_t>(numel) * 4) return Tensor();
        std::memcpy(dst, src, static_cast<size_t>(nbytes));
    } else if (ti->dtype == "F16") {
        if (nbytes != static_cast<uint64_t>(numel) * 2) return Tensor();
        const u16* h = reinterpret_cast<const u16*>(src);
        for (i64 i = 0; i < numel; ++i) dst[i] = fp16_to_fp32(h[i]);
    } else if (ti->dtype == "BF16") {
        if (nbytes != static_cast<uint64_t>(numel) * 2) return Tensor();
        const u16* b = reinterpret_cast<const u16*>(src);
        for (i64 i = 0; i < numel; ++i) dst[i] = bf16_to_fp32(b[i]);
    } else {
        log_error("[safetensors] " + name + " has the unsupported dtype " + ti->dtype);
        return Tensor();
    }
    return out;
}

// ---------------------------------------------------------------------------
// the name mapping
// ---------------------------------------------------------------------------

std::string hf_tensor_name(const std::string& internal, bool is_granite) {
    if (internal == "tok_embeddings") return "model.embed_tokens.weight";
    if (internal == "final_norm")     return "model.norm.weight";
    if (internal == "lm_head")        return "lm_head.weight";

    const std::string pre = "layers.";
    if (internal.compare(0, pre.size(), pre) != 0) return {};
    const size_t dot = internal.find('.', pre.size());
    if (dot == std::string::npos) return {};
    const std::string layer = internal.substr(pre.size(), dot - pre.size());
    const std::string rest  = internal.substr(dot + 1);
    const std::string L = "model.layers." + layer + ".";

    if (rest == "attn_norm") return L + "input_layernorm.weight";
    if (rest == "wq")        return L + "self_attn.q_proj.weight";
    if (rest == "wk")        return L + "self_attn.k_proj.weight";
    if (rest == "wv")        return L + "self_attn.v_proj.weight";
    if (rest == "wo")        return L + "self_attn.o_proj.weight";
    if (rest == "bq")        return L + "self_attn.q_proj.bias";
    if (rest == "bk")        return L + "self_attn.k_proj.bias";
    if (rest == "bv")        return L + "self_attn.v_proj.bias";
    if (rest == "qk_qnorm")  return L + "self_attn.q_norm.weight";
    if (rest == "qk_knorm")  return L + "self_attn.k_norm.weight";
    if (rest == "ffn_norm")  return L + "post_attention_layernorm.weight";

    if (rest == "w_gate")    return L + "mlp.gate_proj.weight";
    if (rest == "w_up")      return L + "mlp.up_proj.weight";
    if (rest == "w_down")    return L + "mlp.down_proj.weight";

    if (is_granite) {
        // IBM Granite-4.0 hybrid packed MoE (all layers, including mamba ones):
        //   block_sparse_moe.router.layer.weight      -> moe_router
        //   block_sparse_moe.input_linear.weight      -> expert gate+up (packed)
        //   block_sparse_moe.output_linear.weight     -> expert down (packed)
        //   shared_mlp.input_linear.weight            -> shared gate+up (packed)
        //   shared_mlp.output_linear.weight           -> shared down
        // The packed expert/shared-MLP weights are shape-driven in load_model().
        if (rest == "moe_router")   return L + "block_sparse_moe.router.layer.weight";
        if (rest == "moe_sh_gate")  return L + "shared_mlp.input_linear.weight";
        if (rest == "moe_sh_up")    return L + "shared_mlp.input_linear.weight";
        if (rest == "moe_sh_down")  return L + "shared_mlp.output_linear.weight";
        if (rest == "moe_gate" || rest == "moe_up" || rest == "moe_down") return {};
    } else {
        // Mixture of experts (Qwen / olmoe naming).
        if (rest == "moe_router")   return L + "mlp.gate.weight";
        if (rest == "moe_sh_gate")  return L + "mlp.shared_expert.gate_proj.weight";
        if (rest == "moe_sh_up")    return L + "mlp.shared_expert.up_proj.weight";
        if (rest == "moe_sh_down")  return L + "mlp.shared_expert.down_proj.weight";
        if (rest == "moe_gate" || rest == "moe_up" || rest == "moe_down") return {};
    }

    // Mamba-2 tensors (granite hybrid mamba layers). The engine's parameter
    // names are "layers.N.mamba.<field>"; the checkpoint packs them under the
    // same namespace.
    if (rest == "mamba.in_proj")       return L + "mamba.in_proj.weight";
    if (rest == "mamba.conv1d.weight") return L + "mamba.conv1d.weight";
    if (rest == "mamba.conv1d.bias")   return L + "mamba.conv1d.bias";
    if (rest == "mamba.dt_bias")       return L + "mamba.dt_bias";
    if (rest == "mamba.A_log")         return L + "mamba.A_log";
    if (rest == "mamba.D")             return L + "mamba.D";
    if (rest == "mamba.norm.weight")   return L + "mamba.norm.weight";
    if (rest == "mamba.out_proj")      return L + "mamba.out_proj.weight";

    return {};
}

// ---------------------------------------------------------------------------
// shared HuggingFace tokenizer builder
// ---------------------------------------------------------------------------

namespace {

struct HfTokenizerInput {
    std::string tokenizer_json;    // raw text, required
    std::string tokenizer_config;  // raw text, optional (dir mode)
    std::string generation_config; // raw text, optional (dir mode)
};

// Builds the engine's byte-level BPE Tokenizer from HuggingFace metadata.
// bos/eos/pad ids are resolved first against the vocab (symbolic names in
// the configs are common), then a single final build runs. Returns false if
// the vocabulary is not a usable byte-level BPE vocabulary.
bool build_hf_tokenizer(const HfTokenizerInput& in, Tokenizer& tk,
                        std::vector<i32>& eos_ids) {
    std::string err;

    // ---- tokenizer.json ----
    Json j;
    if (!Json::parse(in.tokenizer_json, j, &err)) {
        log_error("[hf] bad tokenizer.json (" + err + ")");
        return false;
    }
    const Json& model = j.at("model");
    const Json& vocab = model.at("vocab");
    if (!vocab.is_object()) {
        log_error("[hf] tokenizer.json has no vocabulary");
        return false;
    }
    // Qwen-style added tokens (top-level list of {id, content, special}):
    // they extend the vocabulary beyond model.vocab (<|im_start|> = 151644).
    struct AddedTok { i64 id; std::string content; i64 special_bit; };
    std::vector<AddedTok> added_tokens;
    {
        const Json& at = j.at("added_tokens");
        if (at.is_array()) {
            for (const Json& el : at.elements()) {
                if (!el.is_object()) continue;
                const i64 id = el.at("id").as_int(-1);
                const std::string content = el.at("content").as_string();
                if (id < 0 || content.empty()) continue;
                added_tokens.push_back({id, content,
                    static_cast<i64>(el.at("special").as_bool(false) ? 1 : 0)});
            }
        }
    }

    // HuggingFace matches EVERY added token in the raw text, not only the
    // ones flagged "special": the flag only drives skip_special_tokens.
    // Qwen3 declares 12 non-special added tokens (<think>, <|fim_prefix|>, ...)
    // that must still be tokenized as themselves.
    std::vector<i32> added_ids;
    added_ids.reserve(added_tokens.size());
    for (const auto& a : added_tokens) added_ids.push_back(static_cast<i32>(a.id));

    i64 max_id = -1;
    for (const auto& [tok, id] : vocab.items()) {
        (void)tok;
        max_id = std::max(max_id, id.as_int(-1));
    }
    if (max_id < 0) {
        log_error("[hf] the vocabulary is empty");
        return false;
    }
    i64 max_all = max_id;
    for (const auto& a : added_tokens) max_all = std::max(max_all, a.id);
    std::vector<std::string> ordered(static_cast<size_t>(max_all) + 1);
    std::vector<char> filled(static_cast<size_t>(max_all) + 1, 0);
    for (const auto& [tok, id] : vocab.items()) {
        const i64 i = id.as_int(-1);
        if (i < 0 || i > max_id) continue;
        ordered[static_cast<size_t>(i)] = tok;
        filled[static_cast<size_t>(i)] = 1;
    }
    // The added tokens fill the ids above model.vocab (ChatML markers etc).
    for (const auto& a : added_tokens) {
        if (a.id >= 0 && a.id < static_cast<i64>(ordered.size())) {
            ordered[static_cast<size_t>(a.id)] = a.content;
            filled[static_cast<size_t>(a.id)] = 1;
        }
    }
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (!filled[i]) {
            log_error("[hf] the vocabulary has a hole at the id " + std::to_string(i));
            return false;
        }
    }

    std::vector<std::pair<std::string, std::string>> merges;
    const Json& jm = model.at("merges");
    if (jm.is_array()) {
        for (const Json& m : jm.elements()) {
            if (m.is_string()) {
                const std::string s = m.as_string();
                const size_t sp = s.rfind(' ');
                if (sp == std::string::npos || sp == 0 || sp + 1 >= s.size()) continue;
                merges.emplace_back(s.substr(0, sp), s.substr(sp + 1));
            } else if (m.is_array() && m.size() == 2) {
                merges.emplace_back(m[0].as_string(), m[1].as_string());
            }
        }
    }

    PreTokKind opts_pretok = PreTokKind::Gpt2;

    // Special ids declared by tokenizer.json itself. Newer files define
    // them in the added_tokens list above; older ones in added_tokens_decoder.
    std::vector<i32> special;
    auto collect_added = [&](const Json& cfgj) {
        const Json& added = cfgj.at("added_tokens_decoder");
        if (!added.is_object()) return;
        for (const auto& [id_str, spec] : added.items()) {
            if (!spec.is_object()) continue;
            if (!spec.at("special").as_bool(false)) continue;
            try { special.push_back(static_cast<i32>(std::stoi(id_str))); }
            catch (...) {}
        }
    };
    collect_added(j);
    for (const auto& a : added_tokens) if (a.special_bit) special.push_back(static_cast<i32>(a.id));

    // Resolve the symbolic bos/eos/pad/unk names once the vocab is indexed.
    // Qwen's tokenizer_config.json uses plain strings ("<|im_end|>"), while
    // Llama-3 uses numbers; both are handled.
    // Detect the pre-tokenizer from tokenizer.json's Split regex:
    //   Granite-4   carries \p{N}{1,3}  (digits in groups of up to 3)
    //   Qwen2/3     carries a bare \p{N} clause and no \p{N}+ nor {1,3}
    //   GPT-2/Llama-3 carries \p{N}+
    {
        const std::string blob = j.at("pre_tokenizer").dump();
        const bool granite = blob.find("p{N}{1,3}") != std::string::npos;
        const bool qwen = blob.find("p{N}|") != std::string::npos &&
                          blob.find("p{N}+") == std::string::npos &&
                          blob.find("p{N}{1,3}") == std::string::npos;
        opts_pretok = granite ? PreTokKind::Granite : (qwen ? PreTokKind::Qwen : PreTokKind::Gpt2);
    }

    VocabOptions opts;
    opts.pretok = opts_pretok;
    opts.added_ids = added_ids;
    opts.add_bos = false;
    opts.add_eos = false;
    opts.bos_id  = 151643;  // <|endoftext|>
    opts.eos_id  = 151645;  // <|im_end|>
    opts.pad_id  = 151643;
    opts.unk_id  = 151643;
    opts.special_ids = special;

    tk.build(ordered, merges, opts);

    auto resolve = [&](const Json& cfgj, const char* knum, const char* kname) -> i32 {
        if (!cfgj.is_object()) return -1;
        if (cfgj.has(knum)) {
            const i64 v = cfgj.at(knum).as_int(-1);
            if (v >= 0) return static_cast<i32>(v);
        }
        if (cfgj.has(kname)) {
            const Json& v = cfgj.at(kname);
            if (v.is_null()) return -1;
            if (v.is_number()) return static_cast<i32>(v.as_int(-1));
            if (v.is_string()) return tk.token_to_id(v.as_string());
        }
        return -1;
    };

    Json cfgj;
    const bool have_cfg = !in.tokenizer_config.empty() &&
                          Json::parse(in.tokenizer_config, cfgj) && cfgj.is_object();
    if (have_cfg) {
        collect_added(cfgj);
        auto resolve_name = [&](const char* knum, const char* kname) -> i32 {
            i32 v = resolve(cfgj, knum, kname);
            return v;
        };
        i32 b = resolve_name("bos_token_id",  "bos_token");
        i32 e = resolve_name("eos_token_id",  "eos_token");
        i32 p = resolve_name("pad_token_id",  "pad_token");
        i32 u = resolve_name("unk_token_id",  "unk_token");
        if (b >= 0) opts.bos_id = b;
        if (e >= 0) opts.eos_id = e;
        if (p >= 0) opts.pad_id = p;
        if (u >= 0) opts.unk_id = u;
        opts.add_bos = cfgj.at("add_bos_token").as_bool(false);
        opts.add_eos = cfgj.at("add_eos_token").as_bool(false);
        // Union the config's special ids with the tokenizer.json ones.
        std::unordered_set<i32> uni;
        for (i32 id : special) uni.insert(id);
        for (i32 id : opts.special_ids) uni.insert(id);
        // tokenizer_config.json marks the same ids via added_tokens_decoder;
        // re-collect from cfg in case it adds more.
        // (collect_added(cfgj) already appended into `special`; fold it in.)
        opts.special_ids.assign(uni.begin(), uni.end());
    }

    // Mark bos/eos/pad as special so skip_special hides them.
    auto mark = [&](i32 id) {
        if (id >= 0) opts.special_ids.push_back(id);
    };
    mark(opts.bos_id);
    mark(opts.eos_id);
    mark(opts.pad_id);

    // The tokenizer.json built above may have been built with the original
    // defaults; if the config changed any id, rebuild with the final set.
    tk.build(ordered, merges, opts);

    // ---- generation_config.json: the stop-token list ----
    eos_ids.clear();
    if (!in.generation_config.empty()) {
        Json gj;
        if (Json::parse(in.generation_config, gj) && gj.is_object() && gj.has("eos_token_id")) {
            const Json& e = gj.at("eos_token_id");
            if (e.is_array()) {
                for (const Json& el : e.elements()) {
                    const i64 v = el.as_int(-1);
                    if (v >= 0) eos_ids.push_back(static_cast<i32>(v));
                }
            } else if (e.is_number()) {
                const i64 v = e.as_int(-1);
                if (v >= 0) eos_ids.push_back(static_cast<i32>(v));
            }
        }
    }
    if (eos_ids.empty() && opts.eos_id >= 0) eos_ids.push_back(opts.eos_id);

    return tk.has_byte_map() && tk.vocab_size() > 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// HFCheckpoint
// ---------------------------------------------------------------------------

bool HFCheckpoint::looks_like_checkpoint(const std::string& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    return fs::exists(fs::path(dir) / "config.json", ec) &&
           (fs::exists(fs::path(dir) / "model.safetensors", ec) ||
            fs::exists(fs::path(dir) / "model.safetensors.index.json", ec));
}

bool HFCheckpoint::read_config_json(const std::string& path) {
    bool ok = false;
    const std::string text = read_whole_file(path, &ok);
    if (!ok) {
        log_error("[hf] cannot read " + path);
        return false;
    }
    Json j;
    std::string err;
    if (!Json::parse(text, j, &err) || !j.is_object()) {
        log_error("[hf] bad config.json (" + err + ")");
        return false;
    }

    auto num = [&](const char* key, i64 def) -> i64 {
        return j.has(key) ? j.at(key).as_int(def) : def;
    };
    auto fnum = [&](const char* key, double def) -> double {
        return j.has(key) ? j.at(key).as_number(def) : def;
    };

    cfg_ = ModelConfig{};
    cfg_.hidden_size      = static_cast<int>(num("hidden_size", 0));
    cfg_.num_layers       = static_cast<int>(num("num_hidden_layers", 0));
    cfg_.num_heads        = static_cast<int>(num("num_attention_heads", 0));
    cfg_.num_kv_heads     = static_cast<int>(num("num_key_value_heads",
                                                  num("num_attention_heads", 0)));
    cfg_.intermediate_size= static_cast<int>(num("intermediate_size", 0));
    cfg_.vocab_size       = static_cast<int>(num("vocab_size", 0));
    cfg_.max_seq_len      = static_cast<int>(num("max_position_embeddings", 4096));
    cfg_.rms_eps          = static_cast<float>(fnum("rms_norm_eps", 1e-5));
    cfg_.rope_theta       = static_cast<float>(fnum("rope_theta", 10000.0));

    // RoPE pairing convention.
    //   rope_type 1 = "neox"  : rotate_half(x) = concat(-x2, x1) over the
    //                           first/second half of the head dimension.
    //                           This is what HuggingFace uses for Qwen2/Qwen3,
    //                           Llama, DeepSeek, Mistral, Gemma...
    //   rope_type 0 = "interleaved" : (x0,x1), (x2,x3), ... - the GPT-J style.
    //
    // Every modern HuggingFace architecture uses the neox convention, so it
    // is the default here. Getting this wrong silently destroys the
    // pretrained positional signal without any error.
    cfg_.rope_type        = 1;
    cfg_.tie_embeddings   = j.has("tie_word_embeddings")
                                ? j.at("tie_word_embeddings").as_bool(false)
                                : false;

    // head_dim: a few architectures declare it explicitly (it is then
    // smaller than hidden_size / num_heads).
    if (j.has("head_dim")) {
        const i64 hd = num("head_dim", 0);
        if (hd > 0) cfg_.head_dim_override = static_cast<int>(hd);
    }

    // Sliding-window attention. Without this the model would attend over
    // the whole context and silently produce wrong outputs.
    if (j.has("sliding_window")) {
        const i64 sw = num("sliding_window", 0);
        if (sw > 0) cfg_.sliding_window = static_cast<int>(sw);
    }
    if (j.has("use_sliding_window")) {
        const bool use_sw = j.at("use_sliding_window").as_bool(false);
        if (!use_sw) cfg_.sliding_window = 0;
    }
    if (j.has("use_qkv_bias")) cfg_.use_qkv_bias = j.at("use_qkv_bias").as_bool(cfg_.use_qkv_bias);
    if (j.has("attention_bias")) cfg_.use_qkv_bias = j.at("attention_bias").as_bool(cfg_.use_qkv_bias);

    // Mixture of experts (Qwen3-MoE, Mixtral).
    const i64 experts = num("num_experts", 0);
    if (experts > 0) {
        cfg_.use_moe       = true;
        cfg_.num_experts   = static_cast<int>(experts);
        cfg_.moe_top_k     = static_cast<int>(num("num_experts_per_tok", 8));
        cfg_.moe_expert_dim= static_cast<int>(num("moe_intermediate_size",
                                                   cfg_.intermediate_size));
        // The shared expert has its OWN width, which can differ from the
        // routed one (the shared MLP of a Mixtral/Qwen3-MoE). It used to be
        // dropped here, so the shared expert was rebuilt with the routed
        // width and the load failed on the first layer.
        cfg_.shared_expert_intermediate_size =
            static_cast<int>(num("shared_expert_intermediate_size", 0));
        cfg_.moe_shared    = j.has("shared_expert_intermediate_size");
    }

    // IBM Granite-4.0 hybrid (GraniteMoeHybridForCausalLM, model_type
    // "granitemoehybrid"). Different marker keys: num_local_experts is the
    // routed expert count, shared_intermediate_size the shared expert width,
    // and the layer kind is declared per layer in `layer_types`.
    const std::string model_type =
        j.has("model_type") ? j.at("model_type").as_string("") : "";
    const i64 local_experts = num("num_local_experts", 0);
    if (model_type == "granitemoehybrid" || local_experts > 0) {
        cfg_.arch = ModelConfig::ArchGranite;
        const i64 le = local_experts > 0 ? local_experts : num("num_experts", 0);
        cfg_.use_moe        = true;
        cfg_.num_experts    = static_cast<int>(le);
        cfg_.moe_top_k      = static_cast<int>(num("num_experts_per_tok", 6));
        cfg_.moe_expert_dim = static_cast<int>(num("intermediate_size", 0));
        // Granite-4's MoE has a shared expert whose width is stored under
        // shared_intermediate_size (1024 on h-tiny).
        cfg_.moe_shared     = j.has("shared_intermediate_size");
        cfg_.shared_expert_intermediate_size =
            static_cast<int>(num("shared_intermediate_size", 0));

        // Granite multipliers.
        cfg_.embedding_multiplier = static_cast<float>(fnum("embedding_multiplier", 1.0));
        cfg_.residual_multiplier  = static_cast<float>(fnum("residual_multiplier", 1.0));
        cfg_.attention_multiplier = static_cast<float>(fnum("attention_multiplier", 0.0));
        cfg_.logits_scaling       = static_cast<float>(fnum("logits_scaling", 1.0));

        // Mamba-2 dims.
        cfg_.mamba_n_heads    = static_cast<int>(num("mamba_n_heads", 48));
        cfg_.mamba_d_head     = static_cast<int>(num("mamba_d_head", 64));
        cfg_.mamba_d_state    = static_cast<int>(num("mamba_d_state", 128));
        cfg_.mamba_n_groups   = static_cast<int>(num("mamba_n_groups", 1));
        cfg_.mamba_d_conv     = static_cast<int>(num("mamba_d_conv", 4));
        cfg_.mamba_expand     = static_cast<int>(num("mamba_expand", 2));
        cfg_.mamba_chunk_size = static_cast<int>(num("mamba_chunk_size", 256));
        cfg_.mamba_conv_bias  = j.has("mamba_conv_bias")
                                    ? j.at("mamba_conv_bias").as_bool(true)
                                    : true;
        cfg_.mamba_proj_bias  = j.has("mamba_proj_bias")
                                    ? j.at("mamba_proj_bias").as_bool(false)
                                    : false;

        // position_embedding_type: "nope" (Granite-4.0) disables positional
        // embeddings entirely.
        const std::string pos =
            j.has("position_embedding_type")
                ? j.at("position_embedding_type").as_string("rope")
                : std::string("rope");
        cfg_.position_embedding = (pos == "nope") ? 1 : 0;

        // Layer kinds, per layer.
        if (j.has("layer_types") && j.at("layer_types").is_array()) {
            cfg_.layer_types.clear();
            for (const Json& el : j.at("layer_types").elements()) {
                // The official checkpoint says "mamba" / "attention";
                // transformers also emits "linear_attention" /
                // "full_attention" for the same layers.
                const std::string kind = el.as_string("mamba");
                const bool attn = (kind == "attention" || kind == "full_attention");
                cfg_.layer_types.push_back(attn ? 1 : 0);
            }
        }

        // Shared MLP gate/up are fused into input_linear, shared down into
        // output_linear. Mark the model as Granite so the loader/packing
        // paths know about it.
        log_info(strfmt("[hf] granite-4.0 hybrid recognized: %d layers, %d experts, top-%d, "
                        "d=%d, d_inner=%d, arch=%s", cfg_.num_layers, cfg_.num_experts,
                        cfg_.moe_top_k, cfg_.hidden_size, cfg_.mamba_expand * cfg_.hidden_size,
                        pos.c_str()));
    }

    // RoPE scaling (YaRN). Map the factor onto the engine's YaRN rope path so
    // a long-context checkpoint keeps the intended frequencies instead of
    // silently running the base theta. Only YaRN is supported precisely.
    if (j.has("rope_scaling") && j.at("rope_scaling").is_object()) {
        const Json& rs = j.at("rope_scaling");
        const std::string type = rs.at("rope_type").as_string(rs.at("type").as_string(""));
        const double factor = rs.at("factor").as_number(1.0);
        const double orig_max = rs.at("original_max_position_embeddings").as_number(0.0);

        if (type == "yarn" && factor > 1.0) {
            cfg_.rope_scale = static_cast<float>(factor);
            // YaRN ramps the correction between the short and the long
            // wavelength. The engine's ramp is expressed as the two
            // positions where the correction starts and is full.
            if (orig_max > 0.0) {
                cfg_.rope_yarn_low  = static_cast<float>(orig_max);
                cfg_.rope_yarn_high = static_cast<float>(orig_max * factor);
            }
            // HF's mscale (when present) scales the attention by a constant.
            const double ms = rs.at("mscale").as_number(rs.at("mscale_all_dim").as_number(0.0));
            if (ms > 0.0) cfg_.rope_yarn_mscale = static_cast<float>(ms);
            log_info(strfmt("[hf] rope_scaling yarn factor=%.2f orig_max=%.0f mscale=%.3f",
                            factor, orig_max, ms));
        } else if ((type == "linear" || type == "dynamic") && factor > 1.0) {
            // The engine's NTK-by-factor path is the closest supported
            // behaviour; it is an approximation of HF's linear scaling.
            cfg_.rope_scale = static_cast<float>(factor);
            log_warn(strfmt("[hf] rope_scaling '%s' factor=%.2f mapped onto the NTK-by-factor "
                            "approximation (not bit-exact with HuggingFace)",
                            type.c_str(), factor));
        } else if (!type.empty() && type != "default") {
            log_warn("[hf] rope_scaling type '" + type + "' is not supported; "
                     "running the base theta=" + strfmt("%.0f", (double)cfg_.rope_theta));
        }
    }

    if (cfg_.hidden_size <= 0 || cfg_.num_layers <= 0 || cfg_.num_heads <= 0 ||
        cfg_.vocab_size <= 0) {
        log_error("[hf] config.json is missing the architecture fields");
        return false;
    }
    if (cfg_.num_kv_heads <= 0) cfg_.num_kv_heads = cfg_.num_heads;
    if (cfg_.intermediate_size <= 0) cfg_.intermediate_size = 4 * cfg_.hidden_size;
    return true;
}

bool HFCheckpoint::open(const std::string& dir, bool weights_required) {
    dir_ = dir;
    std::error_code ec;
    if (!fs::is_directory(dir_, ec)) {
        log_error("[hf] not a directory: " + dir_);
        return false;
    }
    if (!read_config_json((fs::path(dir_) / "config.json").string())) return false;

    // The weight files: one file, or the shards listed by the index.
    const fs::path single = fs::path(dir_) / "model.safetensors";
    const fs::path index  = fs::path(dir_) / "model.safetensors.index.json";
    if (fs::exists(single, ec)) {
        weight_files_.push_back(single.string());
    } else if (fs::exists(index, ec)) {
        bool ok = false;
        const std::string text = read_whole_file(index.string(), &ok);
        Json j;
        std::string err;
        if (!ok || !Json::parse(text, j, &err)) {
            log_error("[hf] cannot read the safetensors index");
            return false;
        }
        // The index maps every tensor to its file; the file list is the
        // set of values, in order.
        std::vector<std::string> files;
        const Json& map = j.at("weight_map");
        if (map.is_object()) {
            for (const auto& [tensor, file] : map.items()) {
                (void)tensor;
                const std::string fname = file.as_string();
                if (std::find(files.begin(), files.end(), fname) == files.end()) {
                    files.push_back(fname);
                }
            }
        }
        std::sort(files.begin(), files.end());
        for (const std::string& f : files) {
            weight_files_.push_back((fs::path(dir_) / f).string());
        }
    } else {
        // A dry run (--config-only / the memory estimate / --config-from)
        // only needs the architecture: the weights are not required and
        // their absence must not be an error.
        if (!weights_required) {
            log_info("[hf] no weight file in " + dir_ +
                     " (the architecture only: the weights are not needed)");
            return true;
        }
        log_error("[hf] no model.safetensors (nor an index) in " + dir_);
        return false;
    }

    if (weights_required) {
        for (const std::string& f : weight_files_) {
            SafeTensorsFile sf;
            if (!sf.open(f)) return false;
            for (const auto& [name, ti] : sf.tensors()) {
                tensor_info_[name] = ti;
                tensor_file_[name] = f;
            }
            files_[f] = std::move(sf);
        }
    }
    // A dry run (--config-only / memory estimate) does not need the shards
    // at all: do not open them, and do not pretend anything is broken.
    // The config above is complete by itself.

    // The QK normalization is not in config.json: it is a property of the
    // checkpoint, so it is detected from the tensor names. Without the
    // shards (dry run) the names are unknown and config.json is all we have;
    // that is exactly what the user asked for, so stay quiet about it.
    if (!tensor_info_.empty()) {
        cfg_.use_qk_norm = tensor_info_.count("model.layers.0.self_attn.q_norm.weight") > 0;

        // Detect QKV attention biases (e.g. Qwen 2 / 2.5 / 3) from checkpoint tensors
        if (tensor_info_.count("model.layers.0.self_attn.q_proj.bias") ||
            tensor_info_.count("model.layers.0.self_attn.k_proj.bias") ||
            tensor_info_.count("model.layers.0.self_attn.v_proj.bias")) {
            cfg_.use_qkv_bias = true;
        }
    }

    // The vocabulary of the checkpoint is the first dimension of the
    // embedding, not necessarily the vocab_size of config.json: many
    // checkpoints pad the embedding to a multiple of 64 (or to 151936).
    auto it = tensor_info_.find("model.embed_tokens.weight");
    if (it != tensor_info_.end() && !it->second.shape.empty() &&
        it->second.shape[0] > 0) {
        const int real_vocab = static_cast<int>(it->second.shape[0]);
        if (real_vocab != cfg_.vocab_size) {
            log_info(strfmt("[hf] the embedding has %d rows, config.json says %d: "
                            "using the embedding", real_vocab, cfg_.vocab_size));
            cfg_.vocab_size = real_vocab;
        }
    }
    return true;
}

const SafeTensorInfo* HFCheckpoint::find_tensor(const std::string& name) const {
    auto it = tensor_info_.find(name);
    return it == tensor_info_.end() ? nullptr : &it->second;
}

Tensor HFCheckpoint::read_tensor(const std::string& name) const {
    auto fit = tensor_file_.find(name);
    if (fit == tensor_file_.end()) return Tensor();
    auto it = files_.find(fit->second);
    if (it == files_.end()) return Tensor();
    return it->second.read_f32(name);
}

bool HFCheckpoint::load_model(Model& model) const {
    const ModelConfig& mc = model.config();
    bool lm_head_from_emb = false;

    for (Parameter* p : model.parameters()) {
        const std::string internal = p->name;

        // The LoRA adapters are not part of a HuggingFace checkpoint: they
        // are created by the engine (fresh, random) and trained on top of
        // the loaded weights. Skipping them here keeps the loader working
        // on a LoRA run instead of failing on an unknown tensor name.
        if (internal.find(".lora_") != std::string::npos) continue;

        // The MoE expert weights are flat on the engine side and per-expert on
        // the HuggingFace side: they are gathered here.
        const size_t dot = internal.rfind('.');
        const std::string leaf = (dot == std::string::npos)
                                     ? internal
                                     : internal.substr(dot + 1);
        const bool is_moe_leaf = leaf == "moe_gate" || leaf == "moe_up" || leaf == "moe_down" ||
                                 leaf == "moe_sh_gate" || leaf == "moe_sh_up" || leaf == "moe_sh_down";
        if (is_moe_leaf) {
        // IBM Granite-4.0: the expert weights are packed into two tensors
        // per layer (input_linear = all gate+up, output_linear = all down),
        // and the shared MLP is packed the same way. The packed layout is
        // what the IBM checkpoint uses; it is resolved by shape.
        const bool is_granite = (mc.arch == ModelConfig::ArchGranite);
        {
            const size_t d1 = internal.find('.');
            const size_t d2 = internal.rfind('.');
            const std::string layer = internal.substr(d1 + 1, d2 - d1 - 1);
            const int ne = mc.num_experts;
            const int E  = mc.moe_expert_dim;
            const i64 d  = mc.hidden_size;
            const std::string base  = "model.layers." + layer;

            if (is_granite) {
                const std::string in_name  = base + ".block_sparse_moe.input_linear.weight";
                const std::string out_name = base + ".block_sparse_moe.output_linear.weight";
                Tensor ti = read_tensor(in_name);
                Tensor to = read_tensor(out_name);
                if (!ti.defined() || !to.defined()) {
                    log_error("[hf] granite packed expert tensors missing: " + in_name + " / " + out_name);
                    return false;
                }

                // IBM Granite-4.0 shared MLP (fused gate+up in input_linear, down
                // in output_linear). The engine keeps them as separate parameters.
                const i64 shared = mc.shared_expert_intermediate_size;
                if (leaf == "moe_sh_gate" || leaf == "moe_sh_up" || leaf == "moe_sh_down") {
                    if (shared <= 0) {
                        log_error("[hf] granite config is missing shared_intermediate_size");
                        return false;
                    }
                    if (leaf == "moe_sh_down") {
                        const std::string hn = base + ".shared_mlp.output_linear.weight";
                        Tensor t = read_tensor(hn);
                        if (!t.defined()) {
                            log_error("[hf] granite shared_mlp.output_linear missing: " + hn);
                            return false;
                        }
                        if (t.numel() != static_cast<i64>(shared) * d) {
                            log_error(strfmt("[hf] granite shared_mlp.output_linear numel=%lld != %lld",
                                             (long long)t.numel(), (long long)((i64)shared * d)));
                            return false;
                        }
                        p->w.copy_from_f32(t);
                        continue;
                    }
                    const std::string hn = base + ".shared_mlp.input_linear.weight";
                    Tensor t = read_tensor(hn);
                    if (!t.defined()) {
                        log_error("[hf] granite shared_mlp.input_linear missing: " + hn);
                        return false;
                    }
                    if (t.numel() != static_cast<i64>(2 * shared) * d) {
                        log_error(strfmt("[hf] granite shared_mlp.input_linear numel=%lld != %lld (shared=%d d=%lld)",
                                         (long long)t.numel(), (long long)((i64)(2 * shared) * d),
                                         (int)shared, (long long)d));
                        return false;
                    }
                    const float* src = t.f32();
                    const size_t half = static_cast<size_t>(shared) * static_cast<size_t>(d);
                    // The data is staged on the host and copied in one go:
                    // the copy honours the parameter's storage dtype (fp32 or
                    // half) and moves the bytes to the device properly.
                    Tensor stage(std::vector<i64>{static_cast<i64>(half)}, DType::F32, Device::CPU);
                    std::memcpy(stage.f32(), leaf == "moe_sh_gate" ? src : src + half,
                                half * sizeof(float));
                    p->w.copy_from_f32(stage);
                    continue;
                }

                if (leaf == "moe_down") {
                    if (to.numel() != static_cast<i64>(ne) * E * d) {
                        log_error(strfmt("[hf] granite output_linear numel=%lld != %lld (ne=%d E=%d d=%lld)",
                                         (long long)to.numel(), (long long)((i64)ne * E * d), ne, E, (long long)d));
                        return false;
                    }
                    Tensor stage(std::vector<i64>{p->numel()}, DType::F32, Device::CPU);
                    float* dst = stage.f32();
                    // Layout A: [ne*d, E] — d rows per expert, each [d, E].
                    if (to.ndim() == 2 && to.dim(0) == static_cast<i64>(ne) * d && to.dim(1) == E) {
                        for (int e = 0; e < ne; ++e) {
                            const float* src = to.f32() + static_cast<size_t>(e) * d * E;
                            std::memcpy(dst + static_cast<size_t>(e) * d * E, src,
                                        static_cast<size_t>(d) * E * sizeof(float));
                        }
                        p->w.copy_from_f32(stage);
                        continue;
                    }
                    // Layout B: [ne*E, d] — E rows per expert, each [E, d] (the IBM
                    // packed layout). The engine keeps each down_proj as [d, E], so
                    // transpose per expert.
                    if (to.ndim() == 2 && to.dim(0) == static_cast<i64>(ne) * E && to.dim(1) == d) {
                        const size_t blk = static_cast<size_t>(E) * static_cast<size_t>(d);
                        for (int e = 0; e < ne; ++e) {
                            const float* src = to.f32() + static_cast<size_t>(e) * blk;
                            float* d_e = dst + static_cast<size_t>(e) * blk;
                            for (int i = 0; i < E; ++i)
                                for (int j = 0; j < d; ++j)
                                    d_e[static_cast<size_t>(j) * static_cast<size_t>(E) + i] =
                                        src[static_cast<size_t>(i) * static_cast<size_t>(d) + j];
                        }
                        p->w.copy_from_f32(stage);
                        continue;
                    }
                    // Layout C: [ne, d, E] (3D) — our exact layout per expert.
                    if (to.ndim() == 3 && to.dim(0) == ne && to.dim(1) == d && to.dim(2) == E) {
                        std::memcpy(dst, to.f32(),
                                    static_cast<size_t>(ne) * d * E * sizeof(float));
                        p->w.copy_from_f32(stage);
                        continue;
                    }
                    // Layout D: [ne, E, d] (3D) — per-expert [E, d], transpose.
                    if (to.ndim() == 3 && to.dim(0) == ne && to.dim(1) == E && to.dim(2) == d) {
                        const size_t blk = static_cast<size_t>(E) * static_cast<size_t>(d);
                        for (int e = 0; e < ne; ++e) {
                            const float* src = to.f32() + static_cast<size_t>(e) * blk;
                            float* d_e = dst + static_cast<size_t>(e) * blk;
                            for (int i = 0; i < E; ++i)
                                for (int j = 0; j < d; ++j)
                                    d_e[static_cast<size_t>(j) * static_cast<size_t>(E) + i] =
                                        src[static_cast<size_t>(i) * static_cast<size_t>(d) + j];
                        }
                        p->w.copy_from_f32(stage);
                        continue;
                    }
                    log_error(strfmt("[hf] granite output_linear unsupported shape [%lld,%lld,%lld]",
                                     (long long)to.dim(0), (long long)to.dim(1),
                                     (long long)(to.ndim() > 2 ? to.dim(2) : 0)));
                    return false;
                } else {
                    // moe_gate / moe_up
                    if (ti.numel() != static_cast<i64>(ne) * 2 * E * d) {
                        log_error(strfmt("[hf] granite input_linear numel=%lld != %lld (ne=%d E=%d d=%lld)",
                                         (long long)ti.numel(), (long long)((i64)ne * 2 * E * d), ne, E, (long long)d));
                        return false;
                    }
                    const size_t block = static_cast<size_t>(E) * static_cast<size_t>(d);
                    Tensor stage(std::vector<i64>{p->numel()}, DType::F32, Device::CPU);
                    float* dst = stage.f32();
                    for (int e = 0; e < ne; ++e) {
                        // Per expert: gate block [E*d] then up block [E*d].
                        const float* src = ti.f32() + static_cast<size_t>(e) * 2 * block;
                        const size_t off = static_cast<size_t>(e) * block;
                        std::memcpy(dst + off,
                                    leaf == "moe_gate" ? src : src + block,
                                    block * sizeof(float));
                    }
                    p->w.copy_from_f32(stage);
                    continue;
                }
            }

            // The shared expert is NOT a routed expert: it has its own
            // tensors (mlp.shared_expert.*) and its own width (SH, which
            // can differ from E). It used to fall into the per-expert branch
            // below, which read the wrong tensors and then failed the size
            // check - so a checkpoint with a shared expert could not be
            // loaded at all.
            if (leaf == "moe_sh_gate" || leaf == "moe_sh_up" || leaf == "moe_sh_down") {
                const std::string hn = base + ".mlp.shared_expert." +
                    (leaf == "moe_sh_gate" ? "gate_proj.weight"
                     : leaf == "moe_sh_up" ? "up_proj.weight"
                                           : "down_proj.weight");
                Tensor t = read_tensor(hn);
                if (!t.defined()) {
                    log_error("[hf] missing shared expert tensor: " + hn);
                    return false;
                }
                if (t.numel() != p->numel()) {
                    log_error(strfmt("[hf] shared expert size mismatch %s (%s): "
                                     "file=%lld model=%lld",
                                     internal.c_str(), hn.c_str(),
                                     static_cast<long long>(t.numel()),
                                     static_cast<long long>(p->numel())));
                    return false;
                }
                p->w.copy_from_f32(t);
                continue;
            }

            // Non-granite (Qwen3-MoE / Mixtral): per-expert tensors.
            const std::string suffix = (leaf == "moe_gate")  ? "gate_proj.weight"
                                     : (leaf == "moe_up")    ? "up_proj.weight"
                                     :                         "down_proj.weight";
            const i64 per = (leaf == "moe_down") ? d * E : E * d;
            Tensor flat(std::vector<i64>{static_cast<i64>(ne) * per}, DType::F32,
                        Device::CPU);
            if (!flat.defined()) return false;
            float* dst = flat.f32();
            for (int e = 0; e < ne; ++e) {
                const std::string hn = base + ".mlp.experts." + std::to_string(e) + "." + suffix;
                Tensor t = read_tensor(hn);
                if (!t.defined() || t.numel() != per) {
                    log_error("[hf] missing or bad expert tensor: " + hn);
                    return false;
                }
                std::memcpy(dst + static_cast<size_t>(e) * per, t.f32(),
                            static_cast<size_t>(per) * sizeof(float));
            }
            if (flat.numel() != p->numel()) {
                log_error(strfmt("[hf] expert tensor size mismatch %s: file=%lld model=%lld",
                                 internal.c_str(), static_cast<long long>(flat.numel()),
                                 static_cast<long long>(p->numel())));
                return false;
            }
            p->w.copy_from_f32(flat);
            continue;
        }
        }  // is_moe_leaf

        std::string hn = hf_tensor_name(internal, mc.arch == ModelConfig::ArchGranite);
        if (hn.empty()) {
            log_error("[hf] no HuggingFace name for the parameter " + internal);
            return false;
        }
        if (!find_tensor(hn) && internal == "lm_head") {
            // A tied checkpoint has no separate output projection.
            hn = "model.embed_tokens.weight";
            lm_head_from_emb = true;
        }
        Tensor t = read_tensor(hn);
        if (!t.defined()) {
            log_error("[hf] missing tensor: " + hn);
            return false;
        }
        if (t.numel() != p->numel()) {
            log_error(strfmt("[hf] tensor size mismatch %s (%s): file=%lld model=%lld",
                             internal.c_str(), hn.c_str(),
                             static_cast<long long>(t.numel()),
                             static_cast<long long>(p->numel())));
            return false;
        }
        p->w.copy_from_f32(t);
    }

    if (lm_head_from_emb) {
        log_info("[hf] no lm_head in the checkpoint: the token embedding is used "
                 "as the output projection");
    }
    return true;
}

bool HFCheckpoint::load_tokenizer(Tokenizer& tk) const {
    const fs::path tpath = fs::path(dir_) / "tokenizer.json";
    if (!fs::exists(tpath)) {
        log_error("[hf] no tokenizer.json in " + dir_);
        return false;
    }
    bool ok = false;
    const std::string text = read_whole_file(tpath.string(), &ok);
    if (!ok) {
        log_error("[hf] cannot read " + tpath.string());
        return false;
    }
    auto read_opt = [&](const fs::path& p) {
        std::string s;
        bool good = false;
        s = read_whole_file(p.string(), &good);
        return good ? s : std::string();
    };
    HfTokenizerInput in;
    in.tokenizer_json    = text;
    in.tokenizer_config  = read_opt(fs::path(dir_) / "tokenizer_config.json");
    in.generation_config = read_opt(fs::path(dir_) / "generation_config.json");
    if (!build_hf_tokenizer(in, tk, eos_ids_)) {
        log_error("[hf] the tokenizer in " + dir_ + " has no byte-level alphabet: "
                  "the engine encodes against byte-level BPE only, and this "
                  "vocabulary is not one (SentencePiece/unigram are not supported yet).");
        return false;
    }
    return true;
}

bool HFCheckpoint::load_tokenizer_json(const std::string& path, Tokenizer& tk,
                                       std::vector<i32>* eos_out) {
    bool ok = false;
    const std::string text = read_whole_file(path, &ok);
    if (!ok) {
        log_error("[hf] cannot read " + path);
        return false;
    }
    auto read_opt = [&](const fs::path& p) {
        std::string s;
        bool good = false;
        s = read_whole_file(p.string(), &good);
        return good ? s : std::string();
    };
    const fs::path base = fs::path(path).parent_path();
    HfTokenizerInput in;
    in.tokenizer_json    = text;
    in.tokenizer_config  = read_opt(base / "tokenizer_config.json");
    in.generation_config = read_opt(base / "generation_config.json");
    std::vector<i32> eos;
    if (!build_hf_tokenizer(in, tk, eos)) {
        log_error("[hf] bad or non-byte-level tokenizer.json: " + path);
        return false;
    }
    if (eos_out) *eos_out = std::move(eos);
    return true;
}

}  // namespace gai
