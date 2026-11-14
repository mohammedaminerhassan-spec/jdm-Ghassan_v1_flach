#include "format/hf_writer.h"

#include "core/json.h"
#include "format/hf_checkpoint.h"
#include "quantization/quantize.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

namespace gai {

// ---------------------------------------------------------------------------
// the dtypes
// ---------------------------------------------------------------------------

const char* safe_dtype_name(SafeDType t) {
    switch (t) {
        case SafeDType::F32:  return "F32";
        case SafeDType::F16:  return "F16";
        case SafeDType::BF16: return "BF16";
    }
    return "F32";
}

size_t safe_dtype_size(SafeDType t) {
    switch (t) {
        case SafeDType::F32:  return 4;
        case SafeDType::F16:  return 2;
        case SafeDType::BF16: return 2;
    }
    return 4;
}

// ---------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------

SafeTensorsWriter::SafeTensorsWriter(const std::string& path) : path_(path) {}

SafeTensorsWriter::~SafeTensorsWriter() {
    if (out_.is_open()) {
        out_.flush();
        out_.close();
    }
}

void SafeTensorsWriter::reserve(const std::string& name, const std::vector<i64>& shape,
                                SafeDType dtype, size_t nbytes) {
    Entry e;
    e.name = name;
    e.shape = shape;
    e.dtype = dtype;
    e.nbytes = nbytes;
    e.begin = static_cast<uint64_t>(total_);
    e.end = static_cast<uint64_t>(total_ + nbytes);
    total_ += nbytes;
    entries_.push_back(std::move(e));
}

bool SafeTensorsWriter::begin() {
    if (begun_) return false;
    // The header: {name: {dtype, shape, data_offsets}} for every tensor.
    Json header = Json::object();
    for (const Entry& e : entries_) {
        Json info = Json::object();
        info.set("dtype", Json::string(safe_dtype_name(e.dtype)));
        Json shape = Json::array();
        for (i64 d : e.shape) shape.push(Json::integer(d));
        info.set("shape", shape);
        Json offs = Json::array();
        offs.push(Json::integer(static_cast<i64>(e.begin)));
        offs.push(Json::integer(static_cast<i64>(e.end)));
        info.set("data_offsets", offs);
        header.set(e.name, info);
    }
    std::string head = header.dump();
    // The header is padded with spaces so the data starts on an 8-byte
    // boundary (the reference reader requires it).
    const size_t pad = (8 - ((8 + head.size()) % 8)) % 8;
    head.append(pad, ' ');

    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_.good()) return false;
    const uint64_t hlen = static_cast<uint64_t>(head.size());
    out_.write(reinterpret_cast<const char*>(&hlen), 8);
    out_.write(head.data(), static_cast<std::streamsize>(head.size()));
    if (!out_.good()) { failed_ = true; return false; }
    begun_ = true;
    cursor_ = 0;
    return true;
}

bool SafeTensorsWriter::append(const void* data, size_t nbytes) {
    if (!begun_ || failed_) return false;
    if (cursor_ >= entries_.size()) return false;
    const Entry& e = entries_[cursor_];
    if (nbytes != e.nbytes) return false;
    out_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
    if (!out_.good()) { failed_ = true; return false; }
    ++cursor_;
    return true;
}

bool SafeTensorsWriter::finish() {
    if (!begun_) return false;
    if (cursor_ != entries_.size()) return false;
    out_.flush();
    const bool ok = out_.good();
    out_.close();
    begun_ = false;
    return ok && !failed_;
}

// ---------------------------------------------------------------------------
// the export plan
// ---------------------------------------------------------------------------

namespace {

// The tensor's data on the host, as fp32.
std::vector<float> host_f32(const Tensor& t) {
    if (!t.defined()) return {};
    Tensor cpu = (t.device() == Device::CPU) ? t.clone() : t.to(Device::CPU);
    if (cpu.dtype() == DType::F32) {
        return std::vector<float>(cpu.f32(), cpu.f32() + cpu.numel());
    }
    // The half storage mode: widen to fp32 (exact) and let the encoder
    // narrow it again to the requested dtype.
    Tensor f32 = quant::dequantize(cpu, DType::F32);
    return std::vector<float>(f32.f32(), f32.f32() + f32.numel());
}

void encode(const std::vector<float>& src, SafeDType dt, uint8_t* dst) {
    const size_t n = src.size();
    if (dt == SafeDType::F32) {
        std::memcpy(dst, src.data(), n * sizeof(float));
        return;
    }
    u16* h = reinterpret_cast<u16*>(dst);
    if (dt == SafeDType::F16) {
        for (size_t i = 0; i < n; ++i) h[i] = fp32_to_fp16(src[i]);
        return;
    }
    for (size_t i = 0; i < n; ++i) h[i] = fp32_to_bf16(src[i]);
}

// The layer index of a parameter name ("layers.12.wq" -> 12), or -1.
int layer_index_of(const std::string& internal) {
    if (internal.compare(0, 7, "layers.") != 0) return -1;
    const size_t dot = internal.find('.', 7);
    if (dot == std::string::npos) return -1;
    try {
        return std::stoi(internal.substr(7, dot - 7));
    } catch (...) {
        return -1;
    }
}

}  // namespace

std::vector<ExportTensor> plan_safetensors_export(Model& model, SafeDType dtype) {
    const ModelConfig& cfg = model.config();
    const bool granite = (cfg.arch == ModelConfig::ArchGranite);
    const int  ne = cfg.num_experts;
    const int  E  = cfg.moe_expert_dim;
    const int  SH = cfg.shared_width();
    const i64  d  = cfg.hidden_size;

    std::vector<ExportTensor> out;
    // The parameters that are packed into a shared HuggingFace tensor: they
    // must be emitted ONCE (two tensors with the same name would make the
    // file invalid: the header holds one entry and the data two).
    std::set<std::string> packed;

    for (const Parameter* p : model.parameters()) {
        const std::string& name = p->name;
        // The LoRA adapters are not part of a HuggingFace checkpoint: they
        // are merged into the base weights before the export.
        if (name.find(".lora_") != std::string::npos) continue;
        // Already packed into another tensor (the Granite gate/up pairs).
        if (packed.count(name)) continue;

        const size_t dot = name.rfind('.');
        const std::string leaf = (dot == std::string::npos) ? name : name.substr(dot + 1);
        const int layer = layer_index_of(name);
        const std::string L = (layer >= 0) ? ("model.layers." + std::to_string(layer) + ".") : "";

        // ---- the MoE experts (the only tensors that are not a copy) ----
        if (leaf == "moe_gate" || leaf == "moe_up" || leaf == "moe_down" ||
            leaf == "moe_sh_gate" || leaf == "moe_sh_up" || leaf == "moe_sh_down") {

            if (granite) {
                if (leaf == "moe_gate" || leaf == "moe_up") {
                    // The IBM packed layout: ONE tensor per layer holding the
                    // gate and the up blocks of every expert, interleaved:
                    //   [gate_e0 (E rows) | up_e0 (E rows) | gate_e1 | ...]
                    // (this is what the engine's loader expects back).
                    const Parameter* other = model.find_parameter(
                        (leaf == "moe_gate") ? (name.substr(0, name.rfind('.')) + ".moe_up")
                                             : (name.substr(0, name.rfind('.')) + ".moe_gate"));
                    if (!other) continue;
                    ExportTensor t;
                    t.hf_name = L + "block_sparse_moe.input_linear.weight";
                    t.shape = {2LL * ne * E, d};
                    t.dtype = dtype;
                    t.nbytes = static_cast<size_t>(2 * ne * E * d) * safe_dtype_size(dtype);
                    const Parameter* pg = p;
                    const Parameter* pu = other;
                    t.fill = [pg, pu, ne, E, d, dtype](uint8_t* dst) {
                        std::vector<float> g = host_f32(pg->w);
                        std::vector<float> u = host_f32(pu->w);
                        const size_t block = static_cast<size_t>(E) * static_cast<size_t>(d);
                        std::vector<float> packed(2 * block * static_cast<size_t>(ne));
                        for (int e = 0; e < ne; ++e) {
                            std::memcpy(packed.data() + static_cast<size_t>(e) * 2 * block,
                                        g.data() + static_cast<size_t>(e) * block, block * sizeof(float));
                            std::memcpy(packed.data() + (static_cast<size_t>(e) * 2 + 1) * block,
                                        u.data() + static_cast<size_t>(e) * block, block * sizeof(float));
                        }
                        encode(packed, dtype, dst);
                    };
                    out.push_back(std::move(t));
                    // The partner parameter is packed here: it must not be
                    // emitted a second time.
                    packed.insert(other->name);
                    continue;
                }
                if (leaf == "moe_sh_gate" || leaf == "moe_sh_up") {
                    const Parameter* other = model.find_parameter(
                        (leaf == "moe_sh_gate")
                            ? (name.substr(0, name.rfind('.')) + ".moe_sh_up")
                            : (name.substr(0, name.rfind('.')) + ".moe_sh_gate"));
                    if (!other) continue;
                    ExportTensor t;
                    t.hf_name = L + "shared_mlp.input_linear.weight";
                    t.shape = {2LL * SH, d};
                    t.dtype = dtype;
                    t.nbytes = static_cast<size_t>(2 * SH * d) * safe_dtype_size(dtype);
                    const Parameter* pg = p;
                    const Parameter* pu = other;
                    t.fill = [pg, pu, SH, d, dtype](uint8_t* dst) {
                        std::vector<float> g = host_f32(pg->w);
                        std::vector<float> u = host_f32(pu->w);
                        const size_t half = static_cast<size_t>(SH) * static_cast<size_t>(d);
                        std::vector<float> packed(2 * half);
                        std::memcpy(packed.data(), g.data(), half * sizeof(float));
                        std::memcpy(packed.data() + half, u.data(), half * sizeof(float));
                        encode(packed, dtype, dst);
                    };
                    out.push_back(std::move(t));
                    packed.insert(other->name);
                    continue;
                }
                if (leaf == "moe_down") {
                    // The engine's [ne*d, E] IS the IBM layout (Layout A): a
                    // straight copy.
                    ExportTensor t;
                    t.hf_name = L + "block_sparse_moe.output_linear.weight";
                    t.shape = p->shape;
                    t.dtype = dtype;
                    t.nbytes = static_cast<size_t>(p->numel()) * safe_dtype_size(dtype);
                    const Parameter* pp = p;
                    t.fill = [pp, dtype](uint8_t* dst) {
                        encode(host_f32(pp->w), dtype, dst);
                    };
                    out.push_back(std::move(t));
                    continue;
                }
                if (leaf == "moe_sh_down") {
                    ExportTensor t;
                    t.hf_name = L + "shared_mlp.output_linear.weight";
                    t.shape = p->shape;
                    t.dtype = dtype;
                    t.nbytes = static_cast<size_t>(p->numel()) * safe_dtype_size(dtype);
                    const Parameter* pp = p;
                    t.fill = [pp, dtype](uint8_t* dst) {
                        encode(host_f32(pp->w), dtype, dst);
                    };
                    out.push_back(std::move(t));
                    continue;
                }
                continue;
            }

            // ---- the per-expert layout (Qwen3-MoE, Mixtral) ----
            const bool shared = (leaf == "moe_sh_gate" || leaf == "moe_sh_up" ||
                                 leaf == "moe_sh_down");
            const std::string proj = (leaf == "moe_gate")  ? "gate_proj"
                                   : (leaf == "moe_up")    ? "up_proj"
                                   : (leaf == "moe_down")  ? "down_proj"
                                   : (leaf == "moe_sh_gate") ? "gate_proj"
                                   : (leaf == "moe_sh_up")   ? "up_proj"
                                                             : "down_proj";
            if (shared) {
                ExportTensor t;
                t.hf_name = L + "mlp.shared_expert." + proj + ".weight";
                t.shape = p->shape;
                t.dtype = dtype;
                t.nbytes = static_cast<size_t>(p->numel()) * safe_dtype_size(dtype);
                const Parameter* pp = p;
                t.fill = [pp, dtype](uint8_t* dst) {
                    encode(host_f32(pp->w), dtype, dst);
                };
                out.push_back(std::move(t));
                continue;
            }
            // The engine keeps the experts flat; HuggingFace keeps one
            // tensor per expert. The engine's layout is already
            // expert-major, so each expert is a contiguous row range.
            const int rows_per_expert = (leaf == "moe_down") ? cfg.hidden_size : E;
            const int cols = (leaf == "moe_down") ? E : static_cast<int>(d);
            for (int e = 0; e < ne; ++e) {
                ExportTensor t;
                t.hf_name = L + "mlp.experts." + std::to_string(e) + "." + proj + ".weight";
                t.shape = {rows_per_expert, cols};
                t.dtype = dtype;
                t.nbytes = static_cast<size_t>(rows_per_expert) * static_cast<size_t>(cols) *
                           safe_dtype_size(dtype);
                const Parameter* pp = p;
                const size_t row_off = static_cast<size_t>(e) * static_cast<size_t>(rows_per_expert);
                t.fill = [pp, row_off, rows_per_expert, cols, dtype](uint8_t* dst) {
                    std::vector<float> all = host_f32(pp->w);
                    const size_t off = row_off * static_cast<size_t>(cols);
                    const size_t n = static_cast<size_t>(rows_per_expert) * static_cast<size_t>(cols);
                    std::vector<float> part(all.begin() + static_cast<std::ptrdiff_t>(off),
                                            all.begin() + static_cast<std::ptrdiff_t>(off + n));
                    encode(part, dtype, dst);
                };
                out.push_back(std::move(t));
            }
            continue;
        }

        // ---- everything else is a straight copy ----
        const std::string hn = hf_tensor_name(name, granite);
        if (hn.empty()) continue;   // no HuggingFace counterpart
        ExportTensor t;
        t.hf_name = hn;
        t.shape = p->shape;
        t.dtype = dtype;
        t.nbytes = static_cast<size_t>(p->numel()) * safe_dtype_size(dtype);
        const Parameter* pp = p;
        t.fill = [pp, dtype](uint8_t* dst) {
            encode(host_f32(pp->w), dtype, dst);
        };
        out.push_back(std::move(t));
    }

    return out;
}

// ---------------------------------------------------------------------------
// config.json
// ---------------------------------------------------------------------------

std::string hf_config_json(const ModelConfig& cfg, const std::string& model_name,
                           const std::string& description) {
    Json j = Json::object();
    j.set("architectures", [&] {
        Json a = Json::array();
        a.push(Json::string(cfg.arch == ModelConfig::ArchGranite
                                ? "GraniteMoeHybridForCausalLM"
                                : (cfg.use_moe ? "Qwen3MoeForCausalLM" : "Qwen3ForCausalLM")));
        return a;
    }());
    j.set("model_type", Json::string(cfg.arch == ModelConfig::ArchGranite
                                         ? "granitemoehybrid"
                                         : (cfg.use_moe ? "qwen3_moe" : "qwen3")));
    j.set("hidden_size", Json::integer(cfg.hidden_size));
    j.set("num_hidden_layers", Json::integer(cfg.num_layers));
    j.set("num_attention_heads", Json::integer(cfg.num_heads));
    j.set("num_key_value_heads", Json::integer(cfg.num_kv_heads));
    j.set("intermediate_size", Json::integer(cfg.intermediate_size));
    j.set("vocab_size", Json::integer(cfg.vocab_size));
    j.set("max_position_embeddings", Json::integer(cfg.max_seq_len));
    j.set("rms_norm_eps", Json::number(cfg.rms_eps));
    j.set("rope_theta", Json::number(cfg.rope_theta));
    j.set("tie_word_embeddings", Json::boolean(cfg.tie_embeddings));
    j.set("head_dim", Json::integer(cfg.head_dim()));
    j.set("hidden_act", Json::string("silu"));
    j.set("torch_dtype", Json::string("float16"));
    j.set("use_cache", Json::boolean(true));
    if (!model_name.empty()) {
        j.set("name_or_path", Json::string(model_name));
    }
    if (!description.empty()) {
        j.set("description", Json::string(description));
    }
    if (cfg.sliding_window > 0) {
        j.set("sliding_window", Json::integer(cfg.sliding_window));
        j.set("use_sliding_window", Json::boolean(true));
    }
    if (cfg.use_qkv_bias) j.set("attention_bias", Json::boolean(true));
    if (cfg.use_qk_norm) j.set("use_qk_norm", Json::boolean(true));
    if (cfg.rope_type == 1) j.set("rope_scaling_style", Json::string("neox"));
    if (cfg.rope_scale > 1.0f) {
        Json rs = Json::object();
        rs.set("rope_type", Json::string("yarn"));
        rs.set("factor", Json::number(cfg.rope_scale));
        rs.set("original_max_position_embeddings", Json::integer(cfg.max_seq_len));
        if (cfg.rope_yarn_mscale > 0.0f) rs.set("mscale", Json::number(cfg.rope_yarn_mscale));
        j.set("rope_scaling", rs);
    }

    if (cfg.arch == ModelConfig::ArchGranite) {
        j.set("num_local_experts", Json::integer(cfg.num_experts));
        j.set("num_experts_per_tok", Json::integer(cfg.moe_top_k));
        j.set("shared_intermediate_size", Json::integer(cfg.shared_expert_intermediate_size));
        j.set("embedding_multiplier", Json::number(cfg.embedding_multiplier));
        j.set("residual_multiplier", Json::number(cfg.residual_multiplier));
        j.set("attention_multiplier", Json::number(cfg.attention_multiplier));
        j.set("logits_scaling", Json::number(cfg.logits_scaling));
        j.set("position_embedding_type",
              Json::string(cfg.position_embedding == 1 ? "nope" : "rope"));
        j.set("mamba_n_heads", Json::integer(cfg.mamba_n_heads));
        j.set("mamba_d_head", Json::integer(cfg.mamba_d_head));
        j.set("mamba_d_state", Json::integer(cfg.mamba_d_state));
        j.set("mamba_n_groups", Json::integer(cfg.mamba_n_groups));
        j.set("mamba_d_conv", Json::integer(cfg.mamba_d_conv));
        j.set("mamba_expand", Json::integer(cfg.mamba_expand));
        j.set("mamba_chunk_size", Json::integer(cfg.mamba_chunk_size));
        j.set("mamba_conv_bias", Json::boolean(cfg.mamba_conv_bias));
        j.set("mamba_proj_bias", Json::boolean(cfg.mamba_proj_bias));
        Json lt = Json::array();
        for (int k : cfg.layer_types) {
            lt.push(Json::string(k == 0 ? "mamba" : "attention"));
        }
        j.set("layer_types", lt);
    } else if (cfg.use_moe) {
        j.set("num_experts", Json::integer(cfg.num_experts));
        j.set("num_experts_per_tok", Json::integer(cfg.moe_top_k));
        j.set("moe_intermediate_size", Json::integer(cfg.moe_expert_dim));
        if (cfg.shared_expert_intermediate_size > 0)
            j.set("shared_expert_intermediate_size",
                  Json::integer(cfg.shared_expert_intermediate_size));
    }
    return j.dump_pretty();
}

// ---------------------------------------------------------------------------
// the export
// ---------------------------------------------------------------------------

namespace {

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.good()) return false;
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    f.flush();
    return f.good();
}

}  // namespace

bool export_model_safetensors(const std::string& dir, Model& model,
                              const Tokenizer& tokenizer,
                              const std::string& model_name,
                              const std::string& description,
                              SafeDType dtype,
                              size_t shard_limit) {
    std::error_code ec;
    fs::create_directories(dir, ec);

    const std::vector<ExportTensor> plan = plan_safetensors_export(model, dtype);
    if (plan.empty()) {
        log_error("[hf ] nothing to export: the model has no HuggingFace counterpart");
        return false;
    }

    // ---- group the tensors into shards ----
    struct Shard {
        std::string file;
        std::vector<size_t> tensors;   // the indices in the plan
        size_t bytes = 0;
    };
    std::vector<Shard> shards;
    {
        Shard cur;
        for (size_t i = 0; i < plan.size(); ++i) {
            if (!cur.tensors.empty() && cur.bytes + plan[i].nbytes > shard_limit) {
                shards.push_back(cur);
                cur = Shard();
            }
            cur.tensors.push_back(i);
            cur.bytes += plan[i].nbytes;
        }
        if (!cur.tensors.empty()) shards.push_back(cur);
    }
    const size_t nshards = shards.size();

    // ---- write the shards ----
    std::map<std::string, std::string> weight_map;   // tensor -> file
    uint64_t total_size = 0;
    for (size_t s = 0; s < nshards; ++s) {
        char namebuf[64];
        if (nshards == 1) {
            std::snprintf(namebuf, sizeof(namebuf), "model.safetensors");
        } else {
            std::snprintf(namebuf, sizeof(namebuf), "model-%05zu-of-%05zu.safetensors",
                          s + 1, nshards);
        }
        shards[s].file = namebuf;

        SafeTensorsWriter w((fs::path(dir) / namebuf).string());
        for (size_t idx : shards[s].tensors) {
            w.reserve(plan[idx].hf_name, plan[idx].shape, plan[idx].dtype, plan[idx].nbytes);
        }
        if (!w.begin()) {
            log_error("[hf ] cannot write the shard " + std::string(namebuf));
            return false;
        }
        for (size_t idx : shards[s].tensors) {
            std::vector<uint8_t> buf(plan[idx].nbytes);
            if (plan[idx].fill) plan[idx].fill(buf.data());
            if (!w.append(buf.data(), buf.size())) {
                log_error("[hf ] cannot write the tensor " + plan[idx].hf_name);
                return false;
            }
        }
        if (!w.finish()) {
            log_error("[hf ] cannot finalise the shard " + std::string(namebuf));
            return false;
        }
        for (size_t idx : shards[s].tensors) {
            weight_map[plan[idx].hf_name] = namebuf;
            total_size += plan[idx].nbytes;
        }
    }

    // ---- the index (only when the model is sharded) ----
    if (nshards > 1) {
        Json index = Json::object();
        Json meta = Json::object();
        meta.set("total_size", Json::integer(static_cast<i64>(total_size)));
        index.set("metadata", meta);
        Json wm = Json::object();
        for (const auto& [tensor, file] : weight_map) wm.set(tensor, Json::string(file));
        index.set("weight_map", wm);
        if (!write_file((fs::path(dir) / "model.safetensors.index.json").string(), index.dump_pretty())) {
            log_error("[hf ] cannot write model.safetensors.index.json");
            return false;
        }
    }

    // ---- config.json ----
    const std::string cfg_json = hf_config_json(model.config(), model_name, description);
    if (!write_file((fs::path(dir) / "config.json").string(), cfg_json)) {
        log_error("[hf ] cannot write config.json");
        return false;
    }

    // ---- generation_config.json (the stop tokens) ----
    {
        Json g = Json::object();
        g.set("bos_token_id", Json::integer(tokenizer.bos_id()));
        g.set("eos_token_id", Json::integer(tokenizer.eos_id()));
        g.set("pad_token_id", Json::integer(tokenizer.pad_id()));
        if (!model_name.empty()) g.set("name_or_path", Json::string(model_name));
        write_file((fs::path(dir) / "generation_config.json").string(), g.dump_pretty());
    }

    log_info(strfmt("[hf ] exported %s | %zu tensors | %zu shard(s) | %s | dtype=%s",
                    dir.c_str(), plan.size(), nshards,
                    human_bytes(total_size).c_str(), safe_dtype_name(dtype)));
    return true;
}

}  // namespace gai
