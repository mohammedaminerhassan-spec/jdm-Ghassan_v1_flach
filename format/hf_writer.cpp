#include "format/hf_writer.h"

#include "core/json.h"
#include "format/hf_checkpoint.h"
#include "quantization/quantize.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

namespace gai {

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

SafeTensorsWriter::SafeTensorsWriter(const std::string& path) : path_(path) {}

SafeTensorsWriter::~SafeTensorsWriter() {
    if (out_.is_open()) {
        out_.flush();
        out_.close();
    }
    if (begun_ && !tmp_path_.empty()) {
        std::error_code ec;
        fs::remove(tmp_path_, ec);
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
    const size_t pad = (8 - ((8 + head.size()) % 8)) % 8;
    head.append(pad, ' ');

    tmp_path_ = path_ + ".tmp";
    out_.open(tmp_path_, std::ios::binary | std::ios::trunc);
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
    bool ok = out_.good();
    out_.close();
    ok = ok && !out_.fail() && !failed_;
    begun_ = false;
    if (!ok) {
        std::error_code ec;
        if (!tmp_path_.empty()) fs::remove(tmp_path_, ec);
        return false;
    }
    std::error_code ec;
    fs::path backup;
    if (fs::exists(path_, ec) && !ec) {
        backup = fs::path(path_ + ".bak");
        std::error_code stale;
        fs::remove_all(backup, stale);
        fs::rename(path_, backup, ec);
        if (ec) { log_error("[hf ] cannot back up " + path_ + ": " + ec.message()); return false; }
    }
    fs::rename(tmp_path_, path_, ec);
    if (ec) {
        log_error("[hf ] cannot publish " + path_ + ": " + ec.message());
        if (!backup.empty()) {
            std::error_code rec;
            fs::rename(backup, path_, rec);
            if (rec) log_error("[hf ] cannot restore backup " + backup.string() + ": " + rec.message());
        }
        return false;
    }
    if (!backup.empty()) {
        std::error_code rec;
        fs::remove_all(backup, rec);
        if (rec) log_error("[hf ] cannot remove backup " + backup.string() + ": " + rec.message());
    }
    return true;
}

namespace {

std::vector<float> host_f32(const Tensor& t) {
    if (!t.defined()) return {};
    Tensor cpu = (t.device() == Device::CPU) ? t.clone() : t.to(Device::CPU);
    if (cpu.dtype() == DType::F32) {
        return std::vector<float>(cpu.f32(), cpu.f32() + cpu.numel());
    }
    Tensor f32 = quant::dequantize(cpu, DType::F32);
    return std::vector<float>(f32.f32(), f32.f32() + f32.numel());
}

std::vector<float> host_f32_merged(const Model& model, const Parameter& p) {
    std::vector<float> base = host_f32(p.w);
    std::vector<float> delta;
    if (model.lora_delta_for(p.name, delta)) {
        GAI_CHECK(delta.size() == base.size(),
                  "the LoRA delta and the base weight have different sizes for " + p.name);
        for (size_t i = 0; i < base.size(); ++i) base[i] += delta[i];
    }
    return base;
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

}

std::vector<ExportTensor> plan_safetensors_export(Model& model, SafeDType dtype) {
    const ModelConfig& cfg = model.config();
    const bool granite = (cfg.arch == ModelConfig::ArchGranite);
    const int  ne = cfg.num_experts;
    const int  E  = cfg.moe_expert_dim;
    const int  SH = cfg.shared_width();
    const i64  d  = cfg.hidden_size;

    std::vector<ExportTensor> out;
    std::map<const Parameter*, std::shared_ptr<std::vector<float>>> host_cache;
    auto cached_host = [&](const Parameter* p) -> std::shared_ptr<std::vector<float>> {
        auto it = host_cache.find(p);
        if (it == host_cache.end()) {
            auto v = std::make_shared<std::vector<float>>(host_f32_merged(model, *p));
            it = host_cache.emplace(p, std::move(v)).first;
        }
        return it->second;
    };
    std::set<std::string> packed;

    for (const Parameter* p : model.parameters()) {
        const std::string& name = p->name;
        if (name.find(".lora_") != std::string::npos) continue;
        if (packed.count(name)) continue;

        const size_t dot = name.rfind('.');
        const std::string leaf = (dot == std::string::npos) ? name : name.substr(dot + 1);
        const int layer = layer_index_of(name);
        const std::string L = (layer >= 0) ? ("model.layers." + std::to_string(layer) + ".") : "";

        if (leaf == "moe_gate" || leaf == "moe_up" || leaf == "moe_down" ||
            leaf == "moe_sh_gate" || leaf == "moe_sh_up" || leaf == "moe_sh_down") {

            if (granite) {
                if (leaf == "moe_gate" || leaf == "moe_up") {
                    const Parameter* other = model.find_parameter(
                        (leaf == "moe_gate") ? (name.substr(0, name.rfind('.')) + ".moe_up")
                                             : (name.substr(0, name.rfind('.')) + ".moe_gate"));
                    if (!other) continue;
                    ExportTensor t;
                    t.hf_name = L + "block_sparse_moe.input_linear.weight";
                    t.shape = {2LL * ne * E, d};
                    t.dtype = dtype;
                    t.nbytes = static_cast<size_t>(2 * ne * E * d) * safe_dtype_size(dtype);
                    const Parameter* pg = (leaf == "moe_gate") ? p : other;
                    const Parameter* pu = (leaf == "moe_gate") ? other : p;
                    t.fill = [&model, pg, pu, ne, E, d, dtype](uint8_t* dst) {
                        std::vector<float> g = host_f32_merged(model, *pg);
                        std::vector<float> u = host_f32_merged(model, *pu);
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
                    const Parameter* pg = (leaf == "moe_sh_gate") ? p : other;
                    const Parameter* pu = (leaf == "moe_sh_gate") ? other : p;
                    t.fill = [&model, pg, pu, SH, d, dtype](uint8_t* dst) {
                        std::vector<float> g = host_f32_merged(model, *pg);
                        std::vector<float> u = host_f32_merged(model, *pu);
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
                    ExportTensor t;
                    t.hf_name = L + "block_sparse_moe.output_linear.weight";
                    t.shape = p->shape;
                    t.dtype = dtype;
                    t.nbytes = static_cast<size_t>(p->numel()) * safe_dtype_size(dtype);
                    const Parameter* pp = p;
                    t.fill = [&model, pp, dtype](uint8_t* dst) {
                        encode(host_f32_merged(model, *pp), dtype, dst);
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
                    t.fill = [&model, pp, dtype](uint8_t* dst) {
                        encode(host_f32_merged(model, *pp), dtype, dst);
                    };
                    out.push_back(std::move(t));
                    continue;
                }
                continue;
            }

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
                t.fill = [&model, pp, dtype](uint8_t* dst) {
                    encode(host_f32_merged(model, *pp), dtype, dst);
                };
                out.push_back(std::move(t));
                continue;
            }
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
                auto all_ptr = cached_host(pp);
                t.fill = [all_ptr, row_off, rows_per_expert, cols, dtype](uint8_t* dst) {
                    const std::vector<float>& all = *all_ptr;
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

        const std::string hn = hf_tensor_name(name, granite);
        if (hn.empty()) continue;
        ExportTensor t;
        t.hf_name = hn;
        t.shape = p->shape;
        t.dtype = dtype;
        t.nbytes = static_cast<size_t>(p->numel()) * safe_dtype_size(dtype);
        const Parameter* pp = p;
        t.fill = [&model, pp, dtype](uint8_t* dst) {
            encode(host_f32_merged(model, *pp), dtype, dst);
        };
        out.push_back(std::move(t));
    }

    return out;
}

std::string hf_config_json(const ModelConfig& cfg, const std::string& model_name,
                           const std::string& description) {
    Json j = Json::object();
    j.set("architectures", [&] {
        Json a = Json::array();
        a.push(Json::string(cfg.arch == ModelConfig::ArchGranite
                                ? "GhassanMoeHybridForCausalLM"
                                : (cfg.use_moe ? "GhassanMoeForCausalLM"
                                               : "GhassanForCausalLM")));
        return a;
    }());
    j.set("model_type", Json::string(cfg.arch == ModelConfig::ArchGranite
                                         ? "ghassan_hybrid"
                                         : (cfg.use_moe ? "ghassan_moe" : "ghassan")));
    j.set("hidden_size", Json::integer(cfg.hidden_size));
    j.set("num_hidden_layers", Json::integer(cfg.num_layers));
    j.set("num_attention_heads", Json::integer(cfg.num_heads));
    j.set("num_key_value_heads", Json::integer(cfg.num_kv_heads));
    int inter = cfg.intermediate_size;
    if (cfg.arch == ModelConfig::ArchGranite) {
        inter = cfg.moe_expert_dim;
    } else if (cfg.use_moe) {
        inter = cfg.shared_expert_intermediate_size > 0
                    ? cfg.shared_expert_intermediate_size
                    : cfg.moe_expert_dim;
    }
    j.set("intermediate_size", Json::integer(inter));
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
    if (cfg.moe_aux_free) j.set("moe_aux_free", Json::boolean(true));
    if (cfg.use_qk_norm) j.set("use_qk_norm", Json::boolean(true));
    if (cfg.rope_scale > 1.0f) {
        Json rs = Json::object();
        rs.set("rope_type", Json::string("yarn"));
        rs.set("factor", Json::number(cfg.rope_scale));
        rs.set("original_max_position_embeddings",
               Json::integer(cfg.rope_yarn_low > 0 ? static_cast<i64>(cfg.rope_yarn_low)
                                                   : cfg.max_seq_len));
        if (cfg.rope_yarn_mscale > 0.0f) rs.set("mscale", Json::number(cfg.rope_yarn_mscale));
        j.set("rope_scaling", rs);
    }

    if (cfg.arch == ModelConfig::ArchGranite) {
        j.set("num_local_experts", Json::integer(cfg.num_experts));
        j.set("num_experts_per_tok", Json::integer(cfg.moe_top_k));
        if (cfg.moe_shared)
            j.set("shared_intermediate_size", Json::integer(cfg.shared_width()));
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
        if (cfg.moe_shared)
            j.set("shared_expert_intermediate_size", Json::integer(cfg.shared_width()));
    }
    return j.dump_pretty();
}

namespace {

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.good()) return false;
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    f.flush();
    return f.good();
}

}

bool export_model_safetensors(const std::string& dir, Model& model,
                              const Tokenizer& tokenizer,
                              const std::string& model_name,
                              const std::string& description,
                              SafeDType dtype,
                              size_t shard_limit) {
    {
        const int tok_vocab = tokenizer.vocab_size();
        const int model_vocab = model.config().vocab_size;
        if (tok_vocab > model_vocab) {
            log_error(strfmt("[hf ] tokenizer vocab_size=%d exceeds model vocab_size=%d: "
                             "ids %d..%d have no embedding rows and cannot be generated",
                             tok_vocab, model_vocab, model_vocab, tok_vocab - 1));
            return false;
        }
        if (tok_vocab > 0 && tok_vocab < model_vocab)
            log_info(strfmt("[hf ] tokenizer vocab_size=%d < model vocab_size=%d "
                            "(padded embedding rows; harmless)",
                            tok_vocab, model_vocab));
    }

    std::error_code ec;
    fs::create_directories(dir, ec);

    const fs::path stage = fs::path(dir).string() + ".stage";
    fs::remove_all(stage, ec);
    ec.clear();
    fs::create_directories(stage, ec);
    GAI_CHECK(!ec, "cannot create export staging directory: " + ec.message());
    auto publish_cleanup = [&]() { std::error_code c; fs::remove_all(stage, c); };

    const std::vector<ExportTensor> plan = plan_safetensors_export(model, dtype);
    if (plan.empty()) {
        log_error("[hf ] nothing to export: the model has no HuggingFace counterpart");
        return false;
    }

    struct Shard {
        std::string file;
        std::vector<size_t> tensors;
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

    std::map<std::string, std::string> weight_map;
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

        SafeTensorsWriter w((fs::path(stage) / namebuf).string());
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

    if (nshards > 1) {
        Json index = Json::object();
        Json meta = Json::object();
        meta.set("total_size", Json::integer(static_cast<i64>(total_size)));
        index.set("metadata", meta);
        Json wm = Json::object();
        for (const auto& [tensor, file] : weight_map) wm.set(tensor, Json::string(file));
        index.set("weight_map", wm);
        if (!write_file((fs::path(stage) / "model.safetensors.index.json").string(), index.dump_pretty())) {
            log_error("[hf ] cannot write model.safetensors.index.json");
            return false;
        }
    }

    const std::string cfg_json = hf_config_json(model.config(), model_name, description);
    if (!write_file((fs::path(stage) / "config.json").string(), cfg_json)) {
        log_error("[hf ] cannot write config.json");
        return false;
    }

    {
        Json g = Json::object();
        g.set("bos_token_id", Json::integer(tokenizer.bos_id()));
        g.set("eos_token_id", Json::integer(tokenizer.eos_id()));
        g.set("pad_token_id", Json::integer(tokenizer.pad_id()));
        if (!model_name.empty()) g.set("name_or_path", Json::string(model_name));
        const std::string gc_path = (fs::path(stage) / "generation_config.json").string();
        if (!write_file(gc_path, g.dump_pretty())) {
            log_error("[hf ] failed to write " + gc_path);
            return false;
        }
    }

    {
        std::vector<std::pair<fs::path, fs::path>> backups;
        auto rollback = [&]() {
            for (auto it = backups.rbegin(); it != backups.rend(); ++it) {
                std::error_code r_ec;
                if (fs::exists(it->second, r_ec)) fs::rename(it->second, it->first, r_ec);
                else fs::remove(it->first, r_ec);
            }
        };
        bool ok = true;
        std::set<std::string> published;
        for (const auto& e : fs::directory_iterator(stage, ec))
            published.insert(e.path().filename().string());
        for (const auto& e : fs::directory_iterator(stage, ec)) {
            if (ec) { ok = false; break; }
            const fs::path target = fs::path(dir) / e.path().filename();
            const fs::path backup = target.string() + ".bak";
            std::error_code p_ec;
            if (fs::exists(target, p_ec)) {
                fs::remove(backup, p_ec);
                fs::rename(target, backup, p_ec);
                if (p_ec) { ok = false; break; }
                backups.emplace_back(target, backup);
            }
            fs::rename(e.path(), target, p_ec);
            if (p_ec) { ok = false; break; }
        }
        if (!ok) {
            rollback();
            log_error("[hf ] cannot publish the staged export; previous model restored");
            publish_cleanup();
            return false;
        }
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            const std::string n = e.path().filename().string();
            const bool is_weight = (n == "model.safetensors" ||
                                    n == "model.safetensors.index.json" ||
                                    (n.rfind("model-", 0) == 0 && n.find(".safetensors") != std::string::npos));
            if (is_weight && published.find(n) == published.end()) fs::remove(e.path(), ec);
        }
        for (const auto& b : backups) { std::error_code c; fs::remove(b.second, c); }
    }
    publish_cleanup();

    log_info(strfmt("[hf ] exported %s | %zu tensors | %zu shard(s) | %s | dtype=%s",
                    dir.c_str(), plan.size(), nshards,
                    human_bytes(total_size).c_str(), safe_dtype_name(dtype)));
    return true;
}

}
