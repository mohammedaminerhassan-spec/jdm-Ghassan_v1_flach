#include "format/gguf_codec.h"
#include "quantization/quantize.h"
#include "core/common.h"
#include "core/rng.h"
#include "core/dtype.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

namespace fs = std::filesystem;

namespace gai {

template<typename T>
static void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

static inline uint64_t align_up(uint64_t x, uint64_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

static uint64_t checked_stream_pos(std::ostream& os, const std::string& what) {
    if (!os.good()) GAI_FAIL("gguf_format: bad stream before " + what);
    const std::ostream::pos_type p = os.tellp();
    if (p == std::ostream::pos_type(-1)) GAI_FAIL("gguf_format: tell failed before " + what);
    return static_cast<uint64_t>(p);
}

static std::string escape_merge_piece(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 4);
    for (unsigned char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == ' ') o += "\\x20";
        else o.push_back(static_cast<char>(c));
    }
    return o;
}
static std::string unescape_merge_piece(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i + 1] == '\\') { o.push_back('\\'); i += 2; continue; }
            if (i + 3 < s.size() && s[i + 1] == 'x' && s[i + 2] == '2' && s[i + 3] == '0') {
                o.push_back(' '); i += 4; continue;
            }
        }
        o.push_back(s[i++]);
    }
    return o;
}

GGUFWriter::GGUFWriter(const std::string& path) : path_(path) {}

static std::vector<uint8_t> pack_uint64(uint64_t v) {
    std::vector<uint8_t> d(8);
    memcpy(d.data(), &v, 8);
    return d;
}
static std::vector<uint8_t> pack_float32(float v) {
    std::vector<uint8_t> d(4);
    memcpy(d.data(), &v, 4);
    return d;
}
static std::vector<uint8_t> pack_bool(bool v) {
    return { static_cast<uint8_t>(v ? 1 : 0) };
}
static std::vector<uint8_t> pack_string(const std::string& s) {
    std::vector<uint8_t> d;
    uint64_t len = s.size();
    d.resize(8 + len);
    memcpy(d.data(), &len, 8);
    memcpy(d.data() + 8, s.data(), len);
    return d;
}

static void write_metadata_value(std::ostream& os, const GGUFMetadataValue& v) {
    uint32_t t = static_cast<uint32_t>(v.type);
    write_pod(os, t);
    if (v.type == GGUFType::STRING) {

        os.write(reinterpret_cast<const char*>(v.data.data()),
                 static_cast<std::streamsize>(v.data.size()));
    } else if (v.type == GGUFType::ARRAY) {
        os.write(reinterpret_cast<const char*>(v.data.data()),
                 static_cast<std::streamsize>(v.data.size()));
    } else {
        os.write(reinterpret_cast<const char*>(v.data.data()),
                 static_cast<std::streamsize>(v.data.size()));
    }
}

void GGUFWriter::set_arch(const std::string& arch) {
    add_custom_metadata("general.architecture", arch);
}
void GGUFWriter::set_model_name(const std::string& name) {
    add_custom_metadata("general.name", name);
}
void GGUFWriter::set_description(const std::string& desc) {
    add_custom_metadata("general.description", desc);
}
void GGUFWriter::set_vocab_size(uint32_t v) {
    add_custom_metadata("ghassan.vocab_size", static_cast<uint64_t>(v));
}
void GGUFWriter::set_hidden_size(uint32_t h) {
    add_custom_metadata("ghassan.embedding_length", static_cast<uint64_t>(h));
}
void GGUFWriter::set_num_layers(uint32_t l) {
    add_custom_metadata("ghassan.block_count", static_cast<uint64_t>(l));
}
void GGUFWriter::set_num_heads(uint32_t h) {
    add_custom_metadata("ghassan.attention.head_count", static_cast<uint64_t>(h));
}
void GGUFWriter::set_num_kv_heads(uint32_t kv) {
    add_custom_metadata("ghassan.attention.head_count_kv", static_cast<uint64_t>(kv));
}
void GGUFWriter::set_intermediate_size(uint32_t f) {
    add_custom_metadata("ghassan.feed_forward_length", static_cast<uint64_t>(f));
}
void GGUFWriter::set_moe_config(bool use_moe, uint32_t num_experts, uint32_t top_k,
                                uint32_t expert_dim, bool shared) {
    add_custom_metadata("ghassan.use_moe", use_moe);
    add_custom_metadata("ghassan.moe.expert_count", static_cast<uint64_t>(num_experts));
    add_custom_metadata("ghassan.moe.experts_used_count", static_cast<uint64_t>(top_k));
    add_custom_metadata("ghassan.moe.expert_feed_forward_length",
                        static_cast<uint64_t>(expert_dim));
    add_custom_metadata("ghassan.moe.shared_expert", shared);
}
void GGUFWriter::set_max_seq_len(uint32_t ctx) {
    add_custom_metadata("ghassan.context_length", static_cast<uint64_t>(ctx));
}
void GGUFWriter::set_rope_theta(float theta) {
    add_custom_metadata("ghassan.rope.freq_base", theta);
}
void GGUFWriter::set_rope_scaling(const std::map<std::string, std::string>& kv) {
    // Persist scaling entries instead of silently dropping them; readers
    // that ignore unknown ghassan.* keys are unaffected.
    for (const auto& [k, v] : kv) add_custom_metadata("ghassan.rope." + k, v);
}
void GGUFWriter::set_rms_eps(float eps) {
    add_custom_metadata("ghassan.attention.layer_norm_rms_epsilon", eps);
}
void GGUFWriter::set_tie_embeddings(bool tie) {
    add_custom_metadata("ghassan.tie_word_embeddings", tie);
}
void GGUFWriter::set_norm_type(const std::string& type) {
    add_custom_metadata("ghassan.norm_type", type);
}
void GGUFWriter::set_ffn_type(const std::string& type) {
    add_custom_metadata("ghassan.ffn_type", type);
}
void GGUFWriter::set_attn_type(const std::string& type) {
    add_custom_metadata("ghassan.attention.type", type);
}
void GGUFWriter::set_tokenizer_model(const std::string& model) {
    add_custom_metadata("tokenizer.ggml.model", model);
}
void GGUFWriter::set_tokenizer_tokens(const std::vector<std::string>& tokens) {

    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;

    uint32_t elem_type = static_cast<uint32_t>(GGUFType::STRING);
    uint64_t count     = static_cast<uint64_t>(tokens.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (const auto& t : tokens) {
        uint64_t len = t.size();
        push(&len, 8);
        push(t.data(), t.size());
    }
    metadata_["tokenizer.ggml.tokens"] = std::move(v);
}
void GGUFWriter::set_tokenizer_scores(const std::vector<float>& scores) {
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::FLOAT32);
    uint64_t count     = static_cast<uint64_t>(scores.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (float f : scores) push(&f, 4);
    metadata_["tokenizer.ggml.scores"] = std::move(v);
}
void GGUFWriter::set_tokenizer_token_types(const std::vector<int32_t>& types) {
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::INT32);
    uint64_t count     = static_cast<uint64_t>(types.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (int32_t i : types) push(&i, 4);
    metadata_["tokenizer.ggml.token_type"] = std::move(v);
}
void GGUFWriter::set_tokenizer_bos_id(uint32_t id) {

    set_u32("tokenizer.ggml.bos_token_id", id);
}
void GGUFWriter::set_tokenizer_eos_id(uint32_t id) {
    set_u32("tokenizer.ggml.eos_token_id", id);
}
void GGUFWriter::set_tokenizer_unk_id(uint32_t id) {
    set_u32("tokenizer.ggml.unknown_token_id", id);
}
void GGUFWriter::set_tokenizer_pad_id(uint32_t id) {
    set_u32("tokenizer.ggml.padding_token_id", id);
}
void GGUFWriter::set_tokenizer_merges(const std::vector<std::string>& merges) {
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::STRING);
    uint64_t count     = static_cast<uint64_t>(merges.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (const auto& m : merges) {
        uint64_t len = m.size();
        push(&len, 8);
        push(m.data(), m.size());
    }
    metadata_["tokenizer.ggml.merges"] = std::move(v);
}
void GGUFWriter::set_tokenizer_pre(const std::string& pre) {
    add_custom_metadata("tokenizer.ggml.pre", pre);
}
void GGUFWriter::set_tokenizer_chat_template(const std::string& tmpl) {
    add_custom_metadata("tokenizer.chat_template", tmpl);
}
void GGUFWriter::set_u32(const std::string& key, uint32_t v) {
    GGUFMetadataValue m;
    m.type = GGUFType::UINT32;
    m.data.resize(4);
    memcpy(m.data.data(), &v, 4);
    metadata_[key] = std::move(m);
}
void GGUFWriter::set_general_type(const std::string& t) {
    add_custom_metadata("general.type", t);
}
void GGUFWriter::set_quantization_version(uint32_t v) {

    set_u32("general.quantization_version", v);
}
void GGUFWriter::set_size_label(const std::string& s) {
    add_custom_metadata("general.size_label", s);
}
void GGUFWriter::set_tokenizer_add_bos(bool add) {
    add_custom_metadata("tokenizer.ggml.add_bos_token", add);
}
void GGUFWriter::set_tokenizer_add_eos(bool add) {
    add_custom_metadata("tokenizer.ggml.add_eos_token", add);
}
void GGUFWriter::set_quantization_profile(const std::string& profile) {

    set_quantization_version(2);
    add_custom_metadata("ghassan.quantization.profile", profile);
}
void GGUFWriter::set_file_type(uint32_t type) {

    set_u32("general.file_type", type);
}

void GGUFWriter::add_custom_metadata(const std::string& key, const std::string& value) {
    GGUFMetadataValue v;
    v.type = GGUFType::STRING;
    v.data = pack_string(value);
    metadata_[key] = std::move(v);
}
void GGUFWriter::add_custom_metadata(const std::string& key, uint64_t value) {
    GGUFMetadataValue v;
    v.type = GGUFType::UINT64;
    v.data = pack_uint64(value);
    metadata_[key] = std::move(v);
}
void GGUFWriter::add_custom_metadata(const std::string& key, float value) {
    GGUFMetadataValue v;
    v.type = GGUFType::FLOAT32;
    v.data = pack_float32(value);
    metadata_[key] = std::move(v);
}
void GGUFWriter::add_custom_metadata(const std::string& key, bool value) {
    GGUFMetadataValue v;
    v.type = GGUFType::BOOL;
    v.data = pack_bool(value);
    metadata_[key] = std::move(v);
}

void GGUFWriter::add_tensor(const std::string& name, const Tensor& t, GGMLType dtype) {
    tensors_.emplace_back(name, std::make_pair(t, dtype));
}

static size_t ggml_type_size(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::Q4_0: return 18;
        case GGMLType::Q8_0: return 34;
        default:             GAI_FAIL("ggml_type_size: unsupported GGML type");
    }
}

static size_t ggml_blck_size(GGMLType t) {
    switch (t) {
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q8_0: return 32;
        default:             return 1;
    }
}

static size_t ggml_nbytes(GGMLType t, size_t nelems) {
    size_t blk = ggml_blck_size(t);
    size_t bs  = ggml_type_size(t);
    return (nelems + blk - 1) / blk * bs;
}

static std::vector<uint8_t> tensor_to_ggml_bytes(const Tensor& t, GGMLType target) {

    size_t n = static_cast<size_t>(t.numel());
    const float* src = t.f32();
    GAI_CHECK(src != nullptr, "gguf_format: tensor must be F32 on CPU for export");

    if (target == GGMLType::F32) {
        std::vector<uint8_t> out(n * 4);
        memcpy(out.data(), src, n * 4);
        return out;
    }

    if (target == GGMLType::F16) {
        std::vector<uint8_t> out(n * 2);
        uint16_t* dst = reinterpret_cast<uint16_t*>(out.data());
        for (size_t i = 0; i < n; ++i) {
            dst[i] = fp32_to_fp16(src[i]);
        }
        return out;
    }

    if (target == GGMLType::Q8_0) {

        size_t nblocks = (n + 31) / 32;
        std::vector<uint8_t> out(nblocks * 34);
        for (size_t b = 0; b < nblocks; ++b) {
            size_t start = b * 32;
            size_t end   = std::min(start + 32, n);

            float amax = 0.f;
            for (size_t i = start; i < end; ++i) amax = std::max(amax, std::abs(src[i]));
            uint8_t* block = out.data() + b * 34;
            if (!std::isfinite(amax)) {
                uint16_t zero_scale = fp32_to_fp16(0.0f);
                memcpy(block, &zero_scale, 2);
                std::memset(block + 2, 0, 32);
                continue;
            }
            float scale = amax / 127.f;
            float inv   = (scale != 0.f) ? (1.f / scale) : 0.f;

            uint16_t sh = fp32_to_fp16(scale);
            memcpy(block, &sh, 2);
            for (size_t i = 0; i < 32; ++i) {
                float fv = (start + i < end) ? src[start + i] : 0.f;

                int v = static_cast<int>(std::lround(fv * inv));
                v = std::max(-127, std::min(127, v));
                block[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(v));
            }
        }
        return out;
    }

    if (target == GGMLType::Q4_0) {

        size_t nblocks = (n + 31) / 32;
        std::vector<uint8_t> out(nblocks * 18);
        for (size_t b = 0; b < nblocks; ++b) {
            size_t start = b * 32;
            size_t end   = std::min(start + 32, n);
            uint8_t* block = out.data() + b * 18;

            // GGML's Q4_0 reference (quantize_row_q4_0_ref):
            //   amax = max|x|, d = amax / -8, id = 1/d,
            //   xi = min(15, (int8)(x*id + 8.5f)), d = (x0*xi0 + x1*xi1)
            // The block stores d as f16 and the nibbles as (xi - 8) so the
            // reconstruction is (q - 8) * d.
            float amax = 0.f;
            for (size_t i = start; i < end; ++i) amax = std::max(amax, std::abs(src[i]));
            if (!std::isfinite(amax) || amax == 0.f) {
                uint16_t zero_scale = fp32_to_fp16(0.0f);
                memcpy(block, &zero_scale, 2);
                std::memset(block + 2, 0x88, 16);
                continue;
            }
            const float d    = amax / -8.f;          // negative: the sign carries the scale
            const float id   = 1.0f / d;
            uint16_t sh = fp32_to_fp16(d);
            memcpy(block, &sh, 2);
            for (size_t j = 0; j < 16; ++j) {
                const float f0 = (start + j       < end) ? src[start + j]       : 0.f;
                const float f1 = (start + j + 16  < end) ? src[start + j + 16]  : 0.f;
                const int   xi0 = std::max(0, std::min(15, static_cast<int>(f0 * id + 8.5f)));
                const int   xi1 = std::max(0, std::min(15, static_cast<int>(f1 * id + 8.5f)));
                block[2 + j] = static_cast<uint8_t>((xi1 << 4) | xi0);
            }
        }
        return out;
    }

    std::vector<uint8_t> out(n * 4);
    memcpy(out.data(), src, n * 4);
    return out;
}

void GGUFWriter::write() {
    fs::path p(path_);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "gguf_format: cannot open for write: " + tmp);

        write_pod(f, GGUF_MAGIC);
        write_pod(f, GGUF_VERSION);

        uint64_t tensor_count   = static_cast<uint64_t>(tensors_.size());
        uint64_t metadata_count = static_cast<uint64_t>(metadata_.size());
        write_pod(f, tensor_count);
        write_pod(f, metadata_count);

        for (const auto& [key, val] : metadata_) {
            write_string(f, key);
            write_metadata_value(f, val);
        }

        struct TensorData {
            std::string            name;
            std::vector<uint64_t>  dims;
            GGMLType               type;
            uint64_t               nbytes = 0;
            uint64_t               offset = 0;
        };
        std::vector<TensorData> tdata;
        tdata.reserve(tensors_.size());

        for (auto& [name, pair] : tensors_) {
            const Tensor& t  = pair.first;
            GGMLType dtype   = pair.second;
            TensorData td;
            td.name = name;

            auto shape = t.shape();
            for (auto it = shape.rbegin(); it != shape.rend(); ++it)
                td.dims.push_back(static_cast<uint64_t>(*it));
            td.type = dtype;
            td.nbytes = ggml_nbytes(dtype, static_cast<size_t>(t.numel()));
            tdata.push_back(std::move(td));
        }

        uint64_t info_size = 0;
        for (const auto& td : tdata) {
            info_size += 8 + td.name.size();
            info_size += 4;
            info_size += 8 * td.dims.size();
            info_size += 4;
            info_size += 8;
        }

        const uint64_t info_start = checked_stream_pos(f, "tensor info");

        uint64_t running_offset = 0;
        for (auto& td : tdata) {
            td.offset = running_offset;
            running_offset += td.nbytes;
            running_offset = align_up(running_offset, 32);
        }

        for (const auto& td : tdata) {
            write_string(f, td.name);
            uint32_t ndims = static_cast<uint32_t>(td.dims.size());
            write_pod(f, ndims);
            for (uint64_t d : td.dims) write_pod(f, d);
            uint32_t dtype_u32 = static_cast<uint32_t>(td.type);
            write_pod(f, dtype_u32);
            write_pod(f, td.offset);
        }
        const uint64_t info_end = checked_stream_pos(f, "tensor info");
        GAI_CHECK(info_end >= info_start && info_end - info_start == info_size,
                  "gguf_format: tensor info size mismatch");

        uint64_t cur = checked_stream_pos(f, "tensor data padding");
        uint64_t pad_to = align_up(cur, 32);
        for (uint64_t i = cur; i < pad_to; ++i) f.put('\0');

        for (size_t ti = 0; ti < tdata.size(); ++ti) {
            const auto& td = tdata[ti];
            const Tensor& src = tensors_[ti].second.first;
            Tensor cpu = src.to(Device::CPU);
            if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
            std::vector<uint8_t> payload = tensor_to_ggml_bytes(cpu, td.type);
            GAI_CHECK(payload.size() == td.nbytes, "gguf_format: payload size drift for " + td.name);
            uint64_t before = checked_stream_pos(f, "tensor write for " + td.name);
            f.write(reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
            uint64_t after = checked_stream_pos(f, "tensor padding for " + td.name);
            uint64_t padded = align_up(after, 32);
            for (uint64_t i = after; i < padded; ++i) f.put('\0');
            (void)before;
        }

        GAI_CHECK(f.good(), "gguf_format: write failed");
        f.flush();

    }

    {
        std::error_code ec;
        fs::rename(tmp, path_, ec);
        if (ec) {
#if defined(_WIN32)
            fs::remove(path_, ec);
            ec.clear();
            fs::rename(tmp, path_, ec);
#endif
            GAI_CHECK(!ec, "gguf_format: cannot finalise file: " + ec.message());
        }
    }

    log_info(strfmt("[gguf] wrote %s (%zu tensors, %zu metadata entries)",
                    path_.c_str(), tensors_.size(), metadata_.size()));
}

bool GGUFReader::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    uint32_t magic = 0;
    if (!read_pod(f, magic) || magic != GGUF_MAGIC) return false;

    uint32_t version = 0;
    if (!read_pod(f, version) || (version < 2 || version > 3)) {
        log_warn("gguf_format: unsupported version " + std::to_string(version));
        return false;
    }

    uint64_t tensor_count = 0, metadata_count = 0;
    if (!read_pod(f, tensor_count)) return false;
    if (!read_pod(f, metadata_count)) return false;

    if (tensor_count > 100000 || metadata_count > 100000) {
        log_warn("gguf_format: absurd header counts (tensors/metadata); refusing file");
        return false;
    }

    if (!read_metadata(f, metadata_count)) return false;
    if (!read_tensor_info(f, tensor_count)) return false;

    std::error_code ec;
    file_size_ = static_cast<uint64_t>(fs::file_size(path, ec));
    if (ec || file_size_ == 0) return false;

    const auto end_info = f.tellg();
    if (end_info == std::istream::pos_type(-1)) return false;
    data_start_ = align_up(static_cast<uint64_t>(end_info), 32);
    if (data_start_ > file_size_) return false;
    path_ = path;
    return true;
}

static size_t gguf_scalar_size(GGUFType t) {
    switch (t) {
        case GGUFType::UINT8:
        case GGUFType::INT8:
        case GGUFType::BOOL:    return 1;
        case GGUFType::UINT16:
        case GGUFType::INT16:   return 2;
        case GGUFType::UINT32:
        case GGUFType::INT32:
        case GGUFType::FLOAT32: return 4;
        case GGUFType::UINT64:
        case GGUFType::INT64:
        case GGUFType::FLOAT64: return 8;
        default: return 0;
    }
}

bool GGUFReader::read_metadata(std::istream& is, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        std::string key;
        if (!read_string(is, key)) return false;
        uint32_t type_u32 = 0;
        if (!read_pod(is, type_u32)) return false;
        GGUFType type = static_cast<GGUFType>(type_u32);

        GGUFMetadataValue val;
        val.type = type;

        auto read_scalar = [&](size_t bytes) {
            val.data.resize(bytes);
            return static_cast<bool>(is.read(reinterpret_cast<char*>(val.data.data()),
                                             static_cast<std::streamsize>(bytes)));
        };

        switch (type) {
            case GGUFType::UINT8:   if (!read_scalar(1)) return false; break;
            case GGUFType::INT8:    if (!read_scalar(1)) return false; break;
            case GGUFType::UINT16:  if (!read_scalar(2)) return false; break;
            case GGUFType::INT16:   if (!read_scalar(2)) return false; break;
            case GGUFType::UINT32:  if (!read_scalar(4)) return false; break;
            case GGUFType::INT32:   if (!read_scalar(4)) return false; break;
            case GGUFType::FLOAT32: if (!read_scalar(4)) return false; break;
            case GGUFType::BOOL:    if (!read_scalar(1)) return false; break;
            case GGUFType::UINT64:  if (!read_scalar(8)) return false; break;
            case GGUFType::INT64:   if (!read_scalar(8)) return false; break;
            case GGUFType::FLOAT64: if (!read_scalar(8)) return false; break;
            case GGUFType::STRING: {
                std::string s;
                if (!read_string(is, s)) return false;
                val.data = pack_string(s);
                break;
            }
            case GGUFType::ARRAY: {
                uint32_t elem_type_u32 = 0;
                uint64_t arr_count     = 0;
                if (!read_pod(is, elem_type_u32)) return false;
                if (!read_pod(is, arr_count))     return false;
                GGUFType elem_type = static_cast<GGUFType>(elem_type_u32);

                auto push = [&](const void* d, size_t n) {
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
                    val.data.insert(val.data.end(), p, p + n);
                };
                push(&elem_type_u32, 4);
                push(&arr_count, 8);

                const bool raw_skip = (elem_type != GGUFType::STRING &&
                                       elem_type != GGUFType::FLOAT32 &&
                                       elem_type != GGUFType::INT32 &&
                                       gguf_scalar_size(elem_type) != 0);
                if (raw_skip)
                    log_warn("gguf_format: skipping array '" + key + "' with element type " +
                             std::to_string(elem_type_u32));
                for (uint64_t j = 0; j < arr_count; ++j) {
                    switch (elem_type) {
                        case GGUFType::STRING: {
                            std::string s;
                            if (!read_string(is, s)) return false;
                            uint64_t len = s.size();
                            push(&len, 8);
                            push(s.data(), s.size());
                            if (key == "tokenizer.ggml.tokens") tokenizer_tokens_.push_back(s);
                            if (key == "tokenizer.ggml.merges") tokenizer_merges_.push_back(s);
                            break;
                        }
                        case GGUFType::FLOAT32: {
                            float fv = 0;
                            if (!read_pod(is, fv)) return false;
                            push(&fv, 4);
                            if (key == "tokenizer.ggml.scores") tokenizer_scores_.push_back(fv);
                            break;
                        }
                        case GGUFType::INT32: {
                            int32_t iv = 0;
                            if (!read_pod(is, iv)) return false;
                            push(&iv, 4);
                            if (key == "tokenizer.ggml.token_type") tokenizer_token_types_.push_back(iv);
                            break;
                        }
                        default: {

                            if (!raw_skip) {
                                log_warn("gguf_format: unsupported array element type " +
                                         std::to_string(elem_type_u32) + " for key '" + key + "'");
                                return false;
                            }
                            const size_t esz = gguf_scalar_size(elem_type);
                            uint8_t buf[8];
                            if (!is.read(reinterpret_cast<char*>(buf),
                                         static_cast<std::streamsize>(esz))) return false;
                            push(buf, esz);
                            break;
                        }
                    }
                }
                break;
            }
            default:

                log_warn("gguf_format: unknown metadata type " + std::to_string(type_u32) +
                         " for key '" + key + "'");
                return false;
        }
        metadata_[key] = std::move(val);
    }
    return true;
}

bool GGUFReader::read_tensor_info(std::istream& is, uint64_t count) {
    tensor_infos_.clear();
    tensor_infos_.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        GGUFTensorInfo ti;
        if (!read_string(is, ti.name)) return false;
        uint32_t ndims = 0;
        if (!read_pod(is, ndims) || ndims > 8) return false;
        ti.dimensions.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d)
            if (!read_pod(is, ti.dimensions[d])) return false;
        uint32_t type_u32 = 0;
        if (!read_pod(is, type_u32)) return false;
        ti.type = static_cast<GGMLType>(type_u32);
        if (!read_pod(is, ti.offset)) return false;
        tensor_infos_.push_back(std::move(ti));
    }
    return true;
}

std::string GGUFReader::get_string(const std::string& key, const std::string& def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end() || it->second.type != GGUFType::STRING) return def;
    const auto& d = it->second.data;
    if (d.size() < 8) return def;
    uint64_t len; memcpy(&len, d.data(), 8);
    if (8 + len > d.size()) return def;
    return std::string(reinterpret_cast<const char*>(d.data() + 8), static_cast<size_t>(len));
}

uint64_t GGUFReader::get_uint(const std::string& key, uint64_t def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& d = it->second.data;
    switch (it->second.type) {
        case GGUFType::UINT32: { uint32_t v; if (d.size() >= 4) { memcpy(&v, d.data(), 4); return v; } break; }
        case GGUFType::UINT64: { uint64_t v; if (d.size() >= 8) { memcpy(&v, d.data(), 8); return v; } break; }
        case GGUFType::INT32:  { int32_t  v; if (d.size() >= 4) { memcpy(&v, d.data(), 4); return static_cast<uint64_t>(v); } break; }
        case GGUFType::INT64:  { int64_t  v; if (d.size() >= 8) { memcpy(&v, d.data(), 8); return static_cast<uint64_t>(v); } break; }
        default: break;
    }
    return def;
}

int64_t GGUFReader::get_int(const std::string& key, int64_t def) const {
    return static_cast<int64_t>(get_uint(key, static_cast<uint64_t>(def)));
}

float GGUFReader::get_float(const std::string& key, float def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& d = it->second.data;
    if (it->second.type == GGUFType::FLOAT32 && d.size() >= 4) {
        float v; memcpy(&v, d.data(), 4); return v;
    }
    return def;
}

bool GGUFReader::get_bool(const std::string& key, bool def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& d = it->second.data;
    if (it->second.type == GGUFType::BOOL && !d.empty()) return d[0] != 0;
    return def;
}

std::vector<i32> GGUFReader::eos_ids() const {
    std::vector<i32> out;
    // Only push an id when the key is actually present. Falling back to 0
    // would make the generator stop on the very first sampled token (0 is a
    // perfectly ordinary token in most vocabularies).
    auto it = metadata_.find("tokenizer.ggml.eos_token_id");
    if (it != metadata_.end() && it->second.type == GGUFType::UINT32) {
        out.push_back(static_cast<i32>(it->second.data.size() >= 8
                                           ? (static_cast<uint64_t>(it->second.data[0]) |
                                              (static_cast<uint64_t>(it->second.data[1]) << 8) |
                                              (static_cast<uint64_t>(it->second.data[2]) << 16) |
                                              (static_cast<uint64_t>(it->second.data[3]) << 24))
                                           : 0u));
    }
    it = metadata_.find("tokenizer.ggml.eot_token_id");
    if (it != metadata_.end() && it->second.type == GGUFType::UINT32 &&
        it->second.data.size() >= 4) {
        const i32 eot = static_cast<i32>(it->second.data[0] |
                                         (static_cast<uint32_t>(it->second.data[1]) << 8) |
                                         (static_cast<uint32_t>(it->second.data[2]) << 16) |
                                         (static_cast<uint32_t>(it->second.data[3]) << 24));
        if (std::find(out.begin(), out.end(), eot) == out.end()) out.push_back(eot);
    }
    return out;
}

bool GGUFReader::has_tokenizer() const {
    return !tokenizer_tokens_.empty();
}

bool GGUFReader::load_tokenizer(Tokenizer& tk) const {
    if (tokenizer_tokens_.empty()) return false;
    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(tokenizer_merges_.size());
    for (const auto& m : tokenizer_merges_) {
        size_t sp = m.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= m.size()) continue;
        std::string l = m.substr(0, sp), r = m.substr(sp + 1);

        bool escaped = (m.find('\\') != std::string::npos);
        if (escaped) {
            merges.emplace_back(unescape_merge_piece(l), unescape_merge_piece(r));
        } else {
            size_t rsp = m.rfind(' ');
            merges.emplace_back(m.substr(0, rsp), m.substr(rsp + 1));
        }
    }

    // A byte-level BPE vocabulary (GPT-2/Qwen/Granite family). The normalizer
    // is gone: text goes through the byte-level pre-tokenizer verbatim.
    const std::string pre = get_string("tokenizer.ggml.pre", "gpt-2");
    VocabOptions opts;
    {
        if (pre == "qwen2" || pre == "qwen-2" || pre == "qwen" || pre == "qwen2.5")
            opts.pretok = PreTokKind::Qwen;
        else if (pre == "ghassan" || pre == "granite" || pre == "granite-4" || pre == "granite-4.0")
            opts.pretok = PreTokKind::Granite;
        else
            opts.pretok = PreTokKind::Gpt2;
    }
    opts.add_bos = get_bool("tokenizer.ggml.add_bos_token", false);
    opts.add_eos = get_bool("tokenizer.ggml.add_eos_token", false);
    opts.bos_id  = static_cast<i32>(get_uint("tokenizer.ggml.bos_token_id", 151643));
    opts.eos_id  = static_cast<i32>(get_uint("tokenizer.ggml.eos_token_id", 151645));
    opts.pad_id  = static_cast<i32>(get_uint("tokenizer.ggml.padding_token_id", 151643));
    opts.unk_id  = static_cast<i32>(get_uint("tokenizer.ggml.unknown_token_id", 151643));

    // The exporter writes GGUF token_type 3 = control, 4 = user-defined
    // (added) and 2 = unknown, so the reader recovers both sets from the
    // types table without trusting any name.
    std::vector<i32> special_ids;
    std::vector<i32> added_ids;
    if (tokenizer_token_types_.size() == tokenizer_tokens_.size()) {
        for (size_t i = 0; i < tokenizer_token_types_.size(); ++i) {
            const int t = tokenizer_token_types_[i];
            if (t == 3 || t == 2) special_ids.push_back(static_cast<i32>(i));
            if (t == 3 || t == 4 || t == 2) added_ids.push_back(static_cast<i32>(i));
        }
    }
    auto mark = [&](i32 id) {
        if (id >= 0) special_ids.push_back(id);
    };
    mark(opts.bos_id);
    mark(opts.eos_id);
    mark(opts.pad_id);
    opts.special_ids = std::move(special_ids);
    opts.added_ids  = std::move(added_ids);

    tk.build(tokenizer_tokens_, merges, opts);

    if (!tk.has_byte_map()) {
        log_error(strfmt("gguf_format: the tokenizer in this file has no byte-level "
                         "alphabet (pre='%s', vocab=%d). The Gai engine encodes "
                         "against byte-level BPE only; SentencePiece/unigram "
                         "vocabularies are not supported.",
                         pre.c_str(), tk.vocab_size()));
        return false;
    }
    return tk.vocab_size() > 0;
}

ModelConfig GGUFReader::model_config() const {
    const bool has_native = metadata_.count("ghassan.block_count") != 0;
    const bool has_llama = metadata_.count("llama.block_count") != 0;

    GAI_CHECK(has_native || has_llama,
              "GGUF has no architecture keys (ghassan.block_count / llama.block_count): "
              "not a Ghassan model file");
    auto ukey = [&](const char* native, const char* llama) {
        return has_native ? std::string("ghassan.") + native : std::string("llama.") + llama;
    };
    ModelConfig c;
    c.vocab_size        = static_cast<int>(get_uint(ukey("vocab_size", "vocab_size"), static_cast<uint64_t>(c.vocab_size)));
    c.hidden_size       = static_cast<int>(get_uint(ukey("embedding_length", "embedding_length"), static_cast<uint64_t>(c.hidden_size)));
    c.num_layers        = static_cast<int>(get_uint(ukey("block_count", "block_count"), static_cast<uint64_t>(c.num_layers)));
    c.num_heads         = static_cast<int>(get_uint(ukey("attention.head_count", "attention.head_count"), static_cast<uint64_t>(c.num_heads)));
    c.num_kv_heads      = static_cast<int>(get_uint(ukey("attention.head_count_kv", "attention.head_count_kv"), static_cast<uint64_t>(c.num_kv_heads)));
    c.intermediate_size = static_cast<int>(get_uint(ukey("feed_forward_length", "feed_forward_length"), static_cast<uint64_t>(c.intermediate_size)));
    if (has_native) {
        c.use_moe           = get_bool("ghassan.use_moe", c.use_moe);
        c.num_experts       = static_cast<int>(get_uint("ghassan.moe.expert_count", static_cast<uint64_t>(c.num_experts)));
        c.moe_top_k         = static_cast<int>(get_uint("ghassan.moe.experts_used_count", static_cast<uint64_t>(c.moe_top_k)));
        c.moe_expert_dim    = static_cast<int>(get_uint("ghassan.moe.expert_feed_forward_length", static_cast<uint64_t>(c.moe_expert_dim)));
        c.moe_shared        = get_bool("ghassan.moe.shared_expert", c.moe_shared);
        c.moe_aux_scale     = get_float("ghassan.moe.aux_scale", c.moe_aux_scale);
        c.moe_jitter        = get_float("ghassan.moe.jitter", c.moe_jitter);
        c.max_seq_len       = static_cast<int>(get_uint("ghassan.context_length", static_cast<uint64_t>(c.max_seq_len)));
        c.rope_theta        = get_float("ghassan.rope.freq_base", c.rope_theta);
        c.rope_scale        = get_float("ghassan.rope.scale", c.rope_scale);
        c.rope_yarn_mscale  = get_float("ghassan.rope.yarn_mscale", c.rope_yarn_mscale);
        c.rope_yarn_low     = get_float("ghassan.rope.yarn_low", c.rope_yarn_low);
        c.rope_yarn_high    = get_float("ghassan.rope.yarn_high", c.rope_yarn_high);
        c.sliding_window    = static_cast<int>(get_uint("ghassan.sliding_window", static_cast<uint64_t>(c.sliding_window)));
        c.rope_type         = static_cast<int>(get_uint("ghassan.rope.type", static_cast<uint64_t>(c.rope_type)));
        c.moe_aux_free      = get_bool("ghassan.moe.aux_free", c.moe_aux_free);
        c.rms_eps           = get_float("ghassan.attention.layer_norm_rms_epsilon", c.rms_eps);
        c.use_qk_norm       = get_bool("ghassan.use_qk_norm", c.use_qk_norm);
        c.use_qkv_bias      = get_bool("ghassan.use_qkv_bias", c.use_qkv_bias);
        c.z_loss_scale      = get_float("ghassan.z_loss_scale", c.z_loss_scale);
        c.tie_embeddings    = get_bool("ghassan.tie_word_embeddings", true);
        // IBM Granite-4.0 hybrid: the layer kinds and the multipliers are
        // part of the architecture, so they must survive the round trip.
        c.arch                   = get_bool("ghassan.hybrid",
                                       get_bool("ghassan.granite", false))
                                       ? ModelConfig::ArchGranite : ModelConfig::ArchQwen;
        c.position_embedding     = static_cast<int>(get_uint("ghassan.position_embedding",
                                                               static_cast<uint64_t>(c.position_embedding)));
        c.head_dim_override      = static_cast<int>(get_uint("ghassan.attention.head_dim",
                                                               static_cast<uint64_t>(c.head_dim_override)));
        c.shared_expert_intermediate_size =
            static_cast<int>(get_uint("ghassan.moe.shared_intermediate_size",
                                       static_cast<uint64_t>(c.shared_expert_intermediate_size)));
        c.embedding_multiplier   = get_float("ghassan.embedding_multiplier", c.embedding_multiplier);
        c.residual_multiplier    = get_float("ghassan.residual_multiplier", c.residual_multiplier);
        c.attention_multiplier   = get_float("ghassan.attention_multiplier", c.attention_multiplier);
        c.logits_scaling         = get_float("ghassan.logits_scaling", c.logits_scaling);
        c.mamba_n_heads          = static_cast<int>(get_uint("ghassan.mamba.n_heads",
                                                               static_cast<uint64_t>(c.mamba_n_heads)));
        c.mamba_d_head           = static_cast<int>(get_uint("ghassan.mamba.d_head",
                                                               static_cast<uint64_t>(c.mamba_d_head)));
        c.mamba_d_state          = static_cast<int>(get_uint("ghassan.mamba.d_state",
                                                               static_cast<uint64_t>(c.mamba_d_state)));
        c.mamba_n_groups         = static_cast<int>(get_uint("ghassan.mamba.n_groups",
                                                               static_cast<uint64_t>(c.mamba_n_groups)));
        c.mamba_d_conv           = static_cast<int>(get_uint("ghassan.mamba.d_conv",
                                                               static_cast<uint64_t>(c.mamba_d_conv)));
        c.mamba_expand           = static_cast<int>(get_uint("ghassan.mamba.expand",
                                                               static_cast<uint64_t>(c.mamba_expand)));
        c.mamba_chunk_size       = static_cast<int>(get_uint("ghassan.mamba.chunk_size",
                                                               static_cast<uint64_t>(c.mamba_chunk_size)));
        c.mamba_conv_bias        = get_bool("ghassan.mamba.conv_bias", c.mamba_conv_bias);
        c.mamba_proj_bias        = get_bool("ghassan.mamba.proj_bias", c.mamba_proj_bias);
        {
            // The layer kinds: a comma separated list of 0 (mamba) / 1 (attention).
            const std::string lt = get_string("ghassan.layer_types", "");
            if (!lt.empty()) {
                c.layer_types.clear();
                size_t i = 0;
                while (i < lt.size()) {
                    size_t comma = lt.find(',', i);
                    c.layer_types.push_back(lt[i] == '1' ? 1 : 0);
                    if (comma == std::string::npos) break;
                    i = comma + 1;
                }
            }
        }
    } else {
        c.use_moe           = false;
        c.max_seq_len       = static_cast<int>(get_uint("llama.context_length", static_cast<uint64_t>(c.max_seq_len)));
        c.rope_theta        = get_float("llama.rope.freq_base", c.rope_theta);
        c.rms_eps           = get_float("llama.attention.layer_norm_rms_epsilon", c.rms_eps);
        c.tie_embeddings    = get_bool("llama.tie_word_embeddings", true);

        const uint64_t ne = get_uint("llama.expert_count", 0);
        if (ne > 0 && ne <= 64) {
            c.use_moe        = true;
            c.num_experts    = static_cast<int>(ne);
            c.moe_top_k      = static_cast<int>(get_uint("llama.expert_used_count", 2));
            c.moe_expert_dim = static_cast<int>(get_uint("llama.expert_feed_forward_length",
                                                         static_cast<uint64_t>(c.intermediate_size)));
            c.moe_shared     = false;
            c.moe_aux_scale  = get_float("llama.moe.aux_scale", 0.0f);
        }

        if (!c.use_qk_norm) {
            for (const auto& ti : tensor_infos_) {
                if (ti.name.find("attn_q_norm.weight") != std::string::npos) {
                    c.use_qk_norm = true;
                    break;
                }
            }
        }
    }
    c.validate();
    return c;
}

const GGUFTensorInfo* GGUFReader::find_tensor(const std::string& name) const {
    for (const auto& ti : tensor_infos_)
        if (ti.name == name) return &ti;
    return nullptr;
}

std::string llama_tensor_name(const std::string& internal) {

    if (internal == "tok_embeddings") return "token_embd.weight";
    if (internal == "final_norm")     return "output_norm.weight";
    if (internal == "lm_head")        return "output.weight";
    const std::string pre = "layers.";
    if (internal.compare(0, pre.size(), pre) != 0) return {};
    size_t dot = internal.find('.', pre.size());
    if (dot == std::string::npos) return {};
    const std::string layer = internal.substr(pre.size(), dot - pre.size());
    const std::string rest  = internal.substr(dot + 1);
    const std::string blk = "blk." + layer + ".";
    if (rest == "attn_norm") return blk + "attn_norm.weight";
    if (rest == "wq")        return blk + "attn_q.weight";
    if (rest == "wk")        return blk + "attn_k.weight";
    if (rest == "wv")        return blk + "attn_v.weight";
    if (rest == "wo")        return blk + "attn_output.weight";
    if (rest == "bq")        return blk + "attn_q.bias";
    if (rest == "bk")        return blk + "attn_k.bias";
    if (rest == "bv")        return blk + "attn_v.bias";
    if (rest == "ffn_norm")  return blk + "ffn_norm.weight";
    if (rest == "w_gate")    return blk + "ffn_gate.weight";
    if (rest == "w_up")      return blk + "ffn_up.weight";
    if (rest == "w_down")    return blk + "ffn_down.weight";
    // QK-norm (Qwen3): the query and the key are normalised per head
    // before the attention. llama.cpp stores them next to the projection.
    if (rest == "qk_qnorm")  return blk + "attn_q_norm.weight";
    if (rest == "qk_knorm")  return blk + "attn_k_norm.weight";
    return {};
}

const GGUFTensorInfo* GGUFReader::find_weight(const std::string& internal) const {
    if (const GGUFTensorInfo* ti = find_tensor(internal)) return ti;
    const std::string alt = llama_tensor_name(internal);
    if (!alt.empty()) {
        if (const GGUFTensorInfo* ti = find_tensor(alt)) return ti;
    }
    const std::string alt_moe = llama_moe_tensor_name(internal);
    if (!alt_moe.empty()) {
        if (const GGUFTensorInfo* ti = find_tensor(alt_moe)) return ti;
    }
    return nullptr;
}

bool llama_compat_ok(const ModelConfig& cfg, std::string* reason) {

    auto fail = [&](const std::string& r) {
        if (reason) *reason = r;
        return false;
    };
    if (cfg.use_moe) return fail("MoE has no llama.cpp dense equivalent; "
                                 "export native GGUF for ghassan-ai, --compat llama_moe for MoE Ollama (needs moe_shared=false), "
                                 "or train a dense (use_moe=false) variant for ollama");
    if (cfg.use_qk_norm) return fail("use_qk_norm has no llama-arch equivalent; disable it for --compat llama");
    if (cfg.rope_scale != 1.0f) return fail("rope_scale YaRN has no exact llama mapping here; use rope_scale=1 for --compat llama");

    if (cfg.moe_aux_free) return fail("moe_aux_free (Pro aux-loss-free) has no llama mapping; use native GGUF");
    if (cfg.sliding_window != 0) return fail("sliding_window (Pro SWA) has no llama mapping here; use 0 for --compat llama");
    if (cfg.rope_type != 0) return fail("rope_type=neox (Pro) mismatches this interleaved kernel build; use 0 for --compat llama");
    return true;
}

bool llama_moe_compat_ok(const ModelConfig& cfg, std::string* reason) {
    auto fail = [&](const std::string& r) {
        if (reason) *reason = r;
        return false;
    };
    if (!cfg.use_moe) return fail("llama_moe needs use_moe=true (use --compat llama for dense)");
    if (cfg.moe_shared) return fail("llama_moe has no shared-expert equivalent in llama.cpp; "
                                    "train with moe_shared=false for Ollama, or use native GGUF (full quality)");
    if (cfg.use_qk_norm) return fail("use_qk_norm has no llama-arch equivalent; disable it for --compat llama_moe");
    if (cfg.rope_scale != 1.0f) return fail("rope_scale YaRN has no exact llama mapping here; use rope_scale=1");
    if (cfg.moe_aux_free) return fail("moe_aux_free has no llama mapping; use native GGUF");
    if (cfg.sliding_window != 0) return fail("sliding_window has no llama mapping here; use 0");
    if (cfg.rope_type != 0) return fail("rope_type=neox mismatches interleaved kernels; use 0");
    return true;
}

std::string llama_moe_tensor_name(const std::string& internal) {
    if (internal == "tok_embeddings") return "token_embd.weight";
    if (internal == "final_norm")     return "output_norm.weight";
    if (internal == "lm_head")        return "output.weight";
    const std::string pre = "layers.";
    if (internal.compare(0, pre.size(), pre) != 0) return {};
    size_t dot = internal.find('.', pre.size());
    if (dot == std::string::npos) return {};
    const std::string layer = internal.substr(pre.size(), dot - pre.size());
    const std::string rest  = internal.substr(dot + 1);
    const std::string blk = "blk." + layer + ".";
    if (rest == "attn_norm") return blk + "attn_norm.weight";
    if (rest == "wq")        return blk + "attn_q.weight";
    if (rest == "wk")        return blk + "attn_k.weight";
    if (rest == "wv")        return blk + "attn_v.weight";
    if (rest == "wo")        return blk + "attn_output.weight";
    if (rest == "bq")        return blk + "attn_q.bias";
    if (rest == "bk")        return blk + "attn_k.bias";
    if (rest == "bv")        return blk + "attn_v.bias";
    if (rest == "ffn_norm")  return blk + "ffn_norm.weight";

    if (rest == "moe_router") return blk + "ffn_gate_inp.weight";
    if (rest == "moe_gate")   return blk + "ffn_gate_exps.weight";
    if (rest == "moe_up")     return blk + "ffn_up_exps.weight";
    if (rest == "moe_down")   return blk + "ffn_down_exps.weight";
    return {};
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0)       bits = (sign << 31) | (mant << 13);
    else if (exp == 31) bits = (sign << 31) | 0x7F800000 | (mant << 13);
    else                bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v; memcpy(&v, &bits, 4); return v;
}

// Reinterprets a flat CPU tensor as a 3D one without touching the data. The
// GGUF writer emits the dimensions reversed, so a [a, b, c] tensor becomes
// GGUF dims [c, b, a] -- which is the llama.cpp layout for the MoE expert
// tensors.
static Tensor reshape_3d(const Tensor& src, i64 a, i64 b, i64 c) {
    GAI_CHECK(src.dtype() == DType::F32 && src.device() == Device::CPU,
              "gguf_format: reshape_3d expects an f32 CPU tensor");
    GAI_CHECK(src.numel() == a * b * c, "gguf_format: reshape_3d element count mismatch");
    Tensor out({a, b, c}, DType::F32, Device::CPU);
    memcpy(out.data_ptr(), src.data_ptr(), static_cast<size_t>(src.nbytes()));
    out.set_name(src.name());
    return out;
}

Tensor GGUFReader::read_tensor_raw(const std::string& name) const {
    const GGUFTensorInfo* ti = find_tensor(name);
    GAI_CHECK(ti != nullptr, "gguf_format: tensor not found: " + name);

    u64 nelems64 = 1;
    for (uint64_t d : ti->dimensions) {

        GAI_CHECK(d <= (1ull << 40), "gguf_format: absurd tensor dim for " + name);
        GAI_CHECK(nelems64 <= ((1ull << 40) / (d == 0 ? 1 : d)),
                  "gguf_format: tensor element overflow for " + name);
        nelems64 *= d;
    }
    const size_t nelems = static_cast<size_t>(nelems64);
    size_t nbytes = ggml_nbytes(ti->type, nelems);

    GAI_CHECK(data_start_ <= file_size_, "gguf_format: bad data section for " + name);
    GAI_CHECK(ti->offset <= file_size_ - data_start_, "gguf_format: tensor offset outside file: " + name);
    const uint64_t abs_offset = data_start_ + ti->offset;
    GAI_CHECK(nbytes <= file_size_ - abs_offset, "gguf_format: tensor overruns file: " + name);

    std::ifstream f(path_, std::ios::binary);
    GAI_CHECK(f.good(), "gguf_format: cannot reopen: " + path_);
    f.seekg(static_cast<std::streamoff>(abs_offset));
    GAI_CHECK(f.good(), "gguf_format: seek failed for tensor " + name);

    std::vector<uint8_t> raw(nbytes);
    GAI_CHECK(static_cast<bool>(f.read(reinterpret_cast<char*>(raw.data()),
                                       static_cast<std::streamsize>(nbytes))),
              "gguf_format: short read for tensor " + name);

    std::vector<i64> shape;
    for (auto it = ti->dimensions.rbegin(); it != ti->dimensions.rend(); ++it)
        shape.push_back(static_cast<i64>(*it));

    Tensor out(shape, DType::F32, Device::CPU);
    GAI_CHECK(static_cast<size_t>(out.numel()) == nelems, "gguf shape mismatch for " + name);

    float* dst = out.f32();
    // Statistics are accumulated in the same pass that fills the tensor, so
    // the integrity check costs no extra memory traffic.
    double sumsq = 0.0;
    i64   nonfinite = 0;
    auto acc = [&](float v) {
        if (!std::isfinite(v)) { ++nonfinite; return; }
        sumsq += static_cast<double>(v) * static_cast<double>(v);
    };

    if (ti->type == GGMLType::F32) {
        memcpy(dst, raw.data(), nelems * 4);
        for (size_t i = 0; i < nelems; ++i) acc(dst[i]);
    } else if (ti->type == GGMLType::F16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < nelems; ++i) { dst[i] = fp16_to_f32(src[i]); acc(dst[i]); }
    } else if (ti->type == GGMLType::Q8_0) {
        size_t nblocks = (nelems + 31) / 32;
        for (size_t b = 0; b < nblocks; ++b) {
            const uint8_t* block = raw.data() + b * 34;
            float scale = fp16_to_f32(*reinterpret_cast<const uint16_t*>(block));
            for (size_t i = 0; i < 32 && b*32+i < nelems; ++i) {
                dst[b*32+i] = static_cast<int8_t>(block[2+i]) * scale;
                acc(dst[b*32+i]);
            }
        }
    } else if (ti->type == GGMLType::Q4_0) {
        size_t nblocks = (nelems + 31) / 32;
        for (size_t b = 0; b < nblocks; ++b) {
            const uint8_t* block = raw.data() + b * 18;
            float scale = fp16_to_f32(*reinterpret_cast<const uint16_t*>(block));
            for (size_t j = 0; j < 16; ++j) {
                if (b*32 + j >= nelems) break;
                uint8_t byte = block[2+j];
                int q0 = static_cast<int>(byte & 0xF) - 8;
                int q1 = static_cast<int>(byte >> 4) - 8;
                dst[b*32 + j] = static_cast<float>(q0) * scale;
                acc(dst[b*32 + j]);
                if (b*32 + j + 16 < nelems) {
                    dst[b*32 + j + 16] = static_cast<float>(q1) * scale;
                    acc(dst[b*32 + j + 16]);
                }
            }
        }
    } else {
        GAI_FAIL("gguf_format: unsupported quantization type for reading: " +
                 std::to_string(static_cast<uint32_t>(ti->type)));
    }

    if (nonfinite > 0) {
        GAI_FAIL(strfmt("gguf_format: tensor '%s' contains %lld non-finite value(s); "
                        "the file is corrupt or was written with a different layout",
                        name.c_str(), static_cast<long long>(nonfinite)));
    }
    if (nelems > 0 && sumsq == 0.0) {
        log_warn(strfmt("gguf_format: tensor '%s' is entirely zero (norm 0)", name.c_str()));
    }
    out.set_name(name);
    return out;
}

Tensor GGUFReader::read_tensor_f32(const std::string& name) const {
    return read_tensor_raw(name);
}

bool load_model_from_gguf(const std::string& path, Model& model) {
    GGUFReader r;
    if (!r.open(path)) {
        log_error("gguf_format: cannot open: " + path);
        return false;
    }
    bool lm_head_from_emb = false;
    for (Parameter* p : model.parameters()) {

        const GGUFTensorInfo* ti = r.find_weight(p->name);
        if (!ti && p->name == "lm_head") {
            // No separate output projection in the file: fall back to the
            // token embedding, like llama.cpp does for tied checkpoints that
            // were exported without an output.weight.
            ti = r.find_weight("tok_embeddings");
            if (ti) lm_head_from_emb = true;
        }
        if (!ti) {
            log_error("gguf_format: missing tensor: " + p->name);
            return false;
        }
        Tensor t = r.read_tensor_f32(ti->name);
        if (t.numel() != p->numel()) {
            log_error(strfmt("gguf tensor size mismatch %s: file=%lld model=%lld",
                             p->name.c_str(),
                             static_cast<long long>(t.numel()),
                             static_cast<long long>(p->numel())));
            return false;
        }
        p->w.copy_from(t);
    }
    if (lm_head_from_emb) {
        log_info("[gguf] no output.weight in the file: the token embedding is used "
                 "as the output projection");
    }

    if (model.config().moe_aux_free) {
        model.ensure_moe_bias();
        for (int l = 0; l < model.config().num_layers; ++l) {
            std::string bn = "layers." + std::to_string(l) + ".moe_bias";
            const GGUFTensorInfo* ti = r.find_tensor(bn);
            if (!ti) continue;
            Tensor t = r.read_tensor_f32(bn);
            float* dst = model.moe_bias_ptr_mut(l);
            if (dst && t.numel() == model.config().num_experts)
                std::memcpy(dst, t.data_ptr(), sizeof(float) * static_cast<size_t>(t.numel()));
        }
    }
    return true;
}

ExportProfileGGUF gguf_profile_for(const std::string& name) {
    ExportProfileGGUF p;
    p.name = name;
    if (name == "q8_k" || name == "int8" || name == "q6_k" || name == "q5_k_m" || name == "q5_k_s") {
        // NOTE: true K-quant writers are not implemented; q8_k maps to the
        // Q8_0 codec (GGML-compatible, 8-bit).
        if (name != "q8_k" && name != "int8")
            log_warn("[gguf] profile " + name + " mapped to Q8_0 (nearest implemented; true K-quant writer is roadmap)");
        p.name           = "q8_k";
        p.default_type   = GGMLType::Q8_0;
        p.embedding_type = GGMLType::Q8_0;
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::Q8_0;
    } else if (name == "fp16" || name == "f16" || name == "f16_t" || name == "half") {
        // Compute is fp16 tensor-core on T4; the export is fp16 too, which is
        // bit-consistent with the compute path and needs no lossy quantizer.
        p.name           = "fp16";
        p.default_type   = GGMLType::F16;
        p.embedding_type = GGMLType::F16;
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::F16;
    } else if (name == "q4_0" || name == "int4" || name == "q4_k_m" || name == "q4_k_s" || name == "q4_k") {
        if (name != "q4_0" && name != "int4")
            log_warn("[gguf] profile " + name + " mapped to Q4_0 (nearest implemented; true K-quant writer is roadmap)");
        p.name           = "q4_0";
        p.default_type   = GGMLType::Q4_0;
        p.embedding_type = GGMLType::Q8_0;
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::Q8_0;
    } else {
        GAI_FAIL("gguf_format: unknown profile: " + name + " (q8_k|q4_0|fp16)");
    }
    return p;
}

void export_model_gguf(const std::string& path, Model& model,
                       const Tokenizer& tokenizer,
                       const ExportProfileGGUF& profile,
                       const std::map<std::string, std::string>& extra_meta,
                       const std::string& compat) {

    const ModelConfig& cfg = model.config();
    const bool want_llama = (compat == "llama");
    const bool want_moe = (compat == "llama_moe");
    if (compat != "native" && !want_llama && !want_moe)
        GAI_FAIL("gguf_format: unknown compat '" + compat + "' (native|llama|llama_moe)");
    if (want_llama) {
        std::string reason;
        GAI_CHECK(llama_compat_ok(cfg, &reason),
                  "gguf_format: --compat llama refused: " + reason);
    }
    if (want_moe) {
        std::string reason;
        GAI_CHECK(llama_moe_compat_ok(cfg, &reason),
                  "gguf_format: --compat llama_moe refused: " + reason);
        log_warn("[gguf] llama_moe is experimental Mixtral-style export; verify with `ghassan-ai logits` vs llama.cpp");
    }
    const std::string arch = (want_llama || want_moe) ? "llama" : "ghassan";

    const Tokenizer& tok = tokenizer;
    GAI_CHECK(tok.vocab_size() <= cfg.vocab_size,
              strfmt("gguf_format: the vocabulary (%d tokens) exceeds the model embedding "
                     "(%d rows); refuse to export", tok.vocab_size(), cfg.vocab_size));
    if (tok.vocab_size() < cfg.vocab_size)
        log_info(strfmt("gguf_format: tokenizer has %d tokens, embedding has %d rows "
                        "(padding) - exporting", tok.vocab_size(), cfg.vocab_size));

    GGUFWriter w(path);

    w.set_arch(arch);
    {
        std::string name = "Ghassan v1 Flash";
        {
            u64 n = static_cast<u64>(model.num_parameters());
            std::string scale = n >= 1000000000ull
                ? strfmt("%.1fB", static_cast<double>(n) / 1e9)
                : strfmt("%lluM", static_cast<unsigned long long>((n + 500000ull) / 1000000ull));
            std::string net = cfg.use_moe
                ? strfmt("MoE-%d", cfg.num_experts)
                : "dense";
            name += strfmt(" %s %s L%d", scale.c_str(), net.c_str(), cfg.num_layers);
        }
        auto itn = extra_meta.find("model_name");
        if (itn != extra_meta.end() && !itn->second.empty()) name = itn->second;
        w.set_model_name(name);
    }
    w.set_description("Ghassan v1 Flash");
    w.set_general_type("model");
    {
        u64 n = static_cast<u64>(model.num_parameters());
        std::string label = n >= 1000000000ull
            ? strfmt("%.1fB", static_cast<double>(n) / 1e9)
            : strfmt("%lluM", static_cast<unsigned long long>((n + 500000ull) / 1000000ull));
        w.set_size_label(label);
        w.add_custom_metadata("general.parameter_count", n);
    }

    auto akey = [&](const char* k) { return arch + "." + k; };
    auto au64 = [&](const char* k, uint64_t v) { w.add_custom_metadata(akey(k), v); };
    auto af32 = [&](const char* k, float v) { w.add_custom_metadata(akey(k), v); };
    au64("vocab_size", static_cast<uint64_t>(cfg.vocab_size));
    au64("embedding_length", static_cast<uint64_t>(cfg.hidden_size));
    au64("block_count", static_cast<uint64_t>(cfg.num_layers));

    au64("feed_forward_length", static_cast<uint64_t>(want_moe ? cfg.moe_expert_dim
                                                               : cfg.intermediate_size));
    au64("attention.head_count", static_cast<uint64_t>(cfg.num_heads));
    au64("attention.head_count_kv", static_cast<uint64_t>(cfg.num_kv_heads));
    af32("attention.layer_norm_rms_epsilon", cfg.rms_eps);
    au64("rope.dimension_count", static_cast<uint64_t>(cfg.head_dim()));
    af32("rope.freq_base", cfg.rope_theta);
    au64("context_length", static_cast<uint64_t>(cfg.max_seq_len));
    w.add_custom_metadata(akey("tie_word_embeddings"), cfg.tie_embeddings);
    if (want_moe) {

        au64("expert_count", static_cast<uint64_t>(cfg.num_experts));
        au64("expert_used_count", static_cast<uint64_t>(cfg.moe_top_k));

        w.add_custom_metadata(akey("expert_feed_forward_length"),
                              static_cast<uint64_t>(cfg.moe_expert_dim));
    }
    if (!want_llama && !want_moe) {

        w.set_moe_config(cfg.use_moe, static_cast<uint32_t>(cfg.num_experts),
                         static_cast<uint32_t>(cfg.moe_top_k),
                         static_cast<uint32_t>(cfg.moe_expert_dim), cfg.moe_shared);
        af32("rope.scale", cfg.rope_scale);
        af32("rope.yarn_mscale", cfg.rope_yarn_mscale);
        af32("rope.yarn_low", cfg.rope_yarn_low);
        af32("rope.yarn_high", cfg.rope_yarn_high);
        au64("sliding_window", static_cast<uint64_t>(cfg.sliding_window));
        au64("rope.type", static_cast<uint64_t>(cfg.rope_type));
        w.add_custom_metadata(akey("moe.aux_free"), cfg.moe_aux_free);
        af32("moe.aux_scale", cfg.moe_aux_scale);
        af32("moe.jitter", cfg.moe_jitter);
        af32("z_loss_scale", cfg.z_loss_scale);
        w.add_custom_metadata(akey("use_qk_norm"), cfg.use_qk_norm);
        w.add_custom_metadata(akey("use_qkv_bias"), cfg.use_qkv_bias);
        w.set_norm_type("rmsnorm");
        w.set_ffn_type(cfg.use_moe ? "swiglu_moe" : "swiglu");
        w.set_attn_type("gqa_causal");

        // The hybrid flag: a Ghassan artifact must not carry the upstream
        // architecture's brand. The reader accepts the legacy key too.
        if (cfg.arch == ModelConfig::ArchGranite) {
            w.add_custom_metadata(akey("hybrid"), true);
            au64("position_embedding", static_cast<uint64_t>(cfg.position_embedding));
            au64("attention.head_dim", static_cast<uint64_t>(cfg.head_dim()));
            au64("moe.shared_intermediate_size",
                 static_cast<uint64_t>(cfg.shared_expert_intermediate_size));
            af32("embedding_multiplier", cfg.embedding_multiplier);
            af32("residual_multiplier", cfg.residual_multiplier);
            af32("attention_multiplier", cfg.attention_multiplier);
            af32("logits_scaling", cfg.logits_scaling);
            au64("mamba.n_heads", static_cast<uint64_t>(cfg.mamba_n_heads));
            au64("mamba.d_head", static_cast<uint64_t>(cfg.mamba_d_head));
            au64("mamba.d_state", static_cast<uint64_t>(cfg.mamba_d_state));
            au64("mamba.n_groups", static_cast<uint64_t>(cfg.mamba_n_groups));
            au64("mamba.d_conv", static_cast<uint64_t>(cfg.mamba_d_conv));
            au64("mamba.expand", static_cast<uint64_t>(cfg.mamba_expand));
            au64("mamba.chunk_size", static_cast<uint64_t>(cfg.mamba_chunk_size));
            w.add_custom_metadata(akey("mamba.conv_bias"), cfg.mamba_conv_bias);
            w.add_custom_metadata(akey("mamba.proj_bias"), cfg.mamba_proj_bias);
            {
                std::string lt;
                lt.reserve(cfg.layer_types.size() * 2);
                for (size_t i = 0; i < cfg.layer_types.size(); ++i) {
                    if (i) lt += ',';
                    lt += static_cast<char>('0' + (cfg.layer_types[i] ? 1 : 0));
                }
                w.add_custom_metadata(akey("layer_types"), lt);
            }
        }
    }

    w.set_tokenizer_model("gpt2");
    // The exporter must round-trip the pre-tokenizer the model was trained
    // with: Granite splits digits in groups of up to 3 (\p{N}{1,3}), GPT-2 and
    // Qwen do not. Writing "gpt-2" for a Granite tokenizer re-encodes digits
    // wrongly after a reload.
    switch (tok.pretok_kind()) {
        case PreTokKind::Qwen:    w.set_tokenizer_pre("qwen2"); break;
        case PreTokKind::Granite: w.set_tokenizer_pre("ghassan"); break;
        default:                  w.set_tokenizer_pre("gpt-2"); break;
    }
    w.set_tokenizer_tokens(tok.vocab_entries());
    {
        std::vector<float> scores(static_cast<size_t>(tok.vocab_size()), 0.0f);
        w.set_tokenizer_scores(scores);
        std::vector<int32_t> types;
        types.reserve(static_cast<size_t>(tok.vocab_size()));
        // GGUF token types: 1 = normal, 2 = unknown, 3 = control,
        // 4 = user-defined (an "added" token that is not a control token).
        // The distinction matters: an added token is still matched literally
        // in the text, while a control token is skipped on decode.
        for (int i = 0; i < tok.vocab_size(); ++i) {
            const i32 id = static_cast<i32>(i);
            if (tok.is_special(id))           types.push_back(3);
            else if (tok.is_added(id))         types.push_back(4);
            else                               types.push_back(1);
        }
        w.set_tokenizer_token_types(types);
    }
    {
        std::vector<std::string> merges;
        for (const auto& pr : tok.merge_pairs_ordered())
            merges.push_back(escape_merge_piece(pr.first) + " " + escape_merge_piece(pr.second));
        w.set_tokenizer_merges(merges);
    }
    w.set_tokenizer_bos_id(static_cast<uint32_t>(tok.bos_id()));
    w.set_tokenizer_eos_id(static_cast<uint32_t>(tok.eos_id()));
    w.set_tokenizer_unk_id(static_cast<uint32_t>(tok.unk_id()));
    w.set_tokenizer_pad_id(static_cast<uint32_t>(tok.pad_id()));
    w.set_tokenizer_add_bos(tok.add_bos());
    w.set_tokenizer_add_eos(tok.add_eos());
    {
        // The chat template must match the architecture: Granite speaks the
        // role-tag format (<|start_of_role|>...<|end_of_role|>), Qwen speaks
        // ChatML. Hardcoding ChatML made every Granite GGUF reload with a
        // prompt format the model never saw in training.
        std::string tmpl =
            "{%- for message in messages %}"
            "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}";
        if (cfg.arch == ModelConfig::ArchGranite) {
            // The role-tag format, exactly what the engine's encode_granite
            // renders: an optional system turn (the default sentence when the
            // caller supplies none), one turn per message, always closed by
            // <|end_of_text|>.
            tmpl =
                "{%- if messages[0]['role'] == 'system' %}{%- else %}"
                "{{- '<|start_of_role|>system<|end_of_role|>You are a helpful assistant. "
                "Please ensure responses are professional, accurate, and safe.<|end_of_text|>\\n' }}"
                "{%- endif %}"
                "{%- for message in messages %}"
                "{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + "
                "message['content'] + '<|end_of_text|>\\n' }}"
                "{%- endfor %}"
                "{%- if add_generation_prompt %}{{- '<|start_of_role|>assistant<|end_of_role|>' }}{%- endif %}";
        }
        w.set_tokenizer_chat_template(tmpl);
    }

    {
        const char* pretok_name = "byte_bpe:gpt2_pretokenizer,byte_fallback";
        if (tok.pretok_kind() == PreTokKind::Qwen)
            pretok_name = "byte_bpe:qwen_pretokenizer,byte_fallback";
        else if (tok.pretok_kind() == PreTokKind::Granite)
            pretok_name = "byte_bpe:ghassan_pretokenizer(\\p{N}{1,3}),byte_fallback";
        w.add_custom_metadata("ghassan.tokenizer.pre_tokenizer",
                              std::string(pretok_name));
    }
    w.add_custom_metadata("ghassan.tokenizer.specials",
                          static_cast<uint64_t>(tok.special_ids().size()));

    w.set_quantization_profile(profile.name);
    uint32_t file_type = 1;
    if (profile.default_type == GGMLType::Q8_0) file_type = 7;
    if (profile.default_type == GGMLType::Q4_0) file_type = 2;
    if (profile.default_type == GGMLType::F16)  file_type = 1;
    w.set_file_type(file_type);

    for (const auto& [k, v] : extra_meta) {
        if (k == "model_name") continue;
        w.add_custom_metadata(k, v);
    }

    u64 total_bytes = 0;
    for (Parameter* p : model.parameters()) {
        bool is_norm = p->name.find("norm") != std::string::npos;
        bool is_emb  = (p->name == "tok_embeddings" || p->name == "lm_head");
        GGMLType dt = is_norm ? profile.norm_type
                    : is_emb  ? profile.embedding_type
                              : profile.default_type;
        std::string wname = p->name;
        Tensor cpu = p->w.to(Device::CPU);
        if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
        if (want_llama) {
            wname = llama_tensor_name(p->name);
            GAI_CHECK(!wname.empty(),
                      "gguf_format: no llama.cpp name for tensor: " + p->name);
        } else if (want_moe) {
            wname = llama_moe_tensor_name(p->name);
            GAI_CHECK(!wname.empty(),
                      "gguf_format: no llama_moe name for tensor: " + p->name +
                      " (shared experts / QK-norm have no Ollama equivalent; use native GGUF)");

            // The expert tensors must be 3D (one block per expert) to match
            // llama.cpp:
            //   ffn_gate_exps / ffn_up_exps : [n_embd, n_ff, n_expert]
            //   ffn_down_exps               : [n_ff,  n_embd, n_expert]
            // The Gai layout stores the experts contiguously
            //   moe_gate / moe_up : [ne*E, d] -> per expert [E, d]
            //   moe_down          : [ne*d, E] -> per expert [d, E]
            // which is exactly the per-expert block llama.cpp expects, so the
            // conversion is a reshape and NOT a transpose.
            if (p->name.find("moe_gate") != std::string::npos ||
                p->name.find("moe_up") != std::string::npos) {
                const i64 ne = cfg.num_experts, E = cfg.moe_expert_dim, d = cfg.hidden_size;
                GAI_CHECK(cpu.shape().size() == 2 && cpu.shape()[0] == ne * E &&
                          cpu.shape()[1] == d,
                          "gguf_format: unexpected moe_gate/moe_up shape for llama_moe export");
                cpu = reshape_3d(cpu, ne, E, d);
            } else if (p->name.find("moe_down") != std::string::npos) {
                const i64 ne = cfg.num_experts, E = cfg.moe_expert_dim, d = cfg.hidden_size;
                GAI_CHECK(cpu.shape().size() == 2 && cpu.shape()[0] == ne * d &&
                          cpu.shape()[1] == E,
                          "gguf_format: unexpected moe_down shape for llama_moe export");
                cpu = reshape_3d(cpu, ne, d, E);
            }
        }
        size_t est = ggml_nbytes(dt, static_cast<size_t>(p->numel()));
        total_bytes += static_cast<u64>(est);
        w.add_tensor(wname, cpu, dt);

        if ((want_llama || want_moe) && p->name == "tok_embeddings" && cfg.tie_embeddings) {
            size_t est2 = ggml_nbytes(dt, static_cast<size_t>(p->numel()));
            total_bytes += static_cast<u64>(est2);
            w.add_tensor("output.weight", cpu, dt);
        }
    }

    if (!want_llama && !want_moe && cfg.moe_aux_free) {
        const auto& all = model.moe_bias_all();
        for (int l = 0; l < cfg.num_layers && l < static_cast<int>(all.size()); ++l) {
            const auto& b = all[static_cast<size_t>(l)];
            if (static_cast<int>(b.size()) != cfg.num_experts) continue;
            Tensor t({(i64)b.size()}, DType::F32, Device::CPU);
            std::memcpy(t.data_ptr(), b.data(), sizeof(float) * b.size());
            t.set_name("layers." + std::to_string(l) + ".moe_bias");
            w.add_tensor(t.name(), t, GGMLType::F32);
        }
    }

    w.write();

    log_info(strfmt("[gguf] exported %s  profile=%s compat=%s self_contained=yes  tensors=%zu  weights=%s",
                    path.c_str(), profile.name.c_str(), compat.c_str(),
                    model.parameters().size(),
                    human_bytes(total_bytes).c_str()));
}

}
