#pragma once

#include "core/common.h"
#include "core/tensor.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace gai {

// ---------------------------------------------------------------------------
// The safetensors WRITER.
//
// The engine could READ a HuggingFace checkpoint since the beginning, but
// it could only WRITE a GGUF. A model trained here could therefore not be
// published as a HuggingFace repository - and the whole point of the
// engine is to publish what it trains.
//
// The format is simple:
//
//     [u64 header_len][header_len bytes of JSON][the tensor data]
//
// The header maps a tensor name to {dtype, shape, data_offsets:[begin,end]}
// where the offsets are relative to the start of the data buffer. The
// reference reader validates that the tensors are CONTIGUOUS (each one
// starts where the previous ended) and that the header is padded with
// spaces so the data starts on an 8-byte boundary.
//
// The writer works in two phases so a multi-billion parameter model never
// has to be held in RAM:
//
//   1. reserve() every tensor (the name, the shape, the dtype): the layout
//      and the header are then known;
//   2. write_header(), then append() the bytes of every tensor in the same
//      order.
//
// The reader of this engine (HFCheckpoint) and the reference HuggingFace
// reader both accept what this writer produces.
// ---------------------------------------------------------------------------

enum class SafeDType : u8 { F32 = 0, F16 = 1, BF16 = 2 };

const char* safe_dtype_name(SafeDType t);
size_t      safe_dtype_size(SafeDType t);

class SafeTensorsWriter {
public:
    explicit SafeTensorsWriter(const std::string& path);
    ~SafeTensorsWriter();

    // Phase 1: declares a tensor and reserves its byte range. The calls
    // define the layout order (and the header order).
    void reserve(const std::string& name, const std::vector<i64>& shape,
                 SafeDType dtype, size_t nbytes);

    // Phase 2: writes the header and creates the data area.
    bool begin();
    // Appends the bytes of the next reserved tensor (the reserve order).
    bool append(const void* data, size_t nbytes);
    bool finish();

    size_t tensor_count() const { return entries_.size(); }
    size_t data_bytes() const  { return total_; }
    const std::string& path() const { return path_; }

private:
    struct Entry {
        std::string    name;
        std::vector<i64> shape;
        SafeDType      dtype = SafeDType::F32;
        size_t         nbytes = 0;
        uint64_t       begin = 0;
        uint64_t       end = 0;
    };

    std::string          path_;
    std::vector<Entry>   entries_;
    size_t               total_ = 0;
    std::ofstream        out_;
    size_t               cursor_ = 0;
    bool                 begun_ = false;
    bool                 failed_ = false;
};

// One tensor of the exported file.
struct ExportTensor {
    std::string            hf_name;
    std::vector<i64>       shape;
    SafeDType              dtype = SafeDType::F16;
    size_t                 nbytes = 0;
    // Fills the payload (nbytes bytes) of the tensor.
    std::function<void(uint8_t*)> fill;
};

// Builds the export plan of a model: the list of HuggingFace tensors (their
// names, shapes and how to fill them) for the model's parameters.
//
// The MoE experts are the only tensors that are not a straight copy: the
// engine keeps them flat ([ne*E, d] and [ne*d, E]) while HuggingFace either
// keeps one tensor per expert (Qwen3-MoE, Mixtral) or packs all the experts
// into two tensors (the IBM Granite hybrid).
std::vector<ExportTensor> plan_safetensors_export(Model& model, SafeDType dtype);

// Writes a model as a HuggingFace checkpoint directory:
//
//     <dir>/model.safetensors            (or the shards + the index)
//     <dir>/model.safetensors.index.json (only when sharded)
//     <dir>/config.json
//     <dir>/generation_config.json
//
// The tokenizer files are NOT written here: the caller copies them from the
// source checkpoint (the vocabulary is a property of the tokenizer, not of
// the weights, and re-serialising it would lose the merges).
bool export_model_safetensors(const std::string& dir, Model& model,
                              const Tokenizer& tokenizer,
                              const std::string& model_name,
                              const std::string& description = "",
                              SafeDType dtype = SafeDType::F16,
                              size_t shard_limit = 4ull << 30);

// The config.json of a model (the same fields the engine's own loader
// reads, so a round-trip works).
std::string hf_config_json(const ModelConfig& cfg, const std::string& model_name,
                           const std::string& description);

}  // namespace gai
