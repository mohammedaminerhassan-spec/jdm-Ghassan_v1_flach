#pragma once

#include "core/common.h"
#include "core/mmap.h"
#include "core/tensor.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gai {

// A safetensors file (the format HuggingFace ships).
//
// The layout is simple: a little endian u64 with the length of a JSON
// header, the header itself, then the tensor data. Every tensor in the
// header carries its dtype, its shape and its byte range inside the data
// buffer.
//
//     [u64 header_len][header_len bytes of JSON][data]
//
// The reader keeps the data buffer in memory and converts a tensor to the
// engine's F32 parameter type on demand.

struct SafeTensorInfo {
    std::string name;
    std::string dtype;          // "F32", "F16", "BF16", "I8", ...
    std::vector<i64> shape;
    uint64_t begin = 0;         // byte offset inside the data buffer
    uint64_t end   = 0;
};

class SafeTensorsFile {
public:
    bool open(const std::string& path);

    const std::map<std::string, SafeTensorInfo>& tensors() const { return tensors_; }
    const SafeTensorInfo* find(const std::string& name) const;

    // Reads a tensor and converts it to F32. Returns an undefined tensor
    // when the name is unknown or the dtype is not a float type.
    Tensor read_f32(const std::string& name) const;

    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }

private:
    std::string path_;
    std::map<std::string, SafeTensorInfo> tensors_;
    MappedFile mmap_;
    const uint8_t* data_ = nullptr;
    uint64_t file_size_ = 0;
};

// A HuggingFace checkpoint directory that the engine can serve and train
// directly, with no conversion step:
//
//     <dir>/config.json                  the architecture
//     <dir>/model.safetensors            the weights (or the shards + the index)
//     <dir>/model.safetensors.index.json the shard map
//     <dir>/tokenizer.json                the vocabulary and the merges
//     <dir>/tokenizer_config.json         the special token ids
//
// The tensor names are the HuggingFace ones (model.layers.0.self_attn.q_proj
// .weight, ...). They are mapped onto the engine's own parameter layout at
// load time, so nothing has to be rewritten on disk.

class HFCheckpoint {
public:
    // Opens a checkpoint directory. With `weights_required` = false the
    // weight shards are not mapped (a dry run only needs the config); the
    // tensors are then simply missing and load_model() will say so.
    bool open(const std::string& dir, bool weights_required = true);

    const ModelConfig& config() const { return cfg_; }
    const std::string& dir() const { return dir_; }
    const std::vector<std::string>& weight_files() const { return weight_files_; }
    const std::map<std::string, SafeTensorInfo>& tensors() const { return tensor_info_; }

    // True when the directory looks like a HuggingFace checkpoint.
    static bool looks_like_checkpoint(const std::string& dir);

    // Copies the weights into the model. The model must have been built
    // with the same config (HFCheckpoint::config()).
    bool load_model(Model& model) const;

    // Builds the tokenizer from tokenizer.json. Special (control) token ids
    // are marked from the vocabulary metadata so the decoder knows which
    // ids to hide when skip_special is set.
    bool load_tokenizer(Tokenizer& tk) const;

    // Builds a tokenizer from a standalone HuggingFace tokenizer.json file
    // (useful for GGUF export when there is no checkpoint directory). The
    // special-token ids are taken from the tokenizer.json metadata itself
    // (added_tokens_decoder).
    static bool load_tokenizer_json(const std::string& path, Tokenizer& tk,
                                    std::vector<i32>* eos_out = nullptr);

    // Stop-token ids from generation_config.json ("eos_token_id" may be an
    // int or an array of ints). Empty when the file is absent, in which
    // case callers should fall back to the tokenizer's eos_id().
    const std::vector<i32>& eos_ids() const { return eos_ids_; }

private:
    std::string dir_;
    ModelConfig cfg_{};
    std::vector<std::string> weight_files_;
    std::map<std::string, SafeTensorInfo> tensor_info_;   // name -> info
    std::map<std::string, std::string> tensor_file_;       // name -> file it lives in
    std::map<std::string, SafeTensorsFile> files_;         // file -> reader
    mutable std::vector<i32> eos_ids_;

    const SafeTensorInfo* find_tensor(const std::string& name) const;
    Tensor read_tensor(const std::string& name) const;
    bool read_config_json(const std::string& path);
};

// The HuggingFace tensor name for an engine parameter, or an empty string
// when the parameter has no counterpart (the MoE expert weights are
// per-expert on the HuggingFace side and are gathered instead).
// `is_granite` selects the IBM Granite-4.0 hybrid naming (packed MoE).
std::string hf_tensor_name(const std::string& internal, bool is_granite);

}  // namespace gai
