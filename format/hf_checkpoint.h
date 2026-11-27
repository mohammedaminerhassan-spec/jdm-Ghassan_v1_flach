#pragma once

#include "core/common.h"
#include "core/mmap.h"
#include "core/tensor.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace gai {

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<i64> shape;
    uint64_t begin = 0;
    uint64_t end   = 0;
};

class SafeTensorsFile {
public:
    bool open(const std::string& path);

    const std::map<std::string, SafeTensorInfo>& tensors() const { return tensors_; }
    const SafeTensorInfo* find(const std::string& name) const;

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

class HFCheckpoint {
public:
    bool open(const std::string& dir, bool weights_required = true);

    const ModelConfig& config() const { return cfg_; }
    const std::string& dir() const { return dir_; }
    const std::vector<std::string>& weight_files() const { return weight_files_; }
    uint64_t total_bytes() const { return total_bytes_; }
    const std::map<std::string, SafeTensorInfo>& tensors() const { return tensor_info_; }

    static bool looks_like_checkpoint(const std::string& dir);

    bool load_model(Model& model) const;

    bool load_tokenizer(Tokenizer& tk) const;

    static bool load_tokenizer_json(const std::string& path, Tokenizer& tk,
                                    std::vector<i32>* eos_out = nullptr);

    const std::vector<i32>& eos_ids() const { return eos_ids_; }

private:
    std::string dir_;
    ModelConfig cfg_{};
    std::vector<std::string> weight_files_;
    uint64_t total_bytes_ = 0;
    std::map<std::string, SafeTensorInfo> tensor_info_;
    std::map<std::string, std::string> tensor_file_;
    std::map<std::string, SafeTensorsFile> files_;
    mutable std::vector<i32> eos_ids_;
    mutable std::set<std::string> consumed_;

    const SafeTensorInfo* find_tensor(const std::string& name) const;
    Tensor read_tensor(const std::string& name) const;
    bool read_config_json(const std::string& path);
};

std::string hf_tensor_name(const std::string& internal, bool is_granite);

}
