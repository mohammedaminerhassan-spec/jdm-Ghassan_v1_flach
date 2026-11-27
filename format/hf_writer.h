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

enum class SafeDType : u8 { F32 = 0, F16 = 1, BF16 = 2 };

const char* safe_dtype_name(SafeDType t);
size_t      safe_dtype_size(SafeDType t);

class SafeTensorsWriter {
public:
    explicit SafeTensorsWriter(const std::string& path);
    ~SafeTensorsWriter();

    void reserve(const std::string& name, const std::vector<i64>& shape,
                 SafeDType dtype, size_t nbytes);

    bool begin();
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
    std::string          tmp_path_;
    std::vector<Entry>   entries_;
    size_t               total_ = 0;
    std::ofstream        out_;
    size_t               cursor_ = 0;
    bool                 begun_ = false;
    bool                 failed_ = false;
};

struct ExportTensor {
    std::string            hf_name;
    std::vector<i64>       shape;
    SafeDType              dtype = SafeDType::F16;
    size_t                 nbytes = 0;
    std::function<void(uint8_t*)> fill;
};

std::vector<ExportTensor> plan_safetensors_export(Model& model, SafeDType dtype);

bool export_model_safetensors(const std::string& dir, Model& model,
                              const Tokenizer& tokenizer,
                              const std::string& model_name,
                              const std::string& description = "",
                              SafeDType dtype = SafeDType::F16,
                              size_t shard_limit = 4ull << 30);

std::string hf_config_json(const ModelConfig& cfg, const std::string& model_name,
                           const std::string& description);

}
