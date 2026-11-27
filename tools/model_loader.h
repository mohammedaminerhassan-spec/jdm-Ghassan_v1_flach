#pragma once

#include "core/common.h"
#include "model/model.h"
#include "inference/generator.h"
#include "inference/chat.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "projector/projector.h"
#include "tools/cli_common.h"

#include <memory>
#include <string>
#include <vector>

namespace gai {

struct LoadedModel {
    std::unique_ptr<Model> model;
    Tokenizer               tokenizer;
    ModelConfig             cfg;
    std::string             quant = "q8_k";
    u64                     file_bytes = 0;
    std::string             path;
    std::string             arch = "ghassan";
    std::vector<i32>        eos_ids;
    std::unique_ptr<VisionTower> vision;
};

LoadedModel load_model(const Args& args);

bool is_gguf_file(const std::string& path);

GenerationConfig make_gen_config(const Args& a);

bool load_tokenizer_auto(const std::string& path, Tokenizer& out,
                         std::vector<i32>& eos_out);

bool copy_tokenizer_files(const std::string& src, const std::string& dst_dir);

}
