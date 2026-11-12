#pragma once

#include "core/common.h"
#include "model/model.h"
#include "inference/generator.h"
#include "inference/chat.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "tools/cli_common.h"

#include <memory>
#include <string>
#include <vector>

namespace gai {

// A model loaded from disk and ready to serve.
struct LoadedModel {
    std::unique_ptr<Model> model;
    Tokenizer               tokenizer;
    ModelConfig             cfg;
    std::string             quant = "q8_k";
    u64                     file_bytes = 0;
    std::string             path;
    std::string             arch = "ghassan";
    // Stop tokens from the checkpoint's generation config (e.g. <|im_end|>
    // and <|endoftext|> for Qwen). Empty = tokenizer's eos_id().
    std::vector<i32>        eos_ids;
};

// Reads a GGUF file or a HuggingFace checkpoint directory and loads it on
// the requested device.
LoadedModel load_model(const Args& args);

// True when the file starts with the GGUF magic.
bool is_gguf_file(const std::string& path);

// The generation defaults shared by the CLI commands.
GenerationConfig make_gen_config(const Args& a);

// Loads a HuggingFace-style tokenizer (checkpoint directory or a bare
// tokenizer.json) into `out`. `eos_out` (optional) receives the stop-token
// ids from generation_config.json (directory mode only).
bool load_tokenizer_auto(const std::string& path, Tokenizer& out,
                         std::vector<i32>& eos_out);

}  // namespace gai
