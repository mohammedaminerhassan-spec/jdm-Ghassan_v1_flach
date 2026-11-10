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

namespace gai {

// A model file that has been read into memory and is ready to serve.
struct LoadedModel {
    std::unique_ptr<Model> model;
    Tokenizer               tokenizer;
    ModelConfig             cfg;
    std::string             quant = "fp32";
    u64                     file_bytes = 0;
    std::string             path;
    std::string             arch = "ghassan";
};

// Reads a GGUF (or legacy .gai) file and loads it on the requested device.
LoadedModel load_model(const Args& args);

// True when the file starts with the GGUF magic.
bool is_gguf_file(const std::string& path);

// The generation defaults shared by the CLI and the server.
GenerationConfig make_gen_config(const Args& a);

// The chat style of a loaded vocabulary (native / ChatML / Llama-3 / Mistral).
const char* chat_style_name(ChatStyle s);

// Reads the vision tower of a multimodal GGUF. Returns null when the file
// carries no vision weights.
std::unique_ptr<class VisionTower> load_vision_tower(const std::string& path,
                                                     class VisionConfig* out_cfg);

}  // namespace gai
