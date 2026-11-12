#pragma once

#include "core/common.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"
#include "training/dataloader.h"
#include <string>
#include <vector>

namespace gai {

// Pure C++ JSONL SFT (Supervised Fine-Tuning) converter.
// Parses standard instruction dataset formats and writes a .gbin shard with
// an assistant-span loss mask (1 = supervised token, 0 = prompt token):
//
//   1. OpenAI:   {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
//   2. ShareGPT: {"conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
//   3. Alpaca:   {"instruction":"...","input":"...","output":"..."}
//   4. Raw text: {"text":"..."}

struct JsonlSample {
    std::vector<Message> messages;
};

class JsonlLoader {
public:
    JsonlLoader() = default;

    // Parses a single JSON line into a sample.
    static bool parse_line(const std::string& line, JsonlSample& sample);

    // Converts a .jsonl/.json file directly into a .gbin shard with an
    // assistant-span loss mask.
    static bool convert_to_gbin(const std::string& jsonl_path,
                                const std::string& gbin_out_path,
                                const Tokenizer& tk,
                                int max_seq_len = 8192);

    // Auto-converts every .jsonl/.json file in dir into a sibling .gbin
    // shard. Existing up-to-date .gbin files are skipped. Returns the number
    // of shards converted.
    static int process_directory(const std::string& dir,
                                 const Tokenizer& tk,
                                 int max_seq_len = 8192);
};

} // namespace gai
