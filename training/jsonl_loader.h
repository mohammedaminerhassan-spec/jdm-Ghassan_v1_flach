#pragma once

#include "core/common.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"
#include "training/dataloader.h"
#include <string>
#include <vector>

namespace gai {

struct JsonlSample {
    std::vector<Message> messages;
    bool thinking = false;
};

class JsonlLoader {
public:
    JsonlLoader() = default;

    static bool parse_line(const std::string& line, JsonlSample& sample);

    static bool convert_to_gbin(const std::string& jsonl_path,
                                const std::string& gbin_out_path,
                                const Tokenizer& tk,
                                int max_seq_len = 8192);

    static int process_directory(const std::string& dir,
                                 const Tokenizer& tk,
                                 int max_seq_len = 8192);
};

}
