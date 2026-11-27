#pragma once

#include "core/common.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <vector>

namespace gai {

class ShardBuilder {
public:
    struct Stats {
        int    shards = 0;
        int    documents = 0;
        u64    tokens = 0;
        u64    skipped_docs = 0;
        u64    skipped_tokens = 0;
    };

    static Stats build_directory(const std::string& dir, const Tokenizer& tk,
                                 const std::string& prefix, int seq_len,
                                 u64 target_tokens_per_shard);

    static Stats build_file(const std::string& path, const Tokenizer& tk,
                            const std::string& out_path, int seq_len);
};

}
