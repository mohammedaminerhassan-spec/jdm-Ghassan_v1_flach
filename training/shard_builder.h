#pragma once

#include "core/common.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <vector>

namespace gai {

// Builds the PRETRAINING shards (.gbin, no loss mask) from raw text.
//
// The SFT path already has a converter (JsonlLoader: the instruction
// datasets -> .gbin with an assistant-span loss mask). The pretraining path
// needs the other one: a plain text corpus -> .gbin, every document
// supervised.
//
// The tokenization is done HERE, with the engine's own tokenizer, and not by
// a Python script: the engine implements the byte-level pre-tokenizer by
// hand (the digit grouping of Granite, the Qwen regex, ...), and a Python
// tokenizer would disagree with it on exactly the cases that matter (the
// numbers). A shard built with a different tokenizer is a corpus the model
// was never trained on - and it fails silently.
//
// The input files are plain text, ONE DOCUMENT PER LINE. The dataset
// preparation script (tools/py/prepare_dataset.py) only converts the format
// (Parquet -> text); it never touches the tokens.
class ShardBuilder {
public:
    struct Stats {
        int    shards = 0;
        int    documents = 0;
        u64    tokens = 0;
        u64    skipped_docs = 0;
        u64    skipped_tokens = 0;   // the token ids outside the vocabulary
    };

    // Reads every .txt file of `dir` (sorted by name) and writes the shards
    // `prefix_NNNN.gbin` into the same directory.
    //
    // `target_tokens_per_shard` = 0 means one shard for everything.
    // `max_seq_len` (0 = unlimited) truncates the documents that are longer
    // than the training sequence (a document longer than the sequence is
    // still usable - the loader samples a random window inside it - but a
    // document of 10M tokens would be one huge document).
    static Stats build_directory(const std::string& dir, const Tokenizer& tk,
                                 const std::string& prefix, int seq_len,
                                 u64 target_tokens_per_shard);

    // The same, for one file.
    static Stats build_file(const std::string& path, const Tokenizer& tk,
                            const std::string& out_path, int seq_len);
};

}  // namespace gai
