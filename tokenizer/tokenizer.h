#pragma once

#include "core/common.h"
#include "core/unicode.h"
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace gai {

// GPT-2 byte-level alphabet: maps a raw byte to the single character that
// byte-level BPE vocabularies (GPT-2, Qwen, Llama-3, ...) use to represent
// it. byte 0x20 -> U+0120 ("Ġ"), byte 0x0A -> U+010A ("Ċ"), ...
const std::string& gpt2_byte_token(u8 b);

// Inverse of gpt2_byte_token(): character -> raw byte.
const std::unordered_map<std::string, u8>& gpt2_byte_decoder();

// Which pre-tokenizer splits the text before BPE.
//   Gpt2  - GPT-2 / Llama-3 regex ('s|'t|...| ?\p{L}+| \p{N}+ | ...), 
//   Qwen  - Qwen2/Qwen3 regex (contractions | [^\r\n\p{L}\p{N}]?\p{L}+ | \n//          \p{N} single digit | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | ...)
//   Granite - Granite regex (same as Qwen but \p{N}{1,3}: digits are
//          grouped in runs of at most 3 instead of single digits)
enum class PreTokKind : u8 { Gpt2 = 0, Qwen = 1, Granite = 2 };

// Options for a vocabulary loaded from disk (foreign, byte-level BPE).
struct VocabOptions {
    bool     add_bos = false;
    bool     add_eos = false;
    i32      bos_id  = 0;
    i32      eos_id  = 2;
    i32      pad_id  = 0;
    i32      unk_id  = 3;
    PreTokKind pretok = PreTokKind::Gpt2;
    // Ids that must be treated as control tokens (skipped by skip_special).
    // Filled from the vocabulary metadata (added_tokens_decoder etc.).
    std::vector<i32> special_ids;
    // Every "added" token of the vocabulary, whether it is flagged special
    // or not. HuggingFace matches all of them in the raw text (the `special`
    // flag only drives skip_special_tokens), so the engine must too.
    std::vector<i32> added_ids;
};

class Tokenizer {
public:
    Tokenizer() = default;

    // Builds the tokenizer from a byte-level BPE vocabulary: the list of
    // token strings indexed by id, and the merge list in rank order.
    void build(const std::vector<std::string>& vocab,
               const std::vector<std::pair<std::string, std::string>>& merges,
               const VocabOptions& opts = {});

    std::vector<i32> encode(const std::string& text,
                            bool add_bos = false,
                            bool add_eos = false) const;
    // Convenience: encode with the vocabulary's configured bos/eos flags.
    std::vector<i32> encode_default(const std::string& text) const {
        return encode(text, add_bos_, add_eos_);
    }

    std::string      decode(const std::vector<i32>& ids, bool skip_special = false) const;
    std::string      decode_one(i32 id) const;

    class Stream {
    public:
        explicit Stream(const Tokenizer& tk) : tk_(tk) {}
        std::string push(i32 id);
        std::string flush();
    private:
        const Tokenizer& tk_;
        std::string buf_;
    };

    int  vocab_size() const { return static_cast<int>(vocab_.size()); }
    const std::string& token_text(i32 id) const;
    i32  token_to_id(const std::string& tok) const;

    // Raw bytes a token stands for: every character of the token is mapped
    // back to the byte it encodes.
    std::string token_bytes(i32 id) const;

    const std::vector<std::string>& vocab_entries() const { return vocab_; }

    std::vector<std::pair<std::string, std::string>> merge_pairs_ordered() const;

    // Control tokens marked at build time from the vocabulary metadata.
    bool is_special(i32 id) const { return special_ids_.find(id) != special_ids_.end(); }
    const std::vector<i32>& special_ids() const { return special_vec_; }

    // The "added" tokens: matched literally in the input text (like
    // HuggingFace's AddedVocabulary), even when they are not control
    // tokens.
    bool is_added(i32 id) const { return added_ids_.find(id) != added_ids_.end(); }
    const std::vector<i32>& added_ids() const { return added_vec_; }

    bool add_bos() const { return add_bos_; }
    bool add_eos() const { return add_eos_; }
    i32  bos_id()  const { return bos_id_; }
    i32  eos_id()  const { return eos_id_; }
    i32  pad_id()  const { return pad_id_; }
    i32  unk_id()  const { return unk_id_; }

    // True when the vocabulary carries the full 256-entry byte-level
    // alphabet, i.e. it is a byte-level BPE vocabulary the engine can
    // encode against.
    bool has_byte_map() const { return byte_map_valid_; }
    const std::array<i32, 256>& byte_map() const { return byte_to_id_; }
    i32 byte_id(u8 b) const {
        return byte_map_valid_ ? byte_to_id_[b] : -1;
    }
    const std::vector<int>& missing_bytes() const { return missing_bytes_; }

    // Digits are split one per token (HuggingFace
    // `Digits(individual_digits=true)`).
    PreTokKind pretok_kind() const { return pretok_; }

    bool digits_individual() const { return digits_individual_; }
    void set_digits_individual(bool on) { digits_individual_ = on; }

    struct Fertility {
        double tokens_per_word = 0;
        double bytes_per_token = 0;
        u64    tokens = 0, words = 0, bytes = 0;
    };
    Fertility measure(const std::vector<std::string>& lines) const;

private:
    void   finalize_index();
    void   bpe_chunk(const std::string& piece, std::vector<i32>& out) const;
    void   rebuild_byte_map();
    bool   detect_digits_individual() const;

    std::vector<std::string>                     vocab_;
    std::unordered_map<std::string, i32>         token_ids_;

    std::unordered_map<u64, std::pair<i32, i32>> merges_;
    mutable std::unordered_map<std::string, std::vector<i32>> cache_;
    // bpe_chunk() is const but mutates cache_; without a mutex, sharing one
    // Tokenizer across threads is a data race.
    mutable std::shared_ptr<std::mutex> cache_mu_ = std::make_shared<std::mutex>();
    mutable size_t cache_limit_ = 200000;

    // byte-level BPE alphabet
    std::array<i32, 256> byte_to_id_{};
    bool                 byte_map_valid_ = false;
    std::vector<int>    missing_bytes_;
    bool                 digits_individual_ = false;

    PreTokKind pretok_ = PreTokKind::Gpt2;
    bool add_bos_ = false;
    bool add_eos_ = false;
    i32  bos_id_  = 0;
    i32  eos_id_  = 2;
    i32  pad_id_  = 0;
    i32  unk_id_  = 3;
    std::unordered_set<i32> special_ids_;
    std::vector<i32>       special_vec_;
    std::unordered_set<i32> added_ids_;
    std::vector<i32>       added_vec_;
};

// GPT-2 / Qwen byte-level pre-tokenizer. Reproduces the reference regex
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// by hand: std::regex (ECMAScript) has no lookahead. split_digits mirrors
// HuggingFace's `Digits(individual_digits=true)` pre-tokenizer, which some
// byte-level vocabularies (Qwen2/Qwen3) put in front of the ByteLevel regex.
struct Chunk {
    std::string text;
};
std::vector<Chunk> pre_tokenize_gpt2(const std::string& text, bool split_digits = false);
std::vector<Chunk> pre_tokenize_qwen(const std::string& text, int digits_max = 1);

}
