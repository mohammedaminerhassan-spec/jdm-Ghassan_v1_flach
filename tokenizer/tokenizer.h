#pragma once

#include "core/common.h"
#include "tokenizer/normalizer.h"
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace gai {

namespace special {
constexpr i32 PAD       = 0;
constexpr i32 BOS       = 1;
constexpr i32 EOS       = 2;
constexpr i32 UNK       = 3;
constexpr i32 SYSTEM    = 4;
constexpr i32 USER      = 5;
constexpr i32 ASSISTANT = 6;
constexpr i32 END       = 7;
constexpr i32 NL        = 8;
constexpr i32 FR        = 9;
constexpr i32 AR        = 10;
constexpr i32 DARIJA    = 11;
constexpr i32 LATIN     = 12;
constexpr i32 TOOL      = 13;
constexpr i32 MASK      = 14;
constexpr i32 RESERVED  = 15;
constexpr i32 COUNT     = 16;
}

const std::vector<std::string>& special_token_strings();

// GPT-2 byte-level alphabet: maps a raw byte to the single character that
// byte-level BPE vocabularies (GPT-2, Qwen, Llama-3, ...) use to represent
// it. byte 0x20 -> U+0120 ("Ġ"), byte 0x0A -> U+010A ("Ċ"), ...
const std::string& gpt2_byte_token(u8 b);

// Inverse of gpt2_byte_token(): character -> raw byte.
const std::unordered_map<std::string, u8>& gpt2_byte_decoder();

class Tokenizer {
public:
    // Which pre-tokenizer splits the raw text before BPE.
    enum class PreTokMode : u8 {
        Ghassan = 0,  // Arabic/Latin aware splitter (native .gtok)
        Gpt2    = 1,  // reference GPT-2 regex (foreign / HF vocabularies)
    };

    // Options describing a vocabulary that came from outside the Gai engine.
    struct VocabOptions {
        bool      foreign = false;      // skip the Gai normalizer entirely
        PreTokMode pretok = PreTokMode::Gpt2;
        bool      add_bos = false;
        bool      add_eos = false;
        i32       bos_id = special::BOS;
        i32       eos_id = special::EOS;
        i32       pad_id = special::PAD;
        i32       unk_id = special::UNK;
        // Some chat templates inject a default system message when the
        // conversation does not start with one (SmolLM2, Qwen, ...).
        std::string default_system;
    };

    Tokenizer() = default;

    void init_empty(const NormalizerConfig& ncfg);

    void build(const std::vector<std::string>& vocab,
               const std::vector<std::pair<std::string, std::string>>& merges,
               const NormalizerConfig& ncfg);

    void build(const std::vector<std::string>& vocab,
               const std::vector<std::pair<std::string, std::string>>& merges,
               const NormalizerConfig& ncfg,
               const VocabOptions& opts);

    bool load(const std::string& path);
    void save(const std::string& path) const;

    std::vector<i32> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;

    std::vector<i32> encode_with_specials(const std::string& text) const;
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

    // Raw bytes a token stands for. For a native vocabulary this is the
    // token text; for a byte-level (foreign) vocabulary every character of
    // the token is mapped back to the byte it encodes.
    std::string token_bytes(i32 id) const;

    const std::vector<std::string>& vocab_entries() const { return vocab_; }

    std::vector<std::pair<std::string, std::string>> merge_pairs_ordered() const;
    bool is_special(i32 id) const { return id >= 0 && id < special::COUNT; }
    const NormalizerConfig& normalizer_config() const { return norm_.config(); }
    const Normalizer& normalizer() const { return norm_; }

    // --- foreign vocabulary support -------------------------------------
    bool foreign() const { return foreign_; }
    PreTokMode pretok_mode() const { return pretok_mode_; }
    bool add_bos() const { return add_bos_; }
    bool add_eos() const { return add_eos_; }
    i32 bos_id() const { return bos_id_; }
    i32 eos_id() const { return eos_id_; }
    i32 pad_id() const { return pad_id_; }
    i32 unk_id() const { return unk_id_; }
    // Default system prompt injected when the conversation has no system
    // message (see VocabOptions::default_system).
    const std::string& default_system() const { return default_system_; }

    // True when the vocabulary carries the full 256-entry byte-level
    // alphabet, i.e. it is a byte-level BPE vocabulary that the Gai engine
    // can encode against.
    bool has_byte_map() const { return byte_map_valid_; }
    const std::array<i32, 256>& byte_map() const { return byte_to_id_; }
    // token id for a raw byte, or -1 when the byte is not representable.
    i32 byte_id(u8 b) const {
        return byte_map_valid_ ? byte_to_id_[b] : -1;
    }
    // Bytes that the vocabulary cannot represent (rare: only happens for
    // byte-level alphabets that were pruned during training).
    const std::vector<int>& missing_bytes() const { return missing_bytes_; }

    // Digits are split one per token (HuggingFace
    // `Digits(individual_digits=true)`).
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
    Normalizer                                   norm_;
    mutable std::unordered_map<std::string, std::vector<i32>> cache_;
    // [FIX P2-12] bpe_chunk() is const but mutates cache_. Without a mutex,
    // sharing one Tokenizer across threads is a data race (UB/crash).
    // Shared ownership keeps Tokenizer copyable/movable (BpeTrainer::train
    // returns by value); copies share the lock, each keeps its own map.
    mutable std::shared_ptr<std::mutex> cache_mu_ = std::make_shared<std::mutex>();
    mutable size_t cache_limit_ = 200000;

    // byte-level BPE alphabet (foreign vocabularies)
    std::array<i32, 256> byte_to_id_{};
    bool                 byte_map_valid_ = false;
    std::vector<int>    missing_bytes_;
    bool                 digits_individual_ = false;
    bool                 foreign_ = false;
    PreTokMode           pretok_mode_ = PreTokMode::Ghassan;
    bool                 add_bos_ = false;
    bool                 add_eos_ = false;
    i32                  bos_id_ = special::BOS;
    i32                  eos_id_ = special::EOS;
    i32                  pad_id_ = special::PAD;
    i32                  unk_id_ = special::UNK;
    std::string          default_system_;
};

}
