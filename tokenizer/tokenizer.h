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

const std::string& gpt2_byte_token(u8 b);

const std::unordered_map<std::string, u8>& gpt2_byte_decoder();

enum class PreTokKind : u8 { Gpt2 = 0, Qwen = 1, Granite = 2 };

struct VocabOptions {
    bool     add_bos = false;
    bool     add_eos = false;
    i32      bos_id  = 0;
    i32      eos_id  = 2;
    i32      pad_id  = 0;
    i32      unk_id  = 3;
    PreTokKind pretok = PreTokKind::Gpt2;
    std::vector<i32> special_ids;
    std::vector<i32> added_ids;
};

class Tokenizer {
public:
    Tokenizer() = default;

    void build(const std::vector<std::string>& vocab,
               const std::vector<std::pair<std::string, std::string>>& merges,
               const VocabOptions& opts = {});

    std::vector<i32> encode(const std::string& text,
                            bool add_bos = false,
                            bool add_eos = false) const;
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
    u64  fingerprint() const;
    const std::string& token_text(i32 id) const;
    i32  token_to_id(const std::string& tok) const;

    std::string token_bytes(i32 id) const;

    const std::vector<std::string>& vocab_entries() const { return vocab_; }

    std::vector<std::pair<std::string, std::string>> merge_pairs_ordered() const;

    bool is_special(i32 id) const { return special_ids_.find(id) != special_ids_.end(); }
    const std::vector<i32>& special_ids() const { return special_vec_; }

    bool is_added(i32 id) const { return added_ids_.find(id) != added_ids_.end(); }
    const std::vector<i32>& added_ids() const { return added_vec_; }

    bool add_bos() const { return add_bos_; }
    bool add_eos() const { return add_eos_; }
    i32  bos_id()  const { return bos_id_; }
    i32  eos_id()  const { return eos_id_; }
    i32  pad_id()  const { return pad_id_; }
    i32  unk_id()  const { return unk_id_; }

    bool has_byte_map() const { return byte_map_valid_; }
    const std::array<i32, 256>& byte_map() const { return byte_to_id_; }
    i32 byte_id(u8 b) const {
        return byte_map_valid_ ? byte_to_id_[b] : -1;
    }
    const std::vector<int>& missing_bytes() const { return missing_bytes_; }

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
    mutable std::shared_ptr<std::mutex> cache_mu_ = std::make_shared<std::mutex>();
    mutable size_t cache_limit_ = 200000;

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

struct Chunk {
    std::string text;
};
std::vector<Chunk> pre_tokenize_gpt2(const std::string& text, bool split_digits = false);
std::vector<Chunk> pre_tokenize_qwen(const std::string& text, int digits_max = 1);

}
