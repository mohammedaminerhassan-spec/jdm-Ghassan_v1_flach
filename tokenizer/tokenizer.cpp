#include "tokenizer/tokenizer.h"

#include <fstream>
#include <queue>
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace gai {

// ---------------------------------------------------------------------------
// GPT-2 byte-level alphabet
// ---------------------------------------------------------------------------
// Reference implementation (GPT-2 encoder.py):
//   bs = printable ASCII + 0xA1..0xAC + 0xAE..0xFF
//   every byte not in bs is appended, in increasing order, to the private-use
//   area starting at 0x100.
const std::string& gpt2_byte_token(u8 b) {
    static std::array<std::string, 256> table;
    static bool built = false;
    if (!built) {
        bool in_set[256] = {false};
        auto mark = [&](int lo, int hi) {
            for (int v = lo; v <= hi; ++v) in_set[v] = true;
        };
        mark(0x21, 0x7E);
        mark(0xA1, 0xAC);
        mark(0xAE, 0xFF);

        int n = 0;
        for (int v = 0; v < 256; ++v) {
            if (in_set[v]) {
                if (v < 0x80) {
                    table[v] = std::string(1, static_cast<char>(v));
                } else {
                    // The reference table holds CODEPOINTS, so a byte like
                    // 0xA1 is the character U+00A1 and must be UTF-8 encoded
                    // (2 bytes), not emitted as a raw latin-1 byte.
                    std::string s;
                    utf8_encode(static_cast<u32>(v), s);
                    table[v] = s;
                }
            } else {
                // 0x100 + n, encoded as UTF-8
                u32 cp = static_cast<u32>(256 + n);
                ++n;
                std::string s;
                utf8_encode(cp, s);
                table[v] = s;
            }
        }
        built = true;
    }
    return table[b];
}

// Inverse of gpt2_byte_token(): maps the single character a byte-level
// token is made of back to the raw byte it stands for.
const std::unordered_map<std::string, u8>& gpt2_byte_decoder() {
    static std::unordered_map<std::string, u8> table;
    static bool built = false;
    if (!built) {
        for (int b = 0; b < 256; ++b) table[gpt2_byte_token(static_cast<u8>(b))] = static_cast<u8>(b);
        built = true;
    }
    return table;
}

void Tokenizer::build(const std::vector<std::string>& vocab,
                      const std::vector<std::pair<std::string, std::string>>& merges,
                      const VocabOptions& opts) {
    vocab_ = vocab;
    finalize_index();

    pretok_  = opts.pretok;
    add_bos_ = opts.add_bos;
    add_eos_ = opts.add_eos;
    bos_id_  = opts.bos_id;
    eos_id_  = opts.eos_id;
    pad_id_  = opts.pad_id;
    unk_id_  = opts.unk_id;

    special_ids_.clear();
    special_vec_.clear();
    special_vec_.reserve(opts.special_ids.size());
    for (i32 id : opts.special_ids) {
        if (id < 0) continue;
        if (special_ids_.insert(id).second) special_vec_.push_back(id);
    }
    std::sort(special_vec_.begin(), special_vec_.end());

    // The "added" tokens: every one of them is matched literally in the
    // input text, whether or not it is flagged special. When the caller did
    // not provide the list, the special ids are the best available guess.
    added_ids_.clear();
    added_vec_.clear();
    if (opts.added_ids.empty()) {
        added_vec_ = special_vec_;
    } else {
        added_vec_.reserve(opts.added_ids.size());
        for (i32 id : opts.added_ids) {
            if (id < 0) continue;
            if (added_ids_.insert(id).second) added_vec_.push_back(id);
        }
        std::sort(added_vec_.begin(), added_vec_.end());
    }
    for (i32 id : special_vec_) added_ids_.insert(id);

    merges_.clear();
    merges_.reserve(merges.size() * 2);
    std::unordered_set<i32> merged_ids;
    merged_ids.reserve(merges.size());
    for (size_t r = 0; r < merges.size(); ++r) {
        auto itl = token_ids_.find(merges[r].first);
        auto itr = token_ids_.find(merges[r].second);
        if (itl == token_ids_.end() || itr == token_ids_.end()) continue;
        auto itm = token_ids_.find(merges[r].first + merges[r].second);
        if (itm == token_ids_.end()) continue;
        u64 key = (static_cast<u64>(static_cast<u32>(itl->second)) << 32) |
                   static_cast<u32>(itr->second);
        merges_[key] = {static_cast<i32>(r), itm->second};
        merged_ids.insert(itm->second);
    }

    // Unused "<|unused..." slots that no merge produces: the vocabulary can
    // still reference them by id, but they must never be emitted by encode.
    for (size_t i = 0; i < vocab_.size(); ++i) {
        const std::string& t = vocab_[i];
        if (t.size() > 9 && t.compare(0, 9, "<|unused") == 0 &&
            merged_ids.find(static_cast<i32>(i)) == merged_ids.end()) {
            token_ids_.erase(t);
        }
    }
    { std::lock_guard<std::mutex> lk(*cache_mu_); cache_.clear(); }
    rebuild_byte_map();
    if (!missing_bytes_.empty()) {
        // Only printable ASCII matters in practice; control characters and
        // the rare Latin-1 letters that a pruned alphabet lacks are not worth
        // a warning on every load.
        std::string list;
        size_t printable_missing = 0;
        for (int b : missing_bytes_) {
            if (b >= 0x20 && b <= 0x7E) {
                ++printable_missing;
                if (list.size() < 64) list += strfmt("0x%02X ", b);
            }
        }
        if (printable_missing > 0) {
            log_warn(strfmt("[tokenizer] %zu printable byte(s) have no byte-level entry in "
                            "this vocabulary (%s) and will map to the unknown token",
                            printable_missing, list.c_str()));
        }
    }
}

// Build the byte -> token id table for a byte-level BPE vocabulary.
//
// The GPT-2 layout: the token whose text is gpt2_byte_token(b). Without this
// table the BPE merge table can never match: merges are keyed on real token
// ids but the base nodes were addressed as 16+byte, so every merge lookup
// missed and the encoder emitted raw byte ids that decode to unrelated
// characters.
void Tokenizer::rebuild_byte_map() {
    byte_to_id_.fill(-1);
    byte_map_valid_ = false;
    missing_bytes_.clear();

    bool printable_ok = true;
    for (int b = 0; b < 256; ++b) {
        const std::string& enc = gpt2_byte_token(static_cast<u8>(b));
        auto it = token_ids_.find(enc);
        if (it == token_ids_.end()) {
            if (b >= 0x20 && b <= 0x7E) printable_ok = false;
            missing_bytes_.push_back(b);
            continue;
        }
        byte_to_id_[static_cast<size_t>(b)] = it->second;
    }
    // A vocabulary is usable when every printable ASCII byte is
    // representable. Missing bytes (rare, e.g. a pruned alphabet) fall back
    // to the unknown token.
    byte_map_valid_ = printable_ok;
    digits_individual_ = detect_digits_individual();
}

// A byte-level BPE vocabulary that was trained with HuggingFace's
// `Digits(individual_digits=true)` pre-tokenizer contains the ten single
// digits and no multi-digit token at all. A vocabulary whose trainer did
// merge digits contains multi-digit tokens.
bool Tokenizer::detect_digits_individual() const {
    for (char c = '0'; c <= '9'; ++c) {
        if (token_ids_.find(std::string(1, c)) == token_ids_.end()) return false;
    }
    for (const auto& kv : token_ids_) {
        const std::string& t = kv.first;
        if (t.size() < 2) continue;
        bool all_digits = true;
        for (char c : t) {
            if (c < '0' || c > '9') { all_digits = false; break; }
        }
        if (all_digits) return false;
    }
    return true;
}

void Tokenizer::finalize_index() {
    token_ids_.clear();
    token_ids_.reserve(vocab_.size() * 2);
    for (size_t i = 0; i < vocab_.size(); ++i) token_ids_[vocab_[i]] = static_cast<i32>(i);
}

std::vector<std::pair<std::string, std::string>> Tokenizer::merge_pairs_ordered() const {
    i32 max_rank = -1;
    for (const auto& [key, val] : merges_) {
        if (val.first > max_rank) max_rank = val.first;
    }
    std::vector<std::pair<std::string, std::string>> out;
    if (max_rank < 0) return out;
    std::vector<char> seen(static_cast<size_t>(max_rank) + 1, 0);
    std::vector<std::pair<std::string, std::string>> by_rank(static_cast<size_t>(max_rank) + 1);
    for (const auto& [key, val] : merges_) {
        u32 l = static_cast<u32>(key >> 32);
        u32 r = static_cast<u32>(key & 0xFFFFFFFFu);
        if (l >= vocab_.size() || r >= vocab_.size()) continue;
        by_rank[static_cast<size_t>(val.first)] = {vocab_[l], vocab_[r]};
        seen[static_cast<size_t>(val.first)] = 1;
    }
    out.reserve(merges_.size());
    for (size_t i = 0; i < by_rank.size(); ++i) {
        if (seen[i]) out.push_back(std::move(by_rank[i]));
    }
    return out;
}

namespace {
struct Node {
    i32 id;
    int prev;
    int next;
    bool alive;
};
struct Cand {
    i32 rank;
    int pos;
    i32 merged;
    int left_id;
    int right_id;
    bool operator<(const Cand& o) const {
        if (rank != o.rank) return rank > o.rank;
        return pos > o.pos;
    }
};
}

void Tokenizer::bpe_chunk(const std::string& piece, std::vector<i32>& out) const {
    if (piece.empty()) return;
    GAI_CHECK(piece.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
              "tokenizer chunk exceeds int range");

    auto whole = token_ids_.find(piece);
    if (whole != token_ids_.end() && whole->second >= 0 &&
        special_ids_.find(whole->second) == special_ids_.end()) {
        out.push_back(whole->second);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(*cache_mu_);
        auto cit = cache_.find(piece);
        if (cit != cache_.end()) {
            out.insert(out.end(), cit->second.begin(), cit->second.end());
            return;
        }
    }

    std::vector<Node> nodes;
    nodes.reserve(piece.size());
    for (size_t i = 0; i < piece.size(); ++i) {
        const u8 b = static_cast<u8>(piece[i]);
        i32 id = byte_map_valid_ ? byte_to_id_[b] : -1;
        if (id < 0) id = unk_id_;
        nodes.push_back(Node{id, static_cast<int>(i) - 1,
                             (i + 1 < piece.size()) ? static_cast<int>(i) + 1 : -1, true});
    }

    std::priority_queue<Cand> pq;
    auto try_push = [&](int l) {
        if (l < 0) return;
        int r = nodes[static_cast<size_t>(l)].next;
        if (r < 0) return;
        u64 key = (static_cast<u64>(static_cast<u32>(nodes[static_cast<size_t>(l)].id)) << 32) |
                   static_cast<u32>(nodes[static_cast<size_t>(r)].id);
        auto it = merges_.find(key);
        if (it == merges_.end()) return;
        pq.push(Cand{it->second.first, l, it->second.second,
                     nodes[static_cast<size_t>(l)].id, nodes[static_cast<size_t>(r)].id});
    };

    for (size_t i = 0; i + 1 < nodes.size(); ++i) try_push(static_cast<int>(i));

    while (!pq.empty()) {
        Cand c = pq.top();
        pq.pop();
        Node& L = nodes[static_cast<size_t>(c.pos)];
        if (!L.alive || L.id != c.left_id) continue;
        int r = L.next;
        if (r < 0) continue;
        Node& R = nodes[static_cast<size_t>(r)];
        if (!R.alive || R.id != c.right_id) continue;

        L.id   = c.merged;
        L.next = R.next;
        if (R.next >= 0) nodes[static_cast<size_t>(R.next)].prev = c.pos;
        R.alive = false;

        try_push(L.prev);
        try_push(c.pos);
    }

    std::vector<i32> ids;
    for (int i = 0; i >= 0 && i < static_cast<int>(nodes.size()); i = nodes[static_cast<size_t>(i)].next) {
        if (nodes[static_cast<size_t>(i)].alive) ids.push_back(nodes[static_cast<size_t>(i)].id);
    }

    {
        std::lock_guard<std::mutex> lk(*cache_mu_);
        if (cache_.size() >= cache_limit_) {
            size_t n = 0;
            for (auto it = cache_.begin(); it != cache_.end();) {
                if ((n++ % 8) == 0) it = cache_.erase(it);
                else ++it;
            }
        }
        cache_.emplace(piece, ids);
    }
    out.insert(out.end(), ids.begin(), ids.end());
}

// GPT-2 / Qwen byte-level pre-tokenizer. Reproduces the reference regex
// by hand: std::regex (ECMAScript) has no lookahead, and the reference
// `regex` module is not available in C++.
std::vector<Chunk> pre_tokenize_gpt2(const std::string& text, bool split_digits) {
    std::vector<u32> cps = utf8_to_codepoints(text);
    std::vector<Chunk> out;
    const size_t n = cps.size();
    size_t i = 0;

    // Contractions, longest first. The reference regex lists them as
    // 's|'t|'re|'ve|'m|'ll|'d; none is a prefix of another, so longest-match
    // is equivalent to the ordered alternation.
    static const char* kContractions[] = {"'ll", "'re", "'ve", "'s", "'t", "'m", "'d"};

    auto emit = [&](size_t from, size_t to) {
        if (to <= from) return;
        Chunk c;
        for (size_t k = from; k < to; ++k) utf8_encode(cps[k], c.text);
        out.push_back(std::move(c));
    };

    while (i < n) {
        const size_t before = i;

        // --- alternative 1: contractions ---
        if (cps[i] == '\'' && i + 1 < n) {
            bool matched = false;
            for (const char* con : kContractions) {
                const size_t len = std::strlen(con);
                if (i + len > n) continue;
                bool ok = true;
                for (size_t k = 0; k < len; ++k) {
                    if (cps[i + k] != static_cast<u32>(static_cast<unsigned char>(con[k]))) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) continue;
                emit(i, i + len);
                i += len;
                matched = true;
                break;
            }
            if (matched) continue;
        }

        // --- alternatives 2/3/4: optional single space + a run of one class ---
        {
            size_t j = i;
            const bool spaced = (cps[j] == ' ');
            if (spaced) ++j;
            if (j < n) {
                const u32 c0 = cps[j];
                const bool letter = is_unicode_letter(c0);
                const bool number = is_unicode_number(c0);
                const bool other = !is_whitespace(c0) && !letter && !number;
                if (number && split_digits) {
                    size_t k = i;
                    if (spaced) { emit(k, k + 1); k += 1; }
                    while (k < n && is_unicode_number(cps[k])) { emit(k, k + 1); ++k; }
                    i = k;
                    continue;
                }
                if (letter || number || other) {
                    size_t k = j;
                    while (k < n) {
                        const u32 c = cps[k];
                        const bool same = letter ? is_unicode_letter(c)
                                       : number ? is_unicode_number(c)
                                                 : (!is_whitespace(c) && !is_unicode_letter(c) &&
                                                    !is_unicode_number(c));
                        if (!same) break;
                        ++k;
                    }
                    emit(i, k);
                    i = k;
                    continue;
                }
            }
        }

        // --- alternatives 5/6: whitespace runs ---
        {
            size_t k = i;
            while (k < n && is_whitespace(cps[k])) ++k;
            // \s+(?!\S) matches the run minus its last char when the run is
            // followed by a non-space (that trailing space gets glued to the
            // next token instead).
            if (k < n && (k - i) >= 2) {
                emit(i, k - 1);
                i = k - 1;
            } else {
                emit(i, k);
                i = k;
            }
        }

        // Safety net: a branch that fails to consume anything would spin
        // forever on a very long text.
        if (i == before) break;
    }
    return out;
}

// Qwen2/Qwen3 byte-level pre-tokenizer. Reproduces the reference regex:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//   ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// Differences from GPT-2: the letters class allows any single preceding
// non-newline non-number char (not only a space), digits are always single
// tokens, symbol runs keep trailing newlines, and whitespace runs carrying
// newlines match as one token.
std::vector<Chunk> pre_tokenize_qwen(const std::string& text, int digits_max) {
    std::vector<u32> cps = utf8_to_codepoints(text);
    std::vector<Chunk> out;
    const size_t n = cps.size();
    size_t i = 0;

    static const char* kContractions[] = {"'ll", "'re", "'ve", "'s", "'t", "'m", "'d"};

    auto emit = [&](size_t from, size_t to) {
        if (to <= from) return;
        Chunk c;
        for (size_t k = from; k < to; ++k) utf8_encode(cps[k], c.text);
        out.push_back(std::move(c));
    };
    auto is_letter_or_number = [&](u32 c) { return is_unicode_letter(c) || is_unicode_number(c); };

    while (i < n) {
        const size_t before = i;

        // --- alternative 1: contractions ---
        if (cps[i] == '\'' && i + 1 < n) {
            bool matched = false;
            for (const char* con : kContractions) {
                const size_t len = std::strlen(con);
                if (i + len > n) continue;
                bool ok = true;
                for (size_t k = 0; k < len; ++k) {
                    if (cps[i + k] != static_cast<u32>(static_cast<unsigned char>(con[k]))) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) continue;
                emit(i, i + len);
                i += len;
                matched = true;
                break;
            }
            if (matched) continue;
        }

        // --- alternative 2: [^\r\n\p{L}\p{N}]?\p{L}+ ---
        {
            size_t j = i;
            const u32 c0 = cps[j];
            const bool prefix_ok = (c0 != '\r' && c0 != '\n' &&
                                    !is_letter_or_number(c0));
            if (prefix_ok && j + 1 < n && is_unicode_letter(cps[j + 1])) ++j;
            if (j < n && is_unicode_letter(cps[j])) {
                size_t k = j;
                while (k < n && is_unicode_letter(cps[k])) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // --- alternative 3: \p{N}{1,3} (digits, at most digits_max per token) ---
        if (is_unicode_number(cps[i])) {
            size_t k = i;
            const size_t limit = static_cast<size_t>(digits_max > 0 ? digits_max : 1);
            while (k < n && is_unicode_number(cps[k]) && (k - i) < limit) ++k;
            emit(i, k);
            i = k;
            continue;
        }

        // --- alternative 4: ?[^\s\p{L}\p{N}]+[\r\n]* ---
        {
            size_t j = i;
            if (cps[j] == ' ') ++j;
            if (j < n && !is_whitespace(cps[j]) && !is_letter_or_number(cps[j])) {
                size_t k = j;
                while (k < n && !is_whitespace(cps[k]) && !is_letter_or_number(cps[k])) ++k;
                while (k < n && (cps[k] == '\r' || cps[k] == '\n')) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // --- alternative 5: \s*[\r\n]+ ---
        //
        // The reference regex is greedy: \s* takes as much whitespace as it
        // can and then [\r\n]+ must match at least one newline. The newline
        // does NOT have to be the last character of the whitespace run - it
        // only has to follow the \s* prefix. Example "\n\n  X":
        //   HF      -> "\n\n" | " " | " X"
        //   naive   -> "\n\n " | " X"     (wrong: different BPE input)
        // So: find the last newline inside the run, then consume the maximal
        // run of \r\n characters from there.
        {
            size_t k = i;
            while (k < n && is_whitespace(cps[k])) ++k;
            size_t last_nl = n;  // n == "not found"
            for (size_t t = i; t < k; ++t) {
                if (cps[t] == '\r' || cps[t] == '\n') last_nl = t;
            }
            if (last_nl < n) {
                size_t e = last_nl + 1;
                while (e < n && (cps[e] == '\r' || cps[e] == '\n')) ++e;
                emit(i, e);
                i = e;
                continue;
            }
        }

        // --- alternatives 6/7: \s+(?!\S) | \s+ ---
        {
            size_t k = i;
            while (k < n && is_whitespace(cps[k])) ++k;
            if (k < n && (k - i) >= 2) {
                emit(i, k - 1);
                i = k - 1;
            } else {
                emit(i, k);
                i = k;
            }
        }

        if (i == before) break;
    }
    return out;
}

std::vector<i32> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<i32> out;
    if (add_bos) out.push_back(bos_id_);

    // HuggingFace tokenizes special tokens (added_tokens) even in plain text:
    // a literal "<|im_start|>" in the input becomes the control id. Mirror
    // that semantics exactly: split the input on special strings (longest
    // match wins), BPE the remaining segments.
    if (special_vec_.empty()) {
        if (pretok_ == PreTokKind::Qwen) {
            for (const Chunk& c : pre_tokenize_qwen(text, 1)) bpe_chunk(c.text, out);
        } else if (pretok_ == PreTokKind::Granite) {
            for (const Chunk& c : pre_tokenize_qwen(text, 3)) bpe_chunk(c.text, out);
        } else {
            for (const Chunk& c : pre_tokenize_gpt2(text, digits_individual_)) bpe_chunk(c.text, out);
        }
    } else {
        std::vector<std::pair<std::string, i32>> sp;
        sp.reserve(special_vec_.size());
        for (i32 id : special_vec_) {
            const std::string& s = token_text(id);
            if (!s.empty()) sp.emplace_back(s, id);
        }
        std::sort(sp.begin(), sp.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

        size_t i = 0;
        std::string buf;
        auto flush = [&]() {
            if (buf.empty()) return;
            if (pretok_ == PreTokKind::Qwen) {
                for (const Chunk& c : pre_tokenize_qwen(buf, 1)) bpe_chunk(c.text, out);
            } else if (pretok_ == PreTokKind::Granite) {
                for (const Chunk& c : pre_tokenize_qwen(buf, 3)) bpe_chunk(c.text, out);
            } else {
                for (const Chunk& c : pre_tokenize_gpt2(buf, digits_individual_)) bpe_chunk(c.text, out);
            }
            buf.clear();
        };
        while (i < text.size()) {
            bool matched = false;
            for (const auto& [sp_str, sp_id] : sp) {
                if (text.compare(i, sp_str.size(), sp_str) == 0) {
                    flush();
                    out.push_back(sp_id);
                    i += sp_str.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                buf.push_back(text[i]);
                ++i;
            }
        }
        flush();
    }

    if (add_eos) out.push_back(eos_id_);
    return out;
}

const std::string& Tokenizer::token_text(i32 id) const {
    static const std::string empty;
    if (id < 0 || id >= static_cast<i32>(vocab_.size())) return empty;
    return vocab_[static_cast<size_t>(id)];
}

i32 Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_ids_.find(tok);
    return it == token_ids_.end() ? -1 : it->second;
}

std::string Tokenizer::decode_one(i32 id) const { return token_text(id); }

// Maps the characters of a byte-level token back to the bytes they stand for.
std::string Tokenizer::token_bytes(i32 id) const {
    const std::string& txt = token_text(id);
    const auto& dec = gpt2_byte_decoder();
    std::string out;
    out.reserve(txt.size());
    size_t i = 0;
    while (i < txt.size()) {
        const size_t before = i;
        const u32 cp = utf8_decode(txt, i);
        if (cp == 0xFFFD && i - before <= 1) {
            out.push_back(txt[before]);
            continue;
        }
        std::string key;
        utf8_encode(cp, key);
        auto it = dec.find(key);
        if (it == dec.end()) {
            // Not a byte-level character (e.g. a special token): keep it.
            out.append(key);
        } else {
            out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<i32>& ids, bool skip_special) const {
    // Every token contributes raw bytes, and the concatenation is decoded as
    // UTF-8 at the very end (a character can be split across two tokens).
    std::string raw;
    raw.reserve(ids.size() * 3);
    for (i32 id : ids) {
        if (skip_special && is_special(id)) continue;
        raw += token_bytes(id);
    }
    return utf8_from_bytes(raw);
}

std::string Tokenizer::Stream::push(i32 id) {
    // The token contributes raw bytes; a character may be split across two
    // tokens, so the incomplete tail is kept in buf_ until it becomes a valid
    // UTF-8 sequence.
    buf_ += tk_.token_bytes(id);

    size_t safe = 0, i = 0;
    while (i < buf_.size()) {
        int len = utf8_seq_len(static_cast<u8>(buf_[i]));
        if (len == 0) break;
        if (i + static_cast<size_t>(len) > buf_.size()) break;
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            if ((static_cast<u8>(buf_[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) { ++i; safe = i; continue; }
        i += static_cast<size_t>(len);
        safe = i;
    }
    std::string emit = buf_.substr(0, safe);
    buf_.erase(0, safe);
    return emit;
}

std::string Tokenizer::Stream::flush() {
    if (buf_.empty()) return {};
    std::string s;
    if (utf8_is_complete(buf_)) s = buf_;
    else s = "\xEF\xBF\xBD";
    buf_.clear();
    return s;
}

Tokenizer::Fertility Tokenizer::measure(const std::vector<std::string>& lines) const {
    Fertility f;
    for (const auto& line : lines) {
        if (line.empty()) continue;
        std::vector<i32> ids = encode_default(line);
        f.tokens += ids.size();
        f.bytes  += line.size();

        bool in_word = false;
        size_t i = 0;
        while (i < line.size()) {
            u32 cp = utf8_decode(line, i);
            bool ws = is_whitespace(cp);
            if (!ws && !in_word) { ++f.words; in_word = true; }
            else if (ws) in_word = false;
        }
    }
    f.tokens_per_word = f.words ? static_cast<double>(f.tokens) / static_cast<double>(f.words) : 0.0;
    f.bytes_per_token = f.tokens ? static_cast<double>(f.bytes) / static_cast<double>(f.tokens) : 0.0;
    return f;
}

}
