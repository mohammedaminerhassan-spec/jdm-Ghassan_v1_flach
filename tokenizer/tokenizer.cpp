#include "tokenizer/tokenizer.h"

#include <fstream>
#include <queue>
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace gai {

const std::vector<std::string>& special_token_strings() {
    static const std::vector<std::string> s = {
        "<pad>", "<s>", "</s>", "<unk>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<|nl|>", "<|fr|>", "<|ar|>", "<|darija|>", "<|latin|>",
        "<|tool|>", "<|mask|>", "<|reserved|>"
    };
    return s;
}

static constexpr u32 GTOK_MAGIC   = 0x4B4F5447u;
static constexpr u32 GTOK_VERSION = 1u;

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

void Tokenizer::init_empty(const NormalizerConfig& ncfg) {
    vocab_.clear();
    token_ids_.clear();
    merges_.clear();
    norm_.set_config(ncfg);

    for (const auto& s : special_token_strings()) vocab_.push_back(s);
    for (int b = 0; b < 256; ++b) vocab_.push_back(std::string(1, static_cast<char>(b)));
    finalize_index();
    add_bos_ = true;
    rebuild_byte_map();
}

void Tokenizer::build(const std::vector<std::string>& vocab,
                      const std::vector<std::pair<std::string, std::string>>& merges,
                      const NormalizerConfig& ncfg) {
    VocabOptions o;
    o.foreign = false;
    o.pretok  = PreTokMode::Ghassan;
    o.add_bos = true;
    o.add_eos = false;
    build(vocab, merges, ncfg, o);
}

void Tokenizer::build(const std::vector<std::string>& vocab,
                      const std::vector<std::pair<std::string, std::string>>& merges,
                      const NormalizerConfig& ncfg,
                      const VocabOptions& opts) {
    vocab_ = vocab;
    norm_.set_config(ncfg);
    finalize_index();

    foreign_       = opts.foreign;
    pretok_mode_   = opts.pretok;
    add_bos_       = opts.add_bos;
    add_eos_       = opts.add_eos;
    bos_id_        = opts.bos_id;
    eos_id_        = opts.eos_id;
    pad_id_        = opts.pad_id;
    unk_id_        = opts.unk_id;
    default_system_ = opts.default_system;

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

    for (size_t i = 0; i < vocab_.size(); ++i) {
        const std::string& t = vocab_[i];
        if (t.size() > 9 && t.compare(0, 9, "<|unused") == 0 &&
            merged_ids.find(static_cast<i32>(i)) == merged_ids.end()) {
            token_ids_.erase(t);
        }
    }
    { std::lock_guard<std::mutex> lk(*cache_mu_); cache_.clear(); }
    rebuild_byte_map();
    if (foreign_ && !missing_bytes_.empty()) {
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
// Two layouts are supported:
//   1. the native Gai layout: token id 16+b is the single byte b
//   2. the GPT-2 layout: the token whose text is gpt2_byte_token(b)
//
// Layout 2 is what every HuggingFace byte-level vocabulary uses (GPT-2,
// Qwen, Llama-3, ...). Without this table the BPE merge table can never
// match: merges are keyed on real token ids but the base nodes were
// addressed as 16+byte, so every merge lookup missed and the encoder
// emitted raw byte ids that decode to unrelated characters.
void Tokenizer::rebuild_byte_map() {
    byte_to_id_.fill(-1);
    byte_map_valid_ = false;
    missing_bytes_.clear();

    // Layout 1: native .gtok layout (specials, then the 256 raw bytes).
    bool native_ok = vocab_.size() >= static_cast<size_t>(special::COUNT) + 256;
    if (native_ok) {
        for (int b = 0; b < 256; ++b) {
            const std::string& t = vocab_[static_cast<size_t>(special::COUNT) + b];
            if (t.size() != 1 || static_cast<u8>(t[0]) != static_cast<u8>(b)) {
                native_ok = false;
                break;
            }
            byte_to_id_[static_cast<size_t>(b)] = special::COUNT + b;
        }
    }
    if (native_ok) {
        byte_map_valid_ = true;
        digits_individual_ = false;
        return;
    }

    // Layout 2: GPT-2 byte-level alphabet. The encoded form of a byte is a
    // single character, so it can only collide with another byte's encoding
    // if the mapping is not injective -- it is, by construction.
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
    // to the unknown token, exactly like a HuggingFace tokenizer that has no
    // byte-level entry for them.
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
    if (whole != token_ids_.end() && whole->second >= special::COUNT) {
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
        // Foreign (byte-level) vocabularies address their byte tokens by
        // the real vocab id, not by the native 16+byte convention.
        i32 id = byte_map_valid_ ? byte_to_id_[b] : (special::COUNT + static_cast<i32>(b));
        if (id < 0) id = (unk_id_ >= 0) ? unk_id_ : special::UNK;
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

std::vector<i32> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<i32> out;
    // A foreign vocabulary ships its own normalization behaviour (usually
    // none beyond the byte-level pre-tokenizer). Running the Gai normalizer
    // over its text would strip diacritics / lowercase it and produce token
    // ids the model was never trained on -- a silent train/serve skew.
    if (add_bos) out.push_back(foreign_ ? bos_id_ : special::BOS);

    const std::string norm = foreign_ ? text : norm_.normalize(text);
    if (pretok_mode_ == PreTokMode::Gpt2) {
        for (const Chunk& c : pre_tokenize_gpt2(norm, digits_individual_)) bpe_chunk(c.text, out);
    } else {
        for (const Chunk& c : pre_tokenize(norm)) bpe_chunk(c.text, out);
    }

    if (add_eos) out.push_back(foreign_ ? eos_id_ : special::EOS);
    return out;
}

std::vector<i32> Tokenizer::encode_with_specials(const std::string& text) const {
    std::vector<i32> out;
    const auto& specials = special_token_strings();

    size_t i = 0;
    std::string buf;
    auto flush = [&]() {
        if (buf.empty()) return;
        std::string norm = norm_.normalize(buf);
        for (const Chunk& c : pre_tokenize(norm)) bpe_chunk(c.text, out);
        buf.clear();
    };

    // NOTE: this deliberately treats embedded special-token spellings as
    // control tokens. Callers must only pass trusted template-built text;
    // raw user content must go through encode() so a literal "<|system|>"
    // typed by a user can never inject a control token.
    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<') {
            // Longest-match wins so overlapping spellings (if any are added
            // in the future) resolve deterministically.
            size_t best = specials.size();
            size_t best_len = 0;
            for (size_t s = 0; s < specials.size(); ++s) {
                const std::string& sp = specials[s];
                if (sp.size() > best_len && text.compare(i, sp.size(), sp) == 0) {
                    best = s;
                    best_len = sp.size();
                }
            }
            if (best != specials.size()) {
                flush();
                out.push_back(static_cast<i32>(best));
                i += best_len;
                matched = true;
            }
        }
        if (!matched) { buf.push_back(text[i]); ++i; }
    }
    flush();
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

// Maps the characters of a byte-level token back to the bytes they stand
// for. A native vocabulary stores plain text, so the text is returned as-is.
std::string Tokenizer::token_bytes(i32 id) const {
    const std::string& txt = token_text(id);
    if (!foreign_) return txt;

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
        // Re-encode the codepoint as UTF-8 so it can be looked up as a key.
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
    if (!foreign_) {
        std::string out;
        out.reserve(ids.size() * 3);
        for (i32 id : ids) {
            if (skip_special && is_special(id)) continue;
            out += token_text(id);
        }
        return out;
    }
    // Byte-level vocabulary: every token contributes raw bytes, and the
    // concatenation is decoded as UTF-8 at the very end (a character can be
    // split across two tokens).
    std::string raw;
    raw.reserve(ids.size() * 3);
    for (i32 id : ids) {
        if (skip_special && is_special(id)) continue;
        raw += token_bytes(id);
    }
    return utf8_from_bytes(raw);
}

std::string Tokenizer::Stream::push(i32 id) {
    // For a byte-level vocabulary the token contributes raw bytes; a
    // character may be split across two tokens, so the incomplete tail is
    // kept in buf_ until it becomes a valid UTF-8 sequence.
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

template <typename T>
static void wr(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <typename T>
static bool rd(std::istream& in, T& v) { return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof(T))); }

void Tokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot write tokenizer: " + path);

    wr(f, GTOK_MAGIC);
    wr(f, GTOK_VERSION);
    u32 nflags = norm_.config().pack();
    wr(f, nflags);
    u32 nvocab = static_cast<u32>(vocab_.size());
    wr(f, nvocab);

    for (const auto& t : vocab_) {
        u32 len = static_cast<u32>(t.size());
        wr(f, len);
        f.write(t.data(), static_cast<std::streamsize>(len));
    }

    std::vector<std::pair<i32, u64>> ordered;
    ordered.reserve(merges_.size());
    for (const auto& [key, val] : merges_) ordered.emplace_back(val.first, key);
    std::sort(ordered.begin(), ordered.end());

    u32 nmerges = static_cast<u32>(ordered.size());
    wr(f, nmerges);
    for (const auto& [rank, key] : ordered) {
        u32 l = static_cast<u32>(key >> 32);
        u32 r = static_cast<u32>(key & 0xFFFFFFFFu);
        wr(f, l);
        wr(f, r);
    }
    GAI_CHECK(f.good(), "tokenizer write failed: " + path);
}

bool Tokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0, nflags = 0, nvocab = 0;
    if (!rd(f, magic) || magic != GTOK_MAGIC) return false;
    if (!rd(f, version) || version != GTOK_VERSION) return false;
    if (!rd(f, nflags)) return false;
    if (!rd(f, nvocab) || nvocab == 0 || nvocab > 10000000u) return false;

    norm_.set_config(NormalizerConfig::unpack(nflags));

    vocab_.clear();
    vocab_.reserve(nvocab);
    for (u32 i = 0; i < nvocab; ++i) {
        u32 len = 0;
        if (!rd(f, len) || len > 1024) return false;
        std::string t(len, '\0');
        if (len && !f.read(t.data(), static_cast<std::streamsize>(len))) return false;
        vocab_.push_back(std::move(t));
    }

    if (vocab_.size() < static_cast<size_t>(special::COUNT) + 256) return false;
    {
        const auto& specials = special_token_strings();
        for (int s = 0; s < special::COUNT; ++s) {
            if (vocab_[static_cast<size_t>(s)] != specials[static_cast<size_t>(s)])
                return false;
        }
    }
    for (int b = 0; b < 256; ++b) {
        const std::string& t = vocab_[static_cast<size_t>(special::COUNT) + b];
        if (t.size() != 1 || static_cast<u8>(t[0]) != static_cast<u8>(b)) return false;
    }
    finalize_index();

    // A .gtok file is always a native Gai vocabulary.
    foreign_     = false;
    pretok_mode_ = PreTokMode::Ghassan;
    add_bos_     = true;
    add_eos_     = false;
    bos_id_      = special::BOS;
    eos_id_      = special::EOS;
    pad_id_      = special::PAD;
    unk_id_      = special::UNK;

    u32 nmerges = 0;
    if (!rd(f, nmerges) || nmerges > 10000000u) return false;
    merges_.clear();
    merges_.reserve(static_cast<size_t>(nmerges) * 2);
    for (u32 r = 0; r < nmerges; ++r) {
        u32 l = 0, rr = 0;
        if (!rd(f, l) || !rd(f, rr)) return false;
        if (l >= nvocab || rr >= nvocab) return false;
        const std::string merged = vocab_[l] + vocab_[rr];
        auto it = token_ids_.find(merged);
        if (it == token_ids_.end()) continue;
        merges_[(static_cast<u64>(l) << 32) | rr] = {static_cast<i32>(r), it->second};
    }
    { std::lock_guard<std::mutex> lk(*cache_mu_); cache_.clear(); }
    rebuild_byte_map();
    return true;
}

Tokenizer::Fertility Tokenizer::measure(const std::vector<std::string>& lines) const {
    Fertility f;
    for (const auto& line : lines) {
        if (line.empty()) continue;
        std::vector<i32> ids = encode(line);
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
