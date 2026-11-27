#include "tokenizer/tokenizer.h"

#include <fstream>
#include <queue>
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <mutex>

namespace gai {

static inline u32 ascii_lower(u32 c) {
    return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}

const std::string& gpt2_byte_token(u8 b) {
    static std::array<std::string, 256> table;
    static std::once_flag once;
    std::call_once(once, [] {
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
                    std::string s;
                    utf8_encode(static_cast<u32>(v), s);
                    table[v] = s;
                }
            } else {
                u32 cp = static_cast<u32>(256 + n);
                ++n;
                std::string s;
                utf8_encode(cp, s);
                table[v] = s;
            }
        }
    });
    return table[b];
}

const std::unordered_map<std::string, u8>& gpt2_byte_decoder() {
    static std::unordered_map<std::string, u8> table;
    static std::once_flag once;
    std::call_once(once, [] {
        for (int b = 0; b < 256; ++b) table[gpt2_byte_token(static_cast<u8>(b))] = static_cast<u8>(b);
    });
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

    added_ids_.clear();
    added_vec_.clear();
    if (opts.added_ids.empty()) {
        added_vec_ = special_vec_;
    } else {
        added_vec_.reserve(opts.added_ids.size() + special_vec_.size());
        for (i32 id : opts.added_ids) {
            if (id < 0) continue;
            if (added_ids_.insert(id).second) added_vec_.push_back(id);
        }
        for (i32 id : special_vec_) {
            if (added_ids_.insert(id).second) added_vec_.push_back(id);
        }
        std::sort(added_vec_.begin(), added_vec_.end());
    }

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
    if (!missing_bytes_.empty()) {
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
    byte_map_valid_ = printable_ok;
    digits_individual_ = detect_digits_individual();
}

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

std::vector<Chunk> pre_tokenize_gpt2(const std::string& text, bool split_digits) {
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

    while (i < n) {
        const size_t before = i;

        if (cps[i] == '\'' && i + 1 < n) {
            bool matched = false;
            for (const char* con : kContractions) {
                const size_t len = std::strlen(con);
                if (i + len > n) continue;
                bool ok = true;
                for (size_t k = 0; k < len; ++k) {
                    if (ascii_lower(cps[i + k]) !=
                        ascii_lower(static_cast<u32>(static_cast<unsigned char>(con[k])))) {
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

        if (cps[i] == '\'' && i + 1 < n) {
            bool matched = false;
            for (const char* con : kContractions) {
                const size_t len = std::strlen(con);
                if (i + len > n) continue;
                bool ok = true;
                for (size_t k = 0; k < len; ++k) {
                    if (ascii_lower(cps[i + k]) !=
                        ascii_lower(static_cast<u32>(static_cast<unsigned char>(con[k])))) {
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

        if (is_unicode_number(cps[i])) {
            size_t k = i;
            const size_t limit = static_cast<size_t>(digits_max > 0 ? digits_max : 1);
            while (k < n && is_unicode_number(cps[k]) && (k - i) < limit) ++k;
            emit(i, k);
            i = k;
            continue;
        }

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

        {
            size_t k = i;
            while (k < n && is_whitespace(cps[k])) ++k;
            size_t last_nl = n;
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

    if (special_vec_.empty() && added_vec_.empty()) {
        if (pretok_ == PreTokKind::Qwen) {
            for (const Chunk& c : pre_tokenize_qwen(text, 1)) bpe_chunk(c.text, out);
        } else if (pretok_ == PreTokKind::Granite) {
            for (const Chunk& c : pre_tokenize_qwen(text, 3)) bpe_chunk(c.text, out);
        } else {
            for (const Chunk& c : pre_tokenize_gpt2(text, digits_individual_)) bpe_chunk(c.text, out);
        }
    } else {
        std::vector<std::pair<std::string, i32>> sp;
        const std::vector<i32>* match_ids = added_vec_.empty() ? &special_vec_ : &added_vec_;
        sp.reserve(match_ids->size());
        for (i32 id : *match_ids) {
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

u64 Tokenizer::fingerprint() const {
    u64 h = 1469598103934665603ull;
    auto mix = [&](u64 v) {
        for (int b = 0; b < 8; ++b) { h ^= (v >> (8 * b)) & 0xFF; h *= 1099511628211ull; }
    };
    auto mix_str = [&](const std::string& s) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        mix(0);
    };
    for (const auto& s : vocab_) mix_str(s);
    std::vector<u64> mk;
    mk.reserve(merges_.size());
    for (const auto& [key, val] : merges_)
        mk.push_back(((key << 32) ^ ((static_cast<u64>(static_cast<u32>(val.first)) << 16))) |
                     static_cast<u32>(val.second));
    std::sort(mk.begin(), mk.end());
    for (u64 v : mk) mix(v);
    mix(static_cast<u64>(pretok_) + 1);
    mix(vocab_.size());
    return h;
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
            out.append(key);
        } else {
            out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<i32>& ids, bool skip_special) const {
    std::string raw;
    raw.reserve(ids.size() * 3);
    for (i32 id : ids) {
        if (skip_special && is_special(id)) continue;
        raw += token_bytes(id);
    }
    return utf8_from_bytes(raw);
}

std::string Tokenizer::Stream::push(i32 id) {
    buf_ += tk_.token_bytes(id);

    std::string emit;
    size_t i = 0;
    static const char kReplacement[] = "\xEF\xBF\xBD";
    while (i < buf_.size()) {
        const u8 lead = static_cast<u8>(buf_[i]);
        const int len = utf8_seq_len(lead);
        if (len == 0) {
            emit.append(kReplacement, 3);
            ++i;
            continue;
        }
        if (i + static_cast<size_t>(len) > buf_.size()) break;
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            if ((static_cast<u8>(buf_[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            emit.append(kReplacement, 3);
            ++i;
            continue;
        }
        emit.append(buf_, i, static_cast<size_t>(len));
        i += static_cast<size_t>(len);
    }
    buf_.erase(0, i);
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
