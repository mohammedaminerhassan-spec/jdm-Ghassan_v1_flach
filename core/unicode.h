#pragma once

// The text helpers the engine needs: the UTF-8 codec (the tokenizer works on
// codepoints) and the two Unicode general categories that the GPT-2
// pre-tokenizer uses (\p{L} and \p{N}).

#include "core/common.h"
#include <vector>
#include <string>

namespace gai {

// ---- the UTF-8 codec

int    utf8_seq_len(u8 b);
u32    utf8_decode(const std::string& s, size_t& i);
void   utf8_encode(u32 cp, std::string& out);
std::vector<u32> utf8_to_codepoints(const std::string& s);
// Decodes a raw byte buffer as UTF-8. Invalid or truncated sequences are
// replaced with U+FFFD instead of being dropped.
std::string utf8_from_bytes(const std::string& bytes);
bool   utf8_is_complete(const std::string& s);

// ---- the character classes

bool is_whitespace(u32 cp);

// Unicode general categories used by the GPT-2 pre-tokenizer (\p{L}, \p{N}).
// Table-driven approximations of the full Unicode database: they cover every
// script the engine trains on (Latin, Greek, Cyrillic, Hebrew, Arabic,
// Indic, CJK, Hangul, Kana, ...) and degrade to "not a letter" for the long
// tail instead of shipping a 200 KB table.
bool is_unicode_letter(u32 cp);
bool is_unicode_number(u32 cp);

}
