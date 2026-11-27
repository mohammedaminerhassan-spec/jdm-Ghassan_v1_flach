#pragma once

#include "core/common.h"
#include <vector>
#include <string>

namespace gai {

int    utf8_seq_len(u8 b);
u32    utf8_decode(const std::string& s, size_t& i);
void   utf8_encode(u32 cp, std::string& out);
std::vector<u32> utf8_to_codepoints(const std::string& s);
std::string utf8_from_bytes(const std::string& bytes);
bool   utf8_is_complete(const std::string& s);

bool is_whitespace(u32 cp);

bool is_unicode_letter(u32 cp);
bool is_unicode_number(u32 cp);

}
