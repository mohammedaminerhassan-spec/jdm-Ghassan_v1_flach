#include "core/unicode.h"

namespace gai {

int utf8_seq_len(u8 b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

u32 utf8_decode(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    u8 b0 = static_cast<u8>(s[i]);
    int len = utf8_seq_len(b0);
    if (len == 0 || i + static_cast<size_t>(len) > s.size()) { ++i; return 0xFFFD; }

    u32 cp = 0;
    switch (len) {
        case 1: cp = b0; break;
        case 2: cp = b0 & 0x1Fu; break;
        case 3: cp = b0 & 0x0Fu; break;
        case 4: cp = b0 & 0x07u; break;
        default: break;
    }
    for (int k = 1; k < len; ++k) {
        u8 bk = static_cast<u8>(s[i + static_cast<size_t>(k)]);
        if ((bk & 0xC0) != 0x80) { ++i; return 0xFFFD; }
        cp = (cp << 6) | (bk & 0x3Fu);
    }

    if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
        (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
        ++i;
        return 0xFFFD;
    }
    i += static_cast<size_t>(len);
    return cp;
}

void utf8_encode(u32 cp, std::string& out) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string utf8_from_cp(u32 cp) {
    std::string s;
    utf8_encode(cp, s);
    return s;
}

std::vector<u32> utf8_to_codepoints(const std::string& s) {
    std::vector<u32> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) out.push_back(utf8_decode(s, i));
    return out;
}

std::string codepoints_to_utf8(const std::vector<u32>& cps) {
    std::string out;
    out.reserve(cps.size() * 2);
    for (u32 c : cps) utf8_encode(c, out);
    return out;
}

std::string utf8_from_bytes(const std::string& bytes) {
    std::string out;
    out.reserve(bytes.size());
    const size_t n = bytes.size();
    size_t i = 0;
    while (i < n) {
        const int len = utf8_seq_len(static_cast<u8>(bytes[i]));
        if (len <= 0 || i + static_cast<size_t>(len) > n) {
            utf8_encode(0xFFFD, out);
            ++i;
            continue;
        }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            if ((static_cast<u8>(bytes[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            utf8_encode(0xFFFD, out);
            ++i;
            continue;
        }
        out.append(bytes, i, static_cast<size_t>(len));
        i += static_cast<size_t>(len);
    }
    return out;
}

size_t utf8_length(const std::string& s) {
    size_t i = 0, n = 0;
    while (i < s.size()) { utf8_decode(s, i); ++n; }
    return n;
}

bool utf8_valid(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t before = i;
        u32 cp = utf8_decode(s, i);
        if (cp == 0xFFFD && !(s.size() - before >= 3 &&
                              static_cast<u8>(s[before]) == 0xEF &&
                              static_cast<u8>(s[before + 1]) == 0xBF &&
                              static_cast<u8>(s[before + 2]) == 0xBD)) {
            return false;
        }
    }
    return true;
}

bool utf8_is_complete(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        u8 b = static_cast<u8>(s[i]);
        int len = utf8_seq_len(b);
        if (len == 0) return false;
        if (i + static_cast<size_t>(len) > s.size()) return false;

        for (int k = 1; k < len; ++k) {
            if ((static_cast<u8>(s[i + static_cast<size_t>(k)]) & 0xC0) != 0x80)
                return false;
        }
        i += static_cast<size_t>(len);
    }
    return true;
}

bool is_arabic_letter(u32 cp) {
    if (cp == 0x0620) return true;
    if (cp >= 0x0621 && cp <= 0x063A) return true;
    if (cp >= 0x063B && cp <= 0x063F) return true;
    if (cp >= 0x0641 && cp <= 0x064A) return true;
    if (cp >= 0x066E && cp <= 0x06D3) return true;
    if (cp == 0x0671 || cp == 0x0672 || cp == 0x0673) return true;
    if (cp >= 0x06FA && cp <= 0x06FF) return true;
    if (cp >= 0x0750 && cp <= 0x077F) return true;
    if (cp >= 0x08A0 && cp <= 0x08BF) return true;
    return false;
}

bool is_arabic_diacritic(u32 cp) {
    if (cp >= 0x064B && cp <= 0x065F) return true;
    if (cp == 0x0670) return true;
    if (cp >= 0x06D6 && cp <= 0x06ED) return true;
    if (cp >= 0x08D3 && cp <= 0x08FF) return true;
    return false;
}

bool is_arabic_punct(u32 cp) {
    switch (cp) {
        case 0x060C: case 0x061B: case 0x061F: case 0x066A: case 0x066B:
        case 0x066C: case 0x066D: case 0x06D4: case 0x061E: case 0x0600:
            return true;
        default: return false;
    }
}

bool is_arabic_digit(u32 cp) {
    return (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9);
}

bool is_tatweel(u32 cp) { return cp == 0x0640; }

bool is_latin_letter(u32 cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0x00C0 && cp <= 0x024F) {
        if (cp == 0x00D7 || cp == 0x00F7) return false;
        return true;
    }
    return false;
}

bool is_ascii_digit(u32 cp) { return cp >= '0' && cp <= '9'; }

bool is_whitespace(u32 cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_control(u32 cp) {
    return (cp < 0x20 && cp != '\n' && cp != '\t' && cp != '\r') || (cp >= 0x7F && cp <= 0x9F);
}

bool is_zero_width(u32 cp) {
    return cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0x200E || cp == 0x200F ||
           (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069) || cp == 0xFEFF;
}

bool is_punct(u32 cp) {
    if (cp < 0x80) {
        return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') ||
               (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~');
    }
    if (is_arabic_punct(cp)) return true;
    if (cp >= 0x2010 && cp <= 0x205E) return true;
    if (cp >= 0x00A1 && cp <= 0x00BF) return true;
    return false;
}

bool is_emoji(u32 cp) {
    return (cp >= 0x1F300 && cp <= 0x1FAFF) ||
           (cp >= 0x1F000 && cp <= 0x1F2FF) ||
           (cp >= 0x2600  && cp <= 0x27BF)  ||
           (cp >= 0x2190  && cp <= 0x21FF)  ||
           (cp >= 0xFE00  && cp <= 0xFE0F)  ||
           cp == 0x2764 || cp == 0x203C || cp == 0x2049;
}

bool is_arabizi_digit(u32 cp) {
    return cp == '2' || cp == '3' || cp == '5' || cp == '6' || cp == '7' || cp == '8' || cp == '9';
}

bool is_unicode_letter(u32 cp) {
    if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (is_latin_letter(cp)) return true;

    // Major letter blocks. Gaps inside a block that are NOT letters are
    // listed in `not_letter` below.
    static const u32 ranges[][2] = {
        {0x0180, 0x024F}, {0x02B0, 0x02C1}, {0x0386, 0x0386}, {0x0388, 0x038A},
        {0x038C, 0x038C}, {0x038E, 0x03A1}, {0x03A3, 0x03F5}, {0x03F7, 0x0481},
        {0x048A, 0x052F}, {0x0531, 0x0556}, {0x0561, 0x0587}, {0x05D0, 0x05EA},
        {0x05EF, 0x05F2}, {0x0620, 0x064A}, {0x066E, 0x066F}, {0x0671, 0x06D3},
        {0x06D5, 0x06D5}, {0x06E5, 0x06E6}, {0x06EE, 0x06EF}, {0x06FA, 0x06FC},
        {0x0710, 0x072F}, {0x074D, 0x07A5}, {0x07C9, 0x07EA}, {0x0904, 0x0939},
        {0x093D, 0x093D}, {0x0950, 0x0950}, {0x0958, 0x0961}, {0x0972, 0x097F},
        {0x0985, 0x09B9}, {0x0A05, 0x0A39}, {0x0A85, 0x0AB9}, {0x0B05, 0x0B39},
        {0x0B85, 0x0BB9}, {0x0C05, 0x0C39}, {0x0C85, 0x0CB9}, {0x0D05, 0x0D3A},
        {0x0D85, 0x0DC6}, {0x0E01, 0x0E30}, {0x0E32, 0x0E33}, {0x0E40, 0x0E46},
        {0x0E81, 0x0EB0}, {0x0ED0, 0x0ED9}, {0x0F40, 0x0F6C}, {0x1000, 0x102A},
        {0x10A0, 0x10FA}, {0x1100, 0x1248}, {0x13A0, 0x13F5}, {0x1401, 0x166C},
        {0x1681, 0x169A}, {0x16A0, 0x16EA}, {0x1700, 0x170C}, {0x1780, 0x17B3},
        {0x1820, 0x1877}, {0x1880, 0x18A8}, {0x1E00, 0x1FBC}, {0x1FBE, 0x1FBE},
        {0x1FC2, 0x1FCC}, {0x1FD0, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FFC},
        {0x2071, 0x2071}, {0x207F, 0x207F}, {0x2090, 0x209C}, {0x2102, 0x2102},
        {0x2107, 0x2107}, {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D},
        {0x2124, 0x2139}, {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E},
        {0x2160, 0x2188}, {0x2C00, 0x2CE4}, {0x2D00, 0x2D2D}, {0x2D30, 0x2D67},
        {0x3005, 0x3006}, {0x3041, 0x3096}, {0x309D, 0x309F}, {0x30A1, 0x30FA},
        {0x30FC, 0x30FF}, {0x3105, 0x312F}, {0x3131, 0x318E}, {0x31A0, 0x31BF},
        {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA48C}, {0xA4D0, 0xA4FD},
        {0xA500, 0xA60C}, {0xA640, 0xA66E}, {0xA717, 0xA71F}, {0xA722, 0xA788},
        {0xA78B, 0xA7BF}, {0xAC00, 0xD7A3}, {0xF900, 0xFA6D}, {0xFB00, 0xFB06},
        {0xFB13, 0xFB17}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A}, {0xFF66, 0xFFDC},
        {0x10000, 0x1000B}, {0x1D400, 0x1D7CB}, {0x20000, 0x2A6DF},
    };
    for (const auto& r : ranges) {
        if (cp >= r[0] && cp <= r[1]) return true;
    }

    // Explicit non-letters inside the blocks above.
    static const u32 not_letter[] = {
        0x00B7, 0x0387, 0x0559, 0x058A, 0x05BE, 0x05C0, 0x05C3, 0x05C6, 0x05F3,
        0x05F4, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x0606, 0x0607,
        0x0608, 0x0609, 0x060A, 0x060B, 0x060C, 0x060D, 0x061C, 0x06DD, 0x070F,
        0x1680, 0x180B, 0x180C, 0x180D, 0x180E, 0x180F, 0x200B, 0x200C, 0x200D,
        0x2028, 0x2029, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2060, 0x2061,
        0x2062, 0x2063, 0x2064, 0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F,
        0x20A0, 0x20BF, 0x3000, 0x3001, 0x3002, 0x3003, 0x3004, 0x3007, 0x3008,
        0x3009, 0x300A, 0x300B, 0x300C, 0x300D, 0x300E, 0x300F, 0x3010, 0x3011,
        0x3012, 0x3013, 0x3014, 0x3015, 0x3016, 0x3017, 0x3018, 0x3019, 0x301A,
        0x301B, 0x301C, 0x301D, 0x301E, 0x301F, 0xFEFF,
    };
    for (u32 e : not_letter) if (cp == e) return false;
    return false;
}

bool is_unicode_number(u32 cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    if (is_arabic_digit(cp)) return true;

    static const u32 ranges[][2] = {
        {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE}, {0x0660, 0x0669},
        {0x06F0, 0x06F9}, {0x07C0, 0x07C9}, {0x0966, 0x096F}, {0x09E6, 0x09EF},
        {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F}, {0x0BE6, 0x0BEF},
        {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF}, {0x0D66, 0x0D6F}, {0x0DE6, 0x0DEF},
        {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29}, {0x1040, 0x1049},
        {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x2070, 0x2070}, {0x2074, 0x2079},
        {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189}, {0x2460, 0x249B},
        {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x2CFd, 0x2CFD}, {0x3007, 0x3007},
        {0x3021, 0x3029}, {0x3192, 0x3195}, {0x3220, 0x3229}, {0x3248, 0x324F},
        {0x3251, 0x325F}, {0x3280, 0x3289}, {0x32B1, 0x32BF}, {0xA620, 0xA629},
        {0xA6E6, 0xA6EF}, {0xA830, 0xA835}, {0xA8D0, 0xA8D9}, {0xA900, 0xA909},
        {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9}, {0xAA50, 0xAA59}, {0xABF0, 0xABF9},
        {0xFF10, 0xFF19}, {0x11052, 0x11059}, {0x11066, 0x1106F}, {0x1D7CE, 0x1D7FF},
        {0x16E50, 0x16E59}, {0x1D7F6, 0x1D7F9},
    };
    for (const auto& r : ranges) {
        if (cp >= r[0] && cp <= r[1]) return true;
    }
    return false;
}

u32 arabic_presentation_to_base(u32 cp) {

    static const struct { u32 lo, hi, base; } fb[] = {
        {0xFE80, 0xFE80, 0x0621}, {0xFE81, 0xFE82, 0x0622}, {0xFE83, 0xFE84, 0x0623},
        {0xFE85, 0xFE86, 0x0624}, {0xFE87, 0xFE88, 0x0625}, {0xFE89, 0xFE8C, 0x0626},
        {0xFE8D, 0xFE8E, 0x0627}, {0xFE8F, 0xFE92, 0x0628}, {0xFE93, 0xFE94, 0x0629},
        {0xFE95, 0xFE98, 0x062A}, {0xFE99, 0xFE9C, 0x062B}, {0xFE9D, 0xFEA0, 0x062C},
        {0xFEA1, 0xFEA4, 0x062D}, {0xFEA5, 0xFEA8, 0x062E}, {0xFEA9, 0xFEAA, 0x062F},
        {0xFEAB, 0xFEAC, 0x0630}, {0xFEAD, 0xFEAE, 0x0631}, {0xFEAF, 0xFEB0, 0x0632},
        {0xFEB1, 0xFEB4, 0x0633}, {0xFEB5, 0xFEB8, 0x0634}, {0xFEB9, 0xFEBC, 0x0635},
        {0xFEBD, 0xFEC0, 0x0636}, {0xFEC1, 0xFEC4, 0x0637}, {0xFEC5, 0xFEC8, 0x0638},
        {0xFEC9, 0xFECC, 0x0639}, {0xFECD, 0xFED0, 0x063A}, {0xFED1, 0xFED4, 0x0641},
        {0xFED5, 0xFED8, 0x0642}, {0xFED9, 0xFEDC, 0x0643}, {0xFEDD, 0xFEE0, 0x0644},
        {0xFEE1, 0xFEE4, 0x0645}, {0xFEE5, 0xFEE8, 0x0646}, {0xFEE9, 0xFEEC, 0x0647},
        {0xFEED, 0xFEEE, 0x0648}, {0xFEEF, 0xFEF0, 0x0649}, {0xFEF1, 0xFEF4, 0x064A},
    };
    if (cp >= 0xFE80 && cp <= 0xFEF4) {
        for (auto& e : fb) if (cp >= e.lo && cp <= e.hi) return e.base;
    }
    if (cp >= 0xFEF5 && cp <= 0xFEFC) return 0x0644;
    return 0;
}

u32 arabic_digit_to_ascii(u32 cp) {
    if (cp >= 0x0660 && cp <= 0x0669) return '0' + (cp - 0x0660);
    if (cp >= 0x06F0 && cp <= 0x06F9) return '0' + (cp - 0x06F0);
    return cp;
}

u32 fold_arabic_letter(u32 cp) {
    switch (cp) {
        case 0x0622: case 0x0623: case 0x0625: case 0x0671: return 0x0627;
        case 0x0649: return 0x064A;
        case 0x0629: return 0x0647;
        case 0x0624: case 0x0626: return 0x0621;
        default: return cp;
    }
}

u32 lower_ascii(u32 cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) return cp + 32;
    return cp;
}

u32 fold_fullwidth(u32 cp) {
    if (cp >= 0xFF01 && cp <= 0xFF5E) return cp - 0xFEE0;
    if (cp == 0x3000) return ' ';
    return cp;
}

std::string to_lower_ascii(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        u32 cp = utf8_decode(s, i);
        utf8_encode(lower_ascii(cp), out);
    }
    return out;
}

ScriptStats script_stats(const std::string& s) {
    ScriptStats st;
    size_t i = 0;
    while (i < s.size()) {
        u32 cp = utf8_decode(s, i);
        ++st.total;
        if (is_whitespace(cp))                                   ++st.whitespace;
        else if (is_arabic_letter(cp) || is_arabic_diacritic(cp)) ++st.arabic;
        else if (is_latin_letter(cp))                             ++st.latin;
        else if (is_ascii_digit(cp) || is_arabic_digit(cp))       ++st.digits;
        else if (is_emoji(cp))                                    ++st.emoji;
        else if (is_punct(cp))                                    ++st.punct;
        else                                                      ++st.other;
    }
    return st;
}

}
