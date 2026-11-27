#include "core/json.h"

#include <cmath>
#include <limits>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace gai {

namespace {
const Json kNull;
}

Json Json::null()          { return Json(); }
Json Json::boolean(bool v) { Json j; j.type_ = Type::Bool; j.bool_ = v; return j; }
Json Json::number(double v){ Json j; j.type_ = Type::Number; j.num_ = v; return j; }
Json Json::integer(i64 v)  { Json j; j.type_ = Type::Number; j.num_ = static_cast<double>(v);
                             j.is_int_ = true; j.int_ = v; return j; }
Json Json::string(std::string v) { Json j; j.type_ = Type::String; j.str_ = std::move(v); return j; }
Json Json::array()         { Json j; j.type_ = Type::Array; return j; }
Json Json::object()        { Json j; j.type_ = Type::Object; return j; }

bool Json::as_bool(bool def) const {
    if (type_ == Type::Bool) return bool_;
    if (type_ == Type::Number) return num_ != 0.0;
    return def;
}

double Json::as_number(double def) const {
    return type_ == Type::Number ? num_ : def;
}

i64 Json::as_int(i64 def) const {
    if (type_ != Type::Number) return def;
    if (is_int_) return int_;
    if (!std::isfinite(num_) || num_ < static_cast<double>(std::numeric_limits<i64>::min()) ||
        num_ > static_cast<double>(std::numeric_limits<i64>::max()))
        return def;
    return static_cast<i64>(num_);
}

std::string Json::as_string(const std::string& def) const {
    return type_ == Type::String ? str_ : def;
}

bool Json::has(const std::string& key) const {
    return type_ == Type::Object && object_.find(key) != object_.end();
}

const Json& Json::at(const std::string& key) const {
    if (type_ != Type::Object) return kNull;
    auto it = object_.find(key);
    return it == object_.end() ? kNull : it->second;
}

void Json::set(const std::string& key, const Json& v) {
    type_ = Type::Object;
    object_[key] = v;
}

void Json::push(const Json& v) {
    type_ = Type::Array;
    array_.push_back(v);
}

const Json& Json::operator[](size_t i) const {
    if (i >= array_.size()) return kNull;
    return array_[i];
}

std::string Json::escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

static void dump_string(const std::string& s, std::string& out) {
    out += '"';
    out += Json::escape(s);
    out += '"';
}

static void dump_value(const Json& j, std::string& out, int indent, int depth) {
    const bool pretty = indent > 0;
    const std::string nl = pretty ? "\n" : "";
    const std::string pad = pretty ? std::string(static_cast<size_t>(indent) * (depth + 1), ' ') : "";
    const std::string pad0 = pretty ? std::string(static_cast<size_t>(indent) * depth, ' ') : "";

    switch (j.type()) {
        case Json::Type::Null:   out += "null"; break;
        case Json::Type::Bool:   out += j.as_bool() ? "true" : "false"; break;
        case Json::Type::Number: {
            if (j.is_integer()) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(j.as_int(0)));
                out += buf;
                break;
            }
            double v = j.as_number();
            if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
                out += buf;
            } else if (!std::isfinite(v)) {
                out += "null";
            } else {
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%.17g", v);
                out += buf;
            }
            break;
        }
        case Json::Type::String: dump_string(j.as_string(), out); break;
        case Json::Type::Array: {
            if (j.size() == 0) { out += "[]"; break; }
            out += '[';
            out += nl;
            for (size_t i = 0; i < j.size(); ++i) {
                out += pad;
                dump_value(j[i], out, indent, depth + 1);
                if (i + 1 < j.size()) out += ',';
                out += nl;
            }
            out += pad0;
            out += ']';
            break;
        }
        case Json::Type::Object: {
            if (j.items().empty()) { out += "{}"; break; }
            out += '{';
            out += nl;
            size_t i = 0;
            for (const auto& [k, v] : j.items()) {
                out += pad;
                dump_string(k, out);
                out += pretty ? ": " : ":";
                dump_value(v, out, indent, depth + 1);
                if (++i < j.items().size()) out += ',';
                out += nl;
            }
            out += pad0;
            out += '}';
            break;
        }
    }
}

std::string Json::dump() const {
    std::string out;
    dump_value(*this, out, 0, 0);
    return out;
}

std::string Json::dump_pretty() const {
    std::string out;
    dump_value(*this, out, 2, 0);
    return out;
}

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;
    std::string err;

    explicit Parser(const std::string& text) : s(text) {}

    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }

    bool fail(const char* what) {
        if (err.empty()) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "json: %s at offset %zu", what, i);
            err = buf;
        }
        return false;
    }

    int depth = 0;
    static constexpr int kMaxDepth = 512;

    bool parse_value(Json& out) {
        skip_ws();
        if (i >= s.size()) return fail("unexpected end of input");
        if (++depth > kMaxDepth) return fail("JSON nesting depth limit exceeded");

        const char c = s[i];
        bool ok = false;
        switch (c) {
            case '{': ok = parse_object(out); break;
            case '[': ok = parse_array(out); break;
            case '"': {
                std::string v;
                if (!parse_string(v)) { --depth; return false; }
                out = Json::string(std::move(v));
                ok = true;
                break;
            }
            case 't':
                if (s.compare(i, 4, "true") != 0) { --depth; return fail("bad literal"); }
                i += 4;
                out = Json::boolean(true);
                ok = true;
                break;
            case 'f':
                if (s.compare(i, 5, "false") != 0) { --depth; return fail("bad literal"); }
                i += 5;
                out = Json::boolean(false);
                ok = true;
                break;
            case 'n':
                if (s.compare(i, 4, "null") != 0) { --depth; return fail("bad literal"); }
                i += 4;
                out = Json::null();
                ok = true;
                break;
            default:
                ok = parse_number(out);
                break;
        }
        --depth;
        return ok;
    }

    bool parse_number(Json& out) {
        const size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        if (i >= s.size() || s[i] < '0' || s[i] > '9') return fail("bad number");
        if (s[i] == '0') {
            ++i;
            if (i < s.size() && s[i] >= '0' && s[i] <= '9')
                return fail("leading zero in number");
        } else {
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        bool is_int = true;
        if (i < s.size() && s[i] == '.') {
            is_int = false;
            ++i;
            if (i >= s.size() || s[i] < '0' || s[i] > '9')
                return fail("bad number (want digits after '.')");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            is_int = false;
            ++i;
            if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
            if (i >= s.size() || s[i] < '0' || s[i] > '9')
                return fail("bad exponent (want digits after e/E)");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        const std::string num = s.substr(start, i - start);
        errno = 0;
        char* end = nullptr;
        if (is_int) {
            const long long v = std::strtoll(num.c_str(), &end, 10);
            if (errno == ERANGE || end != num.c_str() + num.size())
                return fail("integer out of range");
            out = Json::integer(v);
        } else {
            const double v = std::strtod(num.c_str(), &end);
            if (end != num.c_str() + num.size() || !std::isfinite(v))
                return fail("number out of range");
            out = Json::number(v);
        }
        return true;
    }

    bool parse_string(std::string& out) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        ++i;
        out.clear();
        while (i < s.size()) {
            const unsigned char c = static_cast<unsigned char>(s[i++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("unescaped control character in string");

            if (c != '\\') {
                if (c < 0x80) {
                    out.push_back(static_cast<char>(c));
                } else if (c >= 0xC2 && c <= 0xDF) {
                    if (i >= s.size()) return fail("truncated UTF-8 sequence");
                    unsigned char c2 = static_cast<unsigned char>(s[i++]);
                    if ((c2 & 0xC0) != 0x80) return fail("invalid UTF-8 continuation byte");
                    out.push_back(static_cast<char>(c));
                    out.push_back(static_cast<char>(c2));
                } else if (c >= 0xE0 && c <= 0xEF) {
                    if (i + 1 >= s.size()) return fail("truncated UTF-8 sequence");
                    unsigned char c2 = static_cast<unsigned char>(s[i++]);
                    unsigned char c3 = static_cast<unsigned char>(s[i++]);
                    if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return fail("invalid UTF-8 continuation byte");
                    if (c == 0xE0 && c2 < 0xA0) return fail("overlong UTF-8 sequence");
                    if (c == 0xED && c2 >= 0xA0) return fail("invalid UTF-8 surrogate codepoint");
                    out.push_back(static_cast<char>(c));
                    out.push_back(static_cast<char>(c2));
                    out.push_back(static_cast<char>(c3));
                } else if (c >= 0xF0 && c <= 0xF4) {
                    if (i + 2 >= s.size()) return fail("truncated UTF-8 sequence");
                    unsigned char c2 = static_cast<unsigned char>(s[i++]);
                    unsigned char c3 = static_cast<unsigned char>(s[i++]);
                    unsigned char c4 = static_cast<unsigned char>(s[i++]);
                    if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80 || (c4 & 0xC0) != 0x80)
                        return fail("invalid UTF-8 continuation byte");
                    if (c == 0xF0 && c2 < 0x90) return fail("overlong UTF-8 sequence");
                    if (c == 0xF4 && c2 > 0x8F) return fail("UTF-8 codepoint above 0x10FFFF");
                    out.push_back(static_cast<char>(c));
                    out.push_back(static_cast<char>(c2));
                    out.push_back(static_cast<char>(c3));
                    out.push_back(static_cast<char>(c4));
                } else {
                    return fail("invalid UTF-8 start byte in string");
                }
                continue;
            }

            if (i >= s.size()) return fail("truncated escape");
            const char e = s[i++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    if (i + 4 > s.size()) return fail("truncated \\u escape");
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        const char h = s[i + static_cast<size_t>(k)];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                        else return fail("bad \\u escape");
                    }
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            unsigned lo = 0;
                            bool ok = true;
                            for (int k = 0; k < 4; ++k) {
                                const char h = s[i + 2 + static_cast<size_t>(k)];
                                lo <<= 4;
                                if (h >= '0' && h <= '9') lo |= static_cast<unsigned>(h - '0');
                                else if (h >= 'a' && h <= 'f') lo |= static_cast<unsigned>(h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') lo |= static_cast<unsigned>(h - 'A' + 10);
                                else { ok = false; break; }
                            }
                            if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                i += 6;
                            } else {
                                return fail("unpaired high surrogate without valid low surrogate");
                            }
                        } else {
                            return fail("unpaired high surrogate at end of string or before non-surrogate");
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("lone low surrogate without preceding high surrogate");
                    } else if (cp > 0x10FFFF) {
                        return fail("unicode codepoint out of range");
                    }

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
                    break;
                }
                default:
                    return fail("unknown escape");
            }
        }
        return fail("unterminated string");
    }

    bool parse_array(Json& out) {
        out = Json::array();
        ++i;
        skip_ws();
        if (i < s.size() && s[i] == ']') { ++i; return true; }
        while (true) {
            Json v;
            if (!parse_value(v)) return false;
            out.push(v);
            skip_ws();
            if (i >= s.size()) return fail("unterminated array");
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == ']') { ++i; return true; }
            return fail("expected ',' or ']'");
        }
    }

    bool parse_object(Json& out) {
        out = Json::object();
        ++i;
        skip_ws();
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (i >= s.size() || s[i] != ':') return fail("expected ':'");
            ++i;
            Json v;
            if (!parse_value(v)) return false;
            out.set(key, v);
            skip_ws();
            if (i >= s.size()) return fail("unterminated object");
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == '}') { ++i; return true; }
            return fail("expected ',' or '}'");
        }
    }
};

}

bool Json::parse(const std::string& text, Json& out, std::string* error) {
    size_t start = 0;
    if (text.size() >= 3 &&
        static_cast<u8>(text[0]) == 0xEF &&
        static_cast<u8>(text[1]) == 0xBB &&
        static_cast<u8>(text[2]) == 0xBF) {
        start = 3;
    }
    const std::string body = (start == 0) ? text : text.substr(start);
    Parser p(body);
    if (!p.parse_value(out)) {
        if (error) *error = p.err;
        return false;
    }
    p.skip_ws();
    if (p.i != body.size()) {
        if (error) *error = "json: trailing characters";
        return false;
    }
    return true;
}

}
