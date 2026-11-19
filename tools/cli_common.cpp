#include "tools/cli_common.h"

#include <cmath>
#include <iostream>
#include <limits>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

namespace gai {

Args::Args(int argc, char** argv) {
    if (argc > 0) prog_ = argv[0];
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto looks_like_key = [](const std::string& s) {
            if (s.empty()) return false;
            // A value like "1e-3", ".5", "-3" or a plain number is NOT a key.
            // Only strings starting with "--" or a single "-" followed by a
            // non-numeric character are options; otherwise `--temp 1e-3` was
            // mis-parsed as the flag `--temp` with value "1" plus a stray key.
            const char c0 = s[0];
            if (c0 == '-') {
                if (s.size() < 2) return true;
                const char c1 = s[1];
                if (c1 >= '0' && c1 <= '9') return false;
                if (c1 == '.' && s.size() > 2 && s[2] >= '0' && s[2] <= '9') return false;
                return true;
            }
            if (c0 == '.' || c0 == '+' || (c0 >= '0' && c0 <= '9')) return false;
            return false;
        };
        if (a.rfind("--", 0) == 0) {
            std::string key = a.substr(2);
            size_t eq = key.find('=');
            if (eq != std::string::npos) {
                kv_[key.substr(0, eq)] = key.substr(eq + 1);
                continue;
            }
            if (i + 1 < argc && !looks_like_key(argv[i + 1])) {
                kv_[key] = argv[++i];
            } else {
                kv_[key] = "1";
            }
        } else if (a.rfind("-", 0) == 0 && a.size() > 1) {
            std::string key = a.substr(1);
            if (i + 1 < argc && !looks_like_key(argv[i + 1])) kv_[key] = argv[++i];
            else kv_[key] = "1";
        } else {
            pos_.push_back(a);
        }
    }
}

bool Args::has(const std::string& k) const { return kv_.find(k) != kv_.end(); }

std::string Args::str(const std::string& k, const std::string& def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
}

static bool parse_cli_i64(const std::string& text, i64& out) {

    try {
        size_t pos = 0;
        const long long v = std::stoll(text, &pos);
        if (pos != text.size()) return false;
        out = static_cast<i64>(v);
        return true;
    } catch (...) {
        return false;
    }
}

i64 Args::num(const std::string& k, i64 def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    i64 v = def;
    if (parse_cli_i64(it->second, v)) return v;
    log_warn("args: ignoring malformed int for '--" + k + "': '" + it->second + "'");
    return def;
}

int Args::num_int(const std::string& k, int def) const {
    const i64 v = num(k, static_cast<i64>(def));
    if (v < static_cast<i64>(std::numeric_limits<int>::min()) ||
        v > static_cast<i64>(std::numeric_limits<int>::max())) {
        log_warn("args: ignoring out-of-range int for '--" + k + "'");
        return def;
    }
    return static_cast<int>(v);
}

u64 Args::num_u64(const std::string& k, u64 def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    // Parse as UNSIGNED: stoll() rejects values above INT64_MAX (large seeds,
    // fingerprints) and silently falls back to the default.
    try {
        size_t pos = 0;
        if (it->second.empty() || it->second[0] == '-') {
            log_warn("args: negative unsigned integer for '--" + k + "', using default");
            return def;
        }
        const unsigned long long v = std::stoull(it->second, &pos, 0);  // accept 0x… seeds
        if (pos != it->second.size()) {
            log_warn("args: trailing characters in '--" + k + "', using parsed value");
        }
        return static_cast<u64>(v);
    } catch (const std::exception&) {
        log_warn("args: ignoring out-of-range integer for '--" + k + "'");
        return def;
    }
}

u64 Args::num_u64_strict(const std::string& k) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) GAI_FAIL("args: missing '--" + k + "'");
    if (it->second.empty() || it->second[0] == '-')
        GAI_FAIL("args: '--" + k + "' must be an unsigned integer");
    try {
        size_t pos = 0;
        const unsigned long long v = std::stoull(it->second, &pos, 0);  // accept 0x… seeds
        if (pos != it->second.size()) GAI_FAIL("args: bad unsigned integer for '--" + k + "'");
        return static_cast<u64>(v);
    } catch (const std::exception&) {
        GAI_FAIL("args: bad unsigned integer for '--" + k + "'");
    }
}

double Args::real(const std::string& k, double def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    try {
        size_t pos = 0;
        const double v = std::stod(it->second, &pos);
        if (pos != it->second.size() || !std::isfinite(v)) {
            log_warn("args: ignoring malformed number for '--" + k + "': '" + it->second + "'");
            return def;
        }
        return v;
    } catch (...) {
        log_warn("args: ignoring malformed number for '--" + k + "': '" + it->second + "'");
        return def;
    }
}

bool Args::flag(const std::string& k, bool def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    const std::string& v = it->second;
    return !(v == "0" || v == "false" || v == "no" || v == "off");
}

i64 Args::num_strict(const std::string& k) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) GAI_FAIL("args: missing required --" + k + " (strict-args)");
    i64 v = 0;
    if (!parse_cli_i64(it->second, v))
        GAI_FAIL("args: malformed int for '--" + k + "': '" + it->second + "' (strict-args)");
    return v;
}

int Args::num_int_strict(const std::string& k) const {
    const i64 v = num_strict(k);
    if (v < static_cast<i64>(std::numeric_limits<int>::min()) ||
        v > static_cast<i64>(std::numeric_limits<int>::max()))
        GAI_FAIL("args: out-of-range int for '--" + k + "' (strict-args)");
    return static_cast<int>(v);
}

double Args::real_strict(const std::string& k) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) GAI_FAIL("args: missing required --" + k + " (strict-args)");
    try {
        size_t pos = 0;
        double v = std::stod(it->second, &pos);
        if (pos != it->second.size() || !std::isfinite(v))
            GAI_FAIL("args: malformed number for '--" + k + "'");
        return v;
    } catch (const Error&) { throw; }
    catch (...) { GAI_FAIL("args: malformed number for '--" + k + "': '" + it->second + "' (strict-args)"); }
    return 0.0;
}

void enable_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    static char buf[1 << 16];
    setvbuf(stdout, buf, _IOFBF, sizeof(buf));
#endif
    std::ios::sync_with_stdio(true);
}

void apply_common_flags(const Args& a) {
    if (a.flag("quiet", false))   set_log_level(LogLevel::Warn);
    if (a.flag("silent", false))  set_log_level(LogLevel::Silent);
    if (a.flag("verbose", false)) set_log_level(LogLevel::Debug);
    i64 t = a.num("threads", 0);
    if (t > 0) {
        set_num_threads(static_cast<int>(t));
    } else if (a.has("threads")) {
        log_warn("args: ignoring non-positive --threads");
    }
}

std::string json_escape(const std::string& s) {
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

}
