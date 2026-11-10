// Tokenizer v2 conformance / unit tests.
//
// Covers the pieces that make the Gai engine able to serve a HuggingFace
// (foreign) vocabulary:
//   * the GPT-2 byte-level alphabet
//   * the GPT-2 pre-tokenizer (reference regex, hand written)
//   * the byte -> token id map and the byte-level decoder
//   * the chat style registry and renderers
//
// The end-to-end check against the real HuggingFace tokenizer lives in
// tools/py/conformance.py (it needs the reference implementation).

#include "core/common.h"
#include "core/unicode.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/normalizer.h"
#include "tokenizer/chat_template.h"
#include "tools/cli_common.h"

#include <iostream>
#include <string>
#include <vector>

using namespace gai;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::cout << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << msg  \
                      << "  [" #cond "]\n";                                     \
        }                                                                       \
    } while (0)

#define CHECK_EQ(a, b, msg)                                                     \
    do {                                                                        \
        ++g_checks;                                                             \
        const auto _a = (a);                                                    \
        const auto _b = (b);                                                    \
        if (!(_a == _b)) {                                                      \
            ++g_failures;                                                       \
            std::cout << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << msg  \
                      << "\n    want: " << _b << "\n    got : " << _a << "\n";   \
        }                                                                       \
    } while (0)

static std::vector<std::string> chunk_texts(const std::vector<Chunk>& chunks) {
    std::vector<std::string> out;
    out.reserve(chunks.size());
    for (const Chunk& c : chunks) out.push_back(c.text);
    return out;
}

static std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

// ---------------------------------------------------------------------------

static void test_gpt2_byte_table() {
    // Reference values from GPT-2's encoder.py bytes_to_unicode().
    CHECK_EQ(gpt2_byte_token('a'), std::string("a"), "ascii byte maps to itself");
    CHECK_EQ(gpt2_byte_token(' '), std::string("\xC4\xA0"), "space -> U+0120");
    CHECK_EQ(gpt2_byte_token('\n'), std::string("\xC4\x8A"), "newline -> U+010A");
    CHECK_EQ(gpt2_byte_token('\t'), std::string("\xC4\x89"), "tab -> U+0109");
    CHECK_EQ(gpt2_byte_token(0x00), std::string("\xC4\x80"), "nul -> U+0100");
    CHECK_EQ(gpt2_byte_token(0x20), std::string("\xC4\xA0"), "0x20 is the space");
    CHECK_EQ(gpt2_byte_token(0xA1), std::string("\xC2\xA1"), "0xA1 -> U+00A1 utf-8");
    CHECK_EQ(gpt2_byte_token(0xFF), std::string("\xC3\xBF"), "0xFF -> U+00FF utf-8");

    // The map is injective.
    const auto& dec = gpt2_byte_decoder();
    CHECK_EQ(dec.size(), static_cast<size_t>(256), "decoder has one entry per byte");
    for (int b = 0; b < 256; ++b) {
        auto it = dec.find(gpt2_byte_token(static_cast<u8>(b)));
        CHECK(it != dec.end() && it->second == static_cast<u8>(b),
              "byte decoder round trip");
    }
}

static void test_gpt2_pretokenizer() {
    struct Case { const char* in; const char* want; };
    // Expected splits of the reference regex
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    const Case cases[] = {
        {"hello world",      "hello| world"},
        {"hello  world",     "hello| | world"},
        {"  hello",          " | hello"},
        {"don't",            "don|'t"},
        {"won't've",         "won|'t|'ve"},
        {"IT'S",             "IT|'|S"},
        {"hello\n\nworld",   "hello|\n|\n|world"},
        {"hello\nworld",     "hello|\n|world"},
        {"123",              "123"},
        {"3.14",             "3|.|14"},
        {"1,000",            "1|,|000"},
        {"!!!",              "!!!"},
        {"a!b",              "a|!|b"},
        {"end.",             "end|."},
        {"  ",               "  "},
        {"\n\n",             "\n\n"},
    };
    for (const Case& c : cases) {
        const std::vector<std::string> got = chunk_texts(pre_tokenize_gpt2(c.in));
        CHECK_EQ(join(got, "|"), std::string(c.want), "gpt2 pretokenizer");
    }

    // Digits(individual_digits=true): every digit is its own pre-token and
    // the space never sticks to a digit.
    const std::vector<std::string> ds = chunk_texts(pre_tokenize_gpt2(" 123", true));
    CHECK_EQ(join(ds, "|"), std::string(" |1|2|3"), "digits split individually");
}

static void test_byte_map_and_decode() {
    // Build a tiny byte-level vocabulary: the 256 GPT-2 byte tokens plus a
    // couple of merged tokens.
    std::vector<std::string> vocab;
    std::unordered_map<std::string, i32> ids;
    for (int b = 0; b < 256; ++b) {
        ids[gpt2_byte_token(static_cast<u8>(b))] = static_cast<i32>(vocab.size());
        vocab.push_back(gpt2_byte_token(static_cast<u8>(b)));
    }
    // "he", "ll" and "llo" -- every merged piece must exist in the
    // vocabulary, otherwise the merge is dropped at load time.
    ids["he"] = static_cast<i32>(vocab.size());
    vocab.push_back("he");
    ids["ll"] = static_cast<i32>(vocab.size());
    vocab.push_back("ll");
    ids["llo"] = static_cast<i32>(vocab.size());
    vocab.push_back("llo");

    std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"}, {"l", "l"}, {"ll", "o"},
    };

    Tokenizer tk;
    Tokenizer::VocabOptions opt;
    opt.foreign = true;
    opt.pretok  = Tokenizer::PreTokMode::Gpt2;
    tk.build(vocab, merges, NormalizerConfig{}, opt);

    CHECK(tk.has_byte_map(), "byte map detected for a byte-level vocabulary");
    CHECK_EQ(tk.byte_id('a'), ids["a"], "byte id for 'a'");
    CHECK_EQ(tk.byte_id(' '), ids[gpt2_byte_token(' ')], "byte id for space");

    // "hello" -> he + llo
    const std::vector<i32> enc = tk.encode("hello");
    CHECK_EQ(enc.size(), static_cast<size_t>(2), "hello encodes to 2 tokens");
    if (enc.size() == 2) {
        CHECK_EQ(tk.token_text(enc[0]), std::string("he"), "first merge");
        CHECK_EQ(tk.token_text(enc[1]), std::string("llo"), "second merge");
    }

    // A space is its own token and decodes back to a space.
    const std::vector<i32> enc2 = tk.encode("a b");
    CHECK_EQ(tk.decode(enc2, false), std::string("a b"), "round trip decode");

    // The byte-level decoder must turn U+0120 back into 0x20.
    const std::vector<i32> enc3 = tk.encode(" ");
    CHECK_EQ(enc3.size(), static_cast<size_t>(1), "space is one token");
    CHECK_EQ(tk.decode(enc3, false), std::string(" "), "space decodes to a space");
}

static void test_native_tokenizer_still_works() {
    // The native .gtok layout keeps working: specials, then the raw bytes.
    std::vector<std::string> vocab;
    for (const auto& s : special_token_strings()) vocab.push_back(s);
    for (int b = 0; b < 256; ++b) vocab.push_back(std::string(1, static_cast<char>(b)));

    Tokenizer tk;
    tk.build(vocab, {}, NormalizerConfig{});
    CHECK(!tk.foreign(), "native vocabulary is not foreign");
    CHECK(tk.has_byte_map(), "native byte map detected");
    CHECK_EQ(tk.byte_id('a'), static_cast<i32>(special::COUNT + 'a'), "native byte id");

    // encode() does not add a BOS on its own; the chat renderer does.
    CHECK(tk.add_bos(), "native vocabulary adds a BOS by default");
    const std::vector<i32> enc = tk.encode("abc");
    CHECK_EQ(enc.size(), static_cast<size_t>(3), "three bytes");
    if (!enc.empty()) CHECK_EQ(enc[0], tk.byte_id('a'), "first byte");
}

static void test_chat_styles() {
    // ChatML vocabulary.
    std::vector<std::string> vocab;
    for (const auto& s : {std::string("<|im_start|>"), std::string("<|im_end|>")}) {
        vocab.push_back(s);
    }
    for (int b = 0; b < 256; ++b) vocab.push_back(gpt2_byte_token(static_cast<u8>(b)));

    Tokenizer tk;
    Tokenizer::VocabOptions opt;
    opt.foreign = true;
    opt.pretok  = Tokenizer::PreTokMode::Gpt2;
    tk.build(vocab, {}, NormalizerConfig{}, opt);

    CHECK(detect_chat_style(tk) == ChatStyle::ChatML, "ChatML detected");

    std::vector<Message> msgs = {
        {Role::System, "sys"},
        {Role::User, "hello"},
    };
    const std::vector<ChatSegment> segs = render_chat(tk, ChatStyle::ChatML, msgs, true);

    // Expect: <|im_start|> "system\n" "sys" <|im_end|> "\n"
    //         <|im_start|> "user\n" "hello" <|im_end|> "\n"
    //         <|im_start|> "assistant\n"
    std::string shape;
    for (const ChatSegment& s : segs) {
        if (s.is_token) shape += "T";
        else shape += "S";
    }
    CHECK_EQ(shape, std::string("TSSTSTSSTSTS"), "ChatML segment layout");

    // The rendered prompt must be exactly what ChatML expects.
    std::vector<i32> ids;
    std::vector<u8> mask;
    ids = encode_chat(tk, ChatStyle::ChatML, msgs, true, &mask);
    const i32 im_start = tk.token_to_id("<|im_start|>");
    const i32 im_end   = tk.token_to_id("<|im_end|>");
    std::string text;
    for (i32 id : ids) {
        if (id == im_start || id == im_end) continue;
        text += tk.token_bytes(id);
    }
    CHECK_EQ(text, std::string("system\nsys\nuser\nhello\nassistant\n"),
             "ChatML rendered prompt");

    // The assistant turn is not supervised, the user turn is not either.
    bool any_supervised = false;
    for (u8 m : mask) if (m) any_supervised = true;
    CHECK(!any_supervised, "no supervised tokens without an assistant turn");

    // With an assistant turn, its content is supervised.
    msgs.push_back({Role::Assistant, "answer"});
    std::vector<u8> mask2;
    std::vector<i32> ids2;
    encode_chat(tk, ChatStyle::ChatML, msgs, false, &mask2);
    size_t sup = 0;
    for (u8 m : mask2) if (m) ++sup;
    CHECK(sup > 0, "assistant content is supervised");
}

int main() {
    enable_utf8_console();

    test_gpt2_byte_table();
    test_gpt2_pretokenizer();
    test_byte_map_and_decode();
    test_native_tokenizer_still_works();
    test_chat_styles();

    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << (g_checks - g_failures)
              << "/" << g_checks << " checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
