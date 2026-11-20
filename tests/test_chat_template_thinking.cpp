#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

using namespace gai;

namespace {

Tokenizer make_tok(bool granite, bool thinking_tokens) {
    std::vector<std::string> vocab;
    for (int i = 0; i < 256; ++i) vocab.push_back(gpt2_byte_token(static_cast<u8>(i)));
    VocabOptions o;
    o.pretok = granite ? PreTokKind::Granite : PreTokKind::Gpt2;
    auto add = [&](const std::string& s) {
        vocab.push_back(s);
        o.special_ids.push_back(static_cast<i32>(vocab.size()) - 1);
        o.added_ids.push_back(static_cast<i32>(vocab.size()) - 1);
        return static_cast<i32>(vocab.size()) - 1;
    };
    if (granite) {
        add("<|start_of_role|>");
        add("<|end_of_role|>");
        add("<|end_of_text|>");
    } else {
        add("<|im_start|>");
        add("<|im_end|>");
    }
    if (thinking_tokens) {
        add("<think_on>");
        add("<think>");
        add("</think>");
    }
    Tokenizer t;
    t.build(vocab, {}, o);
    return t;
}

bool contains(const std::vector<i32>& v, i32 x) {
    for (i32 e : v) if (e == x) return true;
    return false;
}

}  // namespace

int main() {
    int failures = 0;

    // 1. Granite thinking template is selected only with the right tokens.
    {
        Tokenizer t = make_tok(true, true);
        if (chat_template_from_vocab(t, false) != ChatTemplate::Granite) {
            std::cerr << "granite template not selected\n"; ++failures;
        }
        if (chat_template_from_vocab(t, true) != ChatTemplate::GhassanThinking) {
            std::cerr << "thinking template not selected\n"; ++failures;
        }
    }

    // 2. Thinking is opt-in and fails closed on a plain Granite vocabulary.
    {
        Tokenizer t = make_tok(true, false);
        bool threw = false;
        try { (void)chat_template_from_vocab(t, true); }
        catch (...) { threw = true; }
        if (!threw) { std::cerr << "thinking should fail without control tokens\n"; ++failures; }
    }

    // 3. The rendered prompt uses <think_on> / <think> and the assistant span
    //    carries the supervised close token </think>.
    {
        Tokenizer t = make_tok(true, true);
        const i32 think_on   = t.token_to_id("<think_on>");
        const i32 think_open = t.token_to_id("<think>");
        const i32 think_close= t.token_to_id("</think>");
        std::vector<Message> msgs;
        msgs.push_back({Role::User, "Q"});
        msgs.push_back({Role::Assistant, "A", "R"});
        std::vector<u8> mask;
        std::vector<i32> ids = encode_chat(t, ChatTemplate::GhassanThinking,
                                           msgs, false, &mask);
        if (!contains(ids, think_on)) {
            std::cerr << "<think_on> missing from prompt\n"; ++failures;
        }
        if (!contains(ids, think_open) || !contains(ids, think_close)) {
            std::cerr << "<think>/</think> missing from assistant turn\n"; ++failures;
        }
        if (mask.size() != ids.size()) {
            std::cerr << "loss mask size mismatch\n"; ++failures;
        } else {
            for (size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] == think_close && mask[i] != 1) {
                    std::cerr << "</think> not supervised\n"; ++failures;
                }
            }
        }
    }

    // 4. ChatML control tokens are the real ChatML markers, not corrupted text.
    {
        Tokenizer t = make_tok(false, false);
        if (chat_template_from_vocab(t, false) != ChatTemplate::ChatML) {
            std::cerr << "ChatML template not selected\n"; ++failures;
        }
        std::vector<Message> msgs;
        msgs.push_back({Role::User, "Q"});
        std::vector<i32> ids = encode_chatml(t, msgs, true, nullptr);
        if (ids.empty()) { std::cerr << "ChatML encode produced no tokens\n"; ++failures; }
        if (!contains(ids, t.token_to_id("<|im_start|>")) ||
            !contains(ids, t.token_to_id("<|im_end|>"))) {
            std::cerr << "ChatML markers missing\n"; ++failures;
        }
    }

    if (failures == 0) std::cout << "test_chat_template_thinking: PASS\n";
    return failures == 0 ? 0 : 1;
}
