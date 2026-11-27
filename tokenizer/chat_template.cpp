#include "tokenizer/chat_template.h"
#include "core/common.h"

namespace gai {

namespace {

i32 need_ctrl(const Tokenizer& tk, const char* name) {
    const i32 id = tk.token_to_id(name);
    GAI_CHECK(id >= 0,
              strfmt("The chat template needs the control token '%s' but it is not in the "
                     "vocabulary (vocab_size=%d). This vocabulary does not speak this chat format.",
                     name, tk.vocab_size()));
    return id;
}

const char* role_name(Role r) {
    switch (r) {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
    }
    return "user";
}

}

std::vector<ChatSegment> render_chatml(const Tokenizer& tk,
                                       const std::vector<Message>& msgs,
                                       bool add_generation_prompt) {
    std::vector<ChatSegment> segs;
    auto push_tok = [&](i32 id, bool sup) {
        ChatSegment s;
        s.is_token = true;
        s.token = id;
        s.supervised = sup;
        segs.push_back(std::move(s));
    };
    auto push_txt = [&](std::string t, bool sup) {
        if (t.empty()) return;
        ChatSegment s;
        s.text = std::move(t);
        s.supervised = sup;
        segs.push_back(std::move(s));
    };

    if (tk.add_bos()) push_tok(tk.bos_id(), false);

    const i32 im_s = need_ctrl(tk, "<|im_start|>");
    const i32 im_e = need_ctrl(tk, "<|im_end|>");
    for (const Message& m : msgs) {
        const bool sup = (m.role == Role::Assistant);
        push_tok(im_s, false);
        push_txt(std::string(role_name(m.role)) + "\n", false);
        push_txt(m.content, sup);
        push_tok(im_e, sup);
        push_txt("\n", false);
    }
    if (add_generation_prompt) {
        push_tok(im_s, false);
        push_txt("assistant\n", false);
    }
    return segs;
}

std::vector<i32> encode_chatml(const Tokenizer& tk,
                               const std::vector<Message>& msgs,
                               bool add_generation_prompt,
                               std::vector<u8>* loss_mask) {
    std::vector<i32> ids;
    std::vector<u8> mask;

    for (const ChatSegment& s : render_chatml(tk, msgs, add_generation_prompt)) {
        if (s.is_token) {
            ids.push_back(s.token);
            mask.push_back(s.supervised ? 1 : 0);
            continue;
        }
        const std::vector<i32> body = tk.encode(s.text, false, false);
        for (i32 id : body) {
            ids.push_back(id);
            mask.push_back(s.supervised ? 1 : 0);
        }
    }

    if (loss_mask) *loss_mask = std::move(mask);
    return ids;
}

std::vector<ChatSegment> render_granite(const Tokenizer& tk,
                                        const std::vector<Message>& msgs,
                                        bool add_generation_prompt) {
    std::vector<ChatSegment> segs;
    auto push_tok = [&](i32 id, bool sup) {
        ChatSegment s;
        s.is_token = true;
        s.token = id;
        s.supervised = sup;
        segs.push_back(std::move(s));
    };
    auto push_txt = [&](std::string t, bool sup) {
        if (t.empty()) return;
        ChatSegment s;
        s.text = std::move(t);
        s.supervised = sup;
        segs.push_back(std::move(s));
    };

    if (tk.add_bos()) push_tok(tk.bos_id(), false);

    const i32 role_start = need_ctrl(tk, "<|start_of_role|>");
    const i32 role_end   = need_ctrl(tk, "<|end_of_role|>");
    const i32 turn_end   = need_ctrl(tk, "<|end_of_text|>");

    bool has_system = false;
    for (const Message& m : msgs) if (m.role == Role::System) { has_system = true; break; }
    if (!has_system) {
        push_tok(role_start, false);
        push_txt("system", false);
        push_tok(role_end, false);
        push_txt("You are a helpful assistant. Please ensure responses are professional, "
                 "accurate, and safe.", false);
        push_tok(turn_end, false);
        push_txt("\n", false);
    }

    for (const Message& m : msgs) {
        const bool sup = (m.role == Role::Assistant);
        push_tok(role_start, false);
        push_txt(role_name(m.role), false);
        push_tok(role_end, false);
        push_txt(m.content, sup);
        push_tok(turn_end, sup);
        push_txt("\n", false);
    }
    if (add_generation_prompt) {
        push_tok(role_start, false);
        push_txt("assistant", false);
        push_tok(role_end, false);
    }
    return segs;
}

std::vector<i32> encode_granite(const Tokenizer& tk,
                                const std::vector<Message>& msgs,
                                bool add_generation_prompt,
                                std::vector<u8>* loss_mask) {
    std::vector<i32> ids;
    std::vector<u8> mask;

    for (const ChatSegment& s : render_granite(tk, msgs, add_generation_prompt)) {
        if (s.is_token) {
            ids.push_back(s.token);
            mask.push_back(s.supervised ? 1 : 0);
            continue;
        }
        const std::vector<i32> body = tk.encode(s.text, false, false);
        for (i32 id : body) {
            ids.push_back(id);
            mask.push_back(s.supervised ? 1 : 0);
        }
    }

    if (loss_mask) *loss_mask = std::move(mask);
    return ids;
}

std::vector<ChatSegment> render_ghassan_thinking(const Tokenizer& tk,
                                                 const std::vector<Message>& msgs,
                                                 bool add_generation_prompt) {
    std::vector<ChatSegment> segs;
    auto push_tok = [&](i32 id, bool sup) {
        ChatSegment s;
        s.is_token = true;
        s.token = id;
        s.supervised = sup;
        segs.push_back(std::move(s));
    };
    auto push_txt = [&](std::string t, bool sup) {
        if (t.empty()) return;
        ChatSegment s;
        s.text = std::move(t);
        s.supervised = sup;
        segs.push_back(std::move(s));
    };

    if (tk.add_bos()) push_tok(tk.bos_id(), false);

    const i32 role_start = need_ctrl(tk, "<|start_of_role|>");
    const i32 role_end   = need_ctrl(tk, "<|end_of_role|>");
    const i32 turn_end   = need_ctrl(tk, "<|end_of_text|>");
    const i32 think_on   = need_ctrl(tk, "<think_on>");
    const i32 think_open = need_ctrl(tk, "<think>");
    const i32 think_close= need_ctrl(tk, "</think>");

    bool has_system = false;
    for (const Message& m : msgs) if (m.role == Role::System) { has_system = true; break; }
    if (!has_system) {
        push_tok(role_start, false);
        push_txt("system", false);
        push_tok(role_end, false);
        push_txt("You are a helpful assistant. Please ensure responses are professional, "
                 "accurate, and safe.", false);
        push_tok(turn_end, false);
        push_txt("\n", false);
    }
    push_tok(think_on, false);

    for (const Message& m : msgs) {
        const bool sup = (m.role == Role::Assistant);
        push_tok(role_start, false);
        push_txt(role_name(m.role), false);
        push_tok(role_end, false);
        if (sup) {
            push_tok(think_open, sup);
            push_txt(m.thinking, sup);
            push_tok(think_close, sup);
        }
        push_txt(m.content, sup);
        push_tok(turn_end, sup);
        push_txt("\n", false);
    }
    if (add_generation_prompt) {
        push_tok(role_start, false);
        push_txt("assistant", false);
        push_tok(role_end, false);
        push_tok(think_open, false);
    }
    return segs;
}

std::vector<i32> encode_ghassan_thinking(const Tokenizer& tk,
                                         const std::vector<Message>& msgs,
                                         bool add_generation_prompt,
                                         std::vector<u8>* loss_mask) {
    std::vector<i32> ids;
    std::vector<u8> mask;
    for (const ChatSegment& s : render_ghassan_thinking(tk, msgs, add_generation_prompt)) {
        if (s.is_token) {
            ids.push_back(s.token);
            mask.push_back(s.supervised ? 1 : 0);
            continue;
        }
        const std::vector<i32> body = tk.encode(s.text, false, false);
        for (i32 id : body) {
            ids.push_back(id);
            mask.push_back(s.supervised ? 1 : 0);
        }
    }
    if (loss_mask) *loss_mask = std::move(mask);
    return ids;
}

ChatTemplate chat_template_from_vocab(const Tokenizer& tk, bool thinking) {
    const bool role_tags = tk.token_to_id("<|start_of_role|>") >= 0 &&
                           tk.token_to_id("<|end_of_role|>") >= 0;

    if (thinking) {
        if (!role_tags) {
            GAI_FAIL("[chat] thinking=true requires Granite role tags "
                     "(<|start_of_role|> / <|end_of_role|>) but they are missing "
                     "from the vocabulary. Load a tokenizer that declares them "
                     "(e.g. the original Granite tokenizer.json) or pass "
                     "--thinking=false / thinking=false.");
        }
        if (tk.token_to_id("<think_on>") < 0) {
            GAI_FAIL("[chat] thinking=true requires the control token '<think_on>' "
                     "but it is missing from the vocabulary. Add it to "
                     "tokenizer.json (as an added/special token) and re-export, "
                     "or pass --thinking=false / thinking=false.");
        }
        if (tk.token_to_id("<think>") < 0) {
            GAI_FAIL("[chat] thinking=true requires the thinking open control token "
                     "'<think>' but it is missing from the vocabulary. Add it to "
                     "tokenizer.json (as an added/special token) and re-export, "
                     "or pass --thinking=false / thinking=false.");
        }
        if (tk.token_to_id("</think>") < 0) {
            GAI_FAIL("[chat] thinking=true requires the thinking close control token "
                     "'</think>' but it is missing from the vocabulary. Add it to "
                     "tokenizer.json (as an added/special token) and re-export, "
                     "or pass --thinking=false / thinking=false.");
        }
        return ChatTemplate::GhassanThinking;
    }

    if (!role_tags) {
        log_warn("[chat ] the vocabulary has no <|start_of_role|>/<|end_of_role|>; "
                 "falling back to the ChatML template. If this is a hybrid model, "
                 "load its own tokenizer.json or prompts will be out of distribution.");
        return ChatTemplate::ChatML;
    }
    return ChatTemplate::Granite;
}

std::vector<i32> encode_chat(const Tokenizer& tk,
                             ChatTemplate kind,
                             const std::vector<Message>& msgs,
                             bool add_generation_prompt,
                             std::vector<u8>* loss_mask) {
    switch (kind) {
        case ChatTemplate::GhassanThinking:
            return encode_ghassan_thinking(tk, msgs, add_generation_prompt, loss_mask);
        case ChatTemplate::Granite:
            return encode_granite(tk, msgs, add_generation_prompt, loss_mask);
        case ChatTemplate::ChatML:
        default:
            return encode_chatml(tk, msgs, add_generation_prompt, loss_mask);
    }
}

}
