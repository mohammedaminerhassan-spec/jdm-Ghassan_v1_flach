#include "tokenizer/chat_template.h"

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

// Shared renderer: segments are either raw control ids or text to BPE.
// `wrap` renders one message; `terminator` is the per-turn terminator token.
struct RenderSpec {
    i32  start;       // never null in our two templates
    i32  end;
    bool end_supervised;
    std::string start_text_prefix;  // role header rendered as text after `start`
    std::string start_text_suffix;
};

}  // namespace

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

    // The Granite-4 template always opens with a system turn. When the
    // caller supplies none, the reference template uses its own default
    // sentence; the engine reproduces it so the prompt matches the training
    // distribution exactly.
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

std::vector<i32> encode_chat(const Tokenizer& tk,
                             ChatTemplate kind,
                             const std::vector<Message>& msgs,
                             bool add_generation_prompt,
                             std::vector<u8>* loss_mask) {
    switch (kind) {
        case ChatTemplate::Granite:
            return encode_granite(tk, msgs, add_generation_prompt, loss_mask);
        case ChatTemplate::ChatML:
        default:
            return encode_chatml(tk, msgs, add_generation_prompt, loss_mask);
    }
}

}
