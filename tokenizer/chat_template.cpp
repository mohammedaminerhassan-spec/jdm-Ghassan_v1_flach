#include "tokenizer/chat_template.h"
#include "core/unicode.h"

namespace gai {

const char* ChatTemplate::default_system() {
    return "";
}

const char* ChatTemplate::default_system_latin() {
    return "";
}

const char* ChatTemplate::default_system_english() {
    return "";
}

const char* ChatTemplate::system_for_persona(const std::string& persona, ReplyScript s) {
    if (persona == "en" || persona == "english") return default_system_english();
    return system_for_script(s);
}

const char* ChatTemplate::script_directive(ReplyScript s) {
    return (s == ReplyScript::Latin)
        ? "Reply ONLY in Latin letters (Arabizi). Never use Arabic letters."
        : "جاوب غير بالحروف العربية فقط. ممنوع تستعمل الحروف اللاتينية.";
}

ReplyScript ChatTemplate::detect_script(const std::string& user_message) {
    ScriptStats st = script_stats(user_message);

    if (st.arabic == 0 && st.latin == 0) return ReplyScript::Arabic;
    return (st.arabic >= st.latin) ? ReplyScript::Arabic : ReplyScript::Latin;
}

const char* ChatTemplate::system_for_script(ReplyScript s) {
    return (s == ReplyScript::Latin) ? default_system_latin() : default_system();
}

i32 ChatTemplate::role_token(Role r) {
    switch (r) {
        case Role::System:    return special::SYSTEM;
        case Role::User:      return special::USER;
        case Role::Assistant: return special::ASSISTANT;
    }
    return special::USER;
}

// ---------------------------------------------------------------------------
// Foreign chat styles
// ---------------------------------------------------------------------------

namespace {

const char* role_name(Role r) {
    switch (r) {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
    }
    return "user";
}

i32 need_ctrl(const Tokenizer& tk, const char* name) {
    const i32 id = tk.token_to_id(name);
    GAI_CHECK(id >= 0,
              strfmt("chat style needs the control token '%s' but it is not in the "
                     "vocabulary (vocab_size=%d)", name, tk.vocab_size()));
    return id;
}

}  // namespace

ChatStyle detect_chat_style(const Tokenizer& tk) {
    if (tk.token_to_id("<|im_start|>") >= 0 && tk.token_to_id("<|im_end|>") >= 0)
        return ChatStyle::ChatML;
    if (tk.token_to_id("<|start_header_id|>") >= 0 && tk.token_to_id("<|eot_id|>") >= 0)
        return ChatStyle::Llama3;
    if (tk.token_to_id("[INST]") >= 0 && tk.token_to_id("[/INST]") >= 0)
        return ChatStyle::Mistral;
    return ChatStyle::Native;
}

std::vector<ChatSegment> render_chat(const Tokenizer& tk, ChatStyle style,
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

    // Some chat templates (SmolLM2, Qwen, ...) inject a default system
    // message when the conversation does not start with one. The engine
    // reproduces that behaviour from the metadata the converter wrote.
    //
    // An empty system message counts as "no system message": the callers
    // that always push one (the chat session, the CLI) would otherwise
    // suppress the default and change the prompt.
    const bool has_system = [&] {
        for (const Message& m : msgs) {
            if (m.role == Role::System && !m.content.empty()) return true;
        }
        return false;
    }();
    const bool inject_default =
        !has_system && !tk.default_system().empty() &&
        (style == ChatStyle::ChatML || style == ChatStyle::Llama3);
    const bool skip_empty_system =
        (style == ChatStyle::ChatML || style == ChatStyle::Llama3);

    switch (style) {
        case ChatStyle::Native: {
            const i32 sysm  = need_ctrl(tk, "<|system|>");
            const i32 usrm  = need_ctrl(tk, "<|user|>");
            const i32 asstm = need_ctrl(tk, "<|assistant|>");
            const i32 endm  = need_ctrl(tk, "<|end|>");
            for (const Message& m : msgs) {
                const i32 marker = (m.role == Role::System) ? sysm
                                 : (m.role == Role::User)  ? usrm
                                                           : asstm;
                const bool sup = (m.role == Role::Assistant);
                push_tok(marker, false);
                push_txt(m.content, sup);
                push_tok(endm, sup);
            }
            if (add_generation_prompt) push_tok(asstm, false);
            break;
        }
        case ChatStyle::ChatML: {
            const i32 im_s = need_ctrl(tk, "<|im_start|>");
            const i32 im_e = need_ctrl(tk, "<|im_end|>");
            if (inject_default) {
                push_tok(im_s, false);
                push_txt("system\n", false);
                push_txt(tk.default_system(), false);
                push_tok(im_e, false);
                push_txt("\n", false);
            }
            for (const Message& m : msgs) {
                if (skip_empty_system && m.role == Role::System && m.content.empty()) continue;
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
            break;
        }
        case ChatStyle::Llama3: {
            const i32 hs  = need_ctrl(tk, "<|start_header_id|>");
            const i32 he  = need_ctrl(tk, "<|end_header_id|>");
            const i32 eot = need_ctrl(tk, "<|eot_id|>");
            if (inject_default) {
                push_tok(hs, false);
                push_txt("system", false);
                push_tok(he, false);
                push_txt("\n\n", false);
                push_txt(tk.default_system(), false);
                push_tok(eot, false);
            }
            for (const Message& m : msgs) {
                if (skip_empty_system && m.role == Role::System && m.content.empty()) continue;
                const bool sup = (m.role == Role::Assistant);
                push_tok(hs, false);
                push_txt(role_name(m.role), false);
                push_tok(he, false);
                push_txt("\n\n", false);
                push_txt(m.content, sup);
                push_tok(eot, sup);
            }
            if (add_generation_prompt) {
                push_tok(hs, false);
                push_txt("assistant", false);
                push_tok(he, false);
                push_txt("\n\n", false);
            }
            break;
        }
        case ChatStyle::Mistral: {
            const i32 inst  = need_ctrl(tk, "[INST]");
            const i32 einst = need_ctrl(tk, "[/INST]");
            // Mistral has no system role: fold it into the first user turn.
            std::string sys;
            for (const Message& m : msgs) {
                if (m.role == Role::System) { sys = m.content; break; }
            }
            for (const Message& m : msgs) {
                if (m.role == Role::System) continue;
                if (m.role == Role::User) {
                    push_tok(inst, false);
                    push_txt(sys.empty() ? (" " + m.content) : (" " + sys + "\n" + m.content),
                             false);
                    push_tok(einst, false);
                    sys.clear();
                } else {
                    push_txt(" " + m.content, true);
                    push_tok(tk.eos_id(), true);
                }
            }
            break;
        }
    }
    return segs;
}

std::vector<i32> encode_chat(const Tokenizer& tk, ChatStyle style,
                              const std::vector<Message>& msgs,
                              bool add_generation_prompt,
                              std::vector<u8>* loss_mask) {
    std::vector<i32> ids;
    std::vector<u8> mask;

    for (const ChatSegment& s : render_chat(tk, style, msgs, add_generation_prompt)) {
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

static const char* role_marker(Role r) {
    switch (r) {
        case Role::System:    return "<|system|>";
        case Role::User:      return "<|user|>";
        case Role::Assistant: return "<|assistant|>";
    }
    return "<|user|>";
}

std::string ChatTemplate::render(const std::vector<Message>& msgs, bool add_generation_prompt) {
    std::string out = "<s>";
    for (const auto& m : msgs) {
        out += role_marker(m.role);
        out += m.content;
        out += "<|end|>";
    }
    if (add_generation_prompt) out += "<|assistant|>";
    return out;
}

std::vector<i32> ChatTemplate::encode(const Tokenizer& tk,
                                      const std::vector<Message>& msgs,
                                      bool add_generation_prompt,
                                      std::vector<u8>* loss_mask) {
    return encode_chat(tk, ChatStyle::Native, msgs, add_generation_prompt, loss_mask);
}

}
