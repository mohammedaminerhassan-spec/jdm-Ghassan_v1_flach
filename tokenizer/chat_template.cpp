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
    std::vector<i32> ids;
    std::vector<u8>  mask;

    auto push = [&](i32 id, u8 m) { ids.push_back(id); mask.push_back(m); };

    push(special::BOS, 0);

    for (const auto& m : msgs) {
        push(role_token(m.role), 0);
        std::vector<i32> body = tk.encode(m.content, false, false);
        const u8 lm = (m.role == Role::Assistant) ? 1 : 0;
        for (i32 id : body) push(id, lm);

        push(special::END, lm);
    }

    if (add_generation_prompt) push(special::ASSISTANT, 0);

    if (loss_mask) *loss_mask = std::move(mask);
    return ids;
}

}
