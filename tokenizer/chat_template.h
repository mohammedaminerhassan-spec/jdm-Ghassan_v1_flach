#pragma once

#include "tokenizer/tokenizer.h"
#include <vector>
#include <string>

namespace gai {

enum class Role : u8 { System = 0, User = 1, Assistant = 2 };

struct Message {
    Role        role = Role::User;
    std::string content;
};

enum class ReplyScript : u8 { Arabic = 0, Latin = 1 };

// Chat formatting styles.
//
// Native is the Gai style. The other three are the de-facto standards used
// by HuggingFace checkpoints; the engine detects them from the vocabulary so
// a foreign model can be served without hand-editing anything.
enum class ChatStyle : u8 {
    Native  = 0,  // <|system|>c<|end|> ... <|assistant|>
    ChatML  = 1,  // <|im_start|>role\nc<|im_end|>
    Llama3  = 2,  // <|start_header_id|>role<|end_header_id|>\n\nc<|eot_id|>
    Mistral = 3,  // [INST] c [/INST]
};

// One piece of a rendered prompt: either a literal control token or plain
// text that still has to go through the tokenizer.
struct ChatSegment {
    bool        is_token   = false;
    i32         token      = -1;
    std::string text;
    bool        supervised = false;  // contributes to the training loss
};

// Detects the chat style from the marker tokens present in the vocabulary.
ChatStyle detect_chat_style(const Tokenizer& tk);

// Renders the message list into control-token / text segments.
std::vector<ChatSegment> render_chat(const Tokenizer& tk, ChatStyle style,
                                     const std::vector<Message>& msgs,
                                     bool add_generation_prompt);

// Renders and tokenizes. loss_mask (optional) is 1 for the tokens that
// should be predicted (assistant content + the turn terminator).
std::vector<i32> encode_chat(const Tokenizer& tk, ChatStyle style,
                              const std::vector<Message>& msgs,
                              bool add_generation_prompt,
                              std::vector<u8>* loss_mask = nullptr);

class ChatTemplate {
public:
    static const char* default_system();

    static const char* default_system_latin();

    static const char* default_system_english();

    static const char* script_directive(ReplyScript s);

    static ReplyScript detect_script(const std::string& user_message);

    static const char* system_for_script(ReplyScript s);

    static const char* system_for_persona(const std::string& persona, ReplyScript s);

    static std::string render(const std::vector<Message>& msgs, bool add_generation_prompt);

    static std::vector<i32> encode(const Tokenizer& tk,
                                   const std::vector<Message>& msgs,
                                   bool add_generation_prompt,
                                   std::vector<u8>* loss_mask = nullptr);

    static i32 role_token(Role r);
};

}
