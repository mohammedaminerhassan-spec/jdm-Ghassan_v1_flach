#pragma once

#include "tokenizer/tokenizer.h"
#include <vector>
#include <string>

namespace gai {

enum class ChatTemplate : u8 {
    ChatML = 0,
    Granite = 1,
    GhassanThinking = 2,
};

enum class Role : u8 { System = 0, User = 1, Assistant = 2 };

struct Message {
    Role        role = Role::User;
    std::string content;
    std::string thinking;
};

struct ChatSegment {
    bool        is_token   = false;
    i32         token      = -1;
    std::string text;
    bool        supervised = false;
};

std::vector<ChatSegment> render_chatml(const Tokenizer& tk,
                                       const std::vector<Message>& msgs,
                                       bool add_generation_prompt);

std::vector<i32> encode_chatml(const Tokenizer& tk,
                               const std::vector<Message>& msgs,
                               bool add_generation_prompt,
                               std::vector<u8>* loss_mask = nullptr);

std::vector<ChatSegment> render_granite(const Tokenizer& tk,
                                        const std::vector<Message>& msgs,
                                        bool add_generation_prompt);

std::vector<i32> encode_granite(const Tokenizer& tk,
                                const std::vector<Message>& msgs,
                                bool add_generation_prompt,
                                std::vector<u8>* loss_mask = nullptr);

std::vector<ChatSegment> render_ghassan_thinking(const Tokenizer& tk,
                                                 const std::vector<Message>& msgs,
                                                 bool add_generation_prompt);

std::vector<i32> encode_ghassan_thinking(const Tokenizer& tk,
                                         const std::vector<Message>& msgs,
                                         bool add_generation_prompt,
                                         std::vector<u8>* loss_mask = nullptr);

ChatTemplate chat_template_from_vocab(const Tokenizer& tk, bool thinking = false);

std::vector<i32> encode_chat(const Tokenizer& tk,
                             ChatTemplate kind,
                             const std::vector<Message>& msgs,
                             bool add_generation_prompt,
                             std::vector<u8>* loss_mask = nullptr);

}
