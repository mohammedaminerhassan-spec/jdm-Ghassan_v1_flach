#pragma once

#include "tokenizer/tokenizer.h"
#include <vector>
#include <string>

namespace gai {

// The chat formats this engine speaks.
enum class ChatTemplate : u8 {
    // Qwen / ChatML: <|im_start|>role\ncontent<|im_end|>\n
    ChatML = 0,
    // IBM Granite-4: <|start_of_role|>role<|end_of_role|>content<|end_of_text|>\n
    Granite = 1,
};

// The one chat format this engine speaks by default: ChatML, the format Qwen (and
// DeepSeek) checkpoints are trained with:
//
//   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n
//   <|im_start|>assistant\n...<|im_end|>\n
//
// There are no engine-invented prompts: no default system message, no
// language routing, no persona logic. A system message is rendered only
// when the caller (or the user) explicitly supplies one.
enum class Role : u8 { System = 0, User = 1, Assistant = 2 };

struct Message {
    Role        role = Role::User;
    std::string content;
};

// One pre-tokenized piece of a rendered prompt: either a control token to
// emit verbatim, or text that still has to go through the BPE.
struct ChatSegment {
    bool        is_token   = false;
    i32         token      = -1;
    std::string text;
    bool        supervised = false;  // contributes to the training loss
};

std::vector<ChatSegment> render_chatml(const Tokenizer& tk,
                                       const std::vector<Message>& msgs,
                                       bool add_generation_prompt);

// Renders and tokenizes. loss_mask (optional) is 1 for the tokens that
// should be predicted (assistant content + the turn terminator).
std::vector<i32> encode_chatml(const Tokenizer& tk,
                               const std::vector<Message>& msgs,
                               bool add_generation_prompt,
                               std::vector<u8>* loss_mask = nullptr);

// The Granite-4 chat format:
//
//   <|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n
//   <|start_of_role|>user<|end_of_role|>{content}<|end_of_text|>\n
//   <|start_of_role|>assistant<|end_of_role|>{content}<|end_of_text|>\n
//   <|start_of_role|>assistant<|end_of_role|>
//
// The loss mask covers the assistant content and its <|end_of_text|> turn
// terminator only. No engine-invented default system message is added: a
// system turn is rendered only when the caller supplies one.
std::vector<ChatSegment> render_granite(const Tokenizer& tk,
                                        const std::vector<Message>& msgs,
                                        bool add_generation_prompt);

std::vector<i32> encode_granite(const Tokenizer& tk,
                                const std::vector<Message>& msgs,
                                bool add_generation_prompt,
                                std::vector<u8>* loss_mask = nullptr);

// Encode with the requested template.
std::vector<i32> encode_chat(const Tokenizer& tk,
                             ChatTemplate kind,
                             const std::vector<Message>& msgs,
                             bool add_generation_prompt,
                             std::vector<u8>* loss_mask = nullptr);

}
