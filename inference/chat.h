#pragma once

#include "inference/generator.h"

#include <memory>
#include <string>
#include <vector>

namespace gai {

struct ChatOptions {
    // Optional explicit system prompt. Empty = no system message at all
    // (the model speaks as its checkpoint was trained to speak).
    std::string      system;
    GenerationConfig gen;
    bool             stream = true;
    bool             show_stats = false;
    int              max_history_turns = 24;
};

class ChatSession {
public:
    ChatSession(Generator& gen, ChatOptions opts);

    std::string send(const std::string& user_message);

    void clear();
    void set_system(const std::string& s);
    const std::vector<Message>& history() const { return history_; }

    void run_repl();

private:
    void trim_history();

    Generator&           gen_;
    ChatOptions          opts_;
    std::vector<Message> history_;
};

}
