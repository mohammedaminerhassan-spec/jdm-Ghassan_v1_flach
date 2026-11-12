#include "inference/chat.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

namespace gai {

ChatSession::ChatSession(Generator& gen, ChatOptions opts)
    : gen_(gen), opts_(std::move(opts)) {
    GAI_CHECK(opts_.max_history_turns >= 0, "chat history turns must be >= 0");
    GAI_CHECK(opts_.gen.max_new_tokens >= 0, "chat max tokens must be >= 0");
    // A system message is rendered only when one was explicitly configured.
    // The checkpoint speaks as it was trained to speak.
    if (!opts_.system.empty()) {
        history_.push_back({Role::System, opts_.system});
    }
}

void ChatSession::clear() {
    history_.clear();
    if (!opts_.system.empty()) {
        history_.push_back({Role::System, opts_.system});
    }
}

void ChatSession::set_system(const std::string& s) {
    opts_.system = s;
    if (s.empty()) {
        if (!history_.empty() && history_.front().role == Role::System)
            history_.erase(history_.begin());
        return;
    }
    if (!history_.empty() && history_.front().role == Role::System) {
        history_.front().content = s;
    } else {
        history_.insert(history_.begin(), {Role::System, s});
    }
}

void ChatSession::trim_history() {
    // Keep: optional system message (if any) + at most max_history_turns
    // user/assistant pairs.
    const size_t sys = (!history_.empty() && history_.front().role == Role::System) ? 1 : 0;
    const size_t max_msgs = sys + static_cast<size_t>(2 * opts_.max_history_turns);
    if (history_.size() <= max_msgs) return;

    std::vector<Message> trimmed;
    if (sys) trimmed.push_back(history_.front());
    const size_t keep_from = history_.size() - static_cast<size_t>(2 * opts_.max_history_turns);
    for (size_t i = keep_from; i < history_.size(); ++i) trimmed.push_back(history_[i]);
    history_ = std::move(trimmed);
}

std::string ChatSession::send(const std::string& user_message) {
    history_.push_back({Role::User, user_message});
    trim_history();

    GenerationConfig turn_cfg = opts_.gen;
    std::string reply;
    Generator::StreamFn cb = nullptr;
    if (opts_.stream) {
        cb = [&](const std::string& piece, i32) {
            std::cout << piece << std::flush;
            return true;
        };
    }
    reply = gen_.chat(history_, turn_cfg, cb);
    if (opts_.stream) std::cout << std::endl;

    history_.push_back({Role::Assistant, reply});
    return reply;
}

void ChatSession::run_repl() {
    std::cout <<
        "\n  Ghassan AI — chat\n"
        "  /help  /clear  /system <text>  /stats  /quit\n"
        "  ------------------------------------------------------------------\n\n";

    std::string line;
    while (true) {
        std::cout << "\n\033[1muser\033[0m > " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        if (line[0] == '/') {
            std::string cmd = line.substr(1);
            std::string arg;
            size_t sp = cmd.find(' ');
            if (sp != std::string::npos) { arg = cmd.substr(sp + 1); cmd = cmd.substr(0, sp); }

            if (cmd == "quit" || cmd == "exit" || cmd == "q") break;
            if (cmd == "clear") { clear(); std::cout << "  [history cleared]\n"; continue; }
            if (cmd == "system") {
                if (arg.empty()) {
                    const std::string cur = opts_.system.empty() ? "(none)" : opts_.system;
                    std::cout << "  system: " << cur << "\n";
                } else { set_system(arg); std::cout << "  [system prompt updated]\n"; }
                continue;
            }
            if (cmd == "stats") {
                std::cout << "  " << gen_.stats().summary() << "\n";
                std::cout << "  context used: " << gen_.context_used() << " tokens\n";
                std::cout << "  history: " << history_.size() << " messages\n";
                continue;
            }
            if (cmd == "temp") {
                if (!arg.empty()) {
                    try {
                        float t = std::stof(arg);
                        if (!std::isfinite(t)) t = 0.0f;
                        if (t < 0.0f) t = 0.0f;
                        if (t > 5.0f) t = 5.0f;
                        opts_.gen.sampling.temperature = t;
                        std::cout << "  [temperature = " << t << "]\n";
                    } catch (const std::exception&) {
                        std::cout << "  invalid temperature (want 0..5)\n";
                    }
                }
                continue;
            }
            if (cmd == "help") {
                std::cout <<
                    "  /clear          reset the conversation\n"
                    "  /system <text>  set the system prompt (empty = none)\n"
                    "  /temp <f>       set the sampling temperature\n"
                    "  /stats          show timing and context usage\n"
                    "  /quit           exit\n";
                continue;
            }
            std::cout << "  unknown command: " << cmd << " (try /help)\n";
            continue;
        }

        std::cout << "\n\033[1mghassan\033[0m > " << std::flush;
        send(line);
        if (opts_.show_stats) std::cout << "  [" << gen_.stats().summary() << "]\n";
    }
    std::cout << "\n  Goodbye!\n";
}

}
