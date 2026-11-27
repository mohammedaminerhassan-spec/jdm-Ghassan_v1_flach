#include "training/jsonl_loader.h"
#include "tokenizer/chat_template.h"
#include "core/json.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace gai {

namespace {

bool role_from_string(const std::string& s, Role& out) {
    if (s == "system")                        { out = Role::System;    return true; }
    if (s == "user" || s == "human")           { out = Role::User;      return true; }
    if (s == "assistant" || s == "gpt" || s == "bot" || s == "model") {
        out = Role::Assistant; return true;
    }
    return false;
}

}

bool JsonlLoader::parse_line(const std::string& line, JsonlSample& sample) {
    sample.messages.clear();
    sample.thinking = false;

    if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
        return false;
    }

    Json j;
    std::string err;
    if (!Json::parse(line, j, &err) || !j.is_object()) {
        return false;
    }

    if (j.has("messages") && j.at("messages").is_array()) {
        for (const Json& item : j.at("messages").elements()) {
            if (!item.is_object()) continue;
            const std::string role_str = item.at("role").as_string();
            const std::string content  = item.at("content").as_string();

            Role r = Role::User;
            if (!role_from_string(role_str, r)) continue;

            Message msg;
            msg.role = r;
            msg.content = content;
            if (r == Role::Assistant) {
                if (item.has("thinking") && item.at("thinking").is_string() && !item.at("thinking").as_string().empty()) {
                    msg.thinking = item.at("thinking").as_string();
                    sample.thinking = true;
                } else if (item.has("reasoning_content") && item.at("reasoning_content").is_string() && !item.at("reasoning_content").as_string().empty()) {
                    msg.thinking = item.at("reasoning_content").as_string();
                    sample.thinking = true;
                } else if (item.has("thought") && item.at("thought").is_string() && !item.at("thought").as_string().empty()) {
                    msg.thinking = item.at("thought").as_string();
                    sample.thinking = true;
                } else {
                    size_t op = msg.content.find("<think>");
                    if (op != std::string::npos) {
                        size_t cp = msg.content.find("</think>", op + 7);
                        if (cp != std::string::npos) {
                            msg.thinking = msg.content.substr(op + 7, cp - (op + 7));
                            msg.content = msg.content.substr(0, op) + msg.content.substr(cp + 8);
                            sample.thinking = true;
                        }
                    }
                }
            }
            sample.messages.push_back(std::move(msg));
        }
        return !sample.messages.empty();
    }

    if (j.has("conversations") && j.at("conversations").is_array()) {
        for (const Json& item : j.at("conversations").elements()) {
            if (!item.is_object()) continue;
            const std::string from = item.at("from").as_string();
            const std::string val  = item.at("value").as_string();

            Role r = Role::User;
            if (!role_from_string(from, r)) continue;

            Message msg;
            msg.role = r;
            msg.content = val;
            if (r == Role::Assistant) {
                if (item.has("thinking") && item.at("thinking").is_string() && !item.at("thinking").as_string().empty()) {
                    msg.thinking = item.at("thinking").as_string();
                    sample.thinking = true;
                } else if (item.has("reasoning_content") && item.at("reasoning_content").is_string() && !item.at("reasoning_content").as_string().empty()) {
                    msg.thinking = item.at("reasoning_content").as_string();
                    sample.thinking = true;
                } else if (item.has("thought") && item.at("thought").is_string() && !item.at("thought").as_string().empty()) {
                    msg.thinking = item.at("thought").as_string();
                    sample.thinking = true;
                } else {
                    size_t op = msg.content.find("<think>");
                    if (op != std::string::npos) {
                        size_t cp = msg.content.find("</think>", op + 7);
                        if (cp != std::string::npos) {
                            msg.thinking = msg.content.substr(op + 7, cp - (op + 7));
                            msg.content = msg.content.substr(0, op) + msg.content.substr(cp + 8);
                            sample.thinking = true;
                        }
                    }
                }
            }
            sample.messages.push_back(std::move(msg));
        }
        return !sample.messages.empty();
    }

    if (j.has("instruction") && j.has("output")) {
        const std::string inst = j.at("instruction").as_string();
        const std::string in   = j.has("input") ? j.at("input").as_string() : "";
        const std::string out  = j.at("output").as_string();

        std::string user_content = inst;
        if (!in.empty()) {
            user_content += "\n" + in;
        }

        Message u_msg{Role::User, user_content, ""};
        Message a_msg{Role::Assistant, out, ""};
        if (j.has("thinking") && j.at("thinking").is_string() && !j.at("thinking").as_string().empty()) {
            a_msg.thinking = j.at("thinking").as_string();
            sample.thinking = true;
        } else if (j.has("reasoning_content") && j.at("reasoning_content").is_string() && !j.at("reasoning_content").as_string().empty()) {
            a_msg.thinking = j.at("reasoning_content").as_string();
            sample.thinking = true;
        } else if (j.has("thought") && j.at("thought").is_string() && !j.at("thought").as_string().empty()) {
            a_msg.thinking = j.at("thought").as_string();
            sample.thinking = true;
        } else {
            size_t op = a_msg.content.find("<think>");
            if (op != std::string::npos) {
                size_t cp = a_msg.content.find("</think>", op + 7);
                if (cp != std::string::npos) {
                    a_msg.thinking = a_msg.content.substr(op + 7, cp - (op + 7));
                    a_msg.content = a_msg.content.substr(0, op) + a_msg.content.substr(cp + 8);
                    sample.thinking = true;
                }
            }
        }
        sample.messages.push_back(std::move(u_msg));
        sample.messages.push_back(std::move(a_msg));
        return true;
    }

    if (j.has("text")) {
        Message msg{Role::Assistant, j.at("text").as_string(), ""};
        sample.messages.push_back(std::move(msg));
        return true;
    }

    return false;
}

bool JsonlLoader::convert_to_gbin(const std::string& jsonl_path,
                                  const std::string& gbin_out_path,
                                  const Tokenizer& tk,
                                  int max_seq_len) {
    std::ifstream file(jsonl_path);
    if (!file.is_open()) {
        log_error("[jsonl] failed to open input: " + jsonl_path);
        return false;
    }

    ShardWriter writer(gbin_out_path, tk.vocab_size(), true, tk.fingerprint());

    std::string head;
    while (std::getline(file, head)) {
        if (!head.empty() && head.find_first_not_of(" \t\r\n") != std::string::npos) break;
    }
    const bool whole_file_json = !head.empty() && head.find_first_not_of(" \t\r\n") != std::string::npos &&
                                 head.find_first_not_of(" \t") != std::string::npos &&
                                 head.find_first_not_of(" \t") < head.size() &&
                                 head[head.find_first_not_of(" \t")] == '[';
    if (whole_file_json) {
        std::string text(head);
        std::string rest;
        while (std::getline(file, rest)) text += "\n" + rest;
        Json arr;
        std::string err;
        if (!Json::parse(text, arr, &err) || !arr.is_array()) {
            log_error("[jsonl] invalid JSON array in " + jsonl_path + ": " + err);
            return false;
        }
        for (const Json& item : arr.elements()) {
            JsonlSample sample;
            if (!parse_line(item.dump(), sample) || sample.messages.empty()) continue;
            std::vector<u8> loss_mask;
            const ChatTemplate tmpl = chat_template_from_vocab(tk, sample.thinking);
            std::vector<i32> tokens = encode_chat(tk, tmpl, sample.messages, false, &loss_mask);
            if (tokens.empty()) continue;
            if (static_cast<int>(tokens.size()) > max_seq_len) {
                tokens.resize(static_cast<size_t>(max_seq_len));
                loss_mask.resize(static_cast<size_t>(max_seq_len));
            }
            writer.add_document(tokens, &loss_mask);
        }
        writer.close();
        return writer.token_count() > 0;
    }
    file.clear();
    file.seekg(0, std::ios::beg);

    std::string line;
    u64 count = 0;
    while (std::getline(file, line)) {
        JsonlSample sample;
        if (!parse_line(line, sample)) continue;
        if (sample.messages.empty()) continue;

        std::vector<u8> loss_mask;
        const ChatTemplate tmpl = chat_template_from_vocab(tk, sample.thinking);
        if (sample.thinking && tmpl != ChatTemplate::GhassanThinking) {
            log_warn("[jsonl] a sample carries a thinking trace but this vocabulary "
                     "lacks <think_on>/<think>/</think>; the trace will be dropped. "
                     "Use a tokenizer that declares those added tokens.");
        }
        std::vector<i32> tokens = encode_chat(tk, tmpl, sample.messages,
                                              false,
                                              &loss_mask);

        if (tokens.empty()) continue;
        if (static_cast<int>(tokens.size()) > max_seq_len) {
            tokens.resize(static_cast<size_t>(max_seq_len));
            loss_mask.resize(static_cast<size_t>(max_seq_len));
        }

        writer.add_document(tokens, &loss_mask);
        ++count;
    }

    writer.close();
    log_info(strfmt("[jsonl] processed %llu samples -> %s (%llu tokens)",
                    static_cast<unsigned long long>(count),
                    gbin_out_path.c_str(),
                    static_cast<unsigned long long>(writer.token_count())));
    return writer.token_count() > 0;
}

int JsonlLoader::process_directory(const std::string& dir,
                                   const Tokenizer& tk,
                                   int max_seq_len) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return 0;

    int converted = 0;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        const std::string path = entry.path().string();
        const std::string ext = entry.path().extension().string();

        if (ext == ".jsonl" || ext == ".json") {
            fs::path out_path = entry.path();
            out_path.replace_extension(".gbin");
            {
                const std::string parent = entry.path().parent_path().filename().string();
                const std::string stem = out_path.stem().string();
                const bool parent_val = (parent == "val" || parent == "validation" || parent == "eval");
                const bool stem_val = (stem.rfind("val", 0) == 0 || stem.rfind("validation", 0) == 0 ||
                                      stem.rfind("eval", 0) == 0);
                if (parent_val || stem_val) {
                    out_path = entry.path().parent_path() / ("val_" + stem + ".gbin");
                } else {
                    out_path = entry.path().parent_path() / ("train_" + stem + ".gbin");
                }
            }

            if (fs::exists(out_path, ec)) {
                auto t_jsonl = fs::last_write_time(entry.path(), ec);
                auto t_gbin  = fs::last_write_time(out_path, ec);
                if (t_gbin >= t_jsonl) continue;
            }

            if (convert_to_gbin(path, out_path.string(), tk, max_seq_len)) {
                ++converted;
            }
        }
    }
    return converted;
}

}
