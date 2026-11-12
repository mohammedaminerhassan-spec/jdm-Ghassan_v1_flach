#include "training/jsonl_loader.h"
#include "tokenizer/chat_template.h"
#include "core/json.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace gai {

namespace {

// Maps a dataset role onto the three roles this engine speaks.
// Returns false for anything unknown ("tool", "function", "developer", ...):
// an unknown turn must be dropped, never silently turned into a User turn
// (that would teach the model to answer on behalf of the user).
bool role_from_string(const std::string& s, Role& out) {
    if (s == "system")                        { out = Role::System;    return true; }
    if (s == "user" || s == "human")           { out = Role::User;      return true; }
    if (s == "assistant" || s == "gpt" || s == "bot" || s == "model") {
        out = Role::Assistant; return true;
    }
    return false;
}

}  // namespace

bool JsonlLoader::parse_line(const std::string& line, JsonlSample& sample) {
    sample.messages.clear();

    if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
        return false;
    }

    Json j;
    std::string err;
    if (!Json::parse(line, j, &err) || !j.is_object()) {
        return false;
    }

    // Format 1: OpenAI {"messages": [...]}
    if (j.has("messages") && j.at("messages").is_array()) {
        for (const Json& item : j.at("messages").elements()) {
            if (!item.is_object()) continue;
            const std::string role_str = item.at("role").as_string();
            const std::string content  = item.at("content").as_string();

            Role r = Role::User;
            if (!role_from_string(role_str, r)) continue;  // quarantine

            Message msg;
            msg.role = r;
            msg.content = content;
            sample.messages.push_back(std::move(msg));
        }
        return !sample.messages.empty();
    }

    // Format 2: ShareGPT {"conversations": [...]}
    if (j.has("conversations") && j.at("conversations").is_array()) {
        for (const Json& item : j.at("conversations").elements()) {
            if (!item.is_object()) continue;
            const std::string from = item.at("from").as_string();
            const std::string val  = item.at("value").as_string();

            Role r = Role::User;
            if (!role_from_string(from, r)) continue;  // quarantine

            Message msg;
            msg.role = r;
            msg.content = val;
            sample.messages.push_back(std::move(msg));
        }
        return !sample.messages.empty();
    }

    // Format 3: Alpaca {"instruction": "...", "input": "...", "output": "..."}
    if (j.has("instruction") && j.has("output")) {
        const std::string inst = j.at("instruction").as_string();
        const std::string in   = j.has("input") ? j.at("input").as_string() : "";
        const std::string out  = j.at("output").as_string();

        std::string user_content = inst;
        if (!in.empty()) {
            user_content += "\n" + in;
        }

        Message u_msg{Role::User, user_content};
        Message a_msg{Role::Assistant, out};
        sample.messages.push_back(std::move(u_msg));
        sample.messages.push_back(std::move(a_msg));
        return true;
    }

    // Format 4: Raw text {"text": "..."}
    if (j.has("text")) {
        Message msg{Role::Assistant, j.at("text").as_string()};
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

    ShardWriter writer(gbin_out_path, tk.vocab_size(), /*with_loss_mask=*/true);

    std::string line;
    u64 count = 0;
    while (std::getline(file, line)) {
        JsonlSample sample;
        if (!parse_line(line, sample)) continue;
        if (sample.messages.empty()) continue;

        std::vector<u8> loss_mask;
        // The chat format is chosen by the vocabulary, not by a flag: Granite-4
        // tokenizes <|start_of_role|>, Qwen3 tokenizes <|im_start|>.
        const ChatTemplate tmpl =
            (tk.token_to_id("<|start_of_role|>") >= 0 && tk.token_to_id("<|end_of_role|>") >= 0)
                ? ChatTemplate::Granite
                : ChatTemplate::ChatML;
        std::vector<i32> tokens = encode_chat(tk, tmpl, sample.messages,
                                              /*add_generation_prompt=*/false,
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
            // The loader globs "train*.gbin" / "val*.gbin", so an
            // auto-converted shard must carry the prefix of the folder it
            // lives in - otherwise the converted data is never picked up.
            fs::path out_path = entry.path();
            out_path.replace_extension(".gbin");
            {
                const std::string parent = entry.path().parent_path().filename().string();
                const std::string stem = out_path.stem().string();
                if (parent == "val" || parent == "validation" || parent == "eval") {
                    out_path = entry.path().parent_path() / ("val_" + stem + ".gbin");
                } else {
                    out_path = entry.path().parent_path() / ("train_" + stem + ".gbin");
                }
            }

            // Skip if .gbin already exists and is newer
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

} // namespace gai
