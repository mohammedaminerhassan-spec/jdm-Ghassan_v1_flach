#include "tools/serve_cli.h"

#include "tools/model_loader.h"
#include "core/json.h"
#include "server/http.h"
#include "inference/generator.h"
#include "inference/chat.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

namespace gai {

namespace {

std::atomic<unsigned long long> g_req_id{0};

std::string now_iso8601() {
    const auto t = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(t);
    std::tm tmv{};
#ifdef _WIN32
    gmtime_s(&tmv, &tt);
#else
    gmtime_r(&tt, &tmv);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

std::string make_id(const char* prefix) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s-%llu", prefix,
                  (unsigned long long)g_req_id.fetch_add(1) + 1);
    return buf;
}

Json error_json(const std::string& message, const std::string& type = "invalid_request_error") {
    Json err = Json::object();
    err.set("message", Json::string(message));
    err.set("type", Json::string(type));
    err.set("code", Json::null());
    Json root = Json::object();
    root.set("error", err);
    return root;
}

// Turns the request messages into the engine's message list.
bool extract_messages(const Json& body, std::vector<Message>& out, std::string& error) {
    const Json& msgs = body.at("messages");
    if (!msgs.is_array() || msgs.size() == 0) {
        error = "messages must be a non-empty array";
        return false;
    }
    for (size_t i = 0; i < msgs.size(); ++i) {
        const Json& m = msgs[i];
        if (!m.is_object()) {
            error = "messages[" + std::to_string(i) + "] must be an object";
            return false;
        }
        Message msg;
        const std::string role = m.at("role").as_string("user");
        if (role == "system") msg.role = Role::System;
        else if (role == "assistant") msg.role = Role::Assistant;
        else if (role == "tool" || role == "function") msg.role = Role::User;
        else msg.role = Role::User;

        const Json& content = m.at("content");
        if (content.is_string()) {
            msg.content = content.as_string();
        } else if (content.is_array()) {
            // The multimodal content form: [{"type":"text","text":...}, ...]
            for (size_t k = 0; k < content.size(); ++k) {
                const Json& part = content[k];
                if (part.at("type").as_string() == "text") {
                    msg.content += part.at("text").as_string();
                }
            }
        } else {
            msg.content = content.as_string();
        }
        out.push_back(std::move(msg));
    }
    return true;
}

struct ServeState {
    std::unique_ptr<Model>      model;
    Tokenizer                    tokenizer;
    ModelConfig                  cfg;
    std::string                  model_name;
    std::string                  chat_style;
    int                          max_context = 4096;
    GenerationConfig             defaults;
    std::mutex                   mu;      // one request at a time
    std::unique_ptr<Generator>   gen;
    std::atomic<unsigned long long> served{0};
};

void write_json(HttpConn& conn, const Json& j) {
    conn.send_response("application/json", j.dump());
}

void handle_chat(ServeState& st, HttpConn& conn, const HttpRequest& req) {
    Json body;
    std::string perr;
    if (!Json::parse(req.body, body, &perr) || !body.is_object()) {
        write_json(conn, error_json("invalid JSON body: " + perr));
        return;
    }

    std::vector<Message> msgs;
    std::string err;
    if (!extract_messages(body, msgs, err)) {
        write_json(conn, error_json(err));
        return;
    }

    const bool stream = body.at("stream").as_bool(false);
    const int  max_tokens = static_cast<int>(body.at("max_tokens").as_int(st.defaults.max_new_tokens));
    const double temp = body.at("temperature").as_number(st.defaults.sampling.temperature);
    const bool  greedy = body.at("greedy").as_bool(false);
    const std::string model_name = body.at("model").as_string(st.model_name);

    GenerationConfig cfg = st.defaults;
    cfg.max_new_tokens = max_tokens > 0 ? max_tokens : st.defaults.max_new_tokens;
    cfg.sampling.temperature = static_cast<float>(temp);
    cfg.sampling.greedy = greedy;
    if (greedy) cfg.sampling.temperature = 0.0f;
    cfg.sampling.validate();

    const std::string id = make_id("chatcmpl");
    const std::string created = now_iso8601();

    // A system message given by the caller wins over the vocabulary default.
    std::lock_guard<std::mutex> lock(st.mu);

    if (!stream) {
        std::string answer;
        st.gen->reset();
        answer = st.gen->chat(msgs, cfg, nullptr);
        const int ntok = static_cast<int>(st.gen->stats().new_tokens);

        Json choice = Json::object();
        choice.set("index", Json::integer(0));
        Json message = Json::object();
        message.set("role", Json::string("assistant"));
        message.set("content", Json::string(answer));
        choice.set("message", message);
        choice.set("finish_reason", Json::string(ntok >= cfg.max_new_tokens ? "length" : "stop"));
        Json choices = Json::array();
        choices.push(choice);

        Json usage = Json::object();
        usage.set("prompt_tokens", Json::integer(st.gen->stats().prompt_tokens));
        usage.set("completion_tokens", Json::integer(ntok));
        usage.set("total_tokens", Json::integer(st.gen->stats().prompt_tokens + ntok));

        Json root = Json::object();
        root.set("id", Json::string(id));
        root.set("object", Json::string("chat.completion"));
        root.set("created", Json::string(created));
        root.set("model", Json::string(model_name));
        root.set("choices", choices);
        root.set("usage", usage);
        st.served.fetch_add(1);
        write_json(conn, root);
        return;
    }

    // Streaming
    if (!conn.begin_stream("text/event-stream")) return;

    Json head = Json::object();
    head.set("id", Json::string(id));
    head.set("object", Json::string("chat.completion.chunk"));
    head.set("created", Json::string(created));
    head.set("model", Json::string(model_name));

    auto emit = [&](const Json& chunk) {
        conn.send_chunk("data: " + chunk.dump() + "\n\n");
    };

    st.gen->reset();
    {
        Json chunk = head;
        Json choices = Json::array();
        Json choice = Json::object();
        choice.set("index", Json::integer(0));
        Json delta = Json::object();
        delta.set("role", Json::string("assistant"));
        delta.set("content", Json::string(""));
        choice.set("delta", delta);
        choice.set("finish_reason", Json::null());
        choices.push(choice);
        chunk.set("choices", choices);
        emit(chunk);
    }

    std::string pending;
    const Generator::StreamFn on_piece = [&](const std::string& piece, i32) {
        pending += piece;
        // Flush on a UTF-8 boundary so the client never receives half a
        // multi-byte character.
        size_t cut = pending.size();
        while (cut > 0 && (static_cast<u8>(pending[cut - 1]) & 0xC0) == 0x80) --cut;
        if (cut == 0) return true;
        if (cut < pending.size() && (static_cast<u8>(pending[cut - 1]) & 0xC0) == 0x80) {
            // The last byte starts a sequence: keep it for the next piece.
            while (cut > 0 && (static_cast<u8>(pending[cut - 1]) & 0xC0) == 0x80) --cut;
            if (cut > 0) --cut;
        }
        if (cut == 0) return true;
        const std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        if (out.empty()) return true;

        Json chunk = head;
        Json choices = Json::array();
        Json choice = Json::object();
        choice.set("index", Json::integer(0));
        Json delta = Json::object();
        delta.set("content", Json::string(out));
        choice.set("delta", delta);
        choice.set("finish_reason", Json::null());
        choices.push(choice);
        chunk.set("choices", choices);
        emit(chunk);
        return true;
    };

    st.gen->chat(msgs, cfg, on_piece);
    if (!pending.empty()) {
        Json chunk = head;
        Json choices = Json::array();
        Json choice = Json::object();
        choice.set("index", Json::integer(0));
        Json delta = Json::object();
        delta.set("content", Json::string(pending));
        choice.set("delta", delta);
        choice.set("finish_reason", Json::null());
        choices.push(choice);
        chunk.set("choices", choices);
        emit(chunk);
    }

    {
        Json chunk = head;
        Json choices = Json::array();
        Json choice = Json::object();
        choice.set("index", Json::integer(0));
        choice.set("delta", Json::object());
        choice.set("finish_reason", Json::string("stop"));
        choices.push(choice);
        chunk.set("choices", choices);
        emit(chunk);
    }
    conn.send_chunk("data: [DONE]\n\n");
    conn.end_stream();
    st.served.fetch_add(1);
}

}  // namespace

int cmd_serve(const Args& args) {
    const std::string host = args.str("host", "127.0.0.1");
    const int port = args.num_int("port", 8080);

    std::string serr;
    if (!http_socket_init(&serr)) {
        log_error("socket init failed: " + serr);
        return 1;
    }

    LoadedModel lm = load_model(args);
    lm.model->set_rope_runtime(1.0f, 0.0f);

    ServeState st;
    st.model = std::move(lm.model);
    st.tokenizer = std::move(lm.tokenizer);
    st.cfg = lm.cfg;
    st.model_name = lm.path;
    st.chat_style = chat_style_name(detect_chat_style(st.tokenizer));
    st.defaults = make_gen_config(args);
    st.max_context = args.num_int("ctx", 4096);
    if (st.max_context <= 0 || st.max_context > st.cfg.max_seq_len) {
        st.max_context = st.cfg.max_seq_len;
    }
    st.defaults.max_context = st.max_context;
    st.gen = std::make_unique<Generator>(*st.model, st.tokenizer, st.max_context);

    log_info(strfmt("[serve] model %s", st.model_name.c_str()));
    log_info(strfmt("[serve] chat style %s | context %d | vocab %d",
                    st.chat_style.c_str(), st.max_context, st.tokenizer.vocab_size()));

    HttpServer server;
    if (!server.start(host, port, [&st](HttpConn& conn, const HttpRequest& req) {
            if (req.path == "/health" || req.path == "/healthz") {
                Json root = Json::object();
                root.set("status", Json::string("ok"));
                root.set("model", Json::string(st.model_name));
                root.set("chat_style", Json::string(st.chat_style));
                root.set("served", Json::integer(static_cast<i64>(st.served.load())));
                write_json(conn, root);
                return;
            }
            if (req.path == "/v1/models" && req.method == "GET") {
                Json one = Json::object();
                one.set("id", Json::string(st.model_name));
                one.set("object", Json::string("model"));
                one.set("owned_by", Json::string("ghassan-ai"));
                Json list = Json::array();
                list.push(one);
                Json root = Json::object();
                root.set("object", Json::string("list"));
                root.set("data", list);
                write_json(conn, root);
                return;
            }
            if (req.path == "/v1/chat/completions" || req.path == "/v1/completions") {
                if (req.method != "POST") {
                    write_json(conn, error_json("method not allowed", "method_not_allowed"));
                    return;
                }
                handle_chat(st, conn, req);
                return;
            }
            Json root = error_json("unknown endpoint: " + req.path, "not_found");
            write_json(conn, root);
        }, &serr)) {
        log_error("cannot start the server: " + serr);
        http_socket_shutdown();
        return 1;
    }

    log_info(strfmt("[serve] listening on http://%s:%d  (POST /v1/chat/completions)",
                    host.c_str(), server.port()));
    server.serve_forever();
    http_socket_shutdown();
    return 0;
}

}  // namespace gai
