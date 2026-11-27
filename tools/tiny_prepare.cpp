#include "core/json.h"
#include "core/common.h"
#include "tokenizer/tokenizer.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

using namespace gai;

static void build_vocab(Json& vocab, Json& added, int& next_id) {
    bool in_set[256] = {false};
    for (int v = 0x21; v <= 0x7E; ++v) in_set[v] = true;
    for (int v = 0xA1; v <= 0xAC; ++v) in_set[v] = true;
    for (int v = 0xAE; v <= 0xFF; ++v) in_set[v] = true;
    int n = 0;
    next_id = 0;
    for (int b = 0; b < 256; ++b) {
        if (in_set[b]) {
            std::string s;
            if (b < 0x80) s = std::string(1, static_cast<char>(b));
            else {
                s += static_cast<char>(0xC0 | (b >> 6));
                s += static_cast<char>(0x80 | (b & 0x3F));
            }
            vocab.set(s, Json::integer(next_id++));
        } else {
            const int cp = 256 + n++;
            std::string s;
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
            vocab.set(s, Json::integer(next_id++));
        }
    }
    const int eos = next_id++;
    const int pad = next_id++;
    const int unk = next_id++;
    const int bos = next_id++;
    vocab.set("<|endoftext|>", Json::integer(eos));
    vocab.set("<|pad|>", Json::integer(pad));
    vocab.set("<|unk|>", Json::integer(unk));
    vocab.set("<|bos|>", Json::integer(bos));

    auto special = [](int id, const char* content) {
        Json o = Json::object();
        o.set("id", Json::integer(id));
        o.set("content", Json::string(content));
        o.set("special", Json::boolean(true));
        return o;
    };
    added.push(special(eos, "<|endoftext|>"));
    added.push(special(pad, "<|pad|>"));
    added.push(special(unk, "<|unk|>"));
    added.push(special(bos, "<|bos|>"));
}

int main(int argc, char** argv) {
    const std::string out = (argc > 1) ? argv[1] : "/kaggle/working/tok/tokenizer.json";
    const size_t slash = out.find_last_of("/\\");
    if (slash != std::string::npos) {
        std::error_code ec;
        fs::create_directories(out.substr(0, slash), ec);
    }

    Json vocab = Json::object();
    Json added = Json::array();
    int vocab_size = 0;
    build_vocab(vocab, added, vocab_size);

    Json pre = Json::object();
    pre.set("type", Json::string("ByteLevel"));
    pre.set("add_prefix_space", Json::boolean(false));
    pre.set("trim_offsets", Json::boolean(true));

    Json model = Json::object();
    model.set("type", Json::string("BPE"));
    model.set("unk_token", Json::string("<|unk|>"));
    model.set("continuing_subword_prefix", Json::string(""));
    model.set("end_of_word_suffix", Json::string(""));
    model.set("fuse_unk", Json::boolean(false));
    model.set("vocab", vocab);
    model.set("merges", Json::array());

    Json root = Json::object();
    root.set("version", Json::string("1.0"));
    root.set("added_tokens", added);
    root.set("normalizer", Json());
    root.set("pre_tokenizer", pre);
    root.set("post_processor", pre);
    root.set("decoder", pre);
    root.set("model", model);

    std::ofstream f(out, std::ios::binary);
    if (!f.good()) { std::cerr << "cannot write " << out << "\n"; return 1; }
    f << root.dump_pretty();
    f.close();
    if (!f.good()) { std::cerr << "write failed: " << out << "\n"; return 1; }

    std::cout << "wrote " << out << " (vocab=" << vocab_size << ")\n";
    return 0;
}
