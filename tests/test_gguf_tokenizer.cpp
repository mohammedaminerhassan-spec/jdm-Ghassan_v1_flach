// Verifies that a self-contained GGUF's embedded tokenizer can be loaded
// through tools/model_loader.h (load_tokenizer_auto), and that the thinking
// control tokens round-trip so chat_template_from_vocab(thinking=true)
// resolves to GhassanThinking.
#include "format/gguf_codec.h"
#include "tools/model_loader.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace gai;

namespace {
std::string tmp_path() {
    return "test_gguf_tokenizer_" + std::to_string(static_cast<unsigned long long>(std::clock())) + ".gguf";
}
}  // namespace

int main() {
    const std::string path = tmp_path();

    // A minimal byte-level vocabulary with the Granite role tags and the
    // three thinking control tokens, matching the real GGUF's layout.
    std::vector<std::string> vocab;
    for (int i = 0; i < 256; ++i) vocab.push_back(gpt2_byte_token(static_cast<u8>(i)));
    auto add = [&](const std::string& s) { vocab.push_back(s); return static_cast<i32>(vocab.size()) - 1; };
    const i32 role_s = add("<|start_of_role|>");
    const i32 role_e = add("<|end_of_role|>");
    const i32 turn_e = add("<|end_of_text|>");
    const i32 think_on = add("<think_on>");
    const i32 think_open = add("<think>");
    const i32 think_close = add("</think>");

    {
        GGUFWriter w(path);
        w.set_arch("ghassan");
        w.set_vocab_size(static_cast<uint32_t>(vocab.size()));
        w.set_tokenizer_model("gpt2");
        w.set_tokenizer_pre("ghassan");
        w.set_tokenizer_tokens(vocab);
        w.set_tokenizer_scores(std::vector<float>(vocab.size(), 0.0f));
        // Mark the control tokens as special (3) so the reader recovers them.
        std::vector<int32_t> types(vocab.size(), 1);
        types[role_s] = 3; types[role_e] = 3; types[turn_e] = 3;
        types[think_on] = 3; types[think_open] = 3; types[think_close] = 3;
        w.set_tokenizer_token_types(types);
        w.set_tokenizer_bos_id(0);
        w.set_tokenizer_eos_id(static_cast<uint32_t>(turn_e));
        w.set_quantization_profile("fp16");
        w.write();
    }

    int failures = 0;
    {
        Tokenizer tk;
        std::vector<i32> eos;
        bool ok = load_tokenizer_auto(path, tk, eos);
        std::cout << "load_tokenizer_auto(gguf): " << ok << "\n";
        if (!ok) { ++failures; }
        else {
            std::cout << "vocab=" << tk.vocab_size()
                      << " think_on=" << tk.token_to_id("<think_on>")
                      << " think=" << tk.token_to_id("<think>")
                      << " think_close=" << tk.token_to_id("</think>")
                      << " role_start=" << tk.token_to_id("<|start_of_role|>") << "\n";
            if (tk.token_to_id("<think_on>") != think_on ||
                tk.token_to_id("<think>") != think_open ||
                tk.token_to_id("</think>") != think_close) {
                std::cerr << "thinking token ids did not round-trip\n"; ++failures;
            }
            const ChatTemplate ct = chat_template_from_vocab(tk, true);
            if (ct != ChatTemplate::GhassanThinking) {
                std::cerr << "thinking template not selected from GGUF tokenizer\n"; ++failures;
            } else {
                std::cout << "chat_template(thinking)=ghassan-thinking\n";
            }
            if (eos.empty() || eos[0] != turn_e) {
                std::cerr << "eos id did not round-trip\n"; ++failures;
            }
        }
    }

    std::remove(path.c_str());
    if (failures == 0) std::cout << "test_gguf_tokenizer: PASS\n";
    return failures == 0 ? 0 : 1;
}
