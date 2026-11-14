#include "tools/cli_common.h"
#include "tools/model_loader.h"
#include "core/device.h"
#include "core/ops.h"
#include "format/gguf_codec.h"
#include "format/hf_checkpoint.h"
#include "format/hf_writer.h"
#include "inference/chat.h"
#include "evaluation/perplexity.h"
#include "quantization/quantize.h"
#include "training/checkpoint.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <chrono>
#include <cstdlib>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace gai;

static void apply_eos(GenerationConfig& g, const LoadedModel& lm) {
    if (g.stop_tokens.empty() && !lm.eos_ids.empty()) g.stop_tokens = lm.eos_ids;
}

static void usage() {
    std::cout <<
    "ghassan-ai - Ghassan language model engine\n\n"
    "usage:\n"
    "  ghassan-ai chat       --model <hf-dir|model.gguf>\n"
    "  ghassan-ai generate   --model <hf-dir|model.gguf> --prompt \"hello\"\n"
    "  ghassan-ai info       --model <model.gguf|hf-dir>\n"
    "  ghassan-ai quantize   --model <hf-dir|model.gguf> --out out.gguf [--profile q8_k|q4_0]\n"
    "                         [--model-name \"Ghassan v1 Flash\"] [--compat native|llama|llama_moe]\n"
    "  ghassan-ai bench      --model <model.gguf|hf-dir> [--tokens 64]\n"
    "  ghassan-ai tokenize   --model <hf-dir|model.gguf> --text \"salam\"\n"
    "                          [--ids \"1 2 3\"]  decode an explicit token sequence\n"
    "  ghassan-ai tokenizer-info --tokenizer <hf-dir|tokenizer.json>\n"
    "  ghassan-ai export     --checkpoint last.ckpt --tokenizer <hf-dir|tokenizer.json> --out model.gguf\n"
    "                          [--model-name \"Ghassan v1 Flash\"] [--profile q8_k|q4_0|fp16]\n"
    "  ghassan-ai export     --checkpoint last.ckpt --tokenizer <hf-dir> --out <dir>\n"
    "                          --format safetensors [--dtype f16|f32|bf16]\n"
    "                          # a HuggingFace repository: model.safetensors + config.json\n"
    "                          # + the tokenizer files (the weights AND the vocab)\n"
    "  ghassan-ai perplexity --model <model.gguf|hf-dir> (--text \"held out\" | --shards dir --prefix val_)\n"
    "  ghassan-ai devices\n"
    "  ghassan-ai logits     --model <model.gguf|hf-dir> --prompt \"hi\" [--out logits.f32]\n"
    "                          [--ids \"1 2 3\"]  an explicit token sequence (no tokenizer)\n\n"
    "formats:\n"
    "  Serve-time formats: HuggingFace checkpoint directory (config.json + safetensors\n"
    "  + tokenizer.json) or a self-contained .gguf. Training checkpoints live in .ckpt.\n\n"
    "generation options:\n"
    "  --temp <f>        temperature            [0.8]\n"
    "  --top-k <n>       top-k                  [40]\n"
    "  --top-p <f>       nucleus                [0.92]\n"
    "  --min-p <f>       min-p                  [0.05]\n"
    "  --repeat <f>      repetition penalty     [1.12]\n"
    "  --no-repeat-ngram <n>  ban seen n-grams  [0=off]\n"
    "  --max-tokens <n>  max new tokens         [256]\n"
    "  --seed <n>        rng seed (0 = random)\n"
    "  --greedy          deterministic decoding\n"
    "  --no-fast-sample  disable GPU fast sampling (legacy full-vocab path)\n"
    "  --system <text>   system prompt (default: none)\n"
    "  --no-stream       print the reply at once\n"
    "  --print-ids      print the generated token ids (verification)\n"
    "  --device auto|cpu|cuda  (metal/vulkan fall back to portable CPU)\n"
    "  --gemm-fp16 0|1        CUDA tensor-core GEMMs (default 1; 0 = pure fp32)\n"
    "  --no-fp16-cache        disable persistent FP16 inference weights\n"
    "  --weight-f16 0|1      store the big weights in half precision on the device\n"
    "                         (default 1 on the GPU: 12.9 GiB instead of 25.9 GiB for\n"
    "                         a 6.94B Granite; 0 on the CPU, the reference path)\n"
    "  --ctx <n>              KV cache context size (default: model max_seq_len)\n";
}
static std::string ggml_type_str(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        default: return strfmt("t%u", static_cast<unsigned>(t));
    }
}

static u64 ggml_nbytes_of(GGMLType t, u64 nelems) {
    switch (t) {
        case GGMLType::F32:  return nelems * 4;
        case GGMLType::F16:  return nelems * 2;
        case GGMLType::Q8_0: return ((nelems + 31) / 32) * 34;
        case GGMLType::Q4_0: return ((nelems + 31) / 32) * 18;
        default:             return nelems * 4;
    }
}

static int cmd_chat(const Args& args) {
    LoadedModel lm = load_model(args);
    ChatOptions opts;
    opts.gen = make_gen_config(args);
    opts.gen.max_context = args.num_int("ctx", 0);
    GAI_CHECK(opts.gen.max_context >= 0, "--ctx must be >= 0");
    apply_eos(opts.gen, lm);
    Generator gen(*lm.model, lm.tokenizer, opts.gen.max_context);
    opts.stream = !args.flag("no-stream");
    opts.show_stats = args.flag("stats");
    if (args.has("system")) opts.system = args.str("system");

    ChatSession session(gen, opts);
    session.run_repl();
    return 0;
}

static int cmd_generate(const Args& args) {
    LoadedModel lm = load_model(args);
    GenerationConfig g = make_gen_config(args);
    g.max_context = args.num_int("ctx", 0);
    GAI_CHECK(g.max_context >= 0, "--ctx must be >= 0");
    apply_eos(g, lm);
    Generator gen(*lm.model, lm.tokenizer, g.max_context);

    std::string prompt = args.str("prompt");
    if (prompt.empty() && !args.positional().empty() && args.positional().size() > 1) {
        prompt = args.positional()[1];
    }
    GAI_CHECK(!prompt.empty(), "--prompt is required");

    bool stream = !args.flag("no-stream");
    const bool print_ids = args.flag("print-ids", false);
    std::vector<i32> gen_ids;
    auto on_piece = [&](const std::string& p, i32 id) {
        if (stream) std::cout << p << std::flush;
        if (print_ids) gen_ids.push_back(id);
        return true;
    };
    Generator::StreamFn on_tok =
        (stream || print_ids) ? Generator::StreamFn(on_piece) : nullptr;

    std::string out;
    if (args.flag("raw")) {
        out = gen.complete(prompt, g, on_tok);
    } else {
        std::vector<Message> msgs;
        const std::string sys = args.str("system", "");
        if (!sys.empty()) msgs.push_back({Role::System, sys});
        msgs.push_back({Role::User, prompt});
        if (args.flag("dump-prompt", false)) {
            // The chat format follows the vocabulary (Granite role tags vs
            // ChatML).
            const ChatTemplate tmpl =
                (lm.tokenizer.token_to_id("<|start_of_role|>") >= 0 &&
                 lm.tokenizer.token_to_id("<|end_of_role|>") >= 0)
                    ? ChatTemplate::Granite : ChatTemplate::ChatML;
            const std::vector<i32> ids = encode_chat(lm.tokenizer, tmpl, msgs, true, nullptr);
            log_info(strfmt("[prompt] %s | %d tokens",
                            tmpl == ChatTemplate::Granite ? "granite" : "chatml",
                            static_cast<int>(ids.size())));
            log_info("[prompt] " + lm.tokenizer.decode(ids, true));
        }
        out = gen.chat(msgs, g, on_tok);
    }
    if (stream) std::cout << std::endl;
    else std::cout << out << std::endl;
    if (print_ids) {
        std::cout << "  ids     : ";
        for (size_t i = 0; i < gen_ids.size(); ++i) {
            if (i) std::cout << " ";
            std::cout << gen_ids[i];
        }
        std::cout << "\n";
    }

    if (args.flag("stats")) std::cerr << "  " << gen.stats().summary() << "\n";
    return 0;
}

static int cmd_info(const Args& args) {
    std::string path = args.str("model");
    GAI_CHECK(!path.empty(), "--model is required");

    if (is_gguf_file(path)) {
        GGUFReader r;
        GAI_CHECK(r.open(path), "not a valid .gguf file: " + path);

        std::cout << "\n  file        : " << path << "\n";
        std::cout << "  size        : " << human_bytes(r.file_size()) << "\n";
        std::cout << "  format      : GGUF\n";
        std::cout << "  quant       : " << r.get_string("ghassan.quantization.profile",
                     r.get_string("general.quantization_version", "?")) << "\n";
        std::cout << "  arch        : " << r.get_string("general.architecture", "?") << "\n";
        std::cout << "  tokenizer   : " << (r.has_tokenizer() ? "embedded (self-contained)" : "external") << "\n";
        std::cout << "\n  -- config --\n";
        std::cout << r.model_config().summary() << "\n";

        std::cout << "\n  -- tensors (" << r.tensors().size() << ") --\n";
        std::map<std::string, std::pair<u64, u64>> by_dtype;
        u64 total_bytes = 0, total_params = 0;
        for (const auto& e : r.tensors()) {
            i64 n = 1;
            for (u64 d : e.dimensions) n *= static_cast<i64>(d);
            total_params += static_cast<u64>(n);
            u64 nb = ggml_nbytes_of(e.type, static_cast<u64>(n));
            total_bytes += nb;
            auto& slot = by_dtype[ggml_type_str(e.type)];
            slot.first++;
            slot.second += nb;
        }
        for (const auto& [dt, s] : by_dtype)
            std::cout << strfmt("    %-6s %4llu tensors  %s\n", dt.c_str(),
                                (unsigned long long)s.first, human_bytes(s.second).c_str());
        std::cout << strfmt("\n  parameters  : %s (%.2f M)\n",
                            human_count(total_params).c_str(), double(total_params) / 1e6);
        std::cout << strfmt("  weight bytes: %s (%.2f bits/param)\n",
                            human_bytes(total_bytes).c_str(),
                            total_params ? 8.0 * double(total_bytes) / double(total_params) : 0.0);

        if (args.flag("tensors")) {
            std::cout << "\n  -- tensor list --\n";
            for (const auto& e : r.tensors()) {
                std::string dims;
                for (size_t i = 0; i < e.dimensions.size(); ++i) {
                    if (i) dims += "x";
                    dims += std::to_string(e.dimensions[i]);
                }
                i64 n = 1;
                for (u64 d : e.dimensions) n *= static_cast<i64>(d);
                std::cout << strfmt("    %-28s %-14s %-6s %10s\n", e.name.c_str(), dims.c_str(),
                                    ggml_type_str(e.type).c_str(),
                                    human_bytes(ggml_nbytes_of(e.type, static_cast<u64>(n))).c_str());
            }
        }
        std::cout << "\n";
        return 0;
    }

    // HuggingFace checkpoint directory
    if (fs::is_directory(path)) {
        const bool config_only = args.flag("config-only", false);
        HFCheckpoint hf;
        GAI_CHECK(hf.open(path, !config_only),
                  "not a usable HuggingFace checkpoint: " + path);
        const ModelConfig& c = hf.config();
        std::cout << "\n  dir         : " << path << "\n";
        std::cout << "  format      : HuggingFace safetensors\n";
        std::cout << "  weights     : "
                  << (config_only ? std::string("(not loaded: --config-only)")
                                  : strfmt("%zu file(s)", hf.weight_files().size()))
                  << "\n";
        std::cout << "\n  -- config --\n";
        std::cout << c.summary() << "\n";
        const std::string gs = c.hybrid_summary();
        if (!gs.empty()) std::cout << "\n  -- hybrid --\n" << gs << "\n";
        std::cout << "\n  parameters  : " << human_count(Model::count_parameters(c))
                  << " (" << Model::count_parameters(c) << ")\n";
        std::cout << "  tie_embeddings: " << (c.tie_embeddings ? "yes" : "no") << "\n\n";
        return 0;
    }

    GAI_CHECK(false, "not a valid model: " + path);
}

static int cmd_quantize(const Args& args) {
    std::string in = args.str("model");
    std::string out = args.str("out");

    std::string prof = args.str("profile", "q8_k");
    GAI_CHECK(!in.empty() && !out.empty(), "--model and --out are required");

    const bool in_gguf = is_gguf_file(in);
    ModelConfig cfg;
    std::string tok_path = args.str("tokenizer");
    Tokenizer embedded_tok;
    bool have_embedded = false;

    if (in_gguf) {
        GGUFReader r;
        GAI_CHECK(r.open(in), "not a valid .gguf file: " + in);
        cfg = r.model_config();
        if (tok_path.empty() && r.has_tokenizer()) {
            Tokenizer tk;
            GAI_CHECK(r.load_tokenizer(tk), "GGUF tokenizer payload corrupt; pass --tokenizer");
            embedded_tok = std::move(tk);
            have_embedded = true;
        }
    } else if (fs::is_directory(in)) {
        HFCheckpoint hf;
        GAI_CHECK(hf.open(in), "not a usable HuggingFace checkpoint: " + in);
        cfg = hf.config();
        if (tok_path.empty()) {
            GAI_CHECK(hf.load_tokenizer(embedded_tok),
                      "cannot build the tokenizer from " + in + "/tokenizer.json");
            have_embedded = true;
        }
    } else {
        GAI_CHECK(false, "not a valid model for quantize: " + in);
    }
    GAI_CHECK(!tok_path.empty() || have_embedded,
              "--tokenizer is required when the input has no embedded tokenizer");

    Model model(cfg, Device::CPU);
    if (in_gguf) {
        GAI_CHECK(load_model_from_gguf(in, model), "failed to load weights from: " + in);
    } else {
        HFCheckpoint hf;
        GAI_CHECK(hf.open(in), "cannot reopen HuggingFace checkpoint: " + in);
        GAI_CHECK(hf.load_model(model), "failed to load the weights of: " + in);
    }

    log_info(strfmt("[quant] %s -> %s   profile=%s (GGUF, 8-bit max)", in.c_str(), out.c_str(), prof.c_str()));

    // The --model-name flag is honored by --compat exports; wire it up here
    // too, or the artifact ships under the auto-generated label instead of
    // the one the user asked for.
    std::map<std::string, std::string> meta;
    if (args.has("model-name")) meta["model_name"] = args.str("model-name");

    const ExportProfileGGUF profile = gguf_profile_for(prof);
    if (have_embedded) {
        export_model_gguf(out, model, embedded_tok, profile, meta, args.str("compat", "native"));
    } else {
        Tokenizer tk;
        std::vector<i32> eos_ids;
        GAI_CHECK(load_tokenizer_auto(args.str("tokenizer"), tk, eos_ids),
                  "cannot load tokenizer: " + args.str("tokenizer"));
        export_model_gguf(out, model, tk, profile, meta, args.str("compat", "native"));
    }
    return 0;
}

static int cmd_bench(const Args& args) {
    LoadedModel lm = load_model(args);
    const int bench_ctx = args.num_int("ctx", 512);
    const int ntok = args.num_int("tokens", 64);
    const int nprompt = args.num_int("prompt-tokens", 32);
    GAI_CHECK(bench_ctx >= 0, "--ctx must be >= 0");
    GAI_CHECK(ntok >= 0 && ntok <= 1000000, "--tokens must be in [0,1000000]");
    GAI_CHECK(nprompt > 0, "--prompt-tokens must be > 0");
    Generator gen(*lm.model, lm.tokenizer, bench_ctx);

    const int vocab = lm.model->config().vocab_size;
    std::vector<i32> prompt;
    prompt.push_back(lm.tokenizer.encode("hello", false, false).front());
    for (int i = 1; i < nprompt; ++i) prompt.push_back(static_cast<i32>((300 + (i * 37)) % vocab));

    log_info(strfmt("[bench] prefill %d tokens, decode %d tokens on %s",
                    nprompt, ntok, device_name(lm.model->device())));

    gen.reset();
    gen.prefill(prompt);

    GenerationConfig g;
    g.max_new_tokens = ntok;
    g.max_context = bench_ctx;
    g.stop_tokens.clear();
    g.sampling.greedy = true;
    gen.generate(g);

    const GenerationStats& s = gen.stats();
    std::cout << "\n  ---------------- benchmark ----------------\n";
    std::cout << strfmt("  device        : %s\n", device_name(lm.model->device()));
    std::cout << strfmt("  quantization  : %s\n", lm.quant.c_str());
    std::cout << strfmt("  parameters    : %.2f M\n", double(lm.model->num_parameters()) / 1e6);
    std::cout << strfmt("  prefill       : %d tok in %s = %.1f tok/s\n",
                        s.prompt_tokens, human_duration(s.prefill_sec).c_str(), s.prefill_tps());
    std::cout << strfmt("  decode        : %d tok in %s = %.1f tok/s\n",
                        s.new_tokens, human_duration(s.decode_sec).c_str(), s.decode_tps());
    std::cout << strfmt("  ms per token  : %.2f\n",
                        s.new_tokens ? 1000.0 * s.decode_sec / s.new_tokens : 0.0);
    std::cout << strfmt("  kv cache      : %s\n", human_bytes(gen.cache().bytes()).c_str());
    std::cout << "  -------------------------------------------\n\n";
    return 0;
}

static int cmd_tokenizer_info(const Args& args) {
    std::string tp = args.str("tokenizer");
    GAI_CHECK(!tp.empty(), "--tokenizer is required");
    Tokenizer tk;
    std::vector<i32> eos_ids;
    GAI_CHECK(load_tokenizer_auto(tp, tk, eos_ids), "cannot load tokenizer: " + tp);

    std::cout << "path=" << tp << "\n";
    std::cout << "vocab_size=" << tk.vocab_size() << "\n";
    std::cout << "byte_map=" << (tk.has_byte_map() ? 1 : 0) << "\n";
    std::cout << "special_tokens=" << tk.special_ids().size() << "\n";
    std::cout << "add_bos=" << (tk.add_bos() ? 1 : 0) << "\n";
    std::cout << "add_eos=" << (tk.add_eos() ? 1 : 0) << "\n";
    std::cout << "eos_id=" << tk.eos_id() << "\n";
    return 0;
}

static int cmd_tokenize(const Args& args) {
    Tokenizer tk;
    std::string tp = args.str("tokenizer");
    if (tp.empty() && args.has("model")) {
        std::string mp = args.str("model");
        std::vector<i32> eos_ids;
        if (is_gguf_file(mp)) {
            GGUFReader gr;
            GAI_CHECK(gr.open(mp), "cannot open model");
            GAI_CHECK(gr.has_tokenizer() && gr.load_tokenizer(tk),
                      "model has no embedded tokenizer; pass --tokenizer");
        } else {
            GAI_CHECK(load_tokenizer_auto(mp, tk, eos_ids), "cannot load tokenizer from " + mp);
        }
    } else {
        GAI_CHECK(!tp.empty(), "--tokenizer or --model is required");
        std::vector<i32> eos_ids;
        GAI_CHECK(load_tokenizer_auto(tp, tk, eos_ids), "cannot load tokenizer: " + tp);
    }

    if (args.has("batch")) {
        // Batch mode: the file holds [u32 length][utf-8 bytes] records.
        std::ifstream bf(args.str("batch"), std::ios::binary);
        GAI_CHECK(bf.good(), "cannot open batch file: " + args.str("batch"));
        while (true) {
            u32 len = 0;
            bf.read(reinterpret_cast<char*>(&len), 4);
            if (!bf) break;
            std::string s(len, '\0');
            if (len) bf.read(s.data(), len);
            const std::vector<i32> bids = tk.encode_default(s);
            std::cout << "{\"text\":\"" << json_escape(s) << "\",\"ids\":[";
            for (size_t i = 0; i < bids.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << bids[i];
            }
            std::cout << "]}\n";
        }
        return 0;
    }
    std::string text = args.str("text");
    if (text.empty() && args.positional().size() > 1) text = args.positional()[1];

    if (args.has("ids")) {
        // Decode an explicit token sequence (no tokenizer round trip).
        std::vector<i32> ids;
        const std::string spec = args.str("ids");
        size_t pos = 0;
        while (pos < spec.size()) {
            size_t end = spec.find_first_of(", ", pos);
            if (end == std::string::npos) end = spec.size();
            const std::string tok = spec.substr(pos, end - pos);
            if (!tok.empty()) ids.push_back(std::atoi(tok.c_str()));
            pos = end + 1;
        }
        GAI_CHECK(!ids.empty(), "--ids is empty");
        std::cout << "  ids     : ";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << " ";
            std::cout << ids[i];
        }
        std::cout << "\n  pieces  : ";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << " | ";
            std::cout << tk.token_text(ids[i]);
        }
        std::cout << "\n  decoded : " << tk.decode(ids) << "\n";
        return 0;
    }

    GAI_CHECK(!text.empty(), "--text is required");

    auto ids = tk.encode_default(text);
    if (args.flag("json", false)) {
        std::cout << "{\"text\":\"" << json_escape(text) << "\",\"ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << ids[i];
        }
        std::cout << "],\"pieces\":[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << "\"" << json_escape(tk.token_text(ids[i])) << "\"";
        }
        std::cout << "],\"decoded\":\"" << json_escape(tk.decode(ids, true)) << "\"}\n";
        return 0;
    }
    std::cout << "  input   : " << text << "\n";
    std::cout << "  tokens  : " << ids.size() << "\n  pieces  : ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << " | ";
        std::cout << tk.token_text(ids[i]);
    }
    std::cout << "\n  ids     : ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << " ";
        std::cout << ids[i];
    }
    std::cout << "\n  decoded : " << tk.decode(ids) << "\n";
    return 0;
}

static int cmd_export(const Args& args) {
    std::string ckpt = args.str("checkpoint");
    std::string out  = args.str("out");
    GAI_CHECK(!ckpt.empty() && !out.empty(), "--checkpoint and --out are required");

    ModelConfig cfg;
    TrainState st;
    GAI_CHECK(Checkpoint::peek(ckpt, cfg, st), "cannot read checkpoint: " + ckpt);
    log_info(strfmt("[export] checkpoint at step %lld, val %.4f",
                    static_cast<long long>(st.step), st.best_val));

    Model model(cfg, Device::CPU);
    TrainState loaded;
    GAI_CHECK(Checkpoint::load(ckpt, model, static_cast<AdamW*>(nullptr), loaded),
              "cannot load checkpoint weights");
    model.print_parameter_report();

    std::string prof = args.str("profile", "q8_k");
    std::string compat = args.str("compat", "native");
    std::string tok_path = args.str("tokenizer");
    GAI_CHECK(!tok_path.empty(), "--tokenizer is required for GGUF export (vocab is embedded)");
    Tokenizer tok;
    std::vector<i32> eos_ids;
    GAI_CHECK(load_tokenizer_auto(tok_path, tok, eos_ids),
              "cannot load tokenizer: " + tok_path);

    // ---- the HuggingFace safetensors export ----
    //
    // The engine could read a HuggingFace checkpoint but could only write a
    // GGUF, so a model trained here could not be published. This writes the
    // safetensors (plus the config.json) of a real HF repository.
    const std::string format = args.str("format", "gguf");
    if (format == "safetensors" || format == "hf") {
        const std::string name = args.str("model-name", "Ghassan v1 Flash");
        const std::string desc = args.str("description", name);
        const std::string dt = args.str("dtype", "f16");
        SafeDType sd = SafeDType::F16;
        if (dt == "f32" || dt == "float32") sd = SafeDType::F32;
        else if (dt == "bf16" || dt == "bfloat16") sd = SafeDType::BF16;
        GAI_CHECK(export_model_safetensors(out, model, tok, name, desc, sd),
                  "the safetensors export failed");
        // The tokenizer files are a property of the vocabulary, not of the
        // weights: they are copied verbatim (re-serialising them would lose
        // the merges and the added tokens).
        copy_tokenizer_files(tok_path, out);
        return 0;
    }
    GAI_CHECK(format == "gguf", "unknown --format '" + format + "' (gguf|safetensors)");

    std::map<std::string, std::string> meta = {
        {"steps", std::to_string(loaded.step)},
        {"val_loss", strfmt("%.6f", loaded.best_val)},
    };
    if (args.has("model-name")) meta["model_name"] = args.str("model-name");
    export_model_gguf(out, model, tok, gguf_profile_for(prof), meta, compat);
    return 0;
}

static int cmd_perplexity(const Args& args) {
    LoadedModel lm = load_model(args);
    const int ctx = args.num_int("ctx", 0);
    const int seq_len = args.num_int("seq-len", 512);
    const int batch_size = args.num_int("batch-size", 1);
    const int max_batches = args.num_int("batches", 20);
    GAI_CHECK(ctx >= 0 && seq_len > 0 && batch_size > 0 && max_batches > 0,
              "perplexity options must be positive");
    if (args.has("rope-scale") || args.has("yarn-mscale")) {
        float scale = args.has("rope-scale")
            ? static_cast<float>(args.real_strict("rope-scale")) : lm.model->config().rope_scale;
        float yarn = args.has("yarn-mscale")
            ? static_cast<float>(args.real_strict("yarn-mscale")) : lm.model->config().rope_yarn_mscale;
        lm.model->set_rope_runtime(scale, yarn);
    }
    Generator gen(*lm.model, lm.tokenizer, ctx);
    PerplexityResult result;
    if (args.has("text")) {
        std::vector<i32> tokens = lm.tokenizer.encode(args.str("text"), lm.tokenizer.add_bos(), false);
        result = evaluate_perplexity(gen, tokens);
    } else if (args.has("shards")) {
        result = evaluate_shard_perplexity(
            gen, args.str("shards"), args.str("prefix", "val"), batch_size,
            seq_len, max_batches, args.flag("pack", false),
            static_cast<u64>(args.num("seed", 42)));
    } else {
        GAI_FAIL("perplexity requires --text or --shards");
    }
    std::cout << strfmt("  rope_scale  : %.3f\n", (double)lm.model->config().rope_scale);
    std::cout << strfmt("  yarn_mscale : %.3f\n", (double)rope_mscale(lm.model->config()));
    std::cout << strfmt("  mean_nll    : %.8f\n", result.mean_nll);
    std::cout << strfmt("  perplexity  : %.6f\n", result.perplexity);
    std::cout << strfmt("  tokens      : %lld\n", static_cast<long long>(result.tokens));
    std::cout << strfmt("  batches     : %lld\n", static_cast<long long>(result.batches));
    return 0;
}

static int cmd_devices(const Args&) {
    print_device_report();
    return 0;
}

static int cmd_logits(const Args& args) {
    LoadedModel lm = load_model(args);
    std::string prompt = args.str("prompt");
    if (prompt.empty() && args.positional().size() > 1) prompt = args.positional()[1];

    std::vector<i32> ids;
    if (args.has("ids")) {
        // An explicit token sequence (no tokenizer round trip): the way to
        // check the decode path against a full prefill of the same ids.
        const std::string spec = args.str("ids");
        size_t pos = 0;
        while (pos < spec.size()) {
            size_t end = spec.find_first_of(", ", pos);
            if (end == std::string::npos) end = spec.size();
            const std::string tok = spec.substr(pos, end - pos);
            if (!tok.empty()) {
                const int v = std::atoi(tok.c_str());
                GAI_CHECK(v >= 0 && v < lm.model->config().vocab_size,
                          strfmt("--ids: %d is out of the vocabulary (0..%d)",
                                 v, lm.model->config().vocab_size - 1));
                ids.push_back(v);
            }
            pos = end + 1;
        }
        GAI_CHECK(!ids.empty(), "--ids is empty");
    } else {
        GAI_CHECK(!prompt.empty(), "--prompt or --ids is required");
        ids = lm.tokenizer.encode(prompt, lm.tokenizer.add_bos(), false);
    }
    const int T = static_cast<int>(ids.size());
    GAI_CHECK(T > 0, "the prompt encodes to zero tokens");

    GAI_CHECK(T <= 512, "logits prompt too long (T>512 would materialize >64MB); "
                        "split the prompt or use score_tokens/bench instead");
    std::cout << strfmt("  tokens: %d  ids:", T);
    for (i32 id : ids) std::cout << " " << id;
    std::cout << "\n";

    Model& m = *lm.model;
    Activations act = m.make_activations(1, T, false);
    Tensor dev_ids({static_cast<i64>(T)}, DType::I32, m.device());
    device_copy(dev_ids.data_ptr(), m.device(), ids.data(), Device::CPU,
                sizeof(i32) * ids.size());
    Tensor& logits = m.forward(dev_ids.i32p(), 1, T, act);
    const int V = m.config().vocab_size;

    std::vector<float> host_logits;
    device_stage_f32_2d(logits.data_ptr(), m.device(), T, V, host_logits);
    const float* L = host_logits.data();

    std::ofstream out;
    if (args.has("out")) {
        out.open(args.str("out"), std::ios::binary);
        GAI_CHECK(out.good(), "cannot write logits file");
    }
    for (int t = 0; t < T; ++t) {
        const float* row = L + static_cast<size_t>(t) * V;
        if (out) out.write(reinterpret_cast<const char*>(row),
                           static_cast<std::streamsize>(V) * 4);

        int top[5] = {-1, -1, -1, -1, -1};
        for (int v = 0; v < V; ++v) {
            for (int k = 0; k < 5; ++k) {
                if (top[k] < 0 || row[v] > row[top[k]]) {
                    for (int j = 4; j > k; --j) top[j] = top[j - 1];
                    top[k] = v;
                    break;
                }
            }
        }
        std::cout << strfmt("  pos %3d argmax=%6d (%.4f) top5:", t, top[0], row[top[0]]);
        for (int k = 0; k < 5; ++k) std::cout << strfmt(" %d:%.3f", top[k], row[top[k]]);
        std::cout << "\n";
    }
    if (out) {
        out.close();
        GAI_CHECK(out.good(), "logits write failed");
        log_info("[logits] wrote " + args.str("out"));
    }
    return 0;
}

int main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);

    std::string cmd = args.command();
    if (cmd.empty() || args.flag("help") || cmd == "help") { usage(); return cmd.empty() ? 1 : 0; }

    try {
        if (cmd == "chat")     return cmd_chat(args);
        if (cmd == "generate") return cmd_generate(args);
        if (cmd == "info")     return cmd_info(args);
        if (cmd == "quantize") return cmd_quantize(args);
        if (cmd == "bench")    return cmd_bench(args);
        if (cmd == "tokenize") return cmd_tokenize(args);
        if (cmd == "tokenizer-info") return cmd_tokenizer_info(args);
        if (cmd == "export")   return cmd_export(args);
        if (cmd == "perplexity") return cmd_perplexity(args);
        if (cmd == "devices")  return cmd_devices(args);
        if (cmd == "logits")   return cmd_logits(args);
        std::cerr << "unknown command: " << cmd << "\n\n";
        usage();
        return 1;
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
