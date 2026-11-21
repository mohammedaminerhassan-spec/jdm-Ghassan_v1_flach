#include "tools/model_loader.h"

#include "core/device.h"
#include "core/json.h"
#include "core/ops.h"
#include "format/gguf_codec.h"
#include "format/hf_checkpoint.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>

namespace fs = std::filesystem;

namespace gai {

namespace {

// The model is allocated on the device before a single weight is read, so
// a model that does not fit the GPU dies inside cudaMalloc with a cryptic
// "out of memory". This pre-flight turns that into an actionable message.
void check_vram_budget(const ModelConfig& cfg, Device dev, int ctx, bool fp16_cache_on,
                       bool fp16_weights = false, int quant_kind = -1) {
    if (dev != Device::CUDA) return;
    const DeviceInfo& di = device_info();
    if (!di.cuda_available || di.total_mem == 0) return;

    // Use the requested context if given; otherwise fall back to 4096 (a
    // realistic inference budget).  Using cfg.max_seq_len (131 072 for
    // Granite) as the fallback caused the guard to report an absurdly large
    // KV-cache requirement (~40 GiB) even though inference allocates only a
    // fraction of that, refusing every CUDA run on a T4 when --ctx was
    // omitted.
    const int kv_ctx = ctx > 0 ? ctx : std::min(cfg.max_seq_len, 4096);
    const size_t kv   = Model::kv_cache_bytes(cfg, kv_ctx);
    const size_t mam  = Model::mamba_state_bytes(cfg);
    // The hybrid (Granite) prefills TOKEN BY TOKEN (the recurrent state
    // cannot be batched), so its peak activations are per token, not per
    // sequence. The training-style estimate (B x T x V logits) is 20x too
    // big here: it refused a context that fits easily (the 12.93 GiB of
    // the weights leave 1.6 GiB, and the real usage is a few MB).
    const bool hybrid = (cfg.arch == ModelConfig::ArchGranite);
    const int act_T = hybrid ? 1 : (ctx > 0 ? ctx : 512);
    const Model::MemoryPlan plan2 =
        Model::plan_memory(cfg, 1, act_T, false, 1, fp16_cache_on, fp16_weights, quant_kind);
    const Model::WorkspacePlan wsp = Model::workspace_plan(cfg, 1, act_T);
    // Native Q4 keeps a shared FP16 scratch bank: one layer's worth of
    // dequantized weights is live at a time (~350 MB for Granite tiny),
    // rounded up with headroom.
    const size_t qscratch = quant_kind >= 0 ? (512ull << 20) : 0;
    const size_t need = plan2.weights + plan2.fp16_cache + plan2.activations +
                        kv + mam + wsp.gemm_bytes + wsp.moe_bytes + qscratch;

    log_info(strfmt("[mem ] GPU plan: weights %s | fp16 cache %s | activations %s | "
                    "kv %s | mamba %s | workspaces %s = %s of %s (%.0f%%)%s",
                    human_bytes(plan2.weights).c_str(),
                    human_bytes(plan2.fp16_cache).c_str(),
                    human_bytes(plan2.activations).c_str(),
                    human_bytes(kv).c_str(),
                    human_bytes(mam).c_str(),
                    human_bytes(wsp.gemm_bytes + wsp.moe_bytes).c_str(),
                    human_bytes(need).c_str(),
                    human_bytes(di.total_mem).c_str(),
                    100.0 * static_cast<double>(need) / static_cast<double>(di.total_mem),
                    hybrid ? " [the hybrid prefills token by token]" : ""));

    if (need > di.total_mem) {
        GAI_FAIL(strfmt("the model does not fit this GPU: it needs %s but the card has %s "
                        "(%s of %s weights + %s of activations/KV). "
                        "Options: --device cpu (slower, uses the host RAM), a smaller --ctx, "
                        "--weight-f16 1 (the weights in half precision: %s instead of %s), "
                        "or --no-fp16-cache (frees the fp16 copy).",
                        human_bytes(need).c_str(), human_bytes(di.total_mem).c_str(),
                        human_bytes(plan2.weights).c_str(),
                        fp16_weights ? "half-precision" : "fp32",
                        human_bytes(plan2.activations + kv).c_str(),
                        human_bytes(Model::plan_memory(cfg, 1, act_T, false, 1,
                                                       false, true).weights).c_str(),
                        human_bytes(plan2.weights).c_str()));
    }
}

Device pick_device(const Args& a) {
    const std::string d = a.str("device", "auto");
    if (d == "cpu") return Device::CPU;
    if (d == "cuda") {
        GAI_CHECK(cuda_available(), "--device cuda requested but no CUDA device is available (use --device cpu)");
        return Device::CUDA;
    }
    if (d == "metal" || d == "vulkan" || d == "tpu") {
        log_warn("--device " + d + " has no native backend; falling back to CPU "
                 "(portable OpenMP kernels; use --device cuda on NVIDIA GPUs)");
        return Device::CPU;
    }
    if (d != "auto") GAI_FAIL("unknown --device '" + d + "' (auto|cpu|cuda|metal|vulkan)");
    return best_device();
}

}  // namespace

bool is_gguf_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    u32 magic = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), 4) || f.gcount() != 4) return false;
    return magic == GGUF_MAGIC;
}

bool load_tokenizer_auto(const std::string& path, Tokenizer& out,
                         std::vector<i32>& eos_out) {
    eos_out.clear();
    std::error_code ec;

    // GGUF mode: a self-contained artifact carries its tokenizer inline.
    // Without this branch the binary GGUF is handed to the JSON parser and
    // loading fails, which blocks fine-tuning / export when the only
    // tokenizer available is the one embedded in the model itself.
    if (is_gguf_file(path)) {
        GGUFReader r;
        if (!r.open(path)) return false;
        if (!r.has_tokenizer()) return false;
        if (!r.load_tokenizer(out)) return false;
        eos_out = r.eos_ids();
        if (eos_out.empty()) eos_out.push_back(out.eos_id());
        return true;
    }

    // Directory mode: the tokenizer files live next to (not inside) the
    // weights. No need to mmap several GB of safetensors just to tokenize.
    std::string tok_json = path;
    if (fs::is_directory(path, ec)) {
        tok_json = (fs::path(path) / "tokenizer.json").string();
        if (!fs::is_regular_file(tok_json, ec)) return false;
        return HFCheckpoint::load_tokenizer_json(tok_json, out, &eos_out);
    }
    if (!fs::exists(tok_json, ec) || !fs::is_regular_file(tok_json, ec)) return false;
    if (!HFCheckpoint::load_tokenizer_json(tok_json, out, &eos_out)) return false;
    if (eos_out.empty()) eos_out.push_back(out.eos_id());
    return true;
}

bool copy_tokenizer_files(const std::string& src, const std::string& dst_dir) {
    std::error_code ec;
    fs::create_directories(dst_dir, ec);
    if (!fs::exists(src, ec)) {
        log_warn("[hf ] cannot copy the tokenizer files: " + src + " does not exist");
        return false;
    }
    int copied = 0;
    if (fs::is_directory(src, ec)) {
        // The files a HuggingFace repository needs to load the tokenizer.
        // generation_config.json is written by the exporter (the stop tokens
        // belong to OUR artifact), and chat_template.jinja is deliberately
        // skipped: an upstream template would advertise a foreign identity
        // and a prompt format the engine does not use.
        static const char* kFiles[] = {
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
        };
        for (const char* f : kFiles) {
            const fs::path from = fs::path(src) / f;
            if (!fs::exists(from, ec) || ec) continue;
            const fs::path to = fs::path(dst_dir) / f;
            fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                log_warn("[hf ] cannot copy " + from.string() + ": " + ec.message());
                ec.clear();
                continue;
            }
            ++copied;
        }
        // Strip any inherited chat template / provenance from the copied
        // tokenizer config. The engine renders prompts itself; external tools
        // must see the same self-contained artifact.
        const fs::path tc = fs::path(dst_dir) / "tokenizer_config.json";
        if (fs::exists(tc, ec) && !ec) {
            std::ifstream in(tc, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json j;
            std::string err;
            if (Json::parse(text, j, &err) && j.is_object()) {
                for (const char* k : {"chat_template", "_name_or_path", "license",
                                      "license_link", "author", "model_name"}) {
                    if (j.has(k)) j.items().erase(k);
                }
                std::ofstream out(tc, std::ios::binary | std::ios::trunc);
                if (out.good()) out << j.dump_pretty();
            }
        }
    } else {
        // A bare tokenizer.json.
        const fs::path to = fs::path(dst_dir) / "tokenizer.json";
        fs::copy_file(src, to, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            log_warn("[hf ] cannot copy " + src + ": " + ec.message());
            return false;
        }
        ++copied;
    }
    if (copied == 0) {
        log_warn("[hf ] no tokenizer file was copied (the repository will need "
                 "its own tokenizer files)");
        return false;
    }
    log_info(strfmt("[hf ] copied %d tokenizer file(s) into %s", copied, dst_dir.c_str()));
    return true;
}

LoadedModel load_model(const Args& args) {
    LoadedModel lm;
    std::string path = args.str("model");
    GAI_CHECK(!path.empty(), "--model is required");
    GAI_CHECK(fs::exists(path), "model not found: " + path);

    std::ifstream probe(path, std::ios::binary);
    u32 magic = 0;
    probe.read(reinterpret_cast<char*>(&magic), 4);
    const bool is_gguf = (magic == GGUF_MAGIC);

    Device dev = pick_device(args);

    if (args.has("gemm-fp16")) {
        const bool want = args.num("gemm-fp16", 1) != 0;
        ops::set_gemm_fp16(want);
        log_info(strfmt("[prec] inference gemm_fp16=%s (explicit --gemm-fp16)",
                        want ? "on" : "off"));
    } else {
        log_info(strfmt("[prec] inference gemm_fp16=%s (default; --gemm-fp16 0 disables)",
                        ops::gemm_fp16_enabled() ? "on" : "off"));
    }

    lm.path = path;
    // The fp16 weight storage mode: the big matrices live in half precision on
    // the device. It is ON by default on the GPU (it is what makes a 6.94B
    // model fit on a 16 GB card) and OFF on the CPU (the reference path stays
    // bit-identical).
    bool fp16_weights = args.flag("weight-f16", dev == Device::CUDA);
    // Native quantized storage: 0 = Q4_0, 1 = Q4_1, 2 = Q8_0. The packed
    // weights stay on the GPU and are dequantized into FP16 scratch per op,
    // which is what makes a 6.9B model fit an 8 GB card (~4.7 GiB + scratch).
    int quant_kind = -1;
    if (args.has("weight-q4")) {
        quant_kind = static_cast<int>(args.num("weight-q4", 0));
        GAI_CHECK(quant_kind >= 0 && quant_kind <= 2,
                  "--weight-q4 must be 0 (Q4_0), 1 (Q4_1) or 2 (Q8_0)");
        if (fp16_weights && !args.has("weight-f16")) fp16_weights = false;
        GAI_CHECK(!fp16_weights, "--weight-q4 and --weight-f16 are mutually exclusive; "
                                 "pass --weight-f16 0 explicitly");
        // Fused dequant-GEMM reads packed Q4 blocks straight into the GEMM,
        // removing the FP16 scratch round-trip. On by default; --fused-q4 0
        // falls back to dequantize-to-scratch + cuBLAS.
        const bool fused = args.flag("fused-q4", true);
        ops::set_fused_q4(fused);
        log_info(strfmt("[prec] fused Q4 GEMM: %s", fused ? "on" : "off (scratch path)"));
        log_info(strfmt("[prec] native quantized weights: %s (packed on device)",
                        quant_kind == 0 ? "Q4_0" : quant_kind == 1 ? "Q4_1" : "Q8_0"));
    }
    if (args.has("weight-f16"))
        log_info(strfmt("[prec] fp16 weight storage: %s (explicit --weight-f16)",
                        fp16_weights ? "on" : "off"));
    else if (fp16_weights)
        log_info("[prec] fp16 weight storage: on (the default on the GPU: the weights "
                 "are half precision, the activations stay fp32)");
    if (is_gguf) {
        if (args.flag("mmap", false))
            log_warn("--mmap has no effect on GGUF files; loading normally");
        GGUFReader r;
        GAI_CHECK(r.open(path), "not a valid .gguf model file: " + path);
        lm.cfg = r.model_config();
        lm.arch = r.get_string("general.architecture", "ghassan");
        lm.quant = r.get_string("ghassan.quantization.profile",
                     r.get_string("general.quantization_version", "q8_k"));
        lm.file_bytes = r.file_size();

        if (args.has("tokenizer")) {
            GAI_CHECK(load_tokenizer_auto(args.str("tokenizer"), lm.tokenizer, lm.eos_ids),
                      "cannot load tokenizer: " + args.str("tokenizer"));
        } else if (r.has_tokenizer()) {
            GAI_CHECK(r.load_tokenizer(lm.tokenizer),
                      "GGUF tokenizer payload is corrupt; pass --tokenizer explicitly");
            lm.eos_ids = r.eos_ids();
            log_info("[load] using tokenizer embedded in GGUF (self-contained)");
        } else {
            GAI_CHECK(false, "GGUF without embedded tokenizer; pass --tokenizer "
                             "(a HF checkpoint dir or a tokenizer.json file)");
        }
        GAI_CHECK(lm.tokenizer.vocab_size() <= lm.cfg.vocab_size,
                  strfmt("the tokenizer (%d tokens) exceeds the model embedding (%d rows)",
                         lm.tokenizer.vocab_size(), lm.cfg.vocab_size));
        if (lm.tokenizer.vocab_size() < lm.cfg.vocab_size)
            log_info(strfmt("[load] tokenizer has %d tokens; embedding is %d rows (padding)",
                            lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

        Timer t;
        const bool fp16_cache_on = (dev == Device::CUDA) && ops::gemm_fp16_enabled() &&
                                   !args.flag("no-fp16-cache", false) && !fp16_weights &&
                                   quant_kind < 0;
        check_vram_budget(lm.cfg, dev, args.num_int("ctx", 0), fp16_cache_on, fp16_weights,
                          quant_kind);
        lm.model = std::make_unique<Model>(lm.cfg, dev, fp16_weights, quant_kind);
        GAI_CHECK(load_model_from_gguf(path, *lm.model, quant_kind),
                  "failed to load weights from: " + path);
        if (fp16_cache_on) {
            lm.model->enable_fp16_weight_cache(true);
            log_info(strfmt("[prec] persistent fp16 inference cache: %s",
                            human_bytes(lm.model->fp16_weight_cache_bytes()).c_str()));
        }
        log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev)));
        return lm;
    }

    // A HuggingFace checkpoint directory: the engine serves it as it is,
    // with no conversion step.
    if (fs::is_directory(path)) {
        if (quant_kind >= 0) {
            log_warn("[prec] --weight-q4 applies to GGUF inputs only; export a GGUF "
                     "first (ghassan-ai quantize --profile q4_0) to use native Q4");
            quant_kind = -1;
        }
        HFCheckpoint hf;
        GAI_CHECK(hf.open(path), "not a usable HuggingFace checkpoint: " + path);
        lm.cfg = hf.config();
        lm.arch = "hf";
        lm.quant = "q8_k";
        // open() already measured the shard sizes; do not stat them again.
        lm.file_bytes = hf.total_bytes();

        if (args.has("tokenizer")) {
            GAI_CHECK(load_tokenizer_auto(args.str("tokenizer"), lm.tokenizer, lm.eos_ids),
                      "cannot load tokenizer: " + args.str("tokenizer"));
        } else {
            GAI_CHECK(hf.load_tokenizer(lm.tokenizer),
                      "cannot build the tokenizer from " + path +
                      "/tokenizer.json (pass --tokenizer to use another one)");
            lm.eos_ids = hf.eos_ids();
            log_info("[load] using the checkpoint's tokenizer (tokenizer.json)");
        }
        GAI_CHECK(lm.tokenizer.vocab_size() <= lm.cfg.vocab_size,
                  strfmt("the tokenizer has %d tokens but the model has %d: "
                         "the tokenizer would produce ids the model does not have",
                         lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

        Timer t;
        const bool fp16_cache_on = (dev == Device::CUDA) && ops::gemm_fp16_enabled() &&
                                   !args.flag("no-fp16-cache", false) && !fp16_weights;
        check_vram_budget(lm.cfg, dev, args.num_int("ctx", 0), fp16_cache_on, fp16_weights);
        lm.model = std::make_unique<Model>(lm.cfg, dev, fp16_weights);
        GAI_CHECK(hf.load_model(*lm.model),
                  "failed to load the weights of: " + path);
        if (fp16_cache_on) {
            lm.model->enable_fp16_weight_cache(true);
            log_info(strfmt("[prec] persistent fp16 inference cache: %s",
                            human_bytes(lm.model->fp16_weight_cache_bytes()).c_str()));
        }
        log_info(strfmt("[load] %s  (%s, safetensors) in %s on %s",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev)));
        return lm;
    }

    GAI_CHECK(false, "not a valid model: " + path +
                     " (expected a HuggingFace checkpoint directory or a .gguf file)");
}

GenerationConfig make_gen_config(const Args& a) {
    GenerationConfig g;
    g.max_new_tokens = a.num_int("max-tokens", 256);
    GAI_CHECK(g.max_new_tokens >= 0 && g.max_new_tokens <= 1000000,
              "--max-tokens must be in [0,1000000]");
    g.sampling.temperature = static_cast<float>(a.real("temp", 0.8));
    g.sampling.top_k = a.num_int("top-k", 40);
    g.sampling.top_p = static_cast<float>(a.real("top-p", 0.92));
    g.sampling.min_p = static_cast<float>(a.real("min-p", 0.05));
    g.sampling.repetition_penalty = static_cast<float>(a.real("repeat", 1.12));
    g.sampling.repetition_window = a.num_int("repeat-window", 128);
    g.sampling.frequency_penalty = static_cast<float>(a.real("frequency-penalty", 0.0));
    g.sampling.presence_penalty  = static_cast<float>(a.real("presence-penalty", 0.0));
    g.sampling.no_repeat_ngram = a.num_int("no-repeat-ngram", 0);
    g.sampling.seed = a.num_u64("seed", 0);
    g.sampling.greedy = a.flag("greedy", false);
    g.sampling.gpu_fast_sample = !a.flag("no-fast-sample", false);
    // Fail fast on invalid values: validate() coerces silently (e.g. a
    // negative --temp becomes greedy), which makes quality comparisons
    // meaningless.
    const SamplingConfig& s = g.sampling;
    GAI_CHECK(std::isfinite(s.temperature) && s.temperature >= 0.0f && s.temperature <= 5.0f,
              "--temp must be in [0,5]");
    GAI_CHECK(std::isfinite(s.top_p) && s.top_p > 0.0f && s.top_p <= 1.0f,
              "--top-p must be in (0,1]");
    GAI_CHECK(std::isfinite(s.min_p) && s.min_p >= 0.0f && s.min_p < 1.0f,
              "--min-p must be in [0,1)");
    GAI_CHECK(s.top_k >= 0, "--top-k must be >= 0");
    GAI_CHECK(std::isfinite(s.repetition_penalty) && s.repetition_penalty > 0.0f,
              "--repeat must be > 0");
    GAI_CHECK(s.repetition_window >= 0, "--repeat-window must be >= 0");
    g.sampling.validate();
    return g;
}

}  // namespace gai
