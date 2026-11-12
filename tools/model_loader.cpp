#include "tools/model_loader.h"

#include "core/device.h"
#include "core/ops.h"
#include "format/gguf_codec.h"
#include "format/hf_checkpoint.h"

#include <filesystem>
#include <fstream>
#include <map>

namespace fs = std::filesystem;

namespace gai {

namespace {

// The model is allocated on the device before a single weight is read, so
// a model that does not fit the GPU dies inside cudaMalloc with a cryptic
// "out of memory". This pre-flight turns that into an actionable message.
void check_vram_budget(const ModelConfig& cfg, Device dev, int ctx, bool fp16_cache_on) {
    if (dev != Device::CUDA) return;
    const DeviceInfo& di = device_info();
    if (!di.cuda_available || di.total_mem == 0) return;

    const Model::MemoryPlan plan =
        Model::plan_memory(cfg, 1, ctx > 0 ? ctx : 512, false, 1, fp16_cache_on);
    const size_t kv   = Model::kv_cache_bytes(cfg, ctx > 0 ? ctx : cfg.max_seq_len);
    const size_t mam  = Model::mamba_state_bytes(cfg);
    const Model::WorkspacePlan wsp = Model::workspace_plan(cfg, 1, ctx > 0 ? ctx : 512);
    const size_t need = plan.weights + plan.fp16_cache + plan.activations +
                        kv + mam + wsp.gemm_bytes + wsp.moe_bytes;

    log_info(strfmt("[mem ] GPU plan: weights %s | fp16 cache %s | activations %s | "
                    "kv %s | mamba %s | workspaces %s = %s of %s (%.0f%%)",
                    human_bytes(plan.weights).c_str(),
                    human_bytes(plan.fp16_cache).c_str(),
                    human_bytes(plan.activations).c_str(),
                    human_bytes(kv).c_str(),
                    human_bytes(mam).c_str(),
                    human_bytes(wsp.gemm_bytes + wsp.moe_bytes).c_str(),
                    human_bytes(need).c_str(),
                    human_bytes(di.total_mem).c_str(),
                    100.0 * static_cast<double>(need) / static_cast<double>(di.total_mem)));

    if (need > di.total_mem) {
        GAI_FAIL(strfmt("the model does not fit this GPU: it needs %s but the card has %s "
                        "(%s of fp32 weights + %s of fp16 cache + %s of activations/KV). "
                        "Options: --device cpu (slower, uses the host RAM), a smaller --ctx, "
                        "--no-fp16-cache (frees the fp16 copy), or the fp16 weight storage "
                        "mode (not implemented yet: the engine still keeps fp32 masters).",
                        human_bytes(need).c_str(), human_bytes(di.total_mem).c_str(),
                        human_bytes(plan.weights).c_str(),
                        human_bytes(plan.fp16_cache).c_str(),
                        human_bytes(plan.activations + kv).c_str()));
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
    f.read(reinterpret_cast<char*>(&magic), 4);
    return magic == GGUF_MAGIC;
}

bool load_tokenizer_auto(const std::string& path, Tokenizer& out,
                         std::vector<i32>& eos_out) {
    eos_out.clear();
    std::error_code ec;
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
                                   !args.flag("no-fp16-cache", false);
        check_vram_budget(lm.cfg, dev, args.num_int("ctx", 0), fp16_cache_on);
        lm.model = std::make_unique<Model>(lm.cfg, dev);
        GAI_CHECK(load_model_from_gguf(path, *lm.model),
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
        HFCheckpoint hf;
        GAI_CHECK(hf.open(path), "not a usable HuggingFace checkpoint: " + path);
        lm.cfg = hf.config();
        lm.arch = "hf";
        lm.quant = "q8_k";
        for (const std::string& f : hf.weight_files()) {
            std::error_code ec;
            const auto sz = fs::file_size(f, ec);
            if (!ec) lm.file_bytes += static_cast<uint64_t>(sz);
        }

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
                                   !args.flag("no-fp16-cache", false);
        check_vram_budget(lm.cfg, dev, args.num_int("ctx", 0), fp16_cache_on);
        lm.model = std::make_unique<Model>(lm.cfg, dev);
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
    g.sampling.no_repeat_ngram = a.num_int("no-repeat-ngram", 0);
    g.sampling.seed = static_cast<u64>(a.num("seed", 0));
    g.sampling.greedy = a.flag("greedy", false);
    g.sampling.gpu_fast_sample = !a.flag("no-fast-sample", false);
    g.sampling.validate();
    return g;
}

}  // namespace gai
