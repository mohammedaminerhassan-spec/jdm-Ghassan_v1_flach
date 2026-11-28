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
#include <limits>
#include <map>
#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

namespace fs = std::filesystem;

namespace gai {

namespace {

void check_vram_budget(const ModelConfig& cfg, Device dev, int ctx, bool fp16_cache_on,
                       bool fp16_weights = false, int quant_kind = -1) {
    if (dev != Device::CUDA) return;
    const DeviceInfo& di = device_info();
    if (!di.cuda_available || di.total_mem == 0) return;

    auto sat_add = [](size_t a, size_t b) -> size_t {
        const size_t maxv = (std::numeric_limits<size_t>::max)();
        return b > maxv - a ? maxv : a + b;
    };
    auto sat_mul = [](size_t a, size_t b) -> size_t {
        const size_t maxv = (std::numeric_limits<size_t>::max)();
        return a != 0 && b > maxv / a ? maxv : a * b;
    };

    const int kv_ctx = ctx > 0 ? ctx : std::min(cfg.max_seq_len, 4096);
    // Generator prefill uses 512-query chunks that attend to the accumulated KV prefix.
    const int prefill_T = std::max(1, std::min(kv_ctx, 512));
    const size_t kv = Model::kv_cache_bytes(cfg, kv_ctx);
    const Model::MemoryPlan plan =
        Model::plan_memory(cfg, 1, prefill_T, false, 1,
                           fp16_cache_on, fp16_weights, quant_kind);
    const Model::WorkspacePlan wsp = Model::workspace_plan(cfg, 1, prefill_T);
    const size_t qscratch = quant_kind >= 0 ? (512ull << 20) : 0;

    // Cached inference prefill now uses tiled online FlashAttention. Its K/V
    // tiles live in at most 32 KiB of CTA shared memory; it allocates neither a
    // B*H*Tq*Tk score matrix nor a full FP32 copy of the FP16 KV cache.
    // The estimates below are global-memory allocations for this inference path.
    const size_t attention_workspace = 0;
    const size_t kv_stage_bytes = 0;
    const size_t gemm_mirror_bytes = 0;

    const size_t attention_total_workspace = sat_add(
        sat_add(attention_workspace, kv_stage_bytes), gemm_mirror_bytes);
    const size_t workspaces = sat_add(
        sat_add(wsp.gemm_bytes, wsp.moe_bytes),
        sat_add(wsp.mamba_bytes, attention_total_workspace));

    // plan.activations already includes the persistent Mamba recurrent-state allocation.
    const size_t need = sat_add(
        sat_add(plan.weights, plan.fp16_cache),
        sat_add(plan.activations, sat_add(kv, sat_add(workspaces, qscratch))));

    size_t live_free = di.free_mem;
#ifdef GAI_CUDA
    const size_t cuda_free = cuda::free_bytes_live();
    if (cuda_free > 0) live_free = cuda_free;
#endif
    const size_t available = live_free > 0 ? live_free : di.total_mem;
    const size_t reserve = available / 10; // Leave 10% headroom for CUDA/runtime allocations.
    const size_t budget = available - reserve;
    const double pct = budget > 0
        ? 100.0 * static_cast<double>(need) / static_cast<double>(budget)
        : 100.0;

    log_info(strfmt("[mem ] GPU plan: weights %s | fp16 cache %s | activations %s "
                    "(includes Mamba state) | KV %s | CUDA workspaces %s "
                    "(GEMM %s, MoE %s, Mamba %s, FlashAttention global scratch %s, "
                    "KV staging %s, GEMM mirror %s) = %s of %s usable VRAM "
                    "(%s free / %s total, %.0f%%)",
                    human_bytes(plan.weights).c_str(),
                    human_bytes(plan.fp16_cache).c_str(),
                    human_bytes(plan.activations).c_str(),
                    human_bytes(kv).c_str(),
                    human_bytes(workspaces).c_str(),
                    human_bytes(wsp.gemm_bytes).c_str(),
                    human_bytes(wsp.moe_bytes).c_str(),
                    human_bytes(wsp.mamba_bytes).c_str(),
                    human_bytes(attention_workspace).c_str(),
                    human_bytes(kv_stage_bytes).c_str(),
                    human_bytes(gemm_mirror_bytes).c_str(),
                    human_bytes(need).c_str(),
                    human_bytes(budget).c_str(),
                    human_bytes(available).c_str(),
                    human_bytes(di.total_mem).c_str(),
                    pct));

    if (need > budget) {
        GAI_FAIL(strfmt("the model's estimated inference footprint is %s, but only %s of VRAM "
                        "is available for this load after a 10%% safety reserve "
                        "(%s free / %s total). Reduce --ctx, use --weight-f16 1, "
                        "--weight-q4, or disable --fp16-cache.",
                        human_bytes(need).c_str(),
                        human_bytes(budget).c_str(),
                        human_bytes(available).c_str(),
                        human_bytes(di.total_mem).c_str()));
    }
}

Device pick_device(const Args& a) {
    const std::string d = a.str("device", "auto");
    if (d == "cpu") return Device::CPU;
    if (d == "cuda") {
        GAI_CHECK(cuda_available(), "--device cuda requested but no CUDA device is available (use --device cpu)");
        return Device::CUDA;
    }
    if (d == "metal" || d == "vulkan" || d == "tpu" || d == "rocm" || d == "amd" || d == "intel") {
        log_warn("--device " + d + " has no native accelerator driver loaded; falling back automatically to high-performance CPU "
                 "(portable OpenMP SIMD kernels; use --device cuda on NVIDIA GPUs)");
        return Device::CPU;
    }
    if (d != "auto") GAI_FAIL("unknown --device '" + d + "' (auto|cpu|cuda|rocm|amd|intel|metal|vulkan)");
    return best_device();
}

}

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

    if (is_gguf_file(path)) {
        GGUFReader r;
        if (!r.open(path)) return false;
        if (!r.has_tokenizer()) return false;
        if (!r.load_tokenizer(out)) return false;
        eos_out = r.eos_ids();
        if (eos_out.empty()) eos_out.push_back(out.eos_id());
        return true;
    }

    std::string tok_json = path;
    if (fs::is_directory(path, ec)) {
        tok_json = (fs::path(path) / "tokenizer.json").string();
        if (!fs::is_regular_file(tok_json, ec)) return false;
        if (!HFCheckpoint::load_tokenizer_json(tok_json, out, &eos_out)) return false;
        if (eos_out.empty()) eos_out.push_back(out.eos_id());
        return true;
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
    bool fp16_weights = args.flag("weight-f16", dev == Device::CUDA);
    int quant_kind = -1;
    if (args.has("weight-q4")) {
        quant_kind = static_cast<int>(args.num("weight-q4", 0));
        GAI_CHECK(quant_kind >= 0 && quant_kind <= 2,
                  "--weight-q4 must be 0 (Q4_0), 1 (Q4_1) or 2 (Q8_0)");
        if (fp16_weights && !args.has("weight-f16")) fp16_weights = false;
        GAI_CHECK(!fp16_weights, "--weight-q4 and --weight-f16 are mutually exclusive; "
                                 "pass --weight-f16 0 explicitly");
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
        if (args.has("mmproj")) {
            lm.vision = load_vision_from_gguf(args.str("mmproj"), dev);
            if (lm.vision) log_info("[load] multimodal vision projector loaded from: " + args.str("mmproj"));
        } else {
            lm.vision = load_vision_from_gguf(path, dev);
            if (lm.vision) log_info("[load] multimodal vision projector loaded from model GGUF");
        }
        log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev)));
        return lm;
    }

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
        if (args.has("mmproj")) {
            lm.vision = load_vision_from_gguf(args.str("mmproj"), dev);
            if (lm.vision) log_info("[load] multimodal vision projector loaded from: " + args.str("mmproj"));
        } else {
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

}
