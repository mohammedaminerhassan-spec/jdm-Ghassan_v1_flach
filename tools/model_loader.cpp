#include "tools/model_loader.h"

#include "core/device.h"
#include "core/ops.h"
#include "format/legacy_gai_format.h"
#include "format/gguf_codec.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace gai {

namespace {

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

LoadedModel load_model(const Args& args) {
    LoadedModel lm;
    std::string path = args.str("model");
    GAI_CHECK(!path.empty(), "--model is required");
    GAI_CHECK(fs::exists(path), "model file not found: " + path);

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
            log_warn("--mmap currently covers .gai weights only; loading this GGUF normally");
        GGUFReader r;
        GAI_CHECK(r.open(path), "not a valid .gguf model file: " + path);
        lm.cfg = r.model_config();
        lm.arch = r.get_string("general.architecture", "ghassan");
        lm.quant = r.get_string("ghassan.quantization.profile",
                     r.get_string("general.quantization_version", "fp16"));
        lm.file_bytes = r.file_size();

        if (args.has("tokenizer")) {
            GAI_CHECK(lm.tokenizer.load(args.str("tokenizer")),
                      "cannot load tokenizer: " + args.str("tokenizer"));
        } else if (r.has_tokenizer()) {
            GAI_CHECK(r.load_tokenizer(lm.tokenizer),
                      "GGUF tokenizer payload is corrupt; pass --tokenizer explicitly");
            log_info("[load] using tokenizer embedded in GGUF (no sidecar needed)");
        } else {
            std::string def = (fs::path(path).parent_path() / "tokenizer" / "english32k.gtok").string();
            if (!fs::exists(def)) def = "artifacts/tokenizer/english32k.gtok";
            if (!fs::exists(def)) def = "artifacts/tokenizer/darija32k.gtok";
            if (!fs::exists(def)) def = "artifacts/tokenizer/darija.gtok";
            GAI_CHECK(lm.tokenizer.load(def),
                      "legacy GGUF without embedded tokenizer; pass --tokenizer "
                      "(e.g. artifacts/tokenizer/english32k.gtok) or re-export");
        }
        GAI_CHECK(lm.tokenizer.vocab_size() == lm.cfg.vocab_size,
                  strfmt("tokenizer/model vocab mismatch: %d vs %d",
                         lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

        Timer t;
        lm.model = std::make_unique<Model>(lm.cfg, dev);
        GAI_CHECK(load_model_from_gguf(path, *lm.model),
                  "failed to load weights from: " + path);
        if (dev == Device::CUDA && ops::gemm_fp16_enabled() &&
            !args.flag("no-fp16-cache", false)) {
            lm.model->enable_fp16_weight_cache(true);
            log_info(strfmt("[prec] persistent fp16 inference cache: %s",
                            human_bytes(lm.model->fp16_weight_cache_bytes()).c_str()));
        }
        log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev)));
        return lm;
    }

    GaiReader r;
    GAI_CHECK(r.open(path), "not a valid .gai model file: " + path);
    lm.cfg = r.model_config();
    lm.arch = "ghassan";
    lm.quant = r.meta_str("quant_profile", "fp32");
    lm.file_bytes = r.file_size();

    if (args.has("tokenizer")) {
        GAI_CHECK(lm.tokenizer.load(args.str("tokenizer")),
                  "cannot load tokenizer: " + args.str("tokenizer"));
    } else {
        GAI_CHECK(r.has_tokenizer() && r.load_tokenizer(lm.tokenizer),
                  "the model has no embedded tokenizer; pass --tokenizer");
    }
    GAI_CHECK(lm.tokenizer.vocab_size() == lm.cfg.vocab_size,
              strfmt("tokenizer/model vocab mismatch: %d vs %d",
                     lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

    Timer t;
    lm.model = std::make_unique<Model>(lm.cfg, dev);
    bool use_mmap = args.flag("mmap", false);
    if (use_mmap && dev != Device::CPU) {
        log_warn("--mmap is CPU-inference only; loading normally for CUDA");
        use_mmap = false;
    }
    if (use_mmap) {
        int wrapped = 0, converted = 0;
        GAI_CHECK(load_model_from_gai_mmap(path, *lm.model, &wrapped, &converted),
                  "failed to memory-map weights from: " + path);
        log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s [mmap: %d zero-copy, %d converted]",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev),
                        wrapped, converted));
        return lm;
    }
    for (Parameter* p : lm.model->parameters()) {
        Tensor tt = r.read_tensor_f32(p->name);
        GAI_CHECK(tt.numel() == p->numel(), "tensor size mismatch: " + p->name);
        p->w.copy_from(tt);
    }
    if (dev == Device::CUDA && ops::gemm_fp16_enabled() &&
        !args.flag("no-fp16-cache", false)) {
        lm.model->enable_fp16_weight_cache(true);
        log_info(strfmt("[prec] persistent fp16 inference cache: %s",
                        human_bytes(lm.model->fp16_weight_cache_bytes()).c_str()));
    }
    log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                    path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                    human_duration(t.seconds()).c_str(), device_name(dev)));
    return lm;
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

const char* chat_style_name(ChatStyle s) {
    switch (s) {
        case ChatStyle::ChatML:  return "chatml";
        case ChatStyle::Llama3:  return "llama3";
        case ChatStyle::Mistral: return "mistral";
        default:                 return "native";
    }
}

}  // namespace gai
