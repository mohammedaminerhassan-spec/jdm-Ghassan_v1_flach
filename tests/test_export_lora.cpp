// The LoRA export fold: exporting a model that has LoRA adapters must
// produce a SELF-CONTAINED HuggingFace checkpoint whose forward output is
// identical to the model-with-adapters. The exporters fold the adapters
// into the base weights at write time (fp32, before encode()), so:
//
//   * no .lora_ tensor may appear in the file,
//   * the reloaded model (which has no adapters) must produce the same
//     logits as the model that carried them,
//   * the reloaded base weights must equal base + scale * B A.
//
// This is the contract behind "Ghassan v1 Flash ships as one repository,
// not a base + an adapter pair" - and it is exact: the delta never
// round-trips through Q4/Q8/F16 storage.
#include "core/common.h"
#include "core/device.h"
#include "format/hf_checkpoint.h"
#include "format/hf_writer.h"
#include "model/model.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace gai;

static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,      \
                         __LINE__);                                           \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

static std::string unique_dir(const char* tag) {
    const fs::path d = fs::temp_directory_path() /
                       ("gai_export_lora_" + std::string(tag));
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d, ec);
    return d.string();
}

static void run_case(Model& model, const std::string& tag, const i32* ids, int T) {
    // A non-trivial adapter: B carries deterministic non-zero values so the
    // delta is actually exercised (A is filled by init_weights for LoRA).
    for (Parameter* p : model.parameters()) {
        if (p->name.find("lora_B") != std::string::npos) {
            for (i64 i = 0; i < p->numel(); ++i)
                p->w.f32()[i] = 0.02f * static_cast<float>((i % 13) - 6);
        }
    }

    // The merged reference: what the exported model must compute.
    Activations a1 = model.make_activations(1, T, false);
    Tensor logits = model.forward(ids, 1, T, a1);
    std::vector<float> before(static_cast<size_t>(logits.numel()));
    for (i64 i = 0; i < logits.numel(); ++i) before[static_cast<size_t>(i)] = logits.f32()[i];

    // Expected per-parameter values: base + scale * B A for every adapted
    // base, so the reload can be checked tensor by tensor.
    std::vector<std::pair<std::string, std::vector<float>>> expected;
    for (const Parameter* p : model.parameters()) {
        std::vector<float> delta;
        if (model.lora_delta_for(p->name, delta)) {
            const Tensor t = p->w.to(Device::CPU);
            std::vector<float> merged;
            merged.reserve(static_cast<size_t>(t.numel()));
            if (t.dtype() == DType::F32) {
                for (i64 i = 0; i < t.numel(); ++i)
                    merged.push_back(t.f32()[i] + delta[static_cast<size_t>(i)]);
            } else {
                // Storage precision (F16/Q4) would lose bits of the delta;
                // only the F32 master case is checked exactly here - the
                // logits check below is dtype agnostic.
                merged.clear();
            }
            if (!merged.empty()) expected.emplace_back(p->name, std::move(merged));
        }
    }

    const std::string dir = unique_dir(tag.c_str());
    Tokenizer tok;
    CHECK(export_model_safetensors(dir, model, tok, "Ghassan v1 Flash",
                                   "lora export fold test", SafeDType::F32),
          (tag + ": the export failed").c_str());

    HFCheckpoint hf;
    CHECK(hf.open(dir), (tag + ": the exported directory does not open").c_str());
    if (!hf.tensors().empty()) {
        Model reloaded(hf.config(), Device::CPU);
        CHECK(hf.load_model(reloaded), (tag + ": the exported weights do not load").c_str());

        Activations a2 = reloaded.make_activations(1, T, false);
        Tensor after = reloaded.forward(ids, 1, T, a2);
        double worst = 0.0;
        for (i64 i = 0; i < after.numel(); ++i) {
            const double d = std::fabs(static_cast<double>(before[static_cast<size_t>(i)]) -
                                       static_cast<double>(after.f32()[i]));
            if (d > worst) worst = d;
        }
        std::printf("[export-lora] %s: worst |logits diff| = %.3e\n", tag.c_str(), worst);
        CHECK(worst <= 1e-4, (tag + ": the exported model does not compute the adapter").c_str());

        // Per-parameter fold check against the dtype-agnostic expectation.
        for (const auto& [name, merged] : expected) {
            const Parameter* q = reloaded.find_parameter(name);
            if (!q) {
                CHECK(false, (tag + ": reloaded model has no parameter " + name).c_str());
                continue;
            }
            const Tensor t = q->w.to(Device::CPU);
            double w = 0.0;
            if (t.dtype() == DType::F32) {
                for (i64 i = 0; i < t.numel(); ++i) {
                    const double d = std::fabs(static_cast<double>(t.f32()[i]) -
                                               static_cast<double>(merged[static_cast<size_t>(i)]));
                    if (d > w) w = d;
                }
                CHECK(w <= 1e-5, (tag + ": " + name + " does not equal base + BA").c_str());
            }
        }

        // No raw adapter tensor may leak into the artifact.
        for (const auto& [name, info] : hf.tensors()) {
            (void)info;
            CHECK(name.find(".lora_") == std::string::npos &&
                      name.find("lora_A") == std::string::npos &&
                      name.find("lora_B") == std::string::npos,
                  (tag + ": a raw adapter tensor leaked into the export: " + name).c_str());
        }
    }

    std::error_code ec;
    fs::remove_all(dir, ec);
}

int main() {
    const int T = 4;
    std::vector<i32> ids = {3, 11, 17, 29};

    // Case 1: dense model with adapters on the FFN and the projections.
    {
        ModelConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 32;
        cfg.num_layers = 2;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 64;
        cfg.max_seq_len = 16;
        cfg.lora_rank = 4;
        cfg.lora_alpha = 8.0f;
        cfg.lora_target_q = true;
        cfg.lora_target_k = true;
        cfg.lora_target_v = true;
        cfg.lora_target_o = true;
        cfg.lora_target_gate = true;
        cfg.lora_target_up = true;
        cfg.lora_target_down = true;
        cfg.use_moe = false;
        Model m(cfg, Device::CPU, /*fp16_weights=*/false);
        m.init_weights(123);
        m.enable_grad(true);
        run_case(m, "dense", ids.data(), T);
    }

    // Case 2: the Granite hybrid (NoPE + shared MoE + multipliers).
    {
        ModelConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 32;
        cfg.num_layers = 4;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 20;
        cfg.max_seq_len = 16;
        cfg.arch = ModelConfig::ArchGranite;
        cfg.layer_types = {0, 1, 0, 1};
        cfg.use_moe = true;
        cfg.num_experts = 4;
        cfg.moe_top_k = 2;
        cfg.moe_expert_dim = 8;
        cfg.moe_shared = true;
        cfg.shared_expert_intermediate_size = 12;
        cfg.mamba_n_heads = 4;
        cfg.mamba_d_head = 8;
        cfg.mamba_d_state = 16;
        cfg.mamba_n_groups = 1;
        cfg.mamba_d_conv = 4;
        cfg.mamba_expand = 2;
        cfg.position_embedding = 1;             // NoPE
        cfg.embedding_multiplier = 12.0f;
        cfg.residual_multiplier = 0.5f;
        cfg.logits_scaling = 8.0f;
        cfg.lora_rank = 4;
        cfg.lora_alpha = 8.0f;
        cfg.lora_target_q = true;
        cfg.lora_target_k = true;
        cfg.lora_target_v = true;
        cfg.lora_target_o = true;
        cfg.lora_target_gate = false;           // the MoE path has no dense FFN
        cfg.lora_target_up = false;
        cfg.lora_target_down = false;
        Model m(cfg, Device::CPU, /*fp16_weights=*/false);
        m.init_weights(321);
        m.enable_grad(true);
        run_case(m, "granite", ids.data(), T);
    }

    if (g_failures == 0) {
        std::printf("test_export_lora: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_export_lora: %d failure(s)\n", g_failures);
    return 1;
}
