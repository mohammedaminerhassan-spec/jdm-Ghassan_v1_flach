// The safetensors round-trip: write a model, read it back, compare.
//
// The safetensors writer is the only path that publishes a trained model
// as a HuggingFace repository, so it MUST be able to read back what it
// writes - with the engine's own loader (the same code path a user hits
// when they clone the repository).
//
// The test covers the three shapes the engine can export:
//   * a dense model,
//   * a MoE model (the experts are per-expert on the HuggingFace side),
//   * the IBM Granite hybrid (the experts are PACKED into two tensors per
//     layer, and the shared expert into two more).
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
    const fs::path base = fs::temp_directory_path();
    const fs::path d = base / ("gai_safetensors_" + std::string(tag) + "_" +
                                std::to_string(static_cast<long long>(
                                    reinterpret_cast<std::uintptr_t>(&g_failures))));
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d, ec);
    return d.string();
}

// Compares every parameter of two models.
static void compare_models(const Model& a, const Model& b, const std::string& tag) {
    CHECK(a.num_parameters() == b.num_parameters(),
          (tag + ": the parameter count differs").c_str());
    for (const Parameter* p : a.parameters()) {
        const Parameter* q = b.find_parameter(p->name);
        if (!q) {
            CHECK(false, (tag + ": the reloaded model has no parameter " + p->name).c_str());
            continue;
        }
        CHECK(p->w.numel() == q->w.numel(),
              (tag + ": " + p->name + " has a different size").c_str());
        const Tensor ta = p->w.to(Device::CPU);
        const Tensor tb = q->w.to(Device::CPU);
        if (ta.dtype() != DType::F32 || tb.dtype() != DType::F32) continue;
        const float* xa = ta.f32();
        const float* xb = tb.f32();
        double worst = 0.0;
        for (i64 i = 0; i < ta.numel(); ++i) {
            const double d = std::fabs(static_cast<double>(xa[i]) - static_cast<double>(xb[i]));
            if (d > worst) worst = d;
        }
        // The export writes fp16 by default, so the round trip is exact up
        // to the fp16 rounding of the file (the fp32 masters are the
        // reference).
        const double tol = 2e-3;
        CHECK(worst <= tol,
              (tag + ": " + p->name + " differs by " + std::to_string(worst)).c_str());
    }
}

static void roundtrip(Model& model, const std::string& tag) {
    const std::string dir = unique_dir(tag.c_str());
    Tokenizer tok;   // an empty vocabulary: the export only needs the ids

    const bool ok = export_model_safetensors(dir, model, tok, "Ghassan v1 Flash",
                                              "round-trip test", SafeDType::F16);
    CHECK(ok, (tag + ": the export failed").c_str());
    if (!ok) return;

    // The files a HuggingFace repository needs.
    CHECK(fs::exists(fs::path(dir) / "model.safetensors"),
          (tag + ": model.safetensors is missing").c_str());
    CHECK(fs::exists(fs::path(dir) / "config.json"),
          (tag + ": config.json is missing").c_str());

    // Read it back with the engine's own loader.
    HFCheckpoint hf;
    CHECK(hf.open(dir), (tag + ": the exported directory does not open").c_str());
    if (hf.tensors().empty()) {
        CHECK(false, (tag + ": the exported file has no tensor").c_str());
        return;
    }
    Model reloaded(hf.config(), Device::CPU);
    CHECK(hf.load_model(reloaded), (tag + ": the exported weights do not load").c_str());
    compare_models(model, reloaded, tag);

    // The config must describe the same architecture.
    CHECK(hf.config().num_layers == model.config().num_layers,
          (tag + ": the config lost the layer count").c_str());
    CHECK(hf.config().hidden_size == model.config().hidden_size,
          (tag + ": the config lost the hidden size").c_str());
    CHECK(hf.config().vocab_size == model.config().vocab_size,
          (tag + ": the config lost the vocabulary size").c_str());
    CHECK(hf.config().use_moe == model.config().use_moe,
          (tag + ": the config lost the MoE flag").c_str());

    std::error_code ec;
    fs::remove_all(dir, ec);
}

static void test_dense() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 64;
    cfg.max_seq_len = 64;
    cfg.tie_embeddings = true;
    Model m(cfg, Device::CPU);
    m.init_weights(1234);
    roundtrip(m, "dense");
}

static void test_moe() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 64;
    cfg.max_seq_len = 64;
    cfg.use_moe = true;
    cfg.num_experts = 4;
    cfg.moe_top_k = 2;
    cfg.moe_expert_dim = 16;
    cfg.moe_shared = true;
    cfg.shared_expert_intermediate_size = 24;   // different from the routed width
    Model m(cfg, Device::CPU);
    m.init_weights(4321);
    roundtrip(m, "moe");
}

static void test_granite() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 4;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 20;         // NOT the expert width: the Granite
                                     // "intermediate_size" is the ROUTED
                                     // expert width, so the two must differ
                                     // or the round trip cannot see a mix-up
    cfg.max_seq_len = 64;
    cfg.arch = ModelConfig::ArchGranite;
    cfg.layer_types = {0, 1, 0, 1};        // mamba, attention, mamba, attention
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
    Model m(cfg, Device::CPU);
    m.init_weights(99);
    roundtrip(m, "granite");
}

int main() {
    test_dense();
    test_moe();
    test_granite();
    if (g_failures == 0) {
        std::printf("test_safetensors_roundtrip: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_safetensors_roundtrip: %d failure(s)\n", g_failures);
    return 1;
}
