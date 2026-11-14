// The Granite-4.0-H compatibility check.
//
// The reference checkpoint (ibm-granite/granite-4.0-h-tiny) is the geometry
// of Ghassan v1 Flash: the same hybrid (36 Mamba-2 layers + 4 attention
// layers), the same MoE (64 experts, top-6, a shared expert of its own
// width), the same multipliers and the same NoPE positions.
//
// This test builds the engine's ModelConfig from the REFERENCE config.json
// (the file that ships with the checkpoint) and checks that:
//   * every parameter of the engine has a HuggingFace counterpart,
//   * the total parameter count equals the checkpoint's,
//   * the shapes are the ones the loader expects.
//
// It is the check that proves "the safetensors of Granite-4.0-H fit the
// engine": if the geometry ever drifts, this test fails instead of the
// model producing noise.

#include "core/common.h"
#include "format/hf_checkpoint.h"
#include "model/model.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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

// The reference: ibm-granite/granite-4.0-h-tiny, transformers 4.56.
static const char* kReferenceConfig = R"JSON(
{
  "architectures": [
    "GraniteMoeHybridForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attention_multiplier": 0.0078125,
  "bos_token_id": 100257,
  "embedding_multiplier": 12,
  "eos_token_id": 100257,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.1,
  "intermediate_size": 512,
  "layer_types": [
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "attention",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "attention",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "attention",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "mamba",
    "attention",
    "mamba",
    "mamba",
    "mamba",
    "mamba"
  ],
  "logits_scaling": 6,
  "mamba_chunk_size": 256,
  "mamba_conv_bias": true,
  "mamba_d_conv": 4,
  "mamba_d_head": 64,
  "mamba_d_state": 128,
  "mamba_expand": 2,
  "mamba_n_groups": 1,
  "mamba_n_heads": 48,
  "mamba_proj_bias": false,
  "max_position_embeddings": 131072,
  "model_type": "granitemoehybrid",
  "normalization_function": "rmsnorm",
  "num_attention_heads": 12,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 40,
  "num_key_value_heads": 4,
  "num_local_experts": 64,
  "output_router_logits": false,
  "pad_token_id": 100256,
  "position_embedding_type": "nope",
  "residual_multiplier": 0.22,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000,
  "router_aux_loss_coef": 0.0,
  "shared_intermediate_size": 1024,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.56.0",
  "use_cache": true,
  "vocab_size": 100352
}
)JSON";

int main() {
    // The engine's config reader works on a FILE. The reference config is
    // embedded here, so it is written to a temporary file first.
    const std::string tmp = "granite_ref_config.json";
    {
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", tmp.c_str());
            return 1;
        }
        std::fwrite(kReferenceConfig, 1, std::strlen(kReferenceConfig), f);
        std::fclose(f);
    }

    ModelConfig cfg;
    {
        // HFCheckpoint::read_config_json is private; the public path is
        // open() on a directory. A directory with just the config.json is
        // enough (weights_required = false).
        std::filesystem::create_directories("granite_ref_dir");
        std::filesystem::copy_file(tmp,
                                    std::filesystem::path("granite_ref_dir") / "config.json",
                                    std::filesystem::copy_options::overwrite_existing);
        HFCheckpoint hf;
        CHECK(hf.open("granite_ref_dir", /*weights_required=*/false),
              "cannot read the reference config");
        cfg = hf.config();
    }
    std::remove(tmp.c_str());
    std::filesystem::remove_all("granite_ref_dir");

    // ---- the geometry ----
    CHECK(cfg.arch == ModelConfig::ArchGranite, "the architecture is not Granite");
    CHECK(cfg.num_layers == 40, "the layer count is not 40");
    CHECK(cfg.hidden_size == 1536, "the hidden size is not 1536");
    CHECK(cfg.num_heads == 12, "the head count is not 12");
    CHECK(cfg.num_kv_heads == 4, "the KV head count is not 4");
    CHECK(cfg.vocab_size == 100352, "the vocabulary is not 100352");
    CHECK(cfg.use_moe, "the MoE is off");
    CHECK(cfg.num_experts == 64, "the expert count is not 64");
    CHECK(cfg.moe_top_k == 6, "the top-k is not 6");
    CHECK(cfg.moe_expert_dim == 512, "the routed expert width is not 512");
    CHECK(cfg.shared_expert_intermediate_size == 1024, "the shared expert width is not 1024");
    CHECK(cfg.moe_shared, "the shared expert is off");
    CHECK(cfg.position_embedding == 1, "the positions are not NoPE");
    CHECK(cfg.tie_embeddings, "the embeddings are not tied");
    CHECK(cfg.use_qkv_bias == false, "the QKV bias should be off");
    CHECK(cfg.head_dim() == 128, "the head dim is not 128");

    // 36 Mamba layers + 4 attention layers.
    int n_mamba = 0, n_attn = 0;
    for (int k : cfg.layer_types) (k == 0 ? n_mamba : n_attn)++;
    CHECK(n_mamba == 36, "the Mamba layer count is not 36");
    CHECK(n_attn == 4, "the attention layer count is not 4");

    // ---- the parameter count ----
    // The reference checkpoint: 13,878,074,496 bytes of bf16 = 6,939,037,248
    // parameters (the index of the shard map).
    const u64 kCheckpointParams = 6939037248ull;
    const u64 engine = Model::count_parameters(cfg);
    std::printf("the engine expects   %llu parameters (%.3f B)\n",
                (unsigned long long)engine, engine / 1e9);
    std::printf("the checkpoint has   %llu parameters (%.3f B)\n",
                (unsigned long long)kCheckpointParams, kCheckpointParams / 1e9);
    std::printf("the difference       %lld\n",
                (long long)(engine - (i64)kCheckpointParams));
    CHECK(engine == kCheckpointParams,
          "the engine's parameter count differs from the checkpoint's");

    // ---- the shapes the loader expects ----
    const i64 d = cfg.hidden_size;
    const i64 d_inner = (i64)cfg.mamba_expand * d;
    const i64 conv_dim = d_inner + 2LL * cfg.mamba_n_groups * cfg.mamba_d_state;
    const i64 proj = d_inner + conv_dim + cfg.mamba_n_heads;
    const i64 E = cfg.moe_expert_dim;
    const i64 SH = cfg.shared_expert_intermediate_size;
    const i64 ne = cfg.num_experts;

    // The packed expert tensors of the checkpoint:
    //   block_sparse_moe.input_linear  [ne, 2E, d]
    //   block_sparse_moe.output_linear [ne, d, E]
    //   shared_mlp.input_linear        [2SH, d]
    //   shared_mlp.output_linear       [d, SH]
    CHECK(ne * 2 * E * d == 100663296ull, "the packed expert tensor is not the expected size");
    CHECK(ne * d * E == 50331648ull, "the packed down tensor is not the expected size");
    CHECK(2 * SH * d == 3145728ull, "the packed shared input tensor is not the expected size");
    CHECK(d * SH == 1572864ull, "the packed shared down tensor is not the expected size");
    std::printf("the Mamba geometry: d_inner=%lld conv_dim=%lld proj=%lld\n",
                (long long)d_inner, (long long)conv_dim, (long long)proj);
    CHECK(proj == d_inner + conv_dim + cfg.mamba_n_heads,
          "the Mamba projection size is inconsistent");
    CHECK(conv_dim == d_inner + 2LL * cfg.mamba_n_groups * cfg.mamba_d_state,
          "the Mamba conv size is inconsistent");
    CHECK(d_inner == (i64)cfg.mamba_expand * d, "the Mamba inner size is inconsistent");

    // ---- every engine parameter has a HuggingFace name ----
    //
    // The model is NOT built here: it is 6.9B parameters (27 GB of fp32 on
    // the host). The names are what the mapping has to cover, so they are
    // enumerated instead.
    {
        const std::vector<std::string> per_layer = {
            "layers.0.attn_norm", "layers.0.ffn_norm",
            "layers.0.mamba.in_proj", "layers.0.mamba.out_proj",
            "layers.0.mamba.conv1d.weight", "layers.0.mamba.conv1d.bias",
            "layers.0.mamba.dt_bias", "layers.0.mamba.A_log", "layers.0.mamba.D",
            "layers.0.mamba.norm.weight",
            "layers.0.wq", "layers.0.wk", "layers.0.wv", "layers.0.wo",
            "layers.0.moe_router",
        };
        for (const std::string& n : per_layer) {
            const std::string hn = hf_tensor_name(n, true);
            CHECK(!hn.empty(), ("no HuggingFace name for " + n).c_str());
            CHECK(hn.rfind("model.layers.0.", 0) == 0,
                  ("the name of " + n + " is not under model.layers.0: " + hn).c_str());
        }
        // The ROUTED experts have no 1:1 name: the loader gathers them from
        // the two packed tensors (input_linear / output_linear).
        static const char* kPacked[] = {"layers.0.moe_gate", "layers.0.moe_up",
                                        "layers.0.moe_down"};
        for (const char* n : kPacked) {
            CHECK(hf_tensor_name(n, true).empty(),
                  (std::string(n) + " must not have a 1:1 name (it is packed)").c_str());
        }
        // The shared expert points at the packed shared MLP tensors (the
        // loader splits them).
        CHECK(hf_tensor_name("layers.0.moe_sh_gate", true) ==
                  "model.layers.0.shared_mlp.input_linear.weight",
              "the shared gate name is wrong");
        CHECK(hf_tensor_name("layers.0.moe_sh_down", true) ==
                  "model.layers.0.shared_mlp.output_linear.weight",
              "the shared down name is wrong");
        CHECK(hf_tensor_name("layers.0.moe_router", true) ==
                  "model.layers.0.block_sparse_moe.router.layer.weight",
              "the router name is wrong");
        // The global ones.
        CHECK(hf_tensor_name("tok_embeddings", true) == "model.embed_tokens.weight",
              "the embedding name is wrong");
        CHECK(hf_tensor_name("final_norm", true) == "model.norm.weight",
              "the final norm name is wrong");
        CHECK(hf_tensor_name("lm_head", true) == "lm_head.weight",
              "the lm_head name is wrong (the checkpoint is tied: it has none)");
    }

    if (g_failures == 0) {
        std::printf("test_granite_compat: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_granite_compat: %d failure(s)\n", g_failures);
    return 1;
}
