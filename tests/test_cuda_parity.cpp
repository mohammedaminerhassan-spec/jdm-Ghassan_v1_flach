// The CPU <-> CUDA parity of the gradients.
//
// This is the test that validates the CUDA side of the audit on the GPU
// itself (the CPU tests never touch the CUDA kernels):
//
//   * the GEMM chunking (the cuBLAS leading-dimension fix): the LoRA
//     adapters GEMM against a rank of 8..64 with hundreds of tokens, which
//     is exactly the case that used to make cuBLAS refuse the call;
//   * the MoE backward (the shared expert width),
//   * the attention path (the GEMM path, the fp16 mirror).
//
// It is skipped (and it says so) when the build has no CUDA.

#include "core/common.h"
#include "core/device.h"
#include "core/ops.h"
#include "model/model.h"

#include <cmath>
#include <cstdio>
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

static std::vector<i32> make_batch(int V, int B, int T, u64 seed) {
    std::vector<i32> ids(static_cast<size_t>(B) * T);
    u64 s = seed;
    for (auto& id : ids) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id = static_cast<i32>((s >> 33) % static_cast<u64>(V));
    }
    return ids;
}

// The gradients of a model on a device.
static void gradients_on(Model& m, const std::vector<i32>& ids, int B, int T) {
    m.enable_grad(true);
    m.zero_grad();
    Activations act = m.make_activations(B, T, true);
    Tensor d_ids({B * T}, DType::I32, Device::CPU);
    std::memcpy(d_ids.data_ptr(), ids.data(), sizeof(i32) * ids.size());
    Tensor d_tgt({B * T}, DType::I32, Device::CPU);
    for (int i = 0; i < B * T; ++i) d_tgt.i32p()[i] = ids[static_cast<size_t>(i)];
    // The targets live on the device for the CUDA path.
    Tensor dev_ids = d_ids.to(m.device());
    Tensor dev_tgt = d_tgt.to(m.device());
    i64 ntok = 0;
    m.forward_backward(dev_ids.i32p(), dev_tgt.i32p(), B, T, act, &ntok, 1.0f,
                       false, nullptr, d_tgt.i32p());
}

// Compares the gradients of two models (the CPU one and the CUDA one).
static void compare(const Model& cpu, const Model& gpu, const std::string& tag) {
    for (const Parameter* p : cpu.parameters()) {
        const Parameter* q = gpu.find_parameter(p->name);
        if (!q) {
            CHECK(false, (tag + ": the CUDA model has no parameter " + p->name).c_str());
            continue;
        }
        if (!p->g.defined() || !q->g.defined()) continue;
        const Tensor a = p->g.to(Device::CPU);
        const Tensor b = q->g.to(Device::CPU);
        if (a.numel() != b.numel()) {
            CHECK(false, (tag + ": " + p->name + " has a different size").c_str());
            continue;
        }
        double worst = 0.0;
        double scale = 0.0;
        for (i64 i = 0; i < a.numel(); ++i) {
            const double x = a.f32()[i];
            const double y = b.f32()[i];
            worst = std::max(worst, std::fabs(x - y));
            scale = std::max(scale, std::fabs(x));
        }
        // The relative tolerance: the CPU and the CUDA sum in a different
        // order (and the CUDA one uses the tensor cores), so the last bits
        // differ. 2% of the largest gradient of the tensor is the bar.
        const double tol = 0.02 * scale + 1e-6;
        if (worst > tol) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s %s: CPU vs CUDA differ by %.3e (scale %.3e)",
                          tag.c_str(), p->name.c_str(), worst, scale);
            CHECK(false, buf);
        }
    }
}

static void run_case(const ModelConfig& cfg, const std::string& tag, int B, int T) {
    if (!cuda_available()) {
        std::printf("test_cuda_parity: no CUDA device -> skipped\n");
        return;
    }
    const std::vector<i32> ids = make_batch(cfg.vocab_size, B, T, 4242);
    const u64 seed = 777;

    Model cpu(cfg, Device::CPU);
    cpu.init_weights(seed);
    gradients_on(cpu, ids, B, T);

    Model gpu(cfg, Device::CUDA);
    gpu.init_weights(seed);
    gradients_on(gpu, ids, B, T);

    compare(cpu, gpu, tag);
}

// A dense model: the plain path.
static void test_dense() {
    ModelConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 96;
    cfg.max_seq_len = 64;
    cfg.tie_embeddings = true;
    run_case(cfg, "dense", 2, 16);
}

// A MoE model with a shared expert of a DIFFERENT width: the case the
// audit's BUG-02 was about.
static void test_moe() {
    ModelConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 96;
    cfg.max_seq_len = 64;
    cfg.use_moe = true;
    cfg.num_experts = 4;
    cfg.moe_top_k = 2;
    cfg.moe_expert_dim = 32;
    cfg.moe_shared = true;
    cfg.shared_expert_intermediate_size = 48;
    run_case(cfg, "moe", 2, 16);
}

// A LoRA model: the rank is 8 and the tokens are 64, so EVERY LoRA GEMM
// violates the cuBLAS leading dimension (lda = rank < n = tokens). This is
// the case that used to make cuBLAS refuse the call on the GPU.
static void test_lora() {
    ModelConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.intermediate_size = 96;
    cfg.max_seq_len = 64;
    cfg.tie_embeddings = true;
    cfg.lora_rank = 8;
    cfg.lora_alpha = 16.0f;
    cfg.lora_target_q = true;
    cfg.lora_target_k = true;
    cfg.lora_target_v = true;
    cfg.lora_target_o = true;
    run_case(cfg, "lora", 2, 64);
}

int main() {
    test_dense();
    test_moe();
    test_lora();
    if (g_failures == 0) {
        std::printf("test_cuda_parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_cuda_parity: %d failure(s)\n", g_failures);
    return 1;
}
