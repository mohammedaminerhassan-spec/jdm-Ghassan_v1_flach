// LoRA / QLoRA sanity test (CPU reference path).
//
//  1. adapters are the only trainable parameters and receive non-zero grads
//  2. merge_lora() folds BA into the base weight so the forward output is
//     unchanged - this is the contract the finetuned GGUF export relies on.
//
// The same call sites are exercised for the quantized base in QLoRA mode on
// CUDA; the dequantized weights feed the identical *_half ops verified by
// test_quant.cpp.

#include "model/model.h"
#include "core/ops.h"
#include "core/dtype.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace gai;

static float max_abs(const Tensor& t) {
    float m = 0.0f;
    for (i64 i = 0; i < t.numel(); ++i) m = std::max(m, std::fabs(t.f32()[i]));
    return m;
}

int main() {
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
    cfg.lora_target_k = false;
    cfg.lora_target_v = true;
    cfg.lora_target_o = true;
    cfg.use_moe = false;

    Model model(cfg, Device::CPU, /*fp16_weights=*/false);
    model.init_weights(7);
    model.enable_grad(true);

    const i64 trainable = model.trainable_parameters();
    std::printf("[lora] trainable parameters = %lld\n", (long long)trainable);
    if (trainable <= 0 || trainable >= static_cast<i64>(cfg.vocab_size) * cfg.hidden_size) {
        std::printf("FAIL: LoRA did not restrict training to the adapters\n");
        return 1;
    }

    const int B = 1, T = 4;
    std::vector<i32> ids(static_cast<size_t>(B * T));
    std::vector<i32> tgt(static_cast<size_t>(B * T));
    for (int i = 0; i < B * T; ++i) {
        ids[i] = (i * 7 + 3) % cfg.vocab_size;
        tgt[i] = (i * 11 + 5) % cfg.vocab_size;
    }

    Activations act = model.make_activations(B, T, /*with_grad=*/true);
    model.zero_grad();
    const double loss = model.forward_backward(ids.data(), tgt.data(), B, T, act);
    std::printf("[lora] forward_backward loss = %.4f\n", loss);
    if (!std::isfinite(loss)) {
        std::printf("FAIL: non-finite LoRA loss\n");
        return 1;
    }

    float worst_grad = 0.0f;
    for (Parameter* p : model.parameters()) {
        if (p->name.find("lora_") != std::string::npos && p->g.defined())
            worst_grad = std::max(worst_grad, max_abs(p->g));
    }
    std::printf("[lora] worst adapter gradient = %.6f\n", worst_grad);
    if (worst_grad <= 0.0f) {
        std::printf("FAIL: LoRA adapters received no gradient\n");
        return 1;
    }

    // Give B non-zero values so the adapter is active, then verify that
    // folding it into the base preserves the function.
    for (Parameter* p : model.parameters()) {
        if (p->name.find("lora_") != std::string::npos &&
            p->name.size() >= 2 && p->name[p->name.size() - 1] == 'B') {
            for (i64 i = 0; i < p->numel(); ++i)
                p->w.f32()[i] = 0.01f * static_cast<float>((i % 7) - 3);
        }
    }
    Activations a1 = model.make_activations(B, T, false);
    Tensor before = model.forward(ids.data(), B, T, a1);
    std::vector<float> before_v(before.numel());
    for (i64 i = 0; i < before.numel(); ++i) before_v[i] = before.f32()[i];

    model.merge_lora();
    Activations a2 = model.make_activations(B, T, false);
    Tensor after = model.forward(ids.data(), B, T, a2);
    float worst = 0.0f;
    for (i64 i = 0; i < after.numel(); ++i)
        worst = std::max(worst, std::fabs(after.f32()[i] - before_v[i]));
    std::printf("[lora] merge worst |output diff| = %.6f\n", worst);
    if (worst > 2e-3f) {
        std::printf("FAIL: merge_lora changed the model output\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
