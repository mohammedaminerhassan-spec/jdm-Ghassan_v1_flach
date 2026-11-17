#pragma once

#include "core/tensor.h"
#include "core/ops.h"

namespace gai {

// Map a storage dtype to the dequantization kernel kind.
//   -1 : not a quantized dtype (F32 / F16 / BF16 / ...)
//    0 : Q4_0   (32 values / 18 bytes)
//    1 : Q4_1   (32 values / 20 bytes)
//    2 : Q8_0   (32 values / 34 bytes)
inline int quant_kind(DType t) {
    if (t == DType::Q4_0) return 0;
    if (t == DType::Q4_1) return 1;
    if (t == DType::Q8_0) return 2;
    return -1;
}

inline bool is_quant(const Tensor& w) {
    return w.defined() && quant_kind(w.dtype()) >= 0;
}

// True when a weight can be consumed by a *_half / *_fp16 op (either stored
// as FP16, or quantized and materializable into FP16 scratch).
inline bool is_half_or_quant(const Tensor& w) {
    return w.defined() && (w.dtype() == DType::F16 || quant_kind(w.dtype()) >= 0);
}

// Return an FP16 pointer usable by the *_half / *_fp16 ops for a weight.
//
//  * F16 weights  -> the tensor's own storage, no copy.
//  * Q4/Q8 weights -> dequantized once into a shared scratch bank (slot),
//                    which is reused by later ops on the same CUDA stream.
//  * F32 weights  -> nullptr (callers must use w.f32() instead).
//
// `slot` must be distinct only for weights consumed by the SAME kernel
// (e.g. the six MoE expert matrices); sequential ops may all use slot 0
// because CUDA stream ordering guarantees the GEMM reads the scratch before
// the next dequant overwrites it.
inline const u16* weight16(Device dev, const Tensor& w, int slot = 0) {
    if (!w.defined()) return nullptr;
    if (w.dtype() == DType::F16) return w.ptr<u16>();
    const int k = quant_kind(w.dtype());
    if (k >= 0) return ops::quant_materialize(dev, w.data_ptr(), k, w.numel(), slot);
    return nullptr;
}

// A weight ready to feed the fused dequant-GEMM path: when the engine runs
// native Q4 on CUDA with --fused-q4, the PACKED bytes are passed straight to
// the kernel (no FP16 scratch, no 13 GB write); otherwise `ptr` is the same
// FP16 view `weight16()` produces and `qkind` is -1.
struct WeightRef {
    const u16* ptr = nullptr; // nullptr => caller must use w.f32() instead
    int        qkind = -1;    // -1 => fp16 values via ptr; 0/1/2 => packed Q4_0/Q4_1/Q8_0
};

inline WeightRef weight_ref(Device dev, const Tensor& w, int slot = 0) {
    if (!w.defined()) return {};
    const int k = quant_kind(w.dtype());
    if (k >= 0 && dev == Device::CUDA && ops::fused_q4_enabled())
        return { reinterpret_cast<const u16*>(w.data_ptr()), k };
    return { weight16(dev, w, slot), -1 };
}

} // namespace gai
