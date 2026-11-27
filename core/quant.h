#pragma once

#include "core/tensor.h"
#include "core/ops.h"

namespace gai {

inline int quant_kind(DType t) {
    if (t == DType::Q4_0) return 0;
    if (t == DType::Q4_1) return 1;
    if (t == DType::Q8_0) return 2;
    return -1;
}

inline bool is_quant(const Tensor& w) {
    return w.defined() && quant_kind(w.dtype()) >= 0;
}

inline bool is_half_or_quant(const Tensor& w) {
    return w.defined() && (w.dtype() == DType::F16 || quant_kind(w.dtype()) >= 0);
}

inline const u16* weight16(Device dev, const Tensor& w, int slot = 0) {
    if (!w.defined()) return nullptr;
    if (w.dtype() == DType::F16) return w.ptr<u16>();
    const int k = quant_kind(w.dtype());
    if (k >= 0) return ops::quant_materialize(dev, w.data_ptr(), k, w.numel(), slot);
    return nullptr;
}

struct WeightRef {
    const u16* ptr = nullptr;
    int        qkind = -1;
};

inline WeightRef weight_ref(Device dev, const Tensor& w, int slot = 0) {
    if (!w.defined()) return {};
    const int k = quant_kind(w.dtype());
    if (k >= 0 && dev == Device::CUDA && ops::fused_q4_enabled())
        return { reinterpret_cast<const u16*>(w.data_ptr()), k };
    return { weight16(dev, w, slot), -1 };
}

}
