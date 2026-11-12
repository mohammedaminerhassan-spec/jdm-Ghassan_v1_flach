#include "inference/kv_cache.h"
#include "core/device.h"

#include <cstring>
#include <vector>

namespace gai {

KVCache::KVCache(const ModelConfig& cfg, int max_len, Device dev)
    : max_len_(max_len), kv_dim_(cfg.kv_dim()), len_(0),
      pinned_(0), start_(0) {
    // IBM Granite-4.0: only the attention layers keep a KV cache. The mamba
    // layers keep a recurrent state instead, so allocating K/V for them is
    // pure waste (10x more cache than the planner reports on granite-4.0-h-
    // tiny: 40 layers of cache for 4 attention layers). layer_to_cache_ maps
    // the model layer index to the compact cache index, or -1 when the layer
    // keeps no KV cache.
    const int L = cfg.num_layers;
    layer_to_cache_.assign(static_cast<size_t>(L), -1);
    int kv = 0;
    for (int i = 0; i < L; ++i) {
        bool attention = true;
        if (cfg.arch == ModelConfig::ArchGranite &&
            i < static_cast<int>(cfg.layer_types.size())) {
            attention = cfg.layer_types[static_cast<size_t>(i)] != 0;
        }
        if (attention) {
            layer_to_cache_[static_cast<size_t>(i)] = kv;
            ++kv;
        }
    }
    layers_ = kv;
    k_.resize(static_cast<size_t>(layers_));
    v_.resize(static_cast<size_t>(layers_));
    for (int i = 0; i < layers_; ++i) {
        if (fp16_) {
            // fp16 storage: half the bytes of the fp32 cache.
            k_[static_cast<size_t>(i)] = Tensor::zeros({max_len_, kv_dim_}, DType::F16, dev);
            v_[static_cast<size_t>(i)] = Tensor::zeros({max_len_, kv_dim_}, DType::F16, dev);
        } else {
            k_[static_cast<size_t>(i)] = Tensor::zeros({max_len_, kv_dim_}, DType::F32, dev);
            v_[static_cast<size_t>(i)] = Tensor::zeros({max_len_, kv_dim_}, DType::F32, dev);
        }
    }
}

int KVCache::cache_index(int layer) const {
    GAI_CHECK(layer >= 0 && layer < static_cast<int>(layer_to_cache_.size()),
              "KVCache: layer index out of range");
    const int idx = layer_to_cache_[static_cast<size_t>(layer)];
    GAI_CHECK(idx >= 0,
              strfmt("KVCache: layer %d keeps no KV cache (a granite mamba layer)",
                     layer));
    return idx;
}

size_t KVCache::bytes() const {
    size_t n = 0;
    for (const auto& t : k_) n += t.nbytes();
    for (const auto& t : v_) n += t.nbytes();
    return n;
}

}
