#include "inference/paged_kv_cache.h"
#include <cstring>
#include <algorithm>

namespace gai {

PagedKVCache::PagedKVCache(const ModelConfig& cfg, int max_total_tokens, Device dev,
                           int block_size, bool fp16)
    : dev_(dev), block_size_(block_size), fp16_(fp16) {
    GAI_CHECK(block_size_ > 0, "block_size must be positive");
    GAI_CHECK(max_total_tokens > 0, "max_total_tokens must be positive");

    total_blocks_ = (max_total_tokens + block_size_ - 1) / block_size_;
    kv_dim_ = cfg.head_dim() * cfg.num_kv_heads;
    num_layers_ = 0;
    if (!cfg.layer_types.empty()) {
        for (int k : cfg.layer_types) {
            if (k == 1) num_layers_++;
        }
    } else {
        num_layers_ = cfg.num_layers;
    }
    if (num_layers_ == 0) num_layers_ = 1;

    free_block_ids_.clear();
    for (int b = 0; b < total_blocks_; ++b) {
        free_block_ids_.push_back(b);
    }

    const DType dt = fp16_ ? DType::F16 : DType::F32;
    const std::vector<i64> shape = {
        static_cast<i64>(total_blocks_),
        static_cast<i64>(num_layers_),
        static_cast<i64>(block_size_),
        static_cast<i64>(kv_dim_)
    };

    k_storage_ = Tensor(shape, dt, dev_);
    v_storage_ = Tensor(shape, dt, dev_);
    k_storage_.zero_();
    v_storage_.zero_();

    current_length_ = 0;
}

int PagedKVCache::allocate_block() {
    if (free_block_ids_.empty()) return -1;
    int b = free_block_ids_.front();
    free_block_ids_.pop_front();
    return b;
}

void PagedKVCache::free_block(int block_id) {
    if (block_id >= 0 && block_id < total_blocks_) {
        free_block_ids_.push_back(block_id);
    }
}

bool PagedKVCache::reserve_tokens(int num_tokens) {
    const int new_length = current_length_ + num_tokens;
    const int needed_blocks = (new_length + block_size_ - 1) / block_size_;
    while (static_cast<int>(block_table_.size()) < needed_blocks) {
        int b = allocate_block();
        if (b < 0) return false;
        block_table_.push_back(b);
    }
    return true;
}

void PagedKVCache::append_token(int layer_idx, const float* k_vec, const float* v_vec) {
    const int logical_idx = current_length_;
    const int block_idx = logical_idx / block_size_;
    const int offset = logical_idx % block_size_;

    if (block_idx >= static_cast<int>(block_table_.size())) {
        int b = allocate_block();
        GAI_CHECK(b >= 0, "PagedKVCache: out of physical blocks in VRAM pool");
        block_table_.push_back(b);
    }

    const int physical_block = block_table_[block_idx];
    const size_t stride_block = static_cast<size_t>(num_layers_) * block_size_ * kv_dim_;
    const size_t stride_layer = static_cast<size_t>(block_size_) * kv_dim_;
    const size_t element_idx = (static_cast<size_t>(physical_block) * stride_block) +
                               (static_cast<size_t>(layer_idx) * stride_layer) +
                               (static_cast<size_t>(offset) * kv_dim_);

    if (dev_ == Device::CPU) {
        if (fp16_) {
            u16* k_dst = k_storage_.ptr<u16>() + element_idx;
            u16* v_dst = v_storage_.ptr<u16>() + element_idx;
            for (int i = 0; i < kv_dim_; ++i) {
                k_dst[i] = fp32_to_fp16(k_vec[i]);
                v_dst[i] = fp32_to_fp16(v_vec[i]);
            }
        } else {
            float* k_dst = k_storage_.f32() + element_idx;
            float* v_dst = v_storage_.f32() + element_idx;
            std::memcpy(k_dst, k_vec, sizeof(float) * kv_dim_);
            std::memcpy(v_dst, v_vec, sizeof(float) * kv_dim_);
        }
    } else {
        if (fp16_) {
            std::vector<u16> k_h(kv_dim_), v_h(kv_dim_);
            for (int i = 0; i < kv_dim_; ++i) {
                k_h[i] = fp32_to_fp16(k_vec[i]);
                v_h[i] = fp32_to_fp16(v_vec[i]);
            }
            device_copy(k_storage_.ptr<u16>() + element_idx, dev_,
                        k_h.data(), Device::CPU, sizeof(u16) * kv_dim_);
            device_copy(v_storage_.ptr<u16>() + element_idx, dev_,
                        v_h.data(), Device::CPU, sizeof(u16) * kv_dim_);
        } else {
            device_copy(k_storage_.ptr<float>() + element_idx, dev_,
                        k_vec, Device::CPU, sizeof(float) * kv_dim_);
            device_copy(v_storage_.ptr<float>() + element_idx, dev_,
                        v_vec, Device::CPU, sizeof(float) * kv_dim_);
        }
    }

    if (layer_idx == num_layers_ - 1) {
        current_length_++;
    }
}

void PagedKVCache::reset() {
    for (int b : block_table_) {
        free_block(b);
    }
    block_table_.clear();
    current_length_ = 0;
}

size_t PagedKVCache::memory_bytes() const {
    const size_t elem_size = fp16_ ? sizeof(u16) : sizeof(float);
    return static_cast<size_t>(total_blocks_) * num_layers_ * block_size_ * kv_dim_ * elem_size * 2;
}

}
