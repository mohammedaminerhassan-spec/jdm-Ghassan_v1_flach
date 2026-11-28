#include "inference/paged_kv_cache.h"
#include <cstring>
#include <algorithm>
#include <limits>

namespace gai {

PagedKVCache::PagedKVCache(const ModelConfig& cfg, int max_total_tokens, Device dev,
                           int block_size, bool fp16)
    : dev_(dev), block_size_(block_size), fp16_(fp16) {
    GAI_CHECK(block_size_ > 0, "block_size must be positive");
    GAI_CHECK(max_total_tokens > 0, "max_total_tokens must be positive");

    const i64 blocks64 =
        (static_cast<i64>(max_total_tokens) + block_size_ - 1) / block_size_;
    GAI_CHECK(blocks64 > 0 && blocks64 <= (std::numeric_limits<int>::max)(),
              "PagedKVCache: total block count exceeds int range");
    total_blocks_ = static_cast<int>(blocks64);
    max_tokens_ = max_total_tokens;

    const i64 kv_dim64 = static_cast<i64>(cfg.head_dim()) *
                         static_cast<i64>(cfg.num_kv_heads);
    GAI_CHECK(kv_dim64 > 0 && kv_dim64 <= (std::numeric_limits<int>::max)(),
              "PagedKVCache: KV dimension exceeds int range");
    kv_dim_ = static_cast<int>(kv_dim64);
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
    block_allocated_.assign(static_cast<size_t>(total_blocks_), 0);
    layer_written_.assign(static_cast<size_t>(num_layers_), 0);
    for (int b = 0; b < total_blocks_; ++b) {
        free_block_ids_.push_back(b);
    }

    if (dev_ == Device::CUDA && fp16_) {
        host_k_stage_f16_.resize(static_cast<size_t>(kv_dim_));
        host_v_stage_f16_.resize(static_cast<size_t>(kv_dim_));
    }

    const size_t max_size = (std::numeric_limits<size_t>::max)();
    size_t total_elems = static_cast<size_t>(total_blocks_);
    auto checked_mul = [&](size_t factor) {
        GAI_CHECK(factor == 0 || total_elems <= max_size / factor,
                  "PagedKVCache: storage size overflow");
        total_elems *= factor;
    };
    checked_mul(static_cast<size_t>(num_layers_));
    checked_mul(static_cast<size_t>(block_size_));
    checked_mul(static_cast<size_t>(kv_dim_));
    GAI_CHECK(total_elems <= static_cast<size_t>((std::numeric_limits<i64>::max)()),
              "PagedKVCache: tensor element count exceeds i64 range");

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
    const int b = free_block_ids_.front();
    free_block_ids_.pop_front();
    GAI_CHECK(b >= 0 && b < total_blocks_, "PagedKVCache: corrupt free-block id");
    GAI_CHECK(block_allocated_[static_cast<size_t>(b)] == 0,
              "PagedKVCache: duplicate block id in free list");
    block_allocated_[static_cast<size_t>(b)] = 1;
    return b;
}

void PagedKVCache::free_block(int block_id) {
    if (block_id < 0 || block_id >= total_blocks_) return;
    const size_t index = static_cast<size_t>(block_id);
    GAI_CHECK(block_allocated_[index] != 0,
              "PagedKVCache: double free or unowned block");
    block_allocated_[index] = 0;
    free_block_ids_.push_back(block_id);
}

bool PagedKVCache::reserve_tokens(int num_tokens) {
    GAI_CHECK(num_tokens >= 0, "PagedKVCache::reserve_tokens requires a non-negative count");
    const i64 new_length = static_cast<i64>(current_length_) + num_tokens;
    if (new_length > max_tokens_) return false;
    if (num_tokens == 0) return true;

    const i64 needed64 = (new_length + block_size_ - 1) / block_size_;
    GAI_CHECK(needed64 <= total_blocks_, "PagedKVCache: required blocks exceed pool capacity");
    const size_t needed = static_cast<size_t>(needed64);

    // Reserve container capacity before acquiring physical blocks, so vector
    // growth cannot throw after blocks have been removed from the free list.
    const size_t additional = needed > block_table_.size()
        ? needed - block_table_.size() : 0;
    block_table_.reserve(needed);

    // Strong failure guarantee: if the pool cannot satisfy the whole request,
    // return all blocks acquired by this call and leave the block table unchanged.
    std::vector<int> acquired;
    acquired.reserve(additional);
    while (block_table_.size() + acquired.size() < needed) {
        const int b = allocate_block();
        if (b < 0) {
            for (int id : acquired) free_block(id);
            return false;
        }
        acquired.push_back(b);
    }
    block_table_.insert(block_table_.end(), acquired.begin(), acquired.end());
    return true;
}

void PagedKVCache::append_token(int layer_idx, const float* k_vec, const float* v_vec) {
    GAI_CHECK(layer_idx >= 0 && layer_idx < num_layers_,
              "PagedKVCache::append_token layer index out of range");
    GAI_CHECK(k_vec && v_vec, "PagedKVCache::append_token received a null KV vector");
    GAI_CHECK(current_length_ < max_tokens_,
              "PagedKVCache::append_token exceeds configured token capacity");
    GAI_CHECK(layer_written_[static_cast<size_t>(layer_idx)] == 0,
              "PagedKVCache::append_token received a duplicate layer for the current token");
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
            for (int i = 0; i < kv_dim_; ++i) {
                host_k_stage_f16_[static_cast<size_t>(i)] = fp32_to_fp16(k_vec[i]);
                host_v_stage_f16_[static_cast<size_t>(i)] = fp32_to_fp16(v_vec[i]);
            }
            device_copy(k_storage_.ptr<u16>() + element_idx, dev_,
                        host_k_stage_f16_.data(), Device::CPU, sizeof(u16) * kv_dim_);
            device_copy(v_storage_.ptr<u16>() + element_idx, dev_,
                        host_v_stage_f16_.data(), Device::CPU, sizeof(u16) * kv_dim_);
        } else {
            device_copy(k_storage_.ptr<float>() + element_idx, dev_,
                        k_vec, Device::CPU, sizeof(float) * kv_dim_);
            device_copy(v_storage_.ptr<float>() + element_idx, dev_,
                        v_vec, Device::CPU, sizeof(float) * kv_dim_);
        }
    }

    layer_written_[static_cast<size_t>(layer_idx)] = 1;
    const bool token_complete = std::all_of(
        layer_written_.begin(), layer_written_.end(),
        [](u8 written) { return written != 0; });
    if (token_complete) {
        ++current_length_;
        std::fill(layer_written_.begin(), layer_written_.end(), 0);
    }
}

void PagedKVCache::reset() {
    for (int b : block_table_) {
        free_block(b);
    }
    block_table_.clear();
    current_length_ = 0;
    std::fill(layer_written_.begin(), layer_written_.end(), 0);
}

size_t PagedKVCache::memory_bytes() const {
    const size_t kb = k_storage_.nbytes();
    const size_t vb = v_storage_.nbytes();
    const size_t maxv = (std::numeric_limits<size_t>::max)();
    GAI_CHECK(kb <= maxv - vb, "PagedKVCache::memory_bytes overflow");
    return kb + vb;
}

}
