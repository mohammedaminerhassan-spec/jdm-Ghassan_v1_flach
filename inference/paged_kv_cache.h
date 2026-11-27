#pragma once

#include "core/common.h"
#include "core/tensor.h"
#include "core/device.h"
#include "model/model.h"

#include <vector>
#include <memory>
#include <deque>

namespace gai {

constexpr int DEFAULT_BLOCK_SIZE = 16;

struct PhysicalBlock {
    int block_id = -1;
    int ref_count = 0;
};

class PagedKVCache {
public:
    PagedKVCache() = default;
    PagedKVCache(const ModelConfig& cfg, int max_total_tokens, Device dev,
                 int block_size = DEFAULT_BLOCK_SIZE, bool fp16 = true);

    bool reserve_tokens(int num_tokens);
    void append_token(int layer_idx, const float* k_vec, const float* v_vec);

    void reset();

    int block_size() const { return block_size_; }
    int total_blocks() const { return total_blocks_; }
    int free_blocks() const { return static_cast<int>(free_block_ids_.size()); }
    int used_blocks() const { return total_blocks_ - free_blocks(); }
    int current_length() const { return current_length_; }
    size_t memory_bytes() const;

    const std::vector<int>& block_table() const { return block_table_; }

    Tensor& k_storage() { return k_storage_; }
    Tensor& v_storage() { return v_storage_; }

private:
    int allocate_block();
    void free_block(int block_id);

    Device dev_ = Device::CPU;
    int block_size_ = DEFAULT_BLOCK_SIZE;
    int total_blocks_ = 0;
    int kv_dim_ = 0;
    int num_layers_ = 0;
    int current_length_ = 0;
    bool fp16_ = true;

    Tensor k_storage_;
    Tensor v_storage_;

    std::vector<int> block_table_;
    std::deque<int> free_block_ids_;
};

}
