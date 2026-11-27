#pragma once

#include "core/common.h"
#include "core/tensor.h"
#include "model/model.h"

#include <vector>
#include <functional>
#include <map>

namespace gai {

class ActivationCheckpointManager {
public:
    ActivationCheckpointManager(int total_layers = 24, int checkpoint_every_n = 4);

    bool should_checkpoint(int layer_idx) const;

    void save_checkpoint(int layer_idx, const Tensor& act);

    Tensor get_checkpoint(int layer_idx) const;

    void clear();

    int total_layers() const { return total_layers_; }
    int interval() const { return interval_; }
    size_t memory_saved_bytes(size_t layer_activation_bytes) const;

private:
    int total_layers_ = 24;
    int interval_ = 4;
    std::map<int, Tensor> checkpoints_;
};

}
