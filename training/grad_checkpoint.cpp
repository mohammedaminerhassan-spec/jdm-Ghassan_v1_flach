#include "training/grad_checkpoint.h"

namespace gai {

ActivationCheckpointManager::ActivationCheckpointManager(int total_layers, int checkpoint_every_n)
    : total_layers_(total_layers), interval_(checkpoint_every_n) {
    if (interval_ <= 0) interval_ = 1;
}

bool ActivationCheckpointManager::should_checkpoint(int layer_idx) const {
    if (interval_ <= 1) return true;
    return (layer_idx % interval_ == 0);
}

void ActivationCheckpointManager::save_checkpoint(int layer_idx, const Tensor& act) {
    checkpoints_[layer_idx] = act.clone();
}

Tensor ActivationCheckpointManager::get_checkpoint(int layer_idx) const {
    auto it = checkpoints_.find(layer_idx);
    if (it != checkpoints_.end()) {
        return it->second;
    }
    return Tensor();
}

void ActivationCheckpointManager::clear() {
    checkpoints_.clear();
}

size_t ActivationCheckpointManager::memory_saved_bytes(size_t layer_activation_bytes) const {
    if (total_layers_ <= 0 || interval_ <= 1) return 0;
    const int saved_checkpoints = (total_layers_ + interval_ - 1) / interval_;
    const int skipped_layers = total_layers_ - saved_checkpoints;
    return static_cast<size_t>(skipped_layers) * layer_activation_bytes;
}

}
