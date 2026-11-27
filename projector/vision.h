#pragma once

#include "projector/projector.h"
#include "projector/image_video_loader.h"

namespace gai {

inline bool has_vision_tower(const std::string& path) {
    auto vt = load_vision_from_gguf(path, Device::CPU);
    return vt != nullptr && vt->valid();
}

}
