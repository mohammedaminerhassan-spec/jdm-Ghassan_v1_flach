#pragma once

#include "core/common.h"
#include "core/device.h"

namespace gai {

struct MemoryStats {
    size_t free_ram_bytes = 0;
    size_t total_ram_bytes = 0;
    size_t free_vram_bytes = 0;
    size_t total_vram_bytes = 0;
};

MemoryStats get_memory_stats(Device dev = Device::CPU);

void clean_cpu_memory();

void clean_gpu_memory(int device_id = 0);

void purge_all_caches(Device dev = Device::CUDA);

}
