#include "core/memory_cleaner.h"

#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_utils.cuh"
#include "cuda/cuda_memory_pool.cuh"
#include "cuda/cuda_ops.cuh"
#endif

#include <iostream>

#if defined(_WIN32) || defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <malloc.h>
#endif

namespace gai {

MemoryStats get_memory_stats(Device dev) {
    MemoryStats stats;

#if defined(_WIN32) || defined(_MSC_VER)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        stats.total_ram_bytes = static_cast<size_t>(mem_status.ullTotalPhys);
        stats.free_ram_bytes = static_cast<size_t>(mem_status.ullAvailPhys);
    }
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        stats.total_ram_bytes = static_cast<size_t>(info.totalram) * info.mem_unit;
        stats.free_ram_bytes = static_cast<size_t>(info.freeram) * info.mem_unit;
    }
#endif

#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        size_t free_vram = 0, total_vram = 0;
        if (cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess) {
            stats.free_vram_bytes = free_vram;
            stats.total_vram_bytes = total_vram;
        }
    }
#else
    (void)dev;
#endif

    return stats;
}

void clean_cpu_memory() {
#if defined(__linux__) && defined(__GLIBC__)
    malloc_trim(0);
#elif defined(_WIN32) || defined(_MSC_VER)
    HANDLE hHeap = GetProcessHeap();
    if (hHeap) {
        HeapCompact(hHeap, 0);
    }
#endif
}

void clean_gpu_memory(int device_id) {
#ifdef GAI_CUDA
    (void)device_id;
    cuda::pool_trim(0);

    cuda_ops::free_quant_scratch();
    cuda_ops::free_workspaces();

    cudaDeviceSynchronize();
#else
    (void)device_id;
#endif
}

void purge_all_caches(Device dev) {
    MemoryStats before = get_memory_stats(dev);

    clean_cpu_memory();
    if (dev == Device::CUDA) {
        clean_gpu_memory(0);
    }

    MemoryStats after = get_memory_stats(dev);

    size_t ram_freed = (after.free_ram_bytes > before.free_ram_bytes)
        ? (after.free_ram_bytes - before.free_ram_bytes) : 0;
    size_t vram_freed = (after.free_vram_bytes > before.free_vram_bytes)
        ? (after.free_vram_bytes - before.free_vram_bytes) : 0;

    log_info(strfmt("[memory_cleaner] purged caches: reclaimed %zu MB RAM, %zu MB VRAM (free VRAM: %zu MB)",
                    ram_freed / (1024 * 1024),
                    vram_freed / (1024 * 1024),
                    after.free_vram_bytes / (1024 * 1024)));
}

}
