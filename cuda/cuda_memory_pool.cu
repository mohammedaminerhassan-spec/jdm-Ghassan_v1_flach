#include "cuda/cuda_memory_pool.cuh"
#include "cuda/cuda_utils.cuh"
#include <iostream>

#ifdef __CUDACC__

namespace gai {
namespace cuda {

CudaMemoryPool& CudaMemoryPool::instance() {
    static CudaMemoryPool s_pool;
    return s_pool;
}

CudaMemoryPool::~CudaMemoryPool() {
    shutdown();
}

bool CudaMemoryPool::init(int device_id, size_t release_threshold_bytes) {
    if (initialized_) return supported_;
    device_id_ = device_id;

#if (CUDART_VERSION >= 11020)
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    if (driver_version < 11020) {
        supported_ = false;
        initialized_ = true;
        return false;
    }

    int supports_async = 0;
    cudaDeviceGetAttribute(&supports_async, cudaDevAttrMemoryPoolsSupported, device_id);
    if (!supports_async) {
        supported_ = false;
        initialized_ = true;
        return false;
    }

    cudaError_t err = cudaDeviceGetDefaultMemPool(&mem_pool_, device_id);
    if (err != cudaSuccess || !mem_pool_) {
        supported_ = false;
        initialized_ = true;
        return false;
    }

    uint64_t threshold = static_cast<uint64_t>(release_threshold_bytes);
    cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &threshold);

    supported_ = true;
    initialized_ = true;
    return true;
#else
    supported_ = false;
    initialized_ = true;
    return false;
#endif
}

void CudaMemoryPool::shutdown() {
    if (initialized_ && supported_) {
        trim(0);
    }
    initialized_ = false;
    supported_ = false;
}

void* CudaMemoryPool::allocate(size_t bytes, void* stream_handle) {
    if (bytes == 0) return nullptr;
    if (!initialized_) init();

#if (CUDART_VERSION >= 11020)
    if (supported_ && mem_pool_) {
        cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
        if (err == cudaSuccess && ptr) {
            return ptr;
        }
    }
#endif
    return malloc_device(bytes);
}

void CudaMemoryPool::deallocate(void* ptr, void* stream_handle) {
    if (!ptr) return;

#if (CUDART_VERSION >= 11020)
    if (supported_ && mem_pool_) {
        cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        cudaError_t err = cudaFreeAsync(ptr, stream);
        if (err == cudaSuccess) return;
    }
#endif
    free_device(ptr);
}

void CudaMemoryPool::trim(size_t bytes_to_keep) {
#if (CUDART_VERSION >= 11020)
    if (supported_ && mem_pool_) {
        cudaMemPoolTrimTo(mem_pool_, bytes_to_keep);
    }
#else
    (void)bytes_to_keep;
#endif
}

size_t CudaMemoryPool::reserved_bytes() const {
#if (CUDART_VERSION >= 11020)
    if (supported_ && mem_pool_) {
        uint64_t reserved = 0;
        cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrReservedMemCurrent, &reserved);
        return static_cast<size_t>(reserved);
    }
#endif
    return 0;
}

size_t CudaMemoryPool::used_bytes() const {
#if (CUDART_VERSION >= 11020)
    if (supported_ && mem_pool_) {
        uint64_t used = 0;
        cudaMemPoolGetAttribute(mem_pool_, cudaMemPoolAttrUsedMemCurrent, &used);
        return static_cast<size_t>(used);
    }
#endif
    return 0;
}

void* pool_alloc(size_t bytes, void* stream) {
    return CudaMemoryPool::instance().allocate(bytes, stream);
}

void pool_free(void* ptr, void* stream) {
    CudaMemoryPool::instance().deallocate(ptr, stream);
}

void pool_trim(size_t bytes_to_keep) {
    CudaMemoryPool::instance().trim(bytes_to_keep);
}

}
}

#endif
