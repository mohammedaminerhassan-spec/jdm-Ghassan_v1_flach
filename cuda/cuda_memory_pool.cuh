#pragma once

#include "core/common.h"

#if defined(__CUDACC__) || defined(GAI_CUDA)
#include <cuda_runtime.h>
#endif

namespace gai {
namespace cuda {

class CudaMemoryPool {
public:
    static CudaMemoryPool& instance();

    bool init(int device_id = 0, size_t release_threshold_bytes = (1024ULL * 1024 * 1024));
    void shutdown();

    void* allocate(size_t bytes, void* stream_handle = nullptr);
    void  deallocate(void* ptr, void* stream_handle = nullptr);
    void  trim(size_t bytes_to_keep = 0);

    bool is_supported() const { return supported_; }
    size_t reserved_bytes() const;
    size_t used_bytes() const;

private:
    CudaMemoryPool() = default;
    ~CudaMemoryPool();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

#if defined(__CUDACC__) && (CUDART_VERSION >= 11020)
    cudaMemPool_t mem_pool_ = nullptr;
#else
    void* mem_pool_ = nullptr;
#endif
    int device_id_ = 0;
    bool supported_ = false;
    bool initialized_ = false;
};

void* pool_alloc(size_t bytes, void* stream = nullptr);
void  pool_free(void* ptr, void* stream = nullptr);
void  pool_trim(size_t bytes_to_keep = 0);

}
}
