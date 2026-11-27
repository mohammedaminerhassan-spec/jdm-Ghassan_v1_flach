#pragma once

#include "core/common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace gai {
namespace cuda {

class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    CudaGraphRunner(CudaGraphRunner&& other) noexcept;
    CudaGraphRunner& operator=(CudaGraphRunner&& other) noexcept;
    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    static bool is_supported();

    bool begin_capture(void* stream_handle);

    bool end_capture(void* stream_handle);

    bool instantiate();

    bool launch(void* stream_handle);

    void reset();

    bool is_ready() const { return ready_; }

    size_t node_count() const;

private:
#ifdef __CUDACC__
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t instance_ = nullptr;
#else
    void* graph_ = nullptr;
    void* instance_ = nullptr;
#endif
    bool capturing_ = false;
    bool ready_ = false;
};

}
}
