#include "cuda/cuda_graphs.cuh"
#include "cuda/cuda_utils.cuh"
#include <iostream>

#ifdef __CUDACC__

namespace gai {
namespace cuda {

CudaGraphRunner::CudaGraphRunner() = default;

CudaGraphRunner::~CudaGraphRunner() {
    reset();
}

CudaGraphRunner::CudaGraphRunner(CudaGraphRunner&& o) noexcept
    : graph_(o.graph_), instance_(o.instance_),
      capturing_(o.capturing_), ready_(o.ready_) {
    o.graph_ = nullptr;
    o.instance_ = nullptr;
    o.capturing_ = false;
    o.ready_ = false;
}

CudaGraphRunner& CudaGraphRunner::operator=(CudaGraphRunner&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = o.graph_;
        instance_ = o.instance_;
        capturing_ = o.capturing_;
        ready_ = o.ready_;
        o.graph_ = nullptr;
        o.instance_ = nullptr;
        o.capturing_ = false;
        o.ready_ = false;
    }
    return *this;
}

bool CudaGraphRunner::is_supported() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    return prop.major >= 6;
}

bool CudaGraphRunner::begin_capture(void* stream_handle) {
    if (capturing_ || !is_supported()) return false;
    reset();
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        log_warn(strfmt("[cuda_graph] begin_capture failed: %s", cudaGetErrorString(err)));
        return false;
    }
    capturing_ = true;
    return true;
}

bool CudaGraphRunner::end_capture(void* stream_handle) {
    if (!capturing_) return false;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    cudaError_t err = cudaStreamEndCapture(stream, &graph_);
    capturing_ = false;
    if (err != cudaSuccess || !graph_) {
        log_warn(strfmt("[cuda_graph] end_capture failed: %s", cudaGetErrorString(err)));
        reset();
        return false;
    }
    return instantiate();
}

bool CudaGraphRunner::instantiate() {
    if (!graph_) return false;
    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    cudaError_t err = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess || !instance_) {
        log_warn(strfmt("[cuda_graph] instantiate failed: %s", cudaGetErrorString(err)));
        reset();
        return false;
    }
    ready_ = true;
    return true;
}

bool CudaGraphRunner::launch(void* stream_handle) {
    if (!ready_ || !instance_) return false;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    cudaError_t err = cudaGraphLaunch(instance_, stream);
    if (err != cudaSuccess) {
        log_warn(strfmt("[cuda_graph] launch failed: %s", cudaGetErrorString(err)));
        return false;
    }
    return true;
}

void CudaGraphRunner::reset() {
    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    capturing_ = false;
    ready_ = false;
}

size_t CudaGraphRunner::node_count() const {
    if (!graph_) return 0;
    size_t count = 0;
    cudaGraphGetNodes(graph_, nullptr, &count);
    return count;
}

}
}

#endif
