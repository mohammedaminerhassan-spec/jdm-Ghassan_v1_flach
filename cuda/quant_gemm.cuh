#pragma once

#include "core/common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace gai {
namespace cuda {

void gemv_q8_0(void* stream_handle,
               const void* W_q8,
               const float* x,
               float* y,
               int M, int K);

void gemv_q4_0(void* stream_handle,
               const void* W_q4,
               const float* x,
               float* y,
               int M, int K);

}
}
