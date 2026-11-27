#pragma once

#include "core/common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace gai {
namespace cuda {

void flash_decode_attention(void* stream_handle,
                            const float* q,
                            const float* k_cache,
                            const float* v_cache,
                            float* out,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            int seq_len,
                            float scale);

}
}
