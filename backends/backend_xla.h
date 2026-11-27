// ══════════════════════════════════════════════════════════════════════════════
// backend_xla.h — Google TPU / XLA backend via OpenXLA C API
// ══════════════════════════════════════════════════════════════════════════════
// Compiles the model's forward pass as an XLA computation that can run on:
//   • Google TPU (v3/v4/v5)
//   • GPU via XLA JIT (NVIDIA/AMD)
//   • CPU via LLVM JIT (fastest single-socket CPU path)
//
// CMake flag: -DGAI_ENABLE_XLA=ON
// Requires: openxla/xla runtime library  (https://github.com/openxla/xla)
// ══════════════════════════════════════════════════════════════════════════════
#pragma once

#ifdef GAI_XLA

#include "core/common.h"
#include <cstddef>
#include <string>
#include <vector>

namespace gai {
namespace xla_backend {

// ── XLA device target ────────────────────────────────────────────────────────
enum class XlaTarget {
    CPU = 0,   // LLVM JIT — no special hardware
    GPU = 1,   // XLA GPU (NVIDIA or AMD via HLO→triton/cuDNN)
    TPU = 2,   // Google TPU (requires libtpu.so from Google Cloud)
};

// ── Lifecycle ────────────────────────────────────────────────────────────────
void init(XlaTarget target = XlaTarget::CPU);
void finalize();

// ── Buffer: opaque XLA-managed device buffer ─────────────────────────────────
struct XlaBuffer {
    void*  handle  = nullptr;   // opaque PjRtBuffer*
    size_t nbytes  = 0;
    bool   on_host = true;

    bool valid() const { return handle != nullptr; }
};

XlaBuffer buffer_from_host(const void* data, size_t nbytes,
                            const std::vector<int64_t>& shape,
                            bool is_f32 = true);
void       buffer_to_host(const XlaBuffer& buf, void* dst, size_t nbytes);
void       buffer_free(XlaBuffer& buf);

// ── Compiled computations ────────────────────────────────────────────────────
// Each computation is an HLO module compiled once → executed N times.

struct XlaComputation {
    void* executable = nullptr; // opaque PjRtLoadedExecutable*
    bool  valid() const { return executable != nullptr; }
};

// Compile a simple matrix-multiply (y = x @ w.T) as an XLA computation.
// Returns a compiled executable that can be used with run_gemm().
XlaComputation compile_gemm(int M, int K, int N, bool fp16_weights = false);

// Execute a compiled GEMM.
// x [M,K], w [N,K] → y [M,N]
void run_gemm(const XlaComputation& comp,
              const XlaBuffer& x, const XlaBuffer& w, XlaBuffer& y);

// Compile a Flash-Attention forward pass as an XLA computation.
// Returns executable for: q[B,H,T,hd] × k[B,KV,T,hd] × v[B,KV,T,hd] → out[B,H,T,hd]
XlaComputation compile_attention(int B, int T, int H, int KV, int hd, float scale);

void run_attention(const XlaComputation& comp,
                   const XlaBuffer& q, const XlaBuffer& k,
                   const XlaBuffer& v, XlaBuffer& out);

// Free a compiled computation.
void computation_free(XlaComputation& comp);

// ── XLA AOT (Ahead-of-Time) compilation ─────────────────────────────────────
// Serialise a compiled computation to a binary that can be loaded without
// recompilation — ideal for TPU deployment.
bool computation_save(const XlaComputation& comp, const std::string& path);
bool computation_load(XlaComputation& comp, const std::string& path);

// ── Utility ──────────────────────────────────────────────────────────────────
std::string target_name(XlaTarget t);
bool        is_tpu_available();

} // namespace xla_backend
} // namespace gai

#endif // GAI_XLA
