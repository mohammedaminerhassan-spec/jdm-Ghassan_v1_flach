#include "cuda/cuda_ops.h"
#include "core/dtype.h"
#include "cuda/cuda_utils.h"
#include "core/ops.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace gai {
namespace cuda_ops {

#define CU_CHECK2(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

static constexpr int WARP_A = 32;
static constexpr int MAX_HD  = 128;

static constexpr int DECODE_BLOCK = 128;
static constexpr size_t DECODE_SHMEM_FLOATS = 40;

// ---- GEMM-based attention path (training throughput) ----
// This path runs the attention math as strided/grouped cublas GEMMs (tensor
// cores) plus three fused row-wise kernels:
//
//   fwd: S = Q*K^T (1 grouped call) -> fused softmax into probs ->
//        O = P*V (1 grouped call, written directly interleaved)
//   bwd: dP = dO*V^T (1 grouped call) -> fused dS in place ->
//        dq = dS*K, dk = dS^T*Q, dv = P^T*dO (grouped; dk/dv via an expanded
//        buffer plus a deterministic group-reduce, no atomics anywhere)
//
// Layouts are EXACTLY what the CPU reference expects (interleaved
// [B,T,H,hd] for Q/K/V/dO/dq, [B,T,KV,hd] for dk/dv, per-(b,h) T*T probs
// with the same masking semantics), so callers, checkpoints and the CPU
// reference are unaffected. cublas may tile differently than separate calls,
// so results can differ in the last ulp; the parity tests accept
// tolerance-level reassociation. NOTE: bf16 GEMMs are not implemented here
// (T4 has no BF16 tensor cores); if enabled the path falls back to fp32
// with a loud warning.
static __half* g_attn_mir = nullptr;  // fp16 mirror staging (reused per GEMM)
static size_t  g_attn_mir_cap = 0;      // __half elements
static float*  g_attn_exp = nullptr; // expanded dk/dv [B*H*T*hd]
static size_t  g_attn_exp_cap = 0;
static float*  g_attn_dsb = nullptr; // dS/dP [B*H*T*T]
static size_t  g_attn_dsb_cap = 0;

static void attn_pool_ensure(size_t mir_elems, size_t exp_elems, size_t ds_elems) {
    if (mir_elems > g_attn_mir_cap) {
        if (g_attn_mir) CU_CHECK2(cudaFree(g_attn_mir));
        g_attn_mir = nullptr;
        g_attn_mir_cap = 0;
        size_t want = mir_elems + mir_elems / 8 + 1024;
        CU_CHECK2(cudaMalloc(&g_attn_mir, sizeof(__half) * want));
        g_attn_mir_cap = want;
    }
    if (exp_elems > g_attn_exp_cap) {
        if (g_attn_exp) CU_CHECK2(cudaFree(g_attn_exp));
        g_attn_exp = nullptr;
        g_attn_exp_cap = 0;
        size_t want = exp_elems + exp_elems / 8 + 1024;
        CU_CHECK2(cudaMalloc(&g_attn_exp, sizeof(float) * want));
        g_attn_exp_cap = want;
    }
    if (ds_elems > g_attn_dsb_cap) {
        if (g_attn_dsb) CU_CHECK2(cudaFree(g_attn_dsb));
        g_attn_dsb = nullptr;
        g_attn_dsb_cap = 0;
        size_t want = ds_elems + ds_elems / 8 + 1024;
        CU_CHECK2(cudaMalloc(&g_attn_dsb, sizeof(float) * want));
        g_attn_dsb_cap = want;
    }
}

// The backward staging pool (the fp16 mirror, the expanded dk/dv and the
// dS/dP buffer). It is reused across the steps, so it is not a per-step
// leak, but it must be released with the other workspaces: otherwise a
// process that trains and exits keeps its VRAM allocated (and a server that
// reloads a bigger model holds both the old and the new pool).
void attn_free_pool() {
    if (g_attn_mir) { CU_CHECK2(cudaFree(g_attn_mir)); g_attn_mir = nullptr; }
    g_attn_mir_cap = 0;
    if (g_attn_exp) { CU_CHECK2(cudaFree(g_attn_exp)); g_attn_exp = nullptr; }
    g_attn_exp_cap = 0;
    if (g_attn_dsb) { CU_CHECK2(cudaFree(g_attn_dsb)); g_attn_dsb = nullptr; }
    g_attn_dsb_cap = 0;
}

__global__ void k_attn_cvt_flat(const float* __restrict__ src, __half* __restrict__ dst, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
static void attn_cvt(const float* src, __half* dst, size_t n) {
    if (n == 0) return;
    unsigned g = (unsigned)((n + 255) / 256);
    if (g < 1) g = 1;
    k_attn_cvt_flat<<<g, 256>>>(src, dst, (i64)n);
    CU_CHECK2(cudaGetLastError());
}

// Row-major grouped GEMM mirroring gemm()'s convention and precision rule:
// each of the nB problems computes C[M*N] = A[M*K] * B[K*N] (op counts follow
// the (ta,tb) flags exactly like gemm(); fp16 tensor path iff enabled and
// M*N*K >= threshold, else fp32). ONE cublasGemmGroupedBatchedEx call
// (group_count=1; uniform dims, varying pointers). A/Bfp stage through the
// shared fp16 buffers when the fp16 route is taken.
static void attn_sbgemm(bool ta, bool tb, int M, int N, int K,
                        float alpha, const float* const* A, int ldaA, const float* Arange, size_t Aspan,
                        const float* const* B, int ldaB, const float* Brange, size_t Bspan,
                        float beta, float* const* C, int ldaC, int nB,
                        bool force_fp32 = false) {
    if (nB <= 0 || M <= 0 || N <= 0) return;
    if (K <= 0 || alpha == 0.0f) {
        GAI_FAIL("attn_sbgemm: degenerate K<=0/alpha==0 path not implemented");
    }
    if (ops::gemm_bf16_enabled()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[attn] bf16 GEMM requested: attention GEMM path uses fp32 "
                            "instead (T4 has no BF16 tensor cores)\n");
        }
    }
    const i64 mnk = (i64)M * N * K;
    // The fp16 route mirrors the DENSE range of both operands into the
    // shared pool. When the A operand is the [B,H,T,T] score matrix that
    // mirror is tens of megabytes and the conversion pass costs more than
    // the GEMM itself (these GEMMs are memory bound, not compute bound):
    // such a call stays in fp32.
    const bool use_fp16 = !force_fp32 && fp16_gemm_enabled() && mnk >= ops::gemm_fp16_mnk_threshold();
    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(::gai::cuda::cublas_handle());
    // Mirror gemm()'s transpose swap: col-major C^T[N*M] = B^T * A^T.
    cublasOperation_t topA[1] = { tb ? CUBLAS_OP_T : CUBLAS_OP_N };
    cublasOperation_t topB[1] = { ta ? CUBLAS_OP_T : CUBLAS_OP_N };
    int ldaArr[1] = { ldaB }, ldbArr[1] = { ldaA }, ldcArr[1] = { ldaC };
    const void* alp[1] = { &alpha };
    int gs[1] = { nB };
    cublasStatus_t s = CUBLAS_STATUS_SUCCESS;

    // The chunk plan: cuBLAS needs the leading dimension of each operand to
    // be at least the dimension it indexes into (the full contract is
    // documented in cuda/kernels.cu). The GEMM is split along the offending
    // dimension instead of copying the operands: the pointers advance with
    // their own strides and the k-chunks accumulate into the same C.
    int n_chunk = M, m_chunk = N, k_chunk = K;
    if (ta) { if (ldaA < k_chunk) k_chunk = ldaA; }   // cublasB OP_T: ldb >= k
    else     { if (ldaA < n_chunk) n_chunk = ldaA; }   // cublasB OP_N: ldb >= n
    if (tb) { if (ldbB < k_chunk) k_chunk = ldbB; }   // cublasA OP_T: lda >= k
    else     { if (ldbB < m_chunk) m_chunk = ldbB; }   // cublasA OP_N: lda >= m

    // The per-batch base pointers (the packed ones in the fp16 branch).
    std::vector<const void*> Ah(nB), Bh(nB);
    std::vector<void*> Ch(nB);
    if (use_fp16) {
        // Mirror-convert the DENSE ranges once (flat: bitwise-identical to
        // per-matrix conversion); per-batch fp16 pointers follow by offset.
        // NOTE: Ah carries the B operand (gemm transpose swap), so Brange is
        // converted first. Swapping these corrupts shapes under GQA strides.
        attn_pool_ensure(Aspan + Bspan, 0, 0);
        attn_cvt(Brange, g_attn_mir, Bspan);
        attn_cvt(Arange, g_attn_mir + Bspan, Aspan);
        for (int i = 0; i < nB; ++i) {
            Ah[i] = g_attn_mir + (size_t)(B[i] - Brange);
            Bh[i] = g_attn_mir + Bspan + (size_t)(A[i] - Arange);
            Ch[i] = C[i];
        }
    } else {
        // Same operand swap as the fp16 branch: the cublas A-operand carries
        // the row-major B matrix (with B's leading dimension) and the cublas
        // B-operand carries the row-major A matrix (with A's leading
        // dimension). Passing them the other way round mixes the strides and
        // the transpose flags, which corrupts the result under GQA strides.
        for (int i = 0; i < nB; ++i) {
            Ah[i] = B[i];
            Bh[i] = A[i];
            Ch[i] = C[i];
        }
    }

    for (int k0 = 0; k0 < K; k0 += k_chunk) {
        const int kc = std::min(k_chunk, K - k0);
        // The k-chunks accumulate into the same C: the first one applies
        // beta, the rest add.
        const float beta_k = (k0 == 0) ? beta : 1.0f;
        const void* bet[1] = { &beta_k };
        for (int n0 = 0; n0 < M; n0 += n_chunk) {
            const int nc = std::min(n_chunk, M - n0);
            for (int m0 = 0; m0 < N; m0 += m_chunk) {
                const int mc = std::min(m_chunk, N - m0);
                int marr[1] = { mc }, narr[1] = { nc }, karr[1] = { kc };
                // The chunk offsets: every operand advances with its own
                // stride (the element size differs between the fp16 mirror
                // and the raw fp32 buffers).
                const size_t esz = use_fp16 ? sizeof(__half) : sizeof(float);
                std::vector<const void*> Ac(nB), Bc(nB);
                std::vector<void*> Cc(nB);
                for (int i = 0; i < nB; ++i) {
                    // The engine's A (the cublas B-operand): the K dimension
                    // is the columns (ta=0) or the rows (ta=1); the M
                    // dimension is the rows (ta=0) or the columns (ta=1).
                    const size_t a_k = ta ? (size_t)k0 * ldaA : (size_t)k0;
                    const size_t a_n = ta ? (size_t)n0 : (size_t)n0 * ldaA;
                    // The engine's B (the cublas A-operand): the K dimension
                    // is the rows (tb=0) or the columns (tb=1); the N
                    // dimension is the columns (tb=0) or the rows (tb=1).
                    const size_t b_k = tb ? (size_t)k0 : (size_t)k0 * ldaB;
                    const size_t b_m = tb ? (size_t)m0 * ldaB : (size_t)m0;
                    // Ac is the cublas A-operand (it carries the engine's B),
                    // Bc is the cublas B-operand (it carries the engine's A):
                    // the operands are swapped, so the offsets swap with them.
                    Ac[i] = (const char*)Ah[i] + (b_k + b_m) * esz;
                    Bc[i] = (const char*)Bh[i] + (a_k + a_n) * esz;
                    // The engine's C is [M, N] row-major: the M dimension is
                    // the rows and the N dimension is the columns.
                    Cc[i] = (char*)Ch[i] + ((size_t)n0 * ldaC + (size_t)m0) * sizeof(float);
                }
                s = cublasGemmGroupedBatchedEx(h, topA, topB, marr, narr, karr, alp,
                                               Ac.data(), use_fp16 ? CUDA_R_16F : CUDA_R_32F, ldaArr,
                                               Bc.data(), use_fp16 ? CUDA_R_16F : CUDA_R_32F, ldbArr, bet,
                                               Cc.data(), CUDA_R_32F, ldcArr,
                                               1, gs, CUBLAS_COMPUTE_32F);
                if (s != CUBLAS_STATUS_SUCCESS)
                    GAI_FAIL(strfmt("attn grouped GEMM failed: status=%d ta=%d tb=%d M=%d N=%d K=%d "
                                    "ldaA=%d ldaB=%d ldaC=%d nB=%d fp16=%d alpha=%g beta=%g "
                                    "A0=%p B0=%p C0=%p handle=%p",
                                    static_cast<int>(s), ta ? 1 : 0, tb ? 1 : 0, M, N, K,
                                    ldaA, ldaB, ldaC, nB, use_fp16 ? 1 : 0,
                                    (double)alpha, (double)beta,
                                    nB > 0 ? A[0] : nullptr, nB > 0 ? B[0] : nullptr,
                                    nB > 0 ? (const void*)C[0] : nullptr, (const void*)h));
            }
        }
    }
}

// Fused causal/window/segment softmax: reads scores S, writes normalized
// probs P in place. Row (b,h,t) covers j in [j0, t]; anything else is 0.
// Same rules as the CPU reference (j0 bound, seg_t<0 keeps all, div guard).
__global__ void k_attn_softmax_fwd(float* __restrict__ P, const i32* __restrict__ seg,
                                   int T, int H, float scale, int window) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (t >= T) return;
    float* row = P + ((size_t(b) * H + h) * T + t) * T;
    const i32 seg_t = seg ? seg[(size_t)b * T + t] : 0;
    const int j0 = (window > 0 && t + 1 > window) ? t + 1 - window : 0;
    __shared__ float sm[32];
    const int lane = threadIdx.x;
    const int nw = blockDim.x / WARP_A;
    const int wid = lane / WARP_A;
    const int wl = lane % WARP_A;
    float mx = -FLT_MAX;
    for (int j = j0 + lane; j <= t; j += blockDim.x) {
        bool keep = !seg || seg_t < 0 || seg[(size_t)b * T + j] == seg_t;
        float s = keep ? row[j] * scale : -FLT_MAX;
        mx = fmaxf(mx, s);
    }
    float wmax = mx;
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, o));
    if (wl == 0) sm[wid] = wmax;
    __syncthreads();
    if (lane == 0) {
        float m = sm[0];
        for (int i = 1; i < nw; ++i) m = fmaxf(m, sm[i]);
        sm[31] = m;
    }
    __syncthreads();
    mx = sm[31];
    float sum = 0.0f;
    for (int j = j0 + lane; j <= t; j += blockDim.x) {
        bool keep = !seg || seg_t < 0 || seg[(size_t)b * T + j] == seg_t;
        float p = keep ? __expf(row[j] * scale - mx) : 0.0f;
        row[j] = p;
        sum += p;
    }
    float wsum = sum;
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        wsum += __shfl_xor_sync(0xffffffffu, wsum, o);
    if (wl == 0) sm[wid] = wsum;
    __syncthreads();
    float tot = 0.0f;
    if (lane == 0) {
        for (int i = 0; i < nw; ++i) tot += sm[i];
        sm[30] = tot;
    }
    __syncthreads();
    tot = sm[30];
    const float inv = (tot > 1e-30f) ? (1.0f / tot) : 0.0f;
    for (int j = j0 + lane; j <= t; j += blockDim.x) row[j] *= inv;
    for (int j = lane; j < j0; j += blockDim.x) row[j] = 0.0f;
    for (int j = t + 1 + lane; j < T; j += blockDim.x) row[j] = 0.0f;
}

// Fused dS transform: ds[j] = P[j] * (dP[j] - dot) * scale, in place over dP.
// Masked entries have P==0 and contribute exactly 0 (same as the old kernel's
// ds==0 fast path, but written explicitly).
__global__ void k_attn_ds(float* __restrict__ dP, const float* __restrict__ P,
                          int T, int H, float scale) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (t >= T) return;
    float* drow = dP + ((size_t(b) * H + h) * T + t) * T;
    const float* prow = P + ((size_t(b) * H + h) * T + t) * T;
    const int lane = threadIdx.x;
    const int nw = blockDim.x / WARP_A;
    const int wid = lane / WARP_A;
    const int wl = lane % WARP_A;
    __shared__ float sm[32];
    float dot = 0.0f;
    for (int j = lane; j < T; j += blockDim.x) dot += prow[j] * drow[j];
    float wdot = dot;
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        wdot += __shfl_xor_sync(0xffffffffu, wdot, o);
    if (wl == 0) sm[wid] = wdot;
    __syncthreads();
    if (lane == 0) {
        float d = sm[0];
        for (int i = 1; i < nw; ++i) d += sm[i];
        sm[31] = d;
    }
    __syncthreads();
    dot = sm[31];
    for (int j = lane; j < T; j += blockDim.x)
        drow[j] = prow[j] * (drow[j] - dot) * scale;
}

// Deterministic GQA group reduce: small[(b,kvh),j,c] = sum over h in group of
// big[(b,gh),j,c], h ascending. Replaces racy atomic accumulation.
__global__ void k_attn_group_reduce(const float* __restrict__ big, float* __restrict__ small,
                                    int T, int H, int KV, int hd) {
    const int j = blockIdx.x;
    const int kvh = blockIdx.y;
    const int b = blockIdx.z;
    if (j >= T) return;
    const int group = H / KV;
    const int lane = threadIdx.x;
    float* srow = small + ((size_t(b) * T + j) * KV + kvh) * hd;
    for (int c = lane; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int h = 0; h < group; ++h) {
            const int gh = kvh * group + h;
            a += big[((size_t(b) * H + gh) * T + j) * hd + c];
        }
        srow[c] += a;
    }
}

void attention_forward_ex(const float* q, const float* k, const float* v,
                           float* out, float* probs,
                           int B, int T, int H, int KV, int hd, float scale, int window,
                           const i32* segment_ids) {
    if (B <= 0 || T <= 0) return;
    GAI_CHECK(H > 0 && H <= 65535 && B <= 65535,
              "cuda attention_ex: grid dimensions out of range");
    GAI_CHECK(hd <= MAX_HD, "cuda attention_ex: head_dim too large");
    GAI_CHECK(H % KV == 0, "cuda attention_ex: H must be a multiple of KV");
    const int group = H / KV;
    const int BH = B * H;
    // Scores buffer: write into probs, or scratch when probs==nullptr
    // (inference prefill path, which needs no T*T traffic afterwards).
    float* S = probs;
    if (!S) {
        attn_pool_ensure(0, 0, (size_t)BH * T * T);
        S = g_attn_dsb;
    }
    // S = Q * K^T, one grouped call over all (b,h).
    {
        std::vector<const float*> Aq(BH), Bq(BH);
        std::vector<float*> Cs(BH);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                const int i = b * H + h;
                const int kvh = h / group;
                Aq[i] = q + ((size_t)b * T * H + h) * hd;
                Bq[i] = k + ((size_t)b * T * KV + kvh) * hd;
                Cs[i] = S + (size_t)i * T * T;
            }
        }
        attn_sbgemm(false, true, T, T, hd, 1.0f,
                    Aq.data(), H * hd, q, (size_t)B * T * H * hd,
                    Bq.data(), KV * hd, k, (size_t)B * T * KV * hd,
                    0.0f, Cs.data(), T, BH);
    }
    // Fused softmax (masks + guard), in place over S.
    {
        dim3 grid(T, H, B);
        k_attn_softmax_fwd<<<grid, 256>>>(S, segment_ids, T, H, scale, window);
        CU_CHECK2(cudaGetLastError());
    }
    // O = P * V, written directly interleaved; one grouped call.
    {
        std::vector<const float*> Ap(BH), Bp(BH);
        std::vector<float*> Co(BH);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                const int i = b * H + h;
                const int kvh = h / group;
                Ap[i] = S + (size_t)i * T * T;
                Bp[i] = v + ((size_t)b * T * KV + kvh) * hd;
                Co[i] = out + ((size_t)b * T * H + h) * hd;
            }
        }
        attn_sbgemm(false, false, T, hd, T, 1.0f,
                    Ap.data(), T, S, (size_t)BH * T * T,
                    Bp.data(), KV * hd, v, (size_t)B * T * KV * hd,
                    0.0f, Co.data(), H * hd, BH,
                    /*force_fp32=*/true);   // A is the [B,H,T,T] score matrix
    }
}
void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window) {

    // Masked (zero) probs already encode causal/window/segment masking, exactly
    // like the previous kernel path, so window needs no explicit handling.
    (void)window;
    if (B <= 0 || T <= 0) return;
    GAI_CHECK(H > 0 && H <= 65535 && B <= 65535,
              "cuda attention_bwd_ex: grid dimensions out of range");
    GAI_CHECK(hd <= MAX_HD, "cuda attention_bwd_ex: head_dim too large");
    GAI_CHECK(H % KV == 0, "cuda attention_bwd_ex: H must be a multiple of KV");
    GAI_CHECK(probs != nullptr, "cuda attention_bwd_ex requires cached probs");
    const int group = H / KV;
    const int BH = B * H;
    // dS/dP staging (transient).
    //
    // The expanded buffer holds the per-head dk AND dv before the group
    // reduce: the dk half lives at [0, BH*T*hd) and the dv half at
    // [BH*T*hd, 2*BH*T*hd). Asking for one half would let the dv half run
    // past the end of the allocation.
    attn_pool_ensure(0, (size_t)2 * BH * T * hd, (size_t)BH * T * T);
    float* dS = g_attn_dsb;
    float* dE = g_attn_exp;  // expanded per-h dk/dv, then group-reduced
    // dP = dO * V^T into dS.
    {
        std::vector<const float*> Ad(BH), Bd(BH);
        std::vector<float*> Cd(BH);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                const int i = b * H + h;
                const int kvh = h / group;
                Ad[i] = dout + ((size_t)b * T * H + h) * hd;
                Bd[i] = v + ((size_t)b * T * KV + kvh) * hd;
                Cd[i] = dS + (size_t)i * T * T;
            }
        }
        attn_sbgemm(false, true, T, T, hd, 1.0f,
                    Ad.data(), H * hd, dout, (size_t)B * T * H * hd,
                    Bd.data(), KV * hd, v, (size_t)B * T * KV * hd,
                    0.0f, Cd.data(), T, BH);
    }
    // dS = P * (dP - rowdot) * scale, in place over dS.
    {
        dim3 grid(T, H, B);
        k_attn_ds<<<grid, 256>>>(dS, probs, T, H, scale);
        CU_CHECK2(cudaGetLastError());
    }
    // dq = dS * K, written directly interleaved.
    {
        std::vector<const float*> As(BH), Bs(BH);
        std::vector<float*> Cq(BH);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                const int i = b * H + h;
                const int kvh = h / group;
                As[i] = dS + (size_t)i * T * T;
                Bs[i] = k + ((size_t)b * T * KV + kvh) * hd;
                Cq[i] = dq + ((size_t)b * T * H + h) * hd;
            }
        }
        attn_sbgemm(false, false, T, hd, T, 1.0f,
                    As.data(), T, dS, (size_t)BH * T * T,
                    Bs.data(), KV * hd, k, (size_t)B * T * KV * hd,
                    0.0f, Cq.data(), H * hd, BH,
                    /*force_fp32=*/true);   // A is the [B,H,T,T] dS matrix
    }
    // dk = dS^T * Q and dv = P^T * dO, per-h expanded, then group-reduced.
    {
        std::vector<const float*> Ak(BH), Bk(BH), Av(BH), Bv(BH);
        std::vector<float*> Ck(BH), Cv(BH);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                const int i = b * H + h;
                Ak[i] = dS + (size_t)i * T * T;
                Bk[i] = q + ((size_t)b * T * H + h) * hd;
                Ck[i] = dE + (size_t)i * T * hd;
                Av[i] = probs + (size_t)i * T * T;
                Bv[i] = dout + ((size_t)b * T * H + h) * hd;
                Cv[i] = dE + (size_t)BH * T * hd + (size_t)i * T * hd;
            }
        }
        // dk-expanded = dS^T * Q  (ta=true: A=dS transposed in the swap sense)
        attn_sbgemm(true, false, T, hd, T, 1.0f,
                    Ak.data(), T, dS, (size_t)BH * T * T,
                    Bk.data(), H * hd, q, (size_t)B * T * H * hd,
                    0.0f, Ck.data(), hd, BH,
                    /*force_fp32=*/true);   // A is the [B,H,T,T] dS matrix
        // dv-expanded = P^T * dO
        attn_sbgemm(true, false, T, hd, T, 1.0f,
                    Av.data(), T, probs, (size_t)BH * T * T,
                    Bv.data(), H * hd, dout, (size_t)B * T * H * hd,
                    0.0f, Cv.data(), hd, BH,
                    /*force_fp32=*/true);   // A is the [B,H,T,T] probs matrix
        dim3 grid(T, KV, B);
        k_attn_group_reduce<<<grid, 256>>>(dE, dk, T, H, KV, hd);
        CU_CHECK2(cudaGetLastError());
        k_attn_group_reduce<<<grid, 256>>>(dE + (size_t)BH * T * hd, dv, T, H, KV, hd);
        CU_CHECK2(cudaGetLastError());
    }
}
__device__ __forceinline__ size_t decode_ring_slot(int j, int pinned_prefix,
                                                     int ring_start, int ring_capacity) {
    if (j < pinned_prefix) return (size_t)j;
    int cap = ring_capacity > 0 ? ring_capacity : 1;
    return (size_t)pinned_prefix + (size_t)((ring_start + (j - pinned_prefix)) % cap);
}

__global__ void k_attn_decode_ring(const float* __restrict__ q, const float* __restrict__ kc,
                                   const float* __restrict__ vc, float* __restrict__ out,
                                   int H, int KV, int hd, int cur_len, int ring_start,
                                   int pinned_prefix, int ring_capacity, float scale,
                                   float* __restrict__ scratch, int j0) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int group = H / KV;
    const int kvh = h / group;
    const float* qh = q + size_t(h) * hd;
    float* s = scratch + size_t(h) * cur_len;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
        const float* kh = kc + (slot * size_t(KV) + kvh) * hd;
        float d = 0.0f;
#pragma unroll 4
        for (int c = 0; c < hd; ++c) d += qh[c] * kh[c];
        d *= scale;
        s[j] = d;
        local_max = fmaxf(local_max, d);
    }
    int lane = threadIdx.x % WARP_A, wid = threadIdx.x / WARP_A;
#pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
    if (lane == 0) sm[wid] = local_max;
    __syncthreads();
    int nw = (blockDim.x + WARP_A - 1) / WARP_A;
    if (threadIdx.x == 0) {
        float m = sm[0];
        for (int i = 1; i < nw; ++i) m = fmaxf(m, sm[i]);
        sm[32] = m;
    }
    __syncthreads();
    float mx = sm[32];

    float local_sum = 0.0f;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        float p = __expf(s[j] - mx);
        s[j] = p;
        local_sum += p;
    }
#pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
    if (lane == 0) sm[wid] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += sm[i];
        sm[33] = t;
    }
    __syncthreads();
    float inv = sm[33] > 0.0f ? 1.0f / sm[33] : 0.0f;

    float* o = out + size_t(h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            a += s[j] * vc[(slot * size_t(KV) + kvh) * hd + c];
        }
        o[c] = a * inv;
    }
}

// fp16 KV cache variant: the cache is read as packed halves and converted
// on the fly. At a 32k context this halves both the memory footprint and
// the bandwidth of the decode step.
__global__ void k_attn_decode_ring_fp16(const float* __restrict__ q, const u16* __restrict__ kc,
                                         const u16* __restrict__ vc, float* __restrict__ out,
                                         int H, int KV, int hd, int cur_len, int ring_start,
                                         int pinned_prefix, int ring_capacity, float scale,
                                         float* __restrict__ scratch, int j0) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int group = H / KV;
    const int kvh = h / group;
    const float* qh = q + size_t(h) * hd;
    float* s = scratch + size_t(h) * cur_len;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
        const u16* kh = kc + (slot * size_t(KV) + kvh) * hd;
        float d = 0.0f;
#pragma unroll 4
        for (int c = 0; c < hd; ++c) d += qh[c] * __half2float(*reinterpret_cast<const __half*>(kh + c));
        d *= scale;
        s[j] = d;
        local_max = fmaxf(local_max, d);
    }
    int lane = threadIdx.x % WARP_A, wid = threadIdx.x / WARP_A;
#pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
    if (lane == 0) sm[wid] = local_max;
    __syncthreads();
    int nw = (blockDim.x + WARP_A - 1) / WARP_A;
    if (threadIdx.x == 0) {
        float m = sm[0];
        for (int i = 1; i < nw; ++i) m = fmaxf(m, sm[i]);
        sm[32] = m;
    }
    __syncthreads();
    float mx = sm[32];

    float local_sum = 0.0f;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        float p = __expf(s[j] - mx);
        s[j] = p;
        local_sum += p;
    }
#pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
    if (lane == 0) sm[wid] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += sm[i];
        sm[33] = t;
    }
    __syncthreads();
    float inv = sm[33] > 0.0f ? 1.0f / sm[33] : 0.0f;

    float* o = out + size_t(h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            a += s[j] * __half2float(*reinterpret_cast<const __half*>(vc + (slot * size_t(KV) + kvh) * hd + c));
        }
        o[c] = a * inv;
    }
}
void attention_decode_ring(const float* q, const float* kc, const float* vc,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "cuda attention_decode_ring: empty shape");
    GAI_CHECK(H % KV == 0, "cuda attention_decode_ring: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && cache_max >= 0 && cur_len <= cache_max,
              "cuda attention_decode_ring: length exceeds cache capacity");
    GAI_CHECK(pinned_prefix >= 0 && pinned_prefix <= cur_len,
              "cuda attention_decode_ring: pinned prefix out of range");
    const int ring_capacity = cache_max - pinned_prefix;
    if (cur_len > pinned_prefix) {
        GAI_CHECK(ring_capacity > 0, "cuda attention_decode_ring: ring has no rolling slots");
        GAI_CHECK(ring_start >= 0 && ring_start < ring_capacity,
                  "cuda attention_decode_ring: ring start out of range");
    }
    if (H <= 0 || cur_len <= 0) return;
    int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
    int block = DECODE_BLOCK;
    GAI_CHECK((block + WARP_A - 1) / WARP_A + 2 <= (int)DECODE_SHMEM_FLOATS,
              "cuda attention_decode_ring: block exceeds shared-memory budget");
    size_t sh = sizeof(float) * DECODE_SHMEM_FLOATS;
    k_attn_decode_ring<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, ring_start,
                                         pinned_prefix, ring_capacity, scale, scratch, j0);
    CU_CHECK2(cudaGetLastError());
}

void attention_decode_ring_fp16(const float* q, const u16* kc, const u16* vc,
                                float* out, int H, int KV, int hd, int ring_start,
                                int pinned_prefix, int cur_len, int cache_max,
                                float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "cuda attention_decode_ring_fp16: empty shape");
    GAI_CHECK(H % KV == 0, "cuda attention_decode_ring_fp16: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && cache_max >= 0 && cur_len <= cache_max,
              "cuda attention_decode_ring_fp16: length exceeds cache capacity");
    GAI_CHECK(pinned_prefix >= 0 && pinned_prefix <= cur_len,
              "cuda attention_decode_ring_fp16: pinned prefix out of range");
    const int ring_capacity = cache_max - pinned_prefix;
    if (cur_len > pinned_prefix) {
        GAI_CHECK(ring_capacity > 0, "cuda attention_decode_ring_fp16: ring has no rolling slots");
        GAI_CHECK(ring_start >= 0 && ring_start < ring_capacity,
                  "cuda attention_decode_ring_fp16: ring start out of range");
    }
    if (H <= 0 || cur_len <= 0) return;
    int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
    int block = DECODE_BLOCK;
    GAI_CHECK((block + WARP_A - 1) / WARP_A + 2 <= (int)DECODE_SHMEM_FLOATS,
              "cuda attention_decode_ring_fp16: block exceeds shared-memory budget");
    size_t sh = sizeof(float) * DECODE_SHMEM_FLOATS;
    k_attn_decode_ring_fp16<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, ring_start,
                                              pinned_prefix, ring_capacity, scale, scratch, j0);
    CU_CHECK2(cudaGetLastError());
}

}
}
