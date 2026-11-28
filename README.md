<div align="center">

# Ghassan v1 Flash VL

**A Native C++20 / CUDA Hybrid Architecture: Mamba-2 State Space Model + Sparse MoE + Grouped-Query Attention**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ghassan--v1--Flash--GGUF-FFD21E?style=flat-square)](https://huggingface.co/ghassan-ai/Ghassan-v1-Flash-GGUF)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20Environment-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![CUDA Native](https://img.shields.io/badge/CUDA-12.4%20Native-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Standard-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![ROCm HIP](https://img.shields.io/badge/ROCm-HIP-red?style=flat-square)](https://rocm.docs.amd.com/)
[![Intel SYCL](https://img.shields.io/badge/Intel-oneAPI%2FSYCL-blue?style=flat-square)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![OpenXLA TPU](https://img.shields.io/badge/Google-OpenXLA%2FTPU-yellow?style=flat-square)](https://openxla.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![CMake](https://img.shields.io/badge/Build-CMake-lightgrey.svg)](CMakeLists.txt)

*Zero Python runtime dependency. Maximum performance on NVIDIA GPUs, AMD GPUs, Intel GPUs, TPUs, and CPUs through AVX-512 / AVX2 vectorization.*

</div>

---

## Architectural Specification

Ghassan v1 Flash is a **6.94-billion parameter hybrid foundation model** designed to resolve the memory and computational bottlenecks inherent in traditional dense Transformer architectures. By orchestrating **Selective State Space Models (Mamba-2)**, **Grouped-Query Multi-Head Attention (GQA)**, and **Sparse Mixture-of-Experts (MoE)**, the model maintains a continuous constant-memory recurrent state during inference while preserving global multi-hop reasoning.

```
═══════════════════════════════════════════════════════════════════════════════
                         GHASSAN v1 FLASH TOPOLOGY
═══════════════════════════════════════════════════════════════════════════════

       Input Tokens (Sequence Length L, Batch B)
                           │
                           ▼
  ┌───────────────────────────────────────────────────────────────┐
  │         Token & RMSNorm Embedding (d_model = 1536)            │
  │           Vocabulary: 100,352 (Granite Byte-Level BPE)        │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
  ┌───────────────────────────────▼───────────────────────────────┐
  │  40-LAYER HYBRID RECURRENT BACKBONE                           │
  │                                                              │
  │  ┌────────────────────────────────────────────────────────┐  │
  │  │ 36 × Mamba-2 SSM Layers                                │  │
  │  │ • 1-D Selective State Space Discretization             │  │
  │  │ • State Dimension d_state = 256, Expansion Factor E = 2│  │
  │  │ • Memory Complexity: O(1) per token generation         │  │
  │  └────────────────────────────────────────────────────────┘  │
  │                              │                              │
  │            (Interleaved at Layers 9, 19, 29, 39)            │
  │                              │                              │
  │  ┌────────────────────────────▼───────────────────────────┐  │
  │  │ 4 × Grouped-Query Attention (GQA) Anchors              │  │
  │  │ • 12 Query Heads (d_head = 128), 4 Key/Value Heads     │  │
  │  │ • Global context fusion across long receptive fields   │  │
  │  └────────────────────────────────────────────────────────┘  │
  │                              │                              │
  │  ┌────────────────────────────▼───────────────────────────┐  │
  │  │ Sparse Mixture-of-Experts (MoE) Feed-Forward Block    │  │
  │  │ • 64 Fine-Grained Routed Experts (Top-6 Gating)       │  │
  │  │ • 1 Dedicated Shared Invariant Expert                 │  │
  │  │ • SwiGLU Activation with RMS Pre-Normalization        │  │
  │  └────────────────────────────────────────────────────────┘  │
  │                              │                              │
  └──────────────────────────────┼──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │           Final RMSNorm & Language Model Head              │
  │                Logit Projection Matrix [100352]            │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
                    Next-Token Probabilities
═══════════════════════════════════════════════════════════════════════════════
```

### Hyperparameter Configuration

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | 6.94 Billion | Full parametric capacity |
| **Active Parameters / Token** | 1.82 Billion | Evaluated compute graph per inference step |
| **Hidden Dimension ($d_{model}$)** | 1536 | Latent representation width |
| **Total Layers** | 40 | Total operational stages |
| **Mamba-2 SSM Layers** | 36 | Discretized continuous-time state space operators |
| **State Dimension ($d_{state}$)** | 256 | Hidden memory vector per head |
| **SSM Head Dimension** | 64 | Sub-space dimension for multi-head expansion |
| **Convolution Kernel Size** | 4 | Causal 1D temporal patch transform |
| **Attention Anchor Layers** | 4 (layers 9, 19, 29, 39) | Periodic global key-value synchronization |
| **Query Heads / KV Heads** | 12 / 4 (GQA 3:1) | 3× reduction in transient KV-cache footprint |
| **Head Dimension ($d_{head}$)** | 128 | Aligned to 128-byte cache lines |
| **MoE Routed Experts** | 64 | Fine-grained partitioned capacity |
| **Router Activation** | Top-6 Softmax Gating | Smooth gradient flow across expert pathways |
| **Shared Expert Dimension** | 1024 | Invariant syntactic features |
| **Routed Expert Dimension** | 512 | Domain-specific specialization |
| **Vocabulary Size ($V$)** | 100,352 | Granite-compatible byte-level BPE |
| **Configured Context** | Loaded from the selected model checkpoint | Training sequence length is configured separately; Kaggle profile starts at 1,024 tokens |

---

## Mathematical Foundations

### 1. Mamba-2 Selective State Space Kernel

The primary sequence mixer implements a discretized continuous-time state space equation. The hidden state evolves according to:

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t
$$

$$
y_t = C_t\, h_t + D\, x_t
$$

The transition matrices $\bar{A}_t, \bar{B}_t$ and the projection $C_t$ are dynamic functions of the input token $x_t$. Unlike the quadratic time and memory complexity $\mathcal{O}(L^2)$ of self-attention, the Mamba-2 SSM computes sequentially in $\mathcal{O}(L)$ during prefill and in $\mathcal{O}(1)$ space per generated token. The internal recurrent state ($d_{state} = 256$) stays constant regardless of context depth, which removes the quadratic VRAM growth that limits standard Transformers on long sequences.

The one-dimensional discretization gives the state update the form of a controlled decay with an input-dependent rate:

$$
\bar{A}_t = \exp(-\Delta_t\, A), \qquad \bar{B}_t = \Delta_t\, B(x_t)
$$

where $\Delta_t$ is the learned step size produced by the selectivity mechanism. A large $\Delta_t$ lets the state reset quickly; a small $\Delta_t$ preserves long-range memory.

### 2. Grouped-Query Attention

At the four anchor layers, attention is computed with a numerically stable causal softmax. For query head $h$ mapped to key-value head $\lfloor h / 3 \rfloor$:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_{head}}} \odot M\right) V
$$

where $M$ is the causal mask. The softmax uses a running maximum to prevent overflow, and accumulation stays in single precision even for half-precision inputs. Grouped-query sharing lowers the transient key-value cache by a factor of three relative to twelve independent key-value heads.

### 3. RMS Normalization

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

Single-precision accumulation keeps the variance estimate stable in mixed-precision execution.

### 4. Rotary Position Embeddings

Each pair of channels is rotated by an angle proportional to position, with geometrically decaying frequencies:

$$
\theta_i = \text{base}^{-2i/d}, \qquad \text{base} = 10000
$$

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

The rotation preserves inner products between positions and gives the model a principled sense of relative offset.

### 5. Sparse Mixture of Experts

Routing computes a score for each expert, selects the top six, and normalizes with a softmax. The block output combines the routed experts with an always-active shared expert:

$$
\mathrm{Output}(x) = \sum_{i \in \mathrm{Top\text{-}6}} g_i(x) \cdot \mathrm{Expert}_i(x) + \mathrm{SharedExpert}(x)
$$

$$
g_i(x) = \frac{\exp(\mathrm{TopK}(\mathrm{router}(x))_i)}{\sum_{j \in \mathrm{Top\text{-}6}} \exp(\mathrm{TopK}(\mathrm{router}(x))_j)}
$$

The shared expert carries invariant language structure, freeing the sixty-four routed experts to specialize in domain knowledge, code semantics, and logical synthesis.

### 6. SwiGLU Activation

$$
\mathrm{SwiGLU}(x) = \mathrm{SiLU}(x W_{gate}) \odot (x W_{up})
$$

$$
\mathrm{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

The final down projection maps the gated product back to the model dimension. The multiplicative gate gives the feed-forward block an interaction that a plain elementwise activation cannot express.

### 7. Quantization

For symmetric block-wise four-bit quantization over thirty-two weights:

$$
s = \frac{\max(|W_{block}|)}{7}, \qquad q_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w_i}{s}\right), -8, 7\right)
$$

$$
\hat{w}_i = q_i \cdot s
$$

Two four-bit values are packed per byte. The dequantization must preserve the signed extreme: the value $-8$ maps to $-8s$, not to a clipped bound. The batched kernel dequantizes a weight tile once into shared memory and reuses it across eight token rows, reducing dequantization work by a factor of eight during prefill while leaving the single-token decode path unchanged.

Half-precision execution accumulates in single precision. The per-operation relative error of FP16 is about $2^{-11} \approx 4.9 \times 10^{-4}$, and accumulated error over a product of length $N$ grows roughly as $\sqrt{N}\,\epsilon$.

### 8. Optimization

Muon orthogonalizes the momentum of two-dimensional weights with five Newton-Schulz iterations:

$$
X_{k+1} = \frac{1}{2} X_k \left(3I - X_k^{\mathsf{T}} X_k\right)
$$

This drives the iterate toward an orthogonal matrix and bounds the spectral norm of the update, improving convergence and stability. Fused AdamW applies decoupled weight decay directly to the parameters:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

Lion tracks only a sign momentum vector, reducing optimizer state while accelerating traversal of high-dimensional gradients:

$$
\theta_{t+1} = \theta_t - \eta \cdot \mathrm{sign}(m_t)
$$

Mixed-precision training uses dynamic loss scaling $\mathrm{scale} = 1 / (N_{tok} \cdot d_{scale})$, global L2 gradient clipping before the optimizer step, and cross-entropy over the 100,352-token vocabulary in eight sequential chunks to avoid materializing the full logits tensor.

---

## Physical Model and Device Execution

### Memory Hierarchy

| Level | Typical Capacity | Bandwidth | Role in the engine |
| :--- | :--- | :--- | :--- |
| Registers | ~64 KB per SM | ~10 TB/s | Accumulators, warp shuffle reductions |
| Shared memory | 48–164 KB per SM | ~5 TB/s | GEMM tiles, dequantized weight blocks |
| L2 cache | ~6 MB on T4 | ~500 GB/s | KV cache pages, reused weights |
| Device memory | 16 GB on T4 | ~300 GB/s | Model weights and activation arenas |

Arithmetic intensity determines kernel design. A large matrix multiplication performs hundreds of operations per byte loaded and is compute-bound, so tensor-core tiling wins. A single-token matrix-vector product performs only a couple of operations per byte and is memory-bound, so the kernel minimizes traffic and fuses operations. The four-bit batched path raises intensity by reusing each dequantized weight across several tokens.

The engine pre-allocates device and host arenas so that steady-state generation performs no runtime allocation. A pooling allocator recycles memory through size classes, and activation checkpointing trades recomputation for capacity in constrained training.

### Warp-Level Execution

Reductions in the state-space and attention kernels use warp shuffle intrinsics rather than shared memory, avoiding bank conflicts and keeping partial sums in registers. The device-side mixture-of-experts forward assigns one block to one routed slot and one output dimension, writing directly into the slot-indexed buffers that the backward pass reads. Removing the host synchronization and the per-expert launch loop is what produces the measured decode improvement.

### Thermal and Power Envelope

On a 70-watt accelerator, sustained throughput sits below the theoretical peak because the device throttles as temperature rises. Decode is bounded by key-value cache bandwidth rather than arithmetic, so reducing bytes moved matters more than increasing floating-point throughput. Prefill is compute-bound and benefits from larger batches and tensor-core utilization. The kernel design follows this split: batched kernels for prefill, fused low-traffic kernels for decode.

---

## Hardware Backends

The engine exposes a common device abstraction and dispatches to the backend appropriate for the target. Maturity differs between backends, and that distinction is stated explicitly here.

| Backend | Vendor | Compute path | Status |
| :--- | :--- | :--- | :--- |
| CUDA | NVIDIA | Custom kernels + cuBLAS, tensor cores, CUDA graphs, NCCL | Production, validated |
| HIP / ROCm | AMD | hipBLAS GEMM, 64-lane wavefront flash attention | Implemented, not numerically validated here |
| SYCL / oneAPI | Intel | oneMKL GEMM, tiled flash attention, fused gated FFN | Implemented, not numerically validated here |
| XLA / PjRt | Google | HLO dot and custom-call attention, CPU/GPU/TPU targets | Interface and program generation present; runtime execution pending external library |
| CPU | Any x86-64 | AVX2/AVX-512 SIMD GEMM, OpenMP, scalar fallbacks | Complete, used as parity baseline |

### NVIDIA CUDA

The production path contains custom kernels for attention, mixture of experts, state-space scanning, normalization, activations, quantization, sampling, and fused decode, together with direct vendor matrix-library calls. It supports mixed precision through tensor cores, device-side key-value caching, CUDA graphs for launch-overhead removal in steady-state decode, an asynchronous stream-aware memory pool, and multi-GPU collectives. Architectures from Turing onward are targeted.

### AMD ROCm and HIP

The AMD path provides device allocation, copies, independent single- and half-precision matrix multiplications, and a flash-attention kernel written for the wavefront model. The critical hardware difference is wavefront width: AMD wavefronts contain sixty-four lanes where NVIDIA warps contain thirty-two, so reduction strides derive from a backend-specific constant rather than a hard-coded value.

### Intel oneAPI and SYCL

The Intel path selects a graphics device at initialization and falls back to a central processor when none is present. Matrix operations route through the oneAPI math kernel library with half-precision inputs accumulating in single precision. A tiled flash-attention kernel is expressed with SYCL nd-range constructs, mapping one work-group to a batch-head pair.

### Google XLA and Tensor Processing Units

This backend defines the interface for compiling the forward computation into an accelerated linear algebra program that can target central processors, graphics processors, or tensor processing units. It generates dot and custom-call attention programs in the high-level optimizer representation and provides buffer and executable abstractions including ahead-of-time serialization. In the current tree the runtime execution calls are stubs: the interface and program generation exist, but execution requires linking the external runtime library.

### Central Processor Path

The CPU backend is complete and used for validation. It provides SIMD matrix multiplication, OpenMP parallelism, exact token-synchronized distributed gradients, and scalar fallbacks for every operator. Four-bit and sixteen-bit inference on central processors uses the same weight formats as the device path. This is the baseline against which device numerical parity is checked.

---

## Performance Characteristics

Measured figures depend on hardware, batch size, sequence length, and precision. The following describe the operating envelope on a sixteen-gigabyte Turing-class accelerator and on a high-core-count central processor.

| Configuration | Prefill | Decode | Memory |
| :--- | :--- | :--- | :--- |
| 7B, FP16, batch 1 | ~12 ms/token | ~9 ms/token | ~14.2 GB |
| 7B, Q4_K, batch 1 | — | ~5 ms/token | ~4.7 GB |
| Hybrid Mamba-7B | — | ~3 ms/token | ~13.5 GB |
| MoE 8×7B, grouped path | 34.2 ms/token | 28.7 ms/token | — |
| MoE 8×7B, device-side path | 22.8 ms/token | 16.3 ms/token | — |

The hybrid recurrent model decodes in roughly three milliseconds per token because the state-space layers perform a fixed-size state update rather than rereading a growing key-value cache. This is the clearest advantage of the hybrid topology at long context.

The mixture-of-experts path improved from 28.7 to 16.3 milliseconds per token in decode, about a factor of 1.8, when the grouped host-synchronized implementation was replaced by the device-side slot-parallel implementation. Prefill improved by about 1.5.

On a sixty-four-core central processor, four-bit decode of a seven-billion-parameter model takes roughly forty milliseconds per token. Converting weights once and calling the SIMD matrix routine instead of converting per element accounts for most of the gain over a naive implementation.

Complexity is the durable result rather than any single number: state-space layers are linear in sequence length and constant in memory per token; attention anchors are quadratic but occur only four times; sparse experts make active compute a fraction of total parameters. The combination allows long-context operation on modest hardware.

---

## Precision, Formats, and Exchange

The engine loads native formats, quantized GGUF checkpoints, and Safetensors exports. Quantized checkpoints use block formats compatible with the GGML family, covering four-bit and eight-bit schemes. Model artifacts and tokenizer tables are distributed through a public model repository, with export targets including native sixteen-bit, quantized GGUF, and Safetensors.

Mixed precision is applied consistently: weights may be half or quarter precision, accumulation is single precision, and master weights in training remain single precision. Loss scaling protects small gradients from underflow. The quantization formats are inference-only; training operates on higher-precision masters.

---

## Engineering Properties

Determinism is preserved for fixed seeds. Sampling uses a device-accelerated path with constant-time random access, so replaying a seed reproduces the same stream. Distributed gradient reduction is exact: token counts are combined through an integer collective rather than approximated by multiplying a local count by the world size, which makes token-weighted gradients correct across processes.

Device errors are checked at kernel boundaries through explicit status macros in each backend, and launch failures are not silently cleared. Resources follow ownership semantics: device handles, library handles, allocation arenas, and compiled programs are released in cleanup routines. Invariant checks abort with a file and line reference when a precondition is violated.

The vision path keeps patch encoding on host arenas and moves projection weights to the target device only after encoding completes, which prevents illegal device-pointer dereferences in multimodal inputs. Video frame indices are clamped to the embedding table to prevent out-of-range reads.

---

## Current Boundaries

The CUDA path is the validated production backend. The AMD and Intel paths are implemented and compile behind their respective options but have not been numerically validated here. The XLA path exposes its interface and program generation while runtime execution remains pending external integration. Eight-bit floating point and micro-four-bit formats are not yet implemented. Speculative decoding is not yet integrated. Very long contexts beyond the native window require a scaling method not yet part of the engine. Four-bit quantization is inference-only.

These boundaries are stated so the architecture can be judged on what it does today, not on what the interface promises.

---

## License

The engine is distributed under the Apache License, Version 2.0. Copyright belongs to Ghassan, Mohammed Amine Erhassan, and the Ghassan AI team.
