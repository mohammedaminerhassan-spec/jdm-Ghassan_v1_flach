<div align="center">

# Ghassan v1 Flash VL

**A Native C++20 / CUDA Hybrid Architecture: Mamba-2 State Space Model + Sparse MoE + Grouped-Query Attention**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ghassan--v1--Flash--GGUF-FFD21E?style=flat-square)](https://huggingface.co/ghassan-ai/Ghassan-v1-Flash-GGUF)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20Environment-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![CUDA Native](https://img.shields.io/badge/CUDA-12.4%20Native-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Standard-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![CMake](https://img.shields.io/badge/Build-CMake-lightgrey.svg)](CMakeLists.txt)

*Zero Python runtime dependency. The current model execution paths are CPU and NVIDIA CUDA; optional HIP, SYCL, and XLA adapters are experimental and are not yet wired into the core Model device dispatcher.*

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
| **Default Inference Context** | min(checkpoint limit, 4,096 tokens) | Override with --ctx up to the checkpoint limit; a larger setting does not itself create long-context capability |
| **Training Sequence Length** | Trainer default: 1,024 tokens | On 16 GB T4, full fine-tuning above 800M parameters is guarded to batch=1 and seq_len<=512; use LoRA for larger runs and check the memory estimate |

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

The engine reuses device and host workspaces to reduce steady-state allocation. The legacy `training.activation_checkpointing` option currently splits a batch into smaller micro-batches; it does **not** recompute transformer layers. It reduces activation peak only when batch_size is greater than one.

### Warp-Level Execution

Reductions in the state-space and attention kernels use warp shuffle intrinsics rather than shared memory, avoiding bank conflicts and keeping partial sums in registers. The device-side mixture-of-experts forward assigns one block to one routed slot and one output dimension, writing directly into the slot-indexed buffers that the backward pass reads. Removing the host synchronization and the per-expert launch loop is what produces the measured decode improvement.

### Thermal and Power Envelope

On a 70-watt accelerator, sustained throughput sits below the theoretical peak because the device throttles as temperature rises. Decode is bounded by key-value cache bandwidth rather than arithmetic, so reducing bytes moved matters more than increasing floating-point throughput. Prefill is compute-bound and benefits from larger batches and tensor-core utilization. The kernel design follows this split: batched kernels for prefill, fused low-traffic kernels for decode.

---

## Hardware Backends

The core `Model` and `Generator` execution dispatcher currently supports CPU and NVIDIA CUDA. Separate HIP, SYCL, and XLA adapter sources/options exist, but they are experimental and are not complete runtime backends in the core device path.

| Backend | Vendor | Compute path | Status |
| :--- | :--- | :--- | :--- |
| CUDA | NVIDIA | Custom kernels + cuBLAS, tensor cores, CUDA graphs, NCCL (when available) | Primary GPU execution path; runtime correctness/performance must be checked on the target GPU |
| CPU | x86-64 | SIMD GEMM, OpenMP, scalar fallbacks | Supported execution path |
| HIP / ROCm | AMD | Separate adapter and build-probe sources | Experimental; not wired into the core Model dispatcher |
| SYCL / oneAPI | Intel | Separate adapter and build-probe sources | Experimental; not wired into the core Model dispatcher |
| XLA / PjRt | Google | Interface/program-generation prototype | Runtime execution not implemented |

### NVIDIA CUDA

The production path contains custom kernels for attention, mixture of experts, state-space scanning, normalization, activations, quantization, sampling, and fused decode, together with direct vendor matrix-library calls. It supports mixed precision through tensor cores, device-side key-value caching, CUDA graphs for launch-overhead removal in steady-state decode, an asynchronous stream-aware memory pool, and multi-GPU collectives. Architectures from Turing onward are targeted.

### Experimental non-CUDA adapters

HIP and SYCL source files contain backend experiments, while XLA contains interface/program-generation scaffolding. They are not currently selected by the core model runtime and should not be treated as supported inference backends. The CPU and NVIDIA CUDA paths are the execution paths to use today.

### Central Processor Path

The CPU backend is complete and used for validation. It provides SIMD matrix multiplication, OpenMP parallelism, exact token-synchronized distributed gradients, and scalar fallbacks for every operator. Four-bit and sixteen-bit inference on central processors uses the same weight formats as the device path. This is the baseline against which device numerical parity is checked.

---

## Performance Characteristics

Performance depends on the exact checkpoint, quantization, context, batch size, CUDA version, and accelerator. Do not treat a single throughput figure as a hardware-independent guarantee.

Use the built-in `bench` command on the target Kaggle GPU and record the commit, model, precision, context, prompt length, generated-token count, prefill tokens/second, decode tokens/second, and peak VRAM. Prefill throughput and decode throughput are separate metrics: cached chunked prefill improves prompt ingestion, while decode remains a one-token-at-a-time workload.

The default inference context is capped at 4,096 tokens (or the checkpoint's smaller limit). Longer prompts are processed in 512-token query chunks against the accumulated KV cache; the CUDA path currently uses a bounded chunk-by-prefix score workspace. This is not yet a tiled FlashAttention kernel, and no throughput improvement should be claimed until it is benchmarked on the target GPU.

Complexity is the durable result rather than any single number: state-space layers are linear in sequence length and constant in memory per token; attention anchors are quadratic but occur only four times; sparse experts make active compute a fraction of total parameters. The combination allows long-context operation on modest hardware.

---

## Precision, Formats, and Exchange

The engine loads native formats, quantized GGUF checkpoints, and Safetensors exports. Quantized checkpoints use block formats compatible with the GGML family, covering four-bit and eight-bit schemes. Model artifacts and tokenizer tables are distributed through a public model repository, with export targets including native sixteen-bit, quantized GGUF, and Safetensors.

Mixed precision is applied consistently: weights may be half or quarter precision, accumulation is single precision, and master weights in training remain single precision. Loss scaling protects small gradients from underflow. The quantization formats are inference-only; training operates on higher-precision masters.

---

## Engineering Properties

Determinism is preserved for fixed seeds. Sampling uses a device-accelerated path with constant-time random access, so replaying a seed reproduces the same stream. Distributed gradient reduction is exact: token counts are combined through an integer collective rather than approximated by multiplying a local count by the world size, which makes token-weighted gradients correct across processes.

Device errors are checked at kernel boundaries through explicit status macros in each backend, and launch failures are not silently cleared. Resources follow ownership semantics: device handles, library handles, allocation arenas, and compiled programs are released in cleanup routines. Invariant checks abort with a file and line reference when a precondition is violated.

VisionTower can move its weights and encoding operations to CPU or CUDA; image pixels are loaded on the host and copied to the selected device when needed. The current vision trainer updates the multimodal projector and optionally the LLM, but it does not backpropagate through the ViT/VisionTower weights. Video is currently encoded frame-by-frame with time embeddings; a learned cross-frame temporal transformer is not yet implemented.

---

## Current Boundaries

The primary GPU implementation is NVIDIA CUDA; the CPU path is the non-CUDA execution path. HIP/SYCL adapters are experimental and not connected to the core model dispatcher, while XLA runtime execution remains unimplemented. CUDA kernels require runtime validation on the target accelerator; a successful offline Docker build only proves compilation. The current default inference context is capped at 4,096 tokens even when a checkpoint advertises a larger limit. Eight-bit floating point, micro-four-bit formats, and speculative decoding are not implemented. Four-bit quantization is inference-only.

These boundaries are stated so the architecture can be judged on what it does today, not on what the interface promises.

---

## License

The engine is distributed under the Apache License, Version 2.0. Copyright belongs to Ghassan, Mohammed Amine Erhassan, and the Ghassan AI team.
