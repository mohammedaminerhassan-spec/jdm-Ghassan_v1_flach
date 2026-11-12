# Ghassan v1 Flash — a from-scratch LLM engine (C++20 / CUDA)

**The body**: a self-contained inference + training engine — no external
libraries, **no Python, no PyTorch**. **The mind**: the weights, straight from
a HuggingFace safetensors checkpoint.

It **runs, trains and exports**:

- **IBM Granite-4.0-H-Tiny** (`GraniteMoeHybridForCausalLM`: a hybrid
  **Mamba-2 + GQA + MoE** model, 6.939 B parameters) directly from its
  safetensors — the multipliers, the NoPE path, the attention scale and the
  packed MoE layout are implemented exactly, and verified against
  `transformers` and against the real safetensors headers.
- a **self-contained GGUF** named *Ghassan v1 Flash* (Q4_0 / Q8_0 / fp16),
  self-contained: the vocabulary, the merges and the chat template are
  embedded.
- **LoRA fine-tuning** from a dense model (init -> train -> checkpoint ->
  restore -> merge -> export), with AdamW / Lion / Muon and a DDP trainer.

```
core/        tensors · dtypes · device (CPU/CUDA) · ops · mmap · RNG · unicode · JSON
model/       the model: dense + MoE, GQA, RoPE (NTK/YaRN), SWA, Granite hybrid
tokenizer/   byte-level BPE (GPT-2 / Qwen / Llama-3 / Granite) · chat styles
format/      GGUF codec (native + llama.cpp compatible) · HuggingFace safetensors
inference/   generator · paged/ring KV cache · sampler · chat session
quantization/ Q4_0 · Q8_0 · Q4_1
evaluation/  perplexity
cuda/        attention · MoE · Mamba-2 kernels (sm_75 / T4 and up)
training/    trainer · AdamW / Lion / Muon · LoRA · dataloader · checkpoint
tools/       ghassan-ai (serve/inspect/export) · gai_train
```

```bash
ghassan-ai info     --model <hf-dir|model.gguf>
ghassan-ai generate --model <hf-dir> --prompt "hi" --max-tokens 64
ghassan-ai quantize --model <hf-dir> --out ghassan-v1-flash.gguf \
    --profile q8_k --model-name "Ghassan v1 Flash"
gai_train --config <recipe.yaml>
```

### The CPU kernels

They are split per family (`core/ops_cpu.h` maps every declaration to its
file):

| file | what it holds |
|---|---|
| `ops_cpu_dense.cpp` | the shared primitives: the GEMM, the linear ops, the elementwise ops, the embedding, the norms, the RoPE, the SwiGLU — plus the loss, the optimizer steps and the sampler helpers |
| `ops_cpu_attention.cpp` | the attention family: the prefill, the backward, the incremental decode (the ring KV cache, fp32 and packed fp16) |
| `ops_cpu_moe.cpp` | the mixture of experts: the routing, the expert GEMMs, the shared expert |
| `ops_cpu_mamba.cpp` | the IBM Granite Mamba-2 layer |
| `ops_cpu_simd.h` | the AVX2/FMA helpers shared by the above |

A **dense** model and a **MoE** model share the primitives: the dense model is
the same path with the MoE block replaced by a plain MLP.

## Build

**Linux / Kaggle (with CUDA):**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON
cmake --build build -j 2
```

**Windows (MSVC):**

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The build is warning-free (`-Wall -Wextra -Werror`).

## Use

```bash
# what is this model?
./build/bin/ghassan-ai info --model <hf-dir|model.gguf>

# the logits of a prompt / of an explicit token sequence
./build/bin/ghassan-ai logits --model <hf-dir> --prompt "hi" --out logits.f32
./build/bin/ghassan-ai logits --model <hf-dir> --ids "82 17243"

# a completion (greedy, with the token ids for verification)
./build/bin/ghassan-ai generate --model <hf-dir> --prompt "hi" --max-tokens 64 \
    --greedy --no-stream --raw --print-ids

# the deliverable: a self-contained GGUF
./build/bin/ghassan-ai quantize --model <hf-dir> --out ghassan-v1-flash.gguf \
    --profile q8_k --model-name "Ghassan v1 Flash"

# fine-tune (the LoRA path is verified end to end)
./build/bin/gai_train --config <recipe.yaml>
```

## Granite-4.0-H-Tiny

| piece | value |
|---|---|
| layers | 40 = **36 Mamba-2** + **4 attention** (at 5, 15, 25, 35) |
| hidden | 1536, attention head 128 (12 heads / 4 KV) |
| Mamba-2 | d_inner 3072 (expand 2), 48 heads, d_head 64, d_state 128, groups 1, conv k=4, chunk 256 |
| MoE | 64 experts, top-6, expert width 512, **shared expert width 1024** |
| positional | **NoPE** (no RoPE at all) |
| multipliers | embeddings x12, residual x0.22, attention x1/128, logits /6 |
| weights | bf16 on disk (the loader casts them to fp32 masters) |
| vocab | 100352, tied embeddings |

The engine implements it exactly: the multipliers, the NoPE path, the
attention scale (`attention_multiplier`, not `1/sqrt(head_dim)`), the packed
Granite MoE layout (`block_sparse_moe.input_linear` / `output_linear`) and the
shared expert with its own width.

`granite-documentation/granit filles confiques/` holds the **official
HuggingFace files** of `ibm-granite/granite-4.0-h-tiny` (the config, the
tokenizer and the tensor index). The weights themselves (13.9 GB, 3 shards) are
not in the repository: they are attached to the Kaggle notebook as a dataset.

## Verification

The Granite path is checked three ways, not only against itself:

**1. The official config, parsed by the engine == `transformers`' own parse**
(29/29 fields: the vocabulary, the layer kinds, the multipliers, the Mamba-2
geometry, the MoE, the shared expert width, NoPE).

**2. The official tensor index.** The real safetensors headers (fetched from
HuggingFace with a range request, without downloading 13.9 GB) hold **586
tensors** and **6,939,037,248 parameters**; the engine expects exactly the
same names and the same byte layouts, including the 3-D packed MoE experts
(`[64, 1024, 1536]` and `[64, 1536, 512]`).

**3. The forward pass, against independent reference implementations** (a
separate implementation of the Mamba-2 and the attention blocks, written from
the reference modelling code):

* the Mamba-2 block (conv1d + selective scan + gated norm) matches to ~1e-8
* the attention block (GQA + NoPE + the attention multiplier + the causal
  mask) matches to ~1e-7
* the **complete multi-layer Granite forward** (Mamba-2 + attention + the MoE
  + the shared expert + the multipliers) matches to ~4.6e-7
* the decode path (the KV cache + the Mamba-2 recurrent state) reproduces the
  full-prefill greedy continuation
* the LoRA path: init -> train -> checkpoint -> restore -> merge -> export
* the GGUF round trip, the chat template (byte-exact against
  `transformers.apply_chat_template`) and the tokenizer

## Memory (Granite-4.0-H-Tiny, 6.939 B parameters)

| block | size |
|---|---|
| weights, fp32 masters | **25.85 GiB** |
| weights, fp16 (what a GGUF holds) | 12.92 GiB |
| fp16 weight cache (on top of the masters) | 12.95 GiB |
| mamba-2 recurrent state | 55 MiB |
| KV cache, fp16, @131072 ctx | 1.00 GiB (only the 4 attention layers keep one) |

A single 16 GB T4 cannot hold the fp32 masters: the engine says so up front
instead of dying inside `cudaMalloc`. The **fp16 weight storage mode** (no fp32
masters) is the next step to run Granite on a T4.

## Documentation

`granite-documentation/` holds the project documentation (the architecture
notes, the adaptation plan, the state of the project and the official
HuggingFace files of the checkpoint).

The Kaggle runbook is a single cell:

```bash
!bash "kaggle configue/kaggle_granite_run.sh"
```

The verification suite (the reference implementations, the fixture generators
and the comparisons against `transformers` and against the real safetensors
headers) lives outside the repository, in
`C:\Users\Ghassan PC\Documents\ghassan-verify\` — the engine itself stays
C++ only.

## License

MIT (see `LICENSE`).
