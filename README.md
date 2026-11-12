# Ghassan v1 Flash — a from-scratch LLM engine (C++20 / CUDA)

The **body**: a self-contained inference + training engine, no external
libraries, no Python, no PyTorch. The **mind**: the weights, which come
straight from a HuggingFace safetensors checkpoint.

The engine runs **IBM Granite-4.0-H-Tiny** (`GraniteMoeHybridForCausalLM`: a
hybrid Mamba-2 + GQA + MoE model) directly from its safetensors, and exports a
self-contained **GGUF** named *Ghassan v1 Flash*.

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

`granite_tiny/` is a tiny checkpoint with the **same tensor names and layout**
as the real model (random weights): it makes the repo testable without the
13.9 GB of real weights.

## Verification

The Granite path is checked against **independent reference implementations**
(a separate Python implementation of the Mamba-2 and the attention blocks,
written from the reference modelling code), not only against itself:

* the Mamba-2 block (conv1d + selective scan + gated norm) matches to ~1e-8
* the attention block (GQA + NoPE + the attention multiplier + the causal
  mask) matches to ~1e-7
* the decode path (the KV cache + the Mamba-2 recurrent state) reproduces the
  full-prefill greedy continuation
* the LoRA path: init -> train -> checkpoint -> restore -> merge -> export
* the GGUF round trip, the chat template (byte-exact against
  `transformers.apply_chat_template`) and the tokenizer

## Memory (Granite-4.0-H-Tiny, 6.54 B parameters)

| block | size |
|---|---|
| weights, fp32 masters | 24.38 GiB |
| weights, fp16 (what a GGUF holds) | 12.19 GiB |
| fp16 weight cache (on top of the masters) | 12.48 GiB |
| mamba-2 recurrent state | 55 MiB |
| KV cache, fp16, @131072 ctx | 1.00 GiB |

A single 16 GB T4 cannot hold the fp32 masters: the engine says so up front
instead of dying inside `cudaMalloc`. The **fp16 weight storage mode** (no fp32
masters) is the next step to run Granite on a T4.

## Documentation

* `GRANITE.md` — the architecture, the memory budget and the Kaggle runbook
* `CONTEXT.md` — the project state, the decisions and the known gaps

## License

MIT (see `LICENSE`).
