# Ghassan v1 Flash — Granite-4.0-H-Tiny (Kaggle runbook)

## What this is

The engine (`ghassan-ai` / `gai_train`) runs **IBM Granite-4.0-H-Tiny** directly from
its HuggingFace safetensors — no Python conversion step, no PyTorch anywhere.

```
granite-4.0-h-tiny/
├── config.json                 -> GraniteMoeHybridConfig
├── model-0000{1,2,3}-of-00003.safetensors   (the weights, on Kaggle)
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json / generation_config.json
├── special_tokens_map.json / chat_template.jinja / vocab.json / merges.txt
```

The deliverable is a **self-contained GGUF named "Ghassan v1 Flash"**: the vocabulary,
the merges and the chat template are embedded, so it runs with `ghassan-ai` alone.

## Architecture the engine implements for it

| piece | value |
|---|---|
| layers | 40 = **36 Mamba-2** + **4 attention** (at 5, 15, 25, 35) |
| hidden | 1536, attention head 128 (12 heads / 4 KV) |
| Mamba-2 | d_inner 3072 (expand 2), 48 heads, d_head 64, d_state 128, groups 1, conv k=4, chunk 256 |
| MoE | 64 experts, top-6, expert width 512, **shared expert width 1024** |
| positional | **NoPE** (no RoPE at all) |
| multipliers | embeddings x12, residual x0.22, attention x1/128, logits /6 |
| vocab | 100352, tied embeddings, byte-level BPE (digits in groups of 1-3) |
| chat | `<|start_of_role|>role<|end_of_role|>text<|end_of_text|>` |
| parameters | 6.54 B (6.39 B without the embeddings; 92 % of them are the MoE experts) |

## The memory reality (read this before the run)

Measured by the engine's own dry run (`gai_train --dry-run`, no allocation):

| block | size |
|---|---|
| weights, fp32 masters | **24.38 GiB** |
| weights, fp16 (what the GGUF can hold) | 12.19 GiB |
| fp16 weight cache (on top of the masters) | 12.48 GiB |
| mamba-2 recurrent state | 55 MiB |
| KV cache, fp16 packed, @131072 ctx | 1.00 GiB (only the 4 attention layers keep one) |
| KV cache, fp16 packed, @32768 ctx | 0.25 GiB |
| activations, B=1 T=2048, checkpointed | 7.68 GiB |
| **a full fine-tune (AdamW)** | **119.22 GiB** |

**Consequences on Kaggle (2x T4, 16 GB each):**

* **Export / quantize: works.** The exporter reads the safetensors shard by shard and
  writes the GGUF tensor by tensor, so it needs ~25 GiB of **host RAM** (Kaggle has
  ~29 GB) and no GPU at all. `--device cpu` is the right choice there.
* **Inference on a T4: does not fit yet.** The engine keeps **fp32 masters** on the
  device (24.38 GiB) plus an optional fp16 copy (+12.48 GiB). A T4 has 16 GB.
  The engine now says so up front instead of dying inside `cudaMalloc`:
  *"the model does not fit this GPU: it needs … but the card has …"*.
  `--device cpu` runs it (slow but correct); the GPU path needs the **fp16 weight
  storage mode** (see *What is not implemented*).
* **Training: refused.** The Mamba-2 layers have no backward pass yet, so a
  training step on Granite stops with a clear error:
  *"granite mamba layers do not support the backward pass yet (inference only)"*.

## Verified locally (CPU, tiny synthetic checkpoint with the same layout)

The Granite path is checked against **independent reference implementations**
written from the reference modelling code (not against itself):

| check | result |
|---|---|
| the **Mamba-2 block** (conv1d + selective scan + gated norm) | max diff **2.4e-8** |
| the **attention block** (GQA + NoPE + `attention_multiplier` + causal mask) | max diff **1.5e-7** |
| the **decode path** (KV cache + the Mamba-2 recurrent state) = the full-prefill greedy continuation | top-1 at every step |
| the **MoE** (the Granite packing + the top-k routing + the shared expert) | included in the two checks above |
| the **multipliers** (embedding x12, residual x0.22, attention x1/128, logits /6) | included |
| `ghassan-ai info --model granite_tiny` | config + weights parse |
| `ghassan-ai quantize --profile q8_k` -> GGUF -> reload | same logits (Q8: max diff 9.6e-5) |
| `ghassan-ai generate` | token by token, the mamba recurrent state carried between tokens |
| chat template -> **5/5 byte-exact against `transformers` `apply_chat_template`** |
| tokenizer -> **9/9 byte-exact against `transformers`** (Granite digits {1,3}) |

The references and the fixtures live outside the repository
(`$env:TEMP/opencode/`, `data/*_tiny/`); `granite_tiny/` (the same layout with
random weights) is committed so the repo is testable without the 13.9 GB.
* tokenizer -> **9/9 byte-exact against `transformers`** (Granite digits {1,3})

## Kaggle runbook

All the commands below run from the notebook root (the repository is cloned into
`/kaggle/working/Ghassan-Ai-model`).

### Cell 0 — the weights

```python
# the dataset must be attached to the notebook:
#   granite-4.0-h-tiny  (13.9 GB, 3 safetensors shards)
MODEL = "/kaggle/input/granite-4.0-h-tiny/granite-4.0-h-tiny"
```

### Cell 1 — build (the only cell that needs a GPU compiler)

```bash
!cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON
!cmake --build build -j 2
```

`cuda/mamba.cu` and `cuda/attention.cu` are compiled **here** — they have never been
compiled on a machine without a CUDA toolkit. If the build fails, that is the first
thing to look at (see *Verify the CUDA Mamba-2 kernel* below).

### Cell 2 — inspect the real weights

```bash
!./build/bin/ghassan-ai info --model $MODEL
```

Expect: `40 layers, d=1536, H=12/4, V=100352, MoE`, `arch=granite`.

### Cell 3 — the forward pass and a generation (CPU: the GPU does not fit)

```bash
!./build/bin/ghassan-ai logits --model $MODEL --device cpu \
    --prompt "Please list one IBM Research laboratory located in the United States."
!./build/bin/ghassan-ai generate --model $MODEL --device cpu \
    --prompt "Please list one IBM Research laboratory located in the United States." \
    --max-tokens 64 --ctx 2048 --dump-prompt
```

The CPU path is the numerical reference; a 6.9 B model on 4 vCPUs is slow
(roughly a second per token) but it is correct.

### Cell 4 — export the GGUF (the deliverable)

```bash
!./build/bin/ghassan-ai quantize --model $MODEL --device cpu \
    --out /kaggle/working/ghassan-v1-flash.gguf \
    --profile q8_k --model-name "Ghassan v1 Flash"
```

* `--profile q8_k` -> ~7.1 GB, `fp16` -> ~12.2 GB, `q4_0` -> ~3.8 GB.
* `--model-name` is what the GGUF metadata shows. Check it:

```bash
!./build/bin/ghassan-ai info --model /kaggle/working/ghassan-v1-flash.gguf
```

The GGUF is self-contained: the vocabulary, the merges and the chat template are
embedded, so it runs with `ghassan-ai` alone.

### Cell 5 — reload the GGUF and check it

```bash
!./build/bin/ghassan-ai generate --model /kaggle/working/ghassan-v1-flash.gguf \
    --device cpu --max-tokens 32 --ctx 2048 \
    --prompt "Please list one IBM Research laboratory located in the United States."
```

### Cell 6 — the 2x T4 training recipe (dry run only, for now)

```bash
!./build/bin/gai_train --dry-run --config "kaggle configue/ghassan_granite_2xt4.yaml"
```

The recipe is ready for the day the Mamba-2 backward lands; today it prints the
memory plan (119 GiB for a full fine-tune, which is why a LoRA recipe is the
only realistic one).

## Verify the CUDA Mamba-2 kernel (once the build succeeds)

The Mamba-2 CUDA kernels (`cuda/mamba.cu`) are compiled **here** — they had
never been compiled on a machine without a CUDA toolkit, and they carried
three real bugs (host pointers passed to the kernels, the recurrent state
never read, a shuffle reduction that covered 32 of the 256 lanes). They are
now written against the CPU kernel, which is the verified specification:

```bash
# 1) the CPU reference (the tiny fixture has the same layout as Granite)
!./build/bin/ghassan-ai logits --model granite_tiny --device cpu --out /tmp/cpu.f32

# 2) the same model on the GPU
!./build/bin/ghassan-ai logits --model granite_tiny --device cuda --out /tmp/gpu.f32

# 3) the two must match
!python -c "
import numpy as np
a = np.fromfile('/tmp/cpu.f32', dtype='<f4')
b = np.fromfile('/tmp/gpu.f32', dtype='<f4')
print('max |cpu - gpu| =', np.abs(a.astype('float64') - b.astype('float64')).max())
"
```

The same comparison on the mamba-only fixture (`data/mamba_tiny/`) isolates
the Mamba-2 kernel from the rest of the model. If the two differ, the bug is
in `cuda/mamba.cu` (the CPU kernel `core/ops_cpu_mamba.cpp` is the
specification and is verified against an independent reference).

## Precision

- compute: fp16 tensor cores on CUDA (T4), fp32 masters
- export: `q8_k` (8-bit, the default), `fp16` (`--profile fp16`) or `q4_0`;
  the file type and the quantization version are recorded in the GGUF metadata
- the model's own weights are bf16 on disk; the loader casts them to fp32
  masters (bf16 is not a T4 tensor-core format)

## What is NOT implemented yet

1. **fp16 weight storage.** The engine keeps fp32 masters on the device, so a
   6.9 B model needs 24.4 GiB of VRAM. With fp16-only storage it would be
   12.2 GiB and would fit a T4. This needs an fp16 MoE GEMM (the CUDA MoE
   kernel calls the same `linear_forward` as the rest, so it is a contained
   change) plus an fp16 Mamba-2 projection.
2. **The Mamba-2 backward** (the adjoint selective scan). Without it a Granite
   checkpoint cannot be fine-tuned at all: the engine refuses the run with a
   clear message instead of producing wrong gradients.
3. **The fused QKV fp16 path is attention-only**; the MoE and the Mamba
   projections still read the fp32 masters.
