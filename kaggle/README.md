# Ghassan v1 Flash on Kaggle — granite-4.0-h-tiny fine-tune runbook

The honest plan for building "Ghassan v1 Flash" as a **fine-tuned IBM Granite-4.0-H-Tiny**
(Apache-2.0, hybrid Mamba-2/Transformer, 1B active / 6.94B total), not a from-scratch model.

Free Kaggle GPU (verified 2026-11): **T4 ×2, 16 GB each, ~30 h/week, 12 h per session,
4 CPU cores, 29 GB RAM, 20 GB /kaggle/working, P100 retired**.

## 0. Attach the inputs

- Your Granite dataset (the 13.8 GB `granite-4.0-h-tiny` you have, with `config.json`,
  3 safetensors shards, `tokenizer.json`) → `/kaggle/input/<granite-dataset>/`
- Your SFT/CPT data → a Kaggle dataset (JSONL with `messages`/`conversations`, or raw .txt)

## 1. Build and validate the engine (sm_75 = the T4)

In the notebook (GPU on):

```bash
bash /kaggle/working/<copy-of-this-repo>/kaggle/build_kaggle.sh /kaggle/input/<granite-dataset>
```

This clones the repo, builds with `-DGAI_ENABLE_CUDA=ON` (the CMake default targets
`75-real 80-real 90-real`), runs the test suite, loads the real checkpoint with
`ghassan-ai info` and reports `ghassan-ai bench` ms/token. **Report those numbers as
measured — do not trust projections.**

## 2. Build the training shards (or JSONL auto-convert)

```bash
# raw .txt corpus (one document per line):
/kaggle/working/ghassan-src/build/bin/gai_train --build-shards /kaggle/working/shards \
    --shard-prefix train --shard-tokens 20000000

# OR a folder of .jsonl (messages/conversations/alpaca) is auto-converted under --data
```

## 3a. Fine-tune — the fp16 LoRA path (simplest, fits on one T4)

```bash
/kaggle/working/ghassan-src/build/bin/gai_train \
    --config /kaggle/working/<repo>/kaggle/granite_lora.yaml \
    --model /kaggle/input/<granite-dataset> \
    --weight-f16 1 \
    --tokenizer /kaggle/input/<granite-dataset> \
    --export /kaggle/working/ghassan-v1-flash --export-format safetensors --export-dtype bf16
```

Base weights in fp16 = **12.9 GiB**; LoRA adapters fp32 (~10–40 MB); optimizer states only
for the adapters. Fits the 16 GB T4 with headroom for seq 512 and micro-batch 1
(raise `seq_len`/`batch_size` only after a `--dry-run` passes).

## 3b. Fine-tune — the QLoRA path (tightest fit, recommended if fp16 OOMs)

QLoRA in this engine needs the base as **GGUF Q4_0/Q4_1/Q8_0** (not safetensors):

```bash
# 1) quantize the Granite base once (the tool streams; it does not need 27 GB RAM)
/kaggle/working/ghassan-src/build/bin/ghassan-ai quantize \
    --model /kaggle/input/<granite-dataset> --out /kaggle/working/granite_q4_0.gguf --profile q4_0

# 2) QLoRA fine-tune (base packed on device ~4.7 GB; adapters fp32)
/kaggle/working/ghassan-src/build/bin/gai_train \
    --config /kaggle/working/<repo>/kaggle/granite_qlora.yaml \
    --model /kaggle/working/granite_q4_0.gguf --weight-q4 0 \
    --tokenizer /kaggle/input/<granite-dataset> \
    --export /kaggle/working/ghassan-v1-flash --export-format safetensors --export-dtype bf16
```

Both paths fold the LoRA deltas into the exported weights in fp32 at export time —
the exported repository is a complete, self-contained model (no adapter files needed).

## 4. Verify the artifact before publishing

```bash
/kaggle/working/ghassan-src/build/bin/ghassan-ai info       --model /kaggle/working/ghassan-v1-flash
/kaggle/working/ghassan-src/build/bin/ghassan-ai perplexity --model /kaggle/working/ghassan-v1-flash \
    --text "held-out paragraph with no training overlap"
/kaggle/working/ghassan-src/build/bin/ghassan-ai bench      --model /kaggle/working/ghassan-v1-flash --tokens 64
```

Also reload it once in Python (`transformers`) and check the logits match the engine
on a few fixed prompts, and compare against the base model's perplexity on the same
held-out text. **A fine-tune that does not beat the base on its own domain is a
regression, not a release.**

## 5. Serving for speed (honest note)

The Ghassan engine's inference path is **alpha** (single sequence, no batching, no
FlashAttention, Granite prefill is one-token-at-a-time). For the fastest realistic
serving of the artifact:

1. export GGUF (`--export-format gguf --export-profile q8_k` or `q4_0`),
2. serve with llama.cpp (Granite-4.0-H is supported there) or vLLM,
3. measure tokens/s on the target machine.

Groq-class latency comes from Groq's LPU hardware; it is not attainable on a T4 or by
kernel tuning alone. What the hybrid architecture buys you is a **constant-memory
Mamba-2 decode** and **1B active params** — that is the real lever for tokens/s, and
it is already in Granite, not in this repo.

## Expected resource fit (T4 16 GB)

| Stage | fp16 LoRA | QLoRA Q4_0 |
|---|---|---|
| Base weights | 12.9 GB | ~4.7 GB |
| LoRA adapters (fp32) | ~10–40 MB | ~10–40 MB |
| Optimizer (AdamW, adapters only) | ~80 MB | ~80 MB |
| Activations (seq 512, micro-batch 1, ckpt segments) | ~1–2 GB | ~1–2 GB |
| **Total** | **~15 GB (tight)** | **~7–8 GB (comfortable)** |

If fp16 OOMs: switch to 3b, or `--weight-f16 1 --seq-len 384 --ckpt-segments 2`,
or `--weight-q4 0`. Full fine-tuning of ≥ 800 M params is refused by the engine at
this budget by design.

## What was verified where (as of this runbook)

- Build + 13 tests (CPU reference path; CUDA parity tests run only with a GPU):
  verified locally with GNU g++ 16.2, no GPU available in the authoring environment.
- The CUDA build was **never executed** here — the first `build_kaggle.sh` run on the
  T4 is the first real CUDA compile. Report its output before trusting it.
- The exported artifact is self-contained and bit-exact vs the adapter model on the
  CPU reference path (test_export_lora).
