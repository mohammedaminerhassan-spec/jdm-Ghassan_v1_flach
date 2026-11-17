<!--
  Ghassan AI — Ghassan v1 Flash
  Project documentation: verified against the repository source and CMake/CLI interfaces.
-->

<div align="center">

# Ghassan v1 Flash

### A native C++20 and CUDA engine for language-model inference, training, and model conversion

[![Language](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![Build](https://img.shields.io/badge/build-CMake-064F8C?style=flat-square&logo=cmake)](https://cmake.org/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-2ea44f?style=flat-square)](LICENSE)

**Built by Ghassan AI**

[Repository](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_flach) ·
[Issues](https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_flach/issues)

</div>

---

## Overview

**Ghassan v1 Flash** is the name of this project’s C++ language-model engine. The repository contains the runtime and tooling—not a downloadable pretrained checkpoint or a bundled training dataset.

The engine is designed to keep the main model workflow in native code:

- Load supported Hugging Face checkpoint directories or GGUF model files.
- Run text generation and interactive chat on CPU or NVIDIA CUDA.
- Train models from configured datasets and fine-tune supported checkpoints.
- Export model weights to GGUF or Hugging Face safetensors format.
- Tokenize text, inspect model metadata, benchmark generation, and calculate perplexity.

The implementation includes a tensor and device layer, CPU operators, CUDA kernels, model implementations, tokenization, inference, training, checkpoint handling, and model-format codecs.

> **Project scope:** this repository documents the engine implementation. The architecture and behavior of a particular model depend on the checkpoint and its configuration. A model name, parameter count, quality level, speed, or hardware fit should not be inferred from the engine name alone.

## Architecture at a glance

```text
                         Ghassan v1 Flash
                    Native C++20 model engine
                                |
             +------------------+------------------+
             |                  |                  |
          Core / Tensor      Model layer       Tokenizer
       storage, dtypes,    transformer and    BPE, byte map,
       device, dispatch    hybrid components  chat templates
             |                  |                  |
       +-----+------+      +----+-----+       +----+-----+
       |            |      |          |       |          |
      CPU         CUDA   Attention   MoE    ChatML     Granite
    operators    kernels  / Mamba  / shared templates   template
       |            |
       +------------+-------------------------------+
                    |
             Inference and training
                    |
          +---------+----------+
          |                    |
       Inference             Training
     generation, chat,     dataloading, loss,
      sampling, KV cache   optimizers, resume
          |                    |
          +---------+----------+
                    |
             Model formats
          GGUF / safetensors
```

## Capabilities

| Area | Implementation in this repository |
|---|---|
| Runtime language | C++20 |
| GPU backend | Optional NVIDIA CUDA backend |
| CPU execution | Native CPU operators; OpenMP is enabled when found |
| Tensor types | F32, F16, BF16, integer types, Q4_0, Q4_1, and Q8_0 |
| Model families | Qwen-style transformer configuration and IBM Granite hybrid configuration |
| Model components | Attention, Mamba-2, Mixture-of-Experts, shared experts, normalization, RoPE |
| Inference | Generation, streaming chat, sampling controls, KV cache, device selection |
| Training | Pretraining and fine-tuning workflows, gradient accumulation, checkpoint/resume |
| Optimizers | AdamW, Lion, Muon |
| Schedulers | Cosine and WSD |
| Fine-tuning | LoRA; QLoRA path for GGUF base weights |
| Input data | Native GBIN shards; JSONL conversion; text-to-shard tooling |
| Model I/O | Hugging Face safetensors and GGUF |
| Evaluation | Perplexity evaluation and generation benchmark commands |
| Distributed training | Optional NCCL integration when detected by CMake |

The exact operators, architecture constraints, supported tensor mappings, and available options are defined by the source code and the checkpoint being loaded.

## Repository structure

```text
.
├── CMakeLists.txt
├── core/
│   ├── common.*             Shared types, errors, logging, timing
│   ├── config.*             Configuration parsing and validation
│   ├── device.*             CPU/CUDA device and memory management
│   ├── dtype.*              Data types and quantized block layouts
│   ├── tensor.*             Tensor and storage implementation
│   ├── ops.*                Operator API and dispatch
│   └── ops_cpu_*.cpp        CPU operator implementations
│
├── cuda/
│   ├── cuda_utils.*         CUDA runtime utilities
│   ├── kernels.cu           Core CUDA kernels
│   ├── attention.cu         Attention kernels
│   ├── mamba.cu             Mamba kernels
│   └── moe.cu               MoE kernels
│
├── model/
│   └── model.*              Model configuration, parameters, forward/backward
│
├── tokenizer/
│   ├── tokenizer.*          Vocabulary, byte-level BPE and encoding
│   └── chat_template.*      ChatML and Granite message formatting
│
├── inference/
│   ├── generator.*          Token generation
│   ├── chat.*               Interactive chat session
│   ├── sampler.*            Sampling and repetition controls
│   └── kv_cache.*           Inference KV cache
│
├── training/
│   ├── trainer.*            Training loop and configuration
│   ├── dataloader.*         GBIN shard reader/writer
│   ├── jsonl_loader.*       JSONL-to-GBIN conversion
│   ├── optimizer.*          AdamW, Lion and Muon
│   ├── scheduler.h          Learning-rate schedules
│   ├── checkpoint.*         Training checkpoint I/O
│   ├── distributed.*        Distributed training support
│   └── shard_builder.*      Text dataset shard construction
│
├── format/
│   ├── gguf_codec.*         GGUF reader/writer and model mapping
│   ├── hf_checkpoint.*      Hugging Face checkpoint reader
│   └── hf_writer.*          Hugging Face safetensors exporter
│
├── quantization/
│   └── quantize.*           Quantization and dequantization routines
│
├── evaluation/
│   └── perplexity.*         Perplexity evaluation
│
└── tools/
    ├── inference_cli.cpp    ghassan-ai command-line application
    ├── training_cli.cpp     gai_train command-line application
    ├── model_loader.*       Model loading and runtime options
    └── cli_common.*         Shared CLI argument handling
```

## Requirements

### Required

- CMake **3.20 or newer**.
- A C++ compiler with C++20 support.
- A supported operating system and toolchain for the selected compiler.

### Optional

- NVIDIA CUDA Toolkit for the CUDA backend.
- OpenMP for parallel CPU operators.
- NCCL for multi-GPU distributed training (only when CUDA and NCCL are detected).
- Apache Arrow and Parquet libraries for the optional Parquet input path.

CMake options:

| Option | Default | Purpose |
|---|---:|---|
| `GAI_ENABLE_CUDA` | `ON` | Detect and build the CUDA backend when available |
| `GAI_ENABLE_OPENMP` | `ON` | Enable OpenMP CPU parallelism when found |
| `GAI_ENABLE_NCCL` | `ON` | Enable NCCL support when CUDA and NCCL are found |
| `GAI_BUILD_TESTS` | `ON` | Look for `tests/test_*.cpp` test sources |
| `GAI_NATIVE_ARCH` | `OFF` | Compile CPU code for the build machine |
| `GAI_ENABLE_PARQUET` | `OFF` | Enable Parquet input when Arrow/Parquet are found |

The CUDA architecture list is configured by CMake and can be overridden with `CMAKE_CUDA_ARCHITECTURES`. Check the installed toolkit and GPU architecture when configuring a CUDA build.

## Build

Clone the repository:

```bash
git clone https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_flach.git
cd jdm-Ghassan_v1_flach
```

### CPU build

Use this when CUDA is not installed or when you want to build the portable CPU backend explicitly:

```bash
cmake -S . -B build-cpu \
  -DGAI_ENABLE_CUDA=OFF \
  -DGAI_ENABLE_NCCL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cpu --parallel
```

### NVIDIA CUDA build

```bash
cmake -S . -B build-cuda \
  -DGAI_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda --parallel
```

If CMake does not select the architecture you need, set it explicitly. For example, `-DCMAKE_CUDA_ARCHITECTURES=75` targets NVIDIA T4-class GPUs:

```bash
cmake -S . -B build-cuda \
  -DGAI_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda --parallel
```

CMake places executables in the build directory’s `bin` folder:

- `ghassan-ai` — inference, inspection, conversion, and evaluation.
- `gai_train` — model training and fine-tuning.

On Windows, the executable files have the `.exe` extension.

> A successful compilation confirms that the selected source and toolchain build together. It does not, by itself, validate every model architecture, checkpoint, CUDA kernel, or training configuration.

## Inference

The inference CLI accepts either a supported Hugging Face checkpoint directory or a self-contained GGUF file.

### Inspect a model

```bash
./build-cuda/bin/ghassan-ai info --model /path/to/model.gguf
```

### Generate text

```bash
./build-cuda/bin/ghassan-ai generate \
  --model /path/to/model.gguf \
  --prompt "Explain how a mixture-of-experts model works." \
  --device cuda \
  --ctx 2048 \
  --max-tokens 160
```

For CPU inference, use `--device cpu`. To let the engine choose an available backend, use `--device auto`.

### Start an interactive chat

```bash
./build-cuda/bin/ghassan-ai chat \
  --model /path/to/model.gguf \
  --device cuda
```

An optional system message can be supplied with `--system`. The engine does not add a hidden default system prompt.

### Sampling options

The CLI exposes the following generation controls:

| Option | Default | Meaning |
|---|---:|---|
| `--temp` | `0.8` | Sampling temperature |
| `--top-k` | `40` | Top-k candidate limit |
| `--top-p` | `0.92` | Nucleus sampling threshold |
| `--min-p` | `0.05` | Minimum probability threshold |
| `--repeat` | `1.12` | Repetition penalty |
| `--no-repeat-ngram` | `0` | Repeated n-gram blocking; zero disables it |
| `--max-tokens` | `256` | Maximum number of generated tokens |
| `--seed` | `0` | Random seed; zero selects a random seed |
| `--greedy` | Off | Use greedy decoding |
| `--ctx` | Model setting | Inference context/KV-cache capacity |
| `--no-stream` | Off | Print the completed reply instead of streaming |

Run `ghassan-ai --help` for the complete command list and current CLI options.

## Training and fine-tuning

Training is configured through a YAML recipe and the `gai_train` executable.

```bash
./build-cuda/bin/gai_train --help
```

The training configuration supports:

- A model configuration and training schedule.
- A training data directory with train/validation shard prefixes.
- Batch size, sequence length, gradient accumulation, and learning rate.
- AdamW, Lion, or Muon optimization.
- Cosine or WSD learning-rate scheduling.
- FP32, FP16, or BF16 training precision settings.
- Checkpoint output and resume controls.
- Optional pretrained Hugging Face or engine checkpoint weights.
- LoRA adapter configuration.
- CUDA memory and output-budget preflight options.

A run must define **either** `training.max_steps` **or** `training.epochs`; defining both, or neither, is rejected by the configuration parser.

### Preflight before training

Use dry-run mode to inspect the configuration and estimated resource requirements without starting the training run:

```bash
./build-cuda/bin/gai_train \
  --config ./recipes/experiment.yaml \
  --dry-run
```

You can override selected recipe values from the command line:

```bash
./build-cuda/bin/gai_train \
  --config ./recipes/experiment.yaml \
  --max-steps 100 \
  --batch-size 1 \
  --seq-len 512
```

### Fine-tuning a Hugging Face checkpoint

```bash
./build-cuda/bin/gai_train \
  --config ./recipes/finetune.yaml \
  --model /path/to/huggingface-checkpoint
```

The checkpoint directory must contain the architecture configuration, supported safetensors weights, and tokenizer files required by the selected workflow.

### QLoRA with a GGUF base model

The training CLI exposes packed Q4/Q8 base-weight loading for the GGUF QLoRA path. The base weights remain frozen while the configured adapters are trained.

```bash
./build-cuda/bin/gai_train \
  --config ./recipes/qlora.yaml \
  --model /path/to/base-model.gguf \
  --weight-q4 0 \
  --device cuda
```

`--weight-q4` accepts `0` (Q4_0), `1` (Q4_1), or `2` (Q8_0), and requires a GGUF model input. Actual memory use depends on the model, context, batch settings, and available device memory.

## Dataset formats

The training data loader reads the engine’s **GBIN** shard format.

The repository also includes a JSONL converter for common supervised fine-tuning formats:

- OpenAI-style `messages` arrays.
- ShareGPT-style `conversations` arrays.
- Alpaca-style `instruction`, `input`, and `output` fields.
- Raw text records using a `text` field.

When a tokenizer is configured, the training CLI can automatically convert JSONL/JSON files in the data directory to GBIN shards.

For plain-text documents, the CLI provides a shard-building option:

```bash
./build-cuda/bin/gai_train \
  --config ./recipes/pretrain.yaml \
  --build-shards ./data/text \
  --shard-prefix train \
  --shard-tokens 1000000
```

The text shard builder treats text files as document sources. Keep training and validation data separate and configure the corresponding prefixes in the recipe.

## Model formats and conversion

### Supported input workflows

- Hugging Face checkpoint directories containing `config.json`, safetensors weights (single-file or indexed shards), and tokenizer data.
- GGUF model files supported by the repository’s GGUF reader.
- Engine training checkpoints (`.ckpt`) for training resume and export workflows.

### Export a GGUF model

```bash
./build-cuda/bin/ghassan-ai quantize \
  --model /path/to/huggingface-checkpoint \
  --out ./output/ghassan-model-q4.gguf \
  --profile q4_0 \
  --compat native
```

The implemented GGUF export profiles include:

| Profile | Actual implementation |
|---|---|
| `q4_0` | Q4_0 for the default weight class, with selected sensitive tensors stored at higher precision |
| `q4_1` | Q4_1 for the default weight class, with selected tensors stored at higher precision |
| `q8_k` | Q8_0 codec; this is **not** a true GGML K-quant implementation |
| `fp16` | FP16 weights with selected normalization tensors kept at FP32 |

Some familiar profile aliases are mapped to the nearest implemented codec. In particular, names such as `q4_k`, `q4_k_m`, `q5_k_m`, and `q6_k` do not mean that native K-quant encoders are implemented here. Check the CLI output and source before relying on an alias for a particular compatibility target.

### Export a training checkpoint

GGUF export:

```bash
./build-cuda/bin/ghassan-ai export \
  --checkpoint ./checkpoints/last.ckpt \
  --tokenizer /path/to/tokenizer \
  --out ./output/model.gguf
```

Hugging Face safetensors export:

```bash
./build-cuda/bin/ghassan-ai export \
  --checkpoint ./checkpoints/last.ckpt \
  --tokenizer /path/to/tokenizer \
  --out ./output/hf-model \
  --format safetensors \
  --dtype f16
```

## Tokenizer and chat templates

The tokenizer implementation contains byte-level BPE support and pre-tokenizer modes for GPT-2, Qwen, and Granite vocabularies.

The chat-template layer supports:

- **ChatML**, used by supported ChatML-style checkpoints.
- **Granite**, using Granite role and turn markers.

The template must match the model’s tokenizer and training format. A tokenizer/model mismatch can produce invalid token IDs or poor generation behavior.

Inspect a tokenizer:

```bash
./build-cuda/bin/ghassan-ai tokenizer-info \
  --tokenizer /path/to/tokenizer.json
```

Tokenize text:

```bash
./build-cuda/bin/ghassan-ai tokenize \
  --model /path/to/model.gguf \
  --text "Hello from Ghassan AI"
```

## Evaluation and diagnostics

### Device report

```bash
./build-cuda/bin/ghassan-ai devices
```

### Generation benchmark

```bash
./build-cuda/bin/ghassan-ai bench \
  --model /path/to/model.gguf \
  --device cuda \
  --tokens 64
```

The benchmark reports prompt prefill and token decoding measurements for the machine on which it is run. Results depend on hardware, driver/toolkit, model, quantization, context, and runtime settings.

### Perplexity

```bash
./build-cuda/bin/ghassan-ai perplexity \
  --model /path/to/model.gguf \
  --text "A held-out evaluation passage."
```

For shard-based evaluation, use the `--shards` and `--prefix` options shown by `ghassan-ai --help`.

## Important implementation notes

- The engine has CPU and optional CUDA execution paths; CUDA is not required for a CPU build.
- The CUDA backend is NVIDIA-specific. Other device labels do not imply native Metal or Vulkan kernels.
- Model loading is constrained by the architecture mappings and tensor layouts implemented in the repository.
- GGUF support is not equivalent to support for every GGML quantization type. The implemented Q4_0, Q4_1, and Q8_0 paths should be distinguished from format identifiers that the reader may recognize.
- The CMake project searches for `tests/test_*.cpp`. The repository currently contains no `tests/` source directory, so enabling `GAI_BUILD_TESTS` does not create a test suite by itself.
- No benchmark table or hardware-fit guarantee is published here. Run the diagnostics on your own model and target hardware before planning a training or deployment workload.

## Troubleshooting

**CUDA is not detected**

- Confirm that the NVIDIA driver and CUDA Toolkit are installed and visible to CMake.
- Configure with `-DGAI_ENABLE_CUDA=ON`.
- If you only need CPU execution, configure with `-DGAI_ENABLE_CUDA=OFF`.

**The model does not fit in GPU memory**

- Try a smaller `--ctx` value for inference.
- Check the device report with `ghassan-ai devices`.
- Consider a supported quantized GGUF file.
- For training, use `gai_train --dry-run` and the available memory-budget options before launching a run.

**Tokenizer or vocabulary mismatch**

- Verify that the tokenizer belongs to the checkpoint.
- Check the tokenizer vocabulary size against the model configuration.
- Use the matching chat template and special-token metadata.

**A training configuration is rejected**

- Check that exactly one of `training.max_steps` and `training.epochs` is set.
- Check the optimizer and scheduler names.
- Use `--strict-config` to make unknown or unused configuration keys fail explicitly.

## Contributing

Contributions are welcome. For substantial changes:

1. Open an issue describing the problem or proposed improvement.
2. Keep changes focused and consistent with the existing C++20/CUDA architecture.
3. Include reproducible steps and relevant hardware/toolchain details for runtime or CUDA issues.
4. Add tests when introducing or changing behavior, especially for tensor shapes, numerical correctness, model-format compatibility, and checkpoint handling.

## License

This project is distributed under the [MIT License](LICENSE).

---

<div align="center">

**Ghassan AI**  
*Native code. Model engineering. Continuous improvement.*

</div>
