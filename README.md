# Ghassan AI

A from-scratch LLM engine in C++20 / CUDA: the **body** that serves a model.
The **mind** (the weights) can come from a HuggingFace checkpoint.

```
core/        tensors · dtypes · device (CPU/CUDA) · ops · mmap · RNG · unicode
model/       Llama-like dense + mixture-of-experts, GQA, RoPE (NTK/YaRN), SWA
tokenizer/   normalizer (Arabic/Darija aware) · pre-tokenizers · BPE · .gtok
format/      GGUF codec (native + llama.cpp compatible) · legacy .gai
training/    trainer · AdamW/Lion/Muon · dataloader · checkpoints · NCCL  (frozen, see below)
inference/   generator · paged/ring KV cache · sampler · chat session
quantization/ Q4_0 · Q8_0 · Q4_1
evaluation/  benchmark suites · perplexity
cuda/        attention · MoE · quantization kernels (sm_75 T4 and up)
tools/       ghassan-ai · gai_train · train_tokenizer · converters
```

## Build

**MSVC (Visual Studio 17):**

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

**MinGW / MSYS2 (GCC, OpenMP, AVX2/FMA enabled):**

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=OFF
cmake --build build
```

`GAI_ENABLE_CUDA=ON` (default) picks up the CUDA toolkit when present and
targets `sm_75` (T4), `sm_80` and `sm_90`.

## The engine serves HuggingFace models

The engine reads GGUF. `tools/convert_safetensors_to_gguf.py` bridges a
HuggingFace checkpoint to that format:

```bash
pip install gguf numpy
python tools/convert_safetensors_to_gguf.py \
    --model-dir /path/to/hf-checkpoint \
    --out artifacts/my-model.gguf
```

The converter handles dense and mixture-of-experts checkpoints (Mixtral and
Qwen3 layouts), writes the byte-level alphabet, the control token ids and the
default system message, and validates the result.

```bash
./build/bin/ghassan-ai chat --model artifacts/my-model.gguf
./build/bin/ghassan-ai generate --model artifacts/my-model.gguf \
    --prompt "What is the capital of France?" --dump-prompt
```

### Tokenizer compatibility

The engine implements the byte-level BPE used by GPT-2, Qwen and Llama-3:
the GPT-2 byte alphabet, the reference pre-tokenizer regex and the
byte-level decoder. The chat format is detected from the vocabulary
(native Gai, ChatML, Llama-3, Mistral).

**The engine and the reference HuggingFace tokenizer must agree exactly**, or
training and serving drift apart. That contract is enforced by a conformance
harness:

```bash
python tools/py/conformance.py --tokenizer-dir safetensors \
    --out tests/fixtures/tokenizer_conformance.json

python tools/py/conformance.py --tokenizer-dir safetensors \
    --out tests/fixtures/tokenizer_conformance.json --check \
    --binary build/bin/ghassan-ai.exe \
    --model artifacts/Ghassan-v1-pro-360m-q4_0.gguf
```

The unit tests cover the tokenizer pieces in isolation:

```bash
cmake --build build && ctest --test-dir build --output-on-failure
```

## Serving

`ghassan-ai serve` exposes an OpenAI compatible HTTP endpoint. The model is
loaded once and stays resident, so a request never pays the load cost.

```bash
./build/bin/ghassan-ai serve --model artifacts/my-model.gguf --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ghassan","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat completion, streaming (SSE) or not |
| `POST /v1/completions` | same, raw completion |
| `GET /v1/models` | the served model |
| `GET /health` | liveness, chat style, request counter |

The request body follows the OpenAI schema (`messages`, `max_tokens`,
`temperature`, `top_p`, `stream`, `model`).

## Python side

| Tool | Purpose |
|---|---|
| `tools/py/conformance.py` | HuggingFace tokenizer -> engine ground truth |
| `tools/py/make_test_model.py` | tiny HF-style checkpoint for end-to-end tests |
| `tools/convert_safetensors_to_gguf.py` | safetensors -> GGUF |

Training itself runs in Python/PyTorch (accelerate, torchtitan, TRL). The
C++ trainer is kept for research but is **frozen**: it is not the production
training path.

## Known limitations

* The engine encodes against **byte-level BPE** vocabularies. SentencePiece /
  unigram vocabularies (Llama-2 style) are rejected with a clear message.
* Chat templates are reproduced by the style registry; the full Jinja
  template is stored in the GGUF but not interpreted.
* A few byte-level entries can be missing from a pruned vocabulary (rare
  control characters and some Latin-1 letters); they map to the unknown token.
* The CUDA MoE kernels use `atomicAdd` for the expert scatter-add, so MoE
  inference is not bit-reproducible across runs.
