# Ghassan AI

A from-scratch LLM engine in C++20 / CUDA: the **body** that serves a model.
The **mind** (the weights) comes from a HuggingFace checkpoint, and the
training is done in Python with safetensors.

```
core/        tensors · dtypes · device (CPU/CUDA) · ops · mmap · RNG · unicode · JSON
model/       Llama-like dense + mixture-of-experts, GQA, RoPE (NTK/YaRN), SWA
tokenizer/   byte-level BPE (GPT-2 / Qwen / Llama-3) · chat styles · .gtok
format/      GGUF codec (native + llama.cpp compatible) · legacy .gai
inference/   generator · paged/ring KV cache · sampler · chat session
quantization/ Q4_0 · Q8_0 · Q4_1
evaluation/  benchmark suites · perplexity
vision/      ViT tower + projector (the multimodal half)
server/      a minimal HTTP/1.1 server (OpenAI compatible API)
cuda/        attention · MoE · quantization kernels (sm_75 T4 and up)
tools/       ghassan-ai · gai_train · train_tokenizer · converters
tools/py/    the Python side: dataset, GGUF <-> safetensors, QLoRA
```

## Build

**MSVC (Visual Studio 17):**

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

**MinGW / MSYS2 (GCC, OpenMP):**

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=OFF
cmake --build build --parallel
```

`GAI_ENABLE_CUDA=ON` (the default) picks up the CUDA toolkit when it is
there and targets `sm_75` (T4), `sm_80` and `sm_90`.

The unit tests:

```bash
cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

## The engine serves HuggingFace models

The engine reads GGUF. `tools/convert_safetensors_to_gguf.py` bridges a
HuggingFace checkpoint to that format:

```bash
pip install -r requirements.txt
python tools/convert_safetensors_to_gguf.py \
    --model-dir /path/to/hf-checkpoint \
    --out artifacts/my-model.gguf
```

The converter handles dense and mixture-of-experts checkpoints (Mixtral and
Qwen3 layouts), writes the byte-level alphabet, the control token ids and
the default system message, and validates the result.

```bash
./build/bin/ghassan-ai chat --model artifacts/my-model.gguf
./build/bin/ghassan-ai generate --model artifacts/my-model.gguf \
    --prompt "What is the capital of France?" --dump-prompt
./build/bin/ghassan-ai info --model artifacts/my-model.gguf
```

### Tokenizer compatibility

The engine implements the byte-level BPE used by GPT-2, Qwen and Llama-3:
the GPT-2 byte alphabet, the reference pre-tokenizer regex and the
byte-level decoder. The chat format is detected from the vocabulary
(native Gai, ChatML, Llama-3, Mistral).

**The engine and the reference HuggingFace tokenizer must agree exactly**,
or training and serving drift apart. That contract is enforced by a
conformance harness:

```bash
python tools/py/conformance.py --tokenizer-dir safetensors \
    --out tests/fixtures/tokenizer_conformance.json

python tools/py/conformance.py --tokenizer-dir safetensors \
    --out tests/fixtures/tokenizer_conformance.json --check \
    --binary build/bin/ghassan-ai.exe \
    --model artifacts/Ghassan-v1-pro-360m-q4_0.gguf
```

The unit tests (`tests/test_tokenizer_v2.cpp`) cover the tokenizer pieces
in isolation, and `tests/test_vision.cpp` covers the vision tower and the
image reader.

## Vision

`vision/` is a ViT encoder (patch embedding, position embedding, transformer
blocks, projector) whose weights travel in the same GGUF as the language
model. The image features are written over the image placeholder tokens of
the prompt, so the language model sees the picture as embeddings.

The engine has no image decoder: the Python side converts PNG/JPEG to a
binary PPM, and the engine reads that.

```bash
python - <<'PY'
from PIL import Image
Image.open("photo.jpg").resize((336, 336)).save("photo.ppm")
PY
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
`temperature`, `top_p`, `stream`, `model`). A request may also carry an
image (`image_path` or `image_ppm_base64`).

## Training (Python)

The training path is Python and it works on **safetensors**: the engine
serves GGUF, and `tools/py/gguf_to_safetensors.py` is the bridge back.

| Tool | Purpose |
|---|---|
| `tools/py/prepare_dataset.py` | parquet / json / jsonl / txt -> tokenized shards + `train.jsonl` |
| `tools/py/gguf_to_safetensors.py` | GGUF -> safetensors (so a served model can be fine-tuned) |
| `tools/py/qlora_finetune.py` | QLoRA / LoRA fine-tune, merge, export back to GGUF |
| `tools/py/lake_to_corpus.py` | parquet lake -> a text corpus (for the BPE tokenizer) |
| `tools/py/conformance.py` | the HuggingFace tokenizer -> engine ground truth |
| `tools/py/download_model.py` | download a model from the hub |
| `tools/py/make_test_model.py` | a tiny checkpoint for end-to-end tests |
| `tools/convert_safetensors_to_gguf.py` | safetensors -> GGUF |

```bash
# 1. a trainable checkpoint
python tools/py/gguf_to_safetensors.py --gguf artifacts/my-model.gguf \
    --out models/my-model-hf

# 2. the data
python tools/py/prepare_dataset.py --tokenizer models/my-model-hf \
    --lake /kaggle/input/english_parquet --out artifacts/shards_en

# 3. the fine-tune (one 16 GB T4)
python tools/py/qlora_finetune.py --model models/my-model-hf \
    --data artifacts/shards_en/train.jsonl --out runs/exp1 \
    --qlora --merge --export-gguf artifacts/my-model-finetuned.gguf

# 4. back to the engine
./build/bin/ghassan-ai serve --model artifacts/my-model-finetuned.gguf
```

`configs/kaggle/setup.sh` builds the engine and trains the 32k BPE
tokenizer on a Kaggle GPU; `configs/kaggle/build_english_data.sh` turns the
English parquet lake into the training shards.

The C++ trainer in `training/` is **frozen**: it compiles and it is the
native trainer of the project, but it is not the production training path
and it ships no recipes.

## Known limitations

* The engine encodes against **byte-level BPE** vocabularies. SentencePiece /
  unigram vocabularies (Llama-2 style) are rejected with a clear message.
* Chat templates are reproduced by the style registry; the full Jinja
  template is stored in the GGUF but not interpreted.
* A few byte-level entries can be missing from a pruned vocabulary (rare
  control characters and some Latin-1 letters); they map to the unknown token.
* The CUDA MoE kernels use `atomicAdd` for the expert scatter-add, so MoE
  inference is not bit-reproducible across runs.
* The HTTP server is deliberately small: no keep-alive, no TLS, one request
  at a time. Put a reverse proxy in front of it for anything real.

## License

MIT. See [LICENSE](LICENSE).
