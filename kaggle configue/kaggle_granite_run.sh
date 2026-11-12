#!/usr/bin/env bash
# Ghassan v1 Flash — the whole Granite-4.0-H-Tiny flow on Kaggle (2x T4).
#
# Run it from a Kaggle notebook cell:
#
#     !bash "kaggle configue/kaggle_granite_run.sh"
#
# It builds with CUDA, parses the official config, runs the real weights on
# the CPU, exports the GGUF and reloads it.
#
# NOTE on the GPU: the engine keeps the weights as fp32 masters (25.85 GiB for
# Granite-4.0-H-Tiny), so the model does NOT fit in a 16 GB T4. The engine says
# so up front instead of dying inside cudaMalloc. The inference below runs on
# the CPU; the GPU path needs the fp16 weight storage mode (not implemented
# yet).
#
# Env:
#   MODEL          the HuggingFace checkpoint directory (default: the dataset)
#   OUT            the GGUF to write (default: /kaggle/working/ghassan-v1-flash.gguf)
#   PROFILE        q8_k (default) | q4_0 | fp16
#   SKIP_EXPORT    1 = do not export the GGUF
#   CTX            the context length for the generations (default 2048)

set -euo pipefail

MODEL="${MODEL:-/kaggle/input/granite-4.0-h-tiny/granite-4.0-h-tiny}"
OUT="${OUT:-/kaggle/working/ghassan-v1-flash.gguf}"
PROFILE="${PROFILE:-q8_k}"
CTX="${CTX:-2048}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

say "1/7  the CUDA build"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON
cmake --build build -j 2

say "2/7  the devices"
./build/bin/ghassan-ai devices

say "3/7  the official Granite config (the one in the repository)"
./build/bin/ghassan-ai info --model "granite-documentation/granit filles confiques" --config-only

say "4/7  the real weights"
./build/bin/ghassan-ai info --model "$MODEL"

say "5/7  the logits of the real model (CPU: the fp32 masters do not fit a T4)"
./build/bin/ghassan-ai logits --model "$MODEL" --device cpu \
    --prompt "Please list one IBM Research laboratory located in the United States."

if [ "${SKIP_EXPORT:-0}" = "1" ]; then
    say "6/7  the export was skipped (SKIP_EXPORT=1)"
    exit 0
fi

say "6/7  the export (the deliverable)"
mkdir -p "$(dirname "$OUT")"
./build/bin/ghassan-ai quantize --model "$MODEL" --device cpu \
    --out "$OUT" --profile "$PROFILE" --model-name "Ghassan v1 Flash"

say "7/7  the GGUF"
ls -la "$OUT"
./build/bin/ghassan-ai info --model "$OUT"

say "a generation from the GGUF (CPU)"
./build/bin/ghassan-ai generate --model "$OUT" --device cpu --ctx "$CTX" \
    --prompt "Please list one IBM Research laboratory located in the United States." \
    --max-tokens 32 --no-stream --raw

say "done"
