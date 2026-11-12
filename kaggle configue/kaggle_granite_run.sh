#!/usr/bin/env bash
# Ghassan v1 Flash — the whole Granite-4.0-H-Tiny flow on Kaggle (2x T4).
#
# Run it from a Kaggle notebook cell:
#
#     !bash "kaggle configue/kaggle_granite_run.sh"
#
# It builds with CUDA, checks the engine on the tiny fixture, compares the
# CPU and the GPU paths (that is where the Mamba-2 CUDA kernel is verified),
# exports the GGUF and reloads it.
#
# Env:
#   MODEL          the HuggingFace checkpoint directory (default: the dataset)
#   OUT            the GGUF to write (default: /kaggle/working/ghassan-v1-flash.gguf)
#   PROFILE        q8_k (default) | q4_0 | fp16
#   SKIP_EXPORT    1 = do not export the GGUF

set -euo pipefail

MODEL="${MODEL:-/kaggle/input/granite-4.0-h-tiny/granite-4.0-h-tiny}"
OUT="${OUT:-/kaggle/working/ghassan-v1-flash.gguf}"
PROFILE="${PROFILE:-q8_k}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

say "1/7  the CUDA build"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON
cmake --build build -j 2

say "2/7  the devices"
./build/bin/ghassan-ai devices

say "3/7  the engine on the tiny Granite fixture (CPU)"
./build/bin/ghassan-ai info --model granite_tiny
./build/bin/ghassan-ai generate --model granite_tiny --device cpu \
    --prompt "salam" --max-tokens 8 --greedy --no-stream --raw --print-ids

say "4/7  the same fixture on the GPU (the Mamba-2 CUDA kernel is checked here)"
./build/bin/ghassan-ai logits --model granite_tiny --device cpu  --out /tmp/cpu.f32
./build/bin/ghassan-ai logits --model granite_tiny --device cuda --out /tmp/gpu.f32
python - <<'PY'
import numpy as np
a = np.fromfile('/tmp/cpu.f32', dtype='<f4').astype(np.float64)
b = np.fromfile('/tmp/gpu.f32', dtype='<f4').astype(np.float64)
d = np.abs(a - b).max()
scale = max(1e-9, np.abs(a).max())
print('max |cpu - gpu| =', d, ' relative =', d / scale)
print('CPU == GPU' if d / scale < 1e-3 else 'MISMATCH: the CUDA path differs from the CPU reference')
PY

say "5/7  the real weights"
./build/bin/ghassan-ai info --model "$MODEL"

say "6/7  the logits of the real model (CPU: the fp32 masters do not fit a T4)"
./build/bin/ghassan-ai logits --model "$MODEL" --device cpu \
    --prompt "Please list one IBM Research laboratory located in the United States."

if [ "${SKIP_EXPORT:-0}" = "1" ]; then
    say "7/7  the export was skipped (SKIP_EXPORT=1)"
    exit 0
fi

say "7/7  the export (the deliverable)"
mkdir -p "$(dirname "$OUT")"
./build/bin/ghassan-ai quantize --model "$MODEL" --device cpu \
    --out "$OUT" --profile "$PROFILE" --model-name "Ghassan v1 Flash"

say "the GGUF"
ls -la "$OUT"
./build/bin/ghassan-ai info --model "$OUT"

say "a generation from the GGUF (CPU)"
./build/bin/ghassan-ai generate --model "$OUT" --device cpu --ctx 2048 \
    --prompt "Please list one IBM Research laboratory located in the United States." \
    --max-tokens 32 --no-stream --raw

say "done"
