#!/usr/bin/env bash
# Ghassan v1 Flash — the whole Granite-4.0-H-Tiny flow in ONE notebook cell.
#
#     !bash "kaggle configue/kaggle_granite_run.sh"
#
# It builds with CUDA, checks the official config, loads the real weights,
# exports the deliverable GGUF, verifies it (metadata, brand, parity with
# the safetensors logits) and generates from it. Every step prints a verdict,
# so a failure says WHERE it came from.
#
# Env:
#   MODEL      the HuggingFace checkpoint directory (default: the dataset,
#              or /kaggle/working/granite-4.0-h-tiny)
#   OUT        the GGUF to write (default: /kaggle/working/ghassan-v1-flash.gguf)
#   PROFILE    the export profile (default: q4_0, the mixed-precision one)
#   SKIP_EXPORT=1  only run the checks
#   CTX        the context length (default 2048)
#   MAXTOK     the tokens to generate (default 48)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
    for cand in /kaggle/working/granite-4.0-h-tiny \
                /kaggle/input/granite-4.0-h-tiny/granite-4.0-h-tiny \
                /kaggle/input/granite-4.0-h-tiny; do
        if [ -f "$cand/config.json" ]; then MODEL="$cand"; break; fi
    done
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "FATAL: the Granite checkpoint was not found."
    echo "       Set MODEL=/path/to/granite-4.0-h-tiny and run this cell again."
    exit 1
fi

OUT="${OUT:-/kaggle/working/ghassan-v1-flash.gguf}"
PROFILE="${PROFILE:-q4_1}"
CTX="${CTX:-2048}"
MAXTOK="${MAXTOK:-48}"
PROMPT="Please list one IBM Research laboratory located in the United States."

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m  PASS  %s\033[0m\n' "$*"; }
bad() { printf '\033[1;31m  FAIL  %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. the build
say "1/8  the CUDA build"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON -DGAI_ENABLE_OPENMP=ON > /dev/null
cmake --build build -j 2 2>&1 | tail -5
[ -x build/bin/ghassan-ai ] || { bad "the build did not produce ghassan-ai"; exit 1; }
ok "the build (48 targets, CUDA ON)"

# ---------------------------------------------------------------- 2. the devices
say "2/8  the devices"
./build/bin/ghassan-ai devices

# ---------------------------------------------------------------- 3. the config
say "3/8  the official Granite config (the one in the repository)"
./build/bin/ghassan-ai info --model "granite-documentation/granit filles confiques" --config-only

# ---------------------------------------------------------------- 4. the weights
say "4/8  the real weights"
./build/bin/ghassan-ai info --model "$MODEL"

# ---------------------------------------------------------------- 5. the logits
say "5/8  the logits of the safetensors model (the reference)"
# The same ids are used for the GGUF later, so the two runs are comparable.
IDS=$(./build/bin/ghassan-ai tokenize --model "$MODEL" --text "$PROMPT" \
      | sed -n 's/.*ids: *//p')
echo "the ids: $IDS"
./build/bin/ghassan-ai logits --model "$MODEL" --device cpu --ids "$IDS" \
    --out /tmp/logits_hf.f32

# ---------------------------------------------------------------- 6. the export
if [ "${SKIP_EXPORT:-0}" = "1" ]; then
    say "6/8  the export was skipped (SKIP_EXPORT=1)"
    exit 0
fi
say "6/8  the export (the deliverable: $OUT)"
# The old file must go: the writer needs room for the new one next to it.
rm -f "$OUT" "$OUT.tmp"
./build/bin/ghassan-ai quantize --model "$MODEL" --device cpu \
    --out "$OUT" --profile "$PROFILE" --model-name "Ghassan v1 Flash" --compat native
ls -la "$OUT"
df -h /kaggle/working | tail -1

# ---------------------------------------------------------------- 7. the checks
say "7/8  the GGUF checks (the identity, the brand, the parity)"
python scripts/gguf_xray.py "$OUT" | tail -20
python scripts/gguf_xray.py "$OUT" --quiet > /dev/null \
    && ok "the artifact is a clean Ghassan v1 Flash (no upstream brand)" \
    || bad "the artifact carries the upstream brand - see the report above"

./build/bin/ghassan-ai logits --model "$OUT" --device cpu --ids "$IDS" \
    --out /tmp/logits_gguf.f32
python scripts/gguf_parity.py /tmp/logits_hf.f32 /tmp/logits_gguf.f32

# ---------------------------------------------------------------- 8. the model
say "8/8  the generation from the GGUF (the proof)"
./build/bin/ghassan-ai generate --model "$OUT" --device cpu --ctx "$CTX" \
    --max-tokens "$MAXTOK" --no-stream --raw --greedy --repeat 1.0 --print-ids \
    --prompt "$PROMPT"

say "done"
echo "the deliverable: $OUT"
