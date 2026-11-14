#!/usr/bin/env bash
# Compiles every CUDA source WITHOUT a GPU.
#
# Why: the .cu files are the part of the engine that no CPU test can reach.
# A typo in one of them (an undefined identifier, a wrong type) only shows
# up when nvcc runs - which, until now, meant "the Kaggle build fails and
# the run is over". This script is the same check, without the GPU and
# without the 20-minute full build.
#
#   bash scripts/check_cuda_compile.sh
#
# It needs CUDA >= 12.5 (the grouped GEMM API: cublasGemmGroupedBatchedEx).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" > /dev/null 2>&1; then
    for c in /usr/local/cuda/bin/nvcc /usr/lib/nvidia-cuda-toolkit/bin/nvcc; do
        [ -x "$c" ] && NVCC="$c" && break
    done
fi
if ! command -v "$NVCC" > /dev/null 2>&1 && [ ! -x "$NVCC" ]; then
    echo "nvcc not found: set NVCC=/path/to/nvcc" >&2
    exit 2
fi

ARCH="${ARCH:-75}"
STD="${STD:-c++17}"
CCBIN="${CCBIN:-}"
EXTRA=""
[ -n "$CCBIN" ] && EXTRA="-ccbin $CCBIN"

echo "nvcc  : $("$NVCC" --version | tail -2 | head -1)"
echo "arch  : sm_$ARCH   std: $STD"

fail=0
for f in cuda/*.cu; do
    printf '%-24s ' "$f"
    out=$("$NVCC" -std="$STD" -O1 -arch="sm_$ARCH" -DGAI_ENABLE_CUDA=1 $EXTRA \
              -Xcompiler -fopenmp -I"$ROOT" -c "$f" -o /dev/null 2>&1)
    rc=$?
    if [ "$rc" = "0" ]; then
        echo "OK"
    else
        echo "FAILED"
        echo "$out" | head -30
        fail=1
    fi
done

if [ "$fail" = "0" ]; then
    echo "every CUDA source compiles."
else
    echo "the CUDA build is broken (fix it before running on Kaggle)." >&2
fi
exit $fail
