#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Kaggle GPU notebook build + validation for the Ghassan engine
# (Granite-4.0-H-based "Ghassan v1 Flash").
#
# In a Kaggle notebook with GPU = T4 x2:
#   !bash /kaggle/working/<repo-copy>/kaggle/build_kaggle.sh /kaggle/input/<granite-dataset>
#
# What it does:
#   1. clones the repo and builds the engine (sm_75 first = the T4; PTX fallback)
#   2. builds + runs the test suite (CPU-side regression tests)
#   3. checks that the engine loads the REAL granite-4.0-h-tiny checkpoint
#
# What it does NOT do (honest reporting):
#   - The tests exercise the CPU reference path; they are the same math the
#     CUDA kernels implement. CUDA kernels themselves are exercised on the
#     GPU only when a test selects them (test_cuda_parity / mamba_cuda_parity).
#   - Inference latency numbers come from `ghassan-ai bench`, printed below.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_flach.git"
WORK="/kaggle/working"
SRC="${WORK}/ghassan-src"
GRANITE_DIR="${1:-/kaggle/input/granite-4.0-h-tiny}"

if [ ! -d "${GRANITE_DIR}" ]; then
    echo "!! the Granite checkpoint was not found at: ${GRANITE_DIR}"
    echo "!! pass the dataset path explicitly: bash build_kaggle.sh /kaggle/input/<your-granite-dataset-dir>"
    exit 2
fi

echo "==== 1/4 clone ${REPO_URL} ===="
rm -rf "${SRC}"
git clone --depth 1 "${REPO_URL}" "${SRC}"
cd "${SRC}"

echo "==== 2/4 build (CUDA, sm_75) ===="
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGAI_ENABLE_CUDA=ON \
    -DGAI_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"

echo "==== 3/4 tests ===="
ctest --test-dir build --output-on-failure

echo "==== 4/4 engine vs the real checkpoint ===="
"${SRC}/build/bin/ghassan-ai" info --model "${GRANITE_DIR}"
"${SRC}/build/bin/ghassan-ai" bench --model "${GRANITE_DIR}" --tokens 32

echo ""
echo "BUILD + VALIDATION DONE"
echo "Next: see kaggle/README.md for the fine-tune commands."
