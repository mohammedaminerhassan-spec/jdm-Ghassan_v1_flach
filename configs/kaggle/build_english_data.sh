#!/usr/bin/env bash
# Build the training shards from the English lake.
#
# The shards are the Gai format (train_<domain>_NNNN.gbin) that the C++
# trainer reads, plus a tokenized train.jsonl that the Python trainer
# (tools/py/qlora_finetune.py) reads. The domain names are the ones the
# configs declare in data.mix: english_chat, english_instruction, ...
#
# Usage:
#   bash configs/kaggle/build_english_data.sh
#   EN_PARQUET_DIR=/kaggle/input/english_parquet \
#     TOK_PATH=artifacts/tokenizer/english32k.gtok \
#     bash configs/kaggle/build_english_data.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

LAKE_DIR="${EN_PARQUET_DIR:-}"
TOK_PATH="${TOK_PATH:-${REPO_DIR}/artifacts/tokenizer/english32k.gtok}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/artifacts/shards_en}"
VAL_RATIO="${VAL_RATIO:-0.005}"
SHARD_TOKENS="${SHARD_TOKENS:-50000000}"
PYTHON="${PYTHON:-python3}"

if [[ -z "${LAKE_DIR}" ]]; then
    # The usual Kaggle layout: the dataset is attached under /kaggle/input.
    for cand in /kaggle/input/english_parquet /kaggle/input/*parquet* ; do
        if [[ -d "${cand}" ]]; then LAKE_DIR="${cand}"; break; fi
    done
fi

echo "============================================================"
echo "  Ghassan AI - English shard builder"
echo "============================================================"

if [[ -z "${LAKE_DIR}" || ! -d "${LAKE_DIR}" ]]; then
    echo "[ERROR] no English lake found."
    echo "[ERROR] Attach the english_parquet dataset (english_chat_part*.parquet"
    echo "[ERROR] + english_instruction_part*.parquet) or set EN_PARQUET_DIR."
    exit 1
fi

if [[ ! -f "${TOK_PATH}" ]]; then
    echo "[ERROR] no tokenizer at ${TOK_PATH}"
    echo "[ERROR] Run configs/kaggle/setup.sh first (it trains the 32k BPE)."
    exit 1
fi

echo "[lake]      ${LAKE_DIR}"
echo "[tokenizer] ${TOK_PATH}"
echo "[out]       ${OUT_DIR}"
echo ""

mkdir -p "${OUT_DIR}"

# The shard builder needs the HuggingFace form of the tokenizer: the C++
# .gtok file is the engine's own format. A HuggingFace tokenizer directory
# is used directly when one is available; otherwise the .gtok vocabulary is
# exported to a temporary HuggingFace directory first.
HF_TOK="${HF_TOKENIZER_DIR:-}"
if [[ -z "${HF_TOK}" ]]; then
    for cand in /kaggle/input/tokenizer /kaggle/input/*tokenizer* ; do
        if [[ -d "${cand}" && -f "${cand}/tokenizer.json" ]]; then HF_TOK="${cand}"; break; fi
    done
fi

if [[ -z "${HF_TOK}" ]]; then
    echo "[ERROR] no HuggingFace tokenizer directory found."
    echo "[ERROR] Attach a tokenizer dataset (tokenizer.json) or set HF_TOKENIZER_DIR."
    exit 1
fi
echo "[hf tok]    ${HF_TOK}"
echo ""

"${PYTHON}" tools/py/prepare_dataset.py \
    --lake "${LAKE_DIR}" \
    --tokenizer "${HF_TOK}" \
    --out "${OUT_DIR}" \
    --val-ratio "${VAL_RATIO}" \
    --shard-tokens "${SHARD_TOKENS}"

echo ""
echo "=== shards ==="
ls -la "${OUT_DIR}" | head -30
TRAIN_N=$(find "${OUT_DIR}" -name 'train_*.gbin' | wc -l)
VAL_N=$(find "${OUT_DIR}" -name 'val_*.gbin' | wc -l)
echo "  train shards: ${TRAIN_N} | val shards: ${VAL_N}"
if [[ "${TRAIN_N}" -eq 0 ]]; then
    echo "[ERROR] no train shards were produced."
    exit 1
fi

# The trainer addresses the shards by domain; print the names it will see.
echo ""
echo "=== domains (the trainer needs data.mix to match these) ==="
find "${OUT_DIR}" -name 'train_*.gbin' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^train_(.*)_[0-9]+\.gbin$/\1/' | sort -u \
    || find "${OUT_DIR}" -name 'train_*.gbin' -exec basename {} \; \
    | sed -E 's/^train_(.*)_[0-9]+\.gbin$/\1/' | sort -u

echo ""
echo "Done. Next: bash configs/kaggle/train_t4_fast.sh"
