# =============================================================================
# CELL 3 - the reference checkpoint + the dataset
# =============================================================================
# Two things land in /kaggle/working/hf and /kaggle/working/artifacts/shards:
#
#   hf/       the reference checkpoint (the config.json + the tokenizer).
#             Only the CONFIG and the TOKENIZER are needed to train the same
#             architecture from scratch; the weights are downloaded only when
#             a LoRA / SFT run needs them.
#   shards/   the training shards (.gbin), built by the ENGINE's tokenizer.
#
# The tokenizer is the reference one: the vocabulary, the merges and the
# digit grouping all come from it.

import os, sys, subprocess, pathlib, shutil, json, time

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
HF   = WORK / "hf"
DATA = WORK / "artifacts" / "shards"

# The reference checkpoint. The architecture of Ghassan v1 Flash is the
# Granite-4.0-H one: the Mamba-2 hybrid, the MoE, the multipliers.
HF_REPO = os.environ.get("HF_REPO", "ibm-granite/granite-4.0-h-tiny")
HF_REV  = os.environ.get("HF_REV", "main")
# The dataset: a Kaggle dataset with the English parquet files, or a local
# folder. The script only converts the format.
DATA_INPUT = os.environ.get("DATA_INPUT", "")
DATA_MAX_TOKENS = int(os.environ.get("DATA_MAX_TOKENS", "0")) or 0

subprocess.run([sys.executable, "-m", "pip", "-q", "-U", "huggingface_hub"], check=False)
from huggingface_hub import snapshot_download
HF.mkdir(parents=True, exist_ok=True)
print("downloading the reference:", HF_REPO)
t0 = time.time()
# The allow_patterns keeps the download small: the config and the tokenizer
# are what a from-scratch run needs. The weights are a separate step (they
# are tens of GB).
snapshot_download(repo_id=HF_REPO, revision=HF_REV, local_dir=str(HF),
                  allow_patterns=["config.json", "tokenizer*", "generation_config.json",
                                  "special_tokens_map.json", "chat_template*"])
print(f"  done in {time.time()-t0:.0f}s")
for p in sorted(HF.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(HF)}  ({p.stat().st_size/1024:.0f} KB)")

# ---- the vocabulary -------------------------------------------------------
tok_json = HF / "tokenizer.json"
if not tok_json.exists():
    raise SystemExit("the reference has no tokenizer.json: check HF_REPO")

# ---- the dataset ----------------------------------------------------------
if DATA_INPUT:
    print("\npreparing the dataset from", DATA_INPUT)
    r = subprocess.run(
        [sys.executable, str(SRC / "tools" / "py" / "prepare_dataset.py"),
         "--input", DATA_INPUT,
         "--output", str(DATA),
         "--mode", "pretrain",
         "--val-fraction", "0.005"]
        + (["--max-tokens", str(DATA_MAX_TOKENS)] if DATA_MAX_TOKENS else []),
        capture_output=True, text=True)
    print(r.stdout[-4000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit("the dataset preparation failed")
else:
    print("\nDATA_INPUT is empty -> no dataset prepared.")
    print("Point it at a Kaggle dataset (add it on the right side of the")
    print("notebook) or at a local folder of .parquet files.")

# ---- the shards (the ENGINE tokenizes) -------------------------------------
# The shard builder needs a recipe (it checks the vocabulary against the
# model's embedding: the Granite tokenizer has 100352 tokens, so the recipe
# must have a vocab_size >= that).
SHARD_CONFIG = os.environ.get("SHARD_CONFIG",
                              str(SRC / "configs" / "ghassan_v1_flash_smoke.yaml"))
if DATA.exists() and any(DATA.glob("*.txt")):
    print("\nbuilding the shards with the engine's tokenizer ...")
    r = subprocess.run(
        [str(WORK / "build" / "bin" / "gai_train"),
         "--config", SHARD_CONFIG,
         "--build-shards", str(DATA),
         "--tokenizer", str(tok_json),
         "--shard-prefix", "train"],
        capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    if r.returncode != 0:
        raise SystemExit("the shard build failed")
    # The validation shards (the same recipe, the val_ prefix).
    r = subprocess.run(
        [str(WORK / "build" / "bin" / "gai_train"),
         "--config", SHARD_CONFIG,
         "--build-shards", str(DATA),
         "--tokenizer", str(tok_json),
         "--shard-prefix", "val"],
        capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-2000:])

print("\nthe shards:")
for p in sorted(DATA.glob("*.gbin")):
    print(f"  {p.name}  ({p.stat().st_size/1024/1024:.1f} MB)")
