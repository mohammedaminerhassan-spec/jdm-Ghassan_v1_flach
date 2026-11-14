# =============================================================================
# CELL 0 - the Granite weights (the safetensors, 13.9 GB, 3 shards)
# =============================================================================
# This cell puts the reference checkpoint on the disk of the notebook:
#
#   /kaggle/working/granite/
#       model-00001-of-00003.safetensors     <- the 3 shards (13.9 GB)
#       model-00002-of-00003.safetensors
#       model-00003-of-00003.safetensors
#       model.safetensors.index.json          <- the map: the tensor -> the shard
#       config.json                           <- the architecture
#       tokenizer.json ...                    <- the vocabulary
#
# After this cell:
#   * the engine can LOAD the model (the T4 GPU does the forward),
#   * the GGUF export reads it (the deliverable: "Ghassan v1 Flash"),
#   * the safetensors -> GGUF path is tested end to end.
#
# It takes ~10 minutes (13.9 GB over Kaggle's network).
#
# NOTE: /kaggle/working is NOT persistent: the notebook has to download them
# again next time. To avoid that, upload this folder as a Kaggle dataset
# (Add Data -> New Dataset -> the folder) and point MODEL_DIR at it.

import os, sys, subprocess, pathlib, shutil, time

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
BUILD = WORK / "build"
GRANITE = pathlib.Path(os.environ.get("MODEL_DIR", str(WORK / "granite")))

HF_REPO = os.environ.get("HF_REPO", "ibm-granite/granite-4.0-h-tiny")
# The 3 shards + the index + the config + the tokenizer. Only the shards are
# big; the rest is a few MB.
ALLOW = ["*.safetensors", "*.json", "tokenizer*", "*.jinja", "merges.txt", "vocab.json"]

print("=" * 72)
print("the Granite weights:", GRANITE)
print("=" * 72)

subprocess.run([sys.executable, "-m", "pip", "-q", "-U", "huggingface_hub"], check=False)
from huggingface_hub import snapshot_download

GRANITE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
# The download goes to the HuggingFace cache first and is then copied (or
# hard-linked) here. The resume is automatic: an interrupted download
# continues where it stopped.
path = snapshot_download(repo_id=HF_REPO,
                        allow_patterns=ALLOW,
                        local_dir=str(GRANITE),
                        max_workers=8)
print(f"\ndownloaded in {time.time()-t0:.0f}s -> {path}")

print("\nthe files:")
total = 0
for p in sorted(GRANITE.rglob("*")):
    if p.is_file() and ".cache" not in p.parts:
        sz = p.stat().st_size
        total += sz
        print(f"  {p.relative_to(GRANITE)}  ({sz/1024/1024:.1f} MB)")
print(f"  total: {total/1024/1024/1024:.2f} GB")

# ---- the check: the engine must load it -------------------------------------
if not (BUILD / "bin" / "ghassan-ai").exists():
    raise SystemExit("the engine is not built: run cell 2 first")

print("\n" + "=" * 72)
print("the engine loads the checkpoint ...")
print("=" * 72)
r = subprocess.run([str(BUILD / "bin" / "ghassan-ai"), "info", "--model", str(GRANITE)],
                   capture_output=True, text=True)
print(r.stdout[-6000:])
print(r.stderr[-3000:])
if r.returncode != 0:
    raise SystemExit("the engine could not load the Granite weights")

print("\nready. The next cells use:", GRANITE)
