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


def find_installed():
    """The weights are already on the disk: a Kaggle DATASET (read-only, it
    does not use the 20 GB of /kaggle/working) or a previous download."""
    for root in [WORK / "granite"]:
        if (root / "config.json").exists():
            return root
    for p in pathlib.Path("/kaggle/input").rglob("config.json"):
        d = p.parent
        if list(d.glob("*.safetensors")):
            return d
    return None


installed = find_installed()
if installed:
    print("the weights are already here:", installed)
    try:
        st = os.statvfs(str(WORK))
        free = st.f_bavail * st.f_frsize / (1024 ** 3)
        print(f"the free space on /kaggle/working: {free:.1f} GB")
    except Exception:
        pass
    if str(installed).startswith("/kaggle/input"):
        print("(a dataset: it is read-only and it does NOT use the 20 GB of\n"
              " /kaggle/working - so the GGUF export has the room it needs)")
    GRANITE = installed
else:
    print("=" * 72)
    print("the Granite weights (into the HuggingFace cache, NOT /kaggle/working:")
    print("the working directory has 20 GB and the GGUF export needs 13 of them)")
    print("=" * 72)

    subprocess.run([sys.executable, "-m", "pip", "-q", "-U", "huggingface_hub"], check=False)
    from huggingface_hub import snapshot_download

    t0 = time.time()
    # No local_dir: the files land in the HuggingFace cache (outside
    # /kaggle/working, so the 20 GB there stay free for the GGUF export).
    # The resume is automatic: an interrupted download continues where it
    # stopped.
    path = snapshot_download(repo_id=HF_REPO,
                            allow_patterns=ALLOW,
                            max_workers=8)
    GRANITE = pathlib.Path(path)
    print(f"\ndownloaded in {time.time()-t0:.0f}s -> {GRANITE}")

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
