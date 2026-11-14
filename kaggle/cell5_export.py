# =============================================================================
# CELL 5 - the export (the safetensors) + the upload to the Hub
# =============================================================================
# The trained model becomes a real HuggingFace repository:
#
#   <out>/model.safetensors      the weights
#   <out>/config.json            the architecture (the Granite-4.0-H geometry)
#   <out>/generation_config.json  the stop tokens
#   <out>/tokenizer.json ...      the vocabulary (copied verbatim)
#
# Then it is uploaded (a private repository by default: the model is not
# public until you say so).

import os, sys, subprocess, pathlib, shutil, time, json

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
BUILD = WORK / "build"
HF   = WORK / "hf"

CKPT = os.environ.get("CKPT", str(WORK / "artifacts" / "checkpoints" / "smoke" / "last.ckpt"))
OUT  = pathlib.Path(os.environ.get("OUT", str(WORK / "ghassan-v1-flash")))
MODEL_NAME = os.environ.get("MODEL_NAME", "Ghassan v1 Flash")
REPO_ID = os.environ.get("HF_REPO_OUT", "")     # <-- the target repository
PUBLIC = os.environ.get("HF_PUBLIC", "0") == "1"

ckpt = pathlib.Path(CKPT)
if not ckpt.exists():
    cands = sorted((WORK / "artifacts" / "checkpoints").rglob("last.ckpt"))
    if not cands:
        raise SystemExit("no checkpoint found: run the training first")
    ckpt = cands[-1]
    print("using the checkpoint:", ckpt)

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

cmd = [str(BUILD / "bin" / "ghassan-ai"), "export",
       "--checkpoint", str(ckpt),
       "--tokenizer", str(HF),          # the directory: the tokenizer files
       "--out", str(OUT),
       "--format", "safetensors",
       "--dtype", os.environ.get("DTYPE", "f16"),
       "--model-name", MODEL_NAME]
print("$", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-4000:])
if r.returncode != 0:
    raise SystemExit("the export failed")

print("\nthe exported repository:")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  ({p.stat().st_size/1024/1024:.1f} MB)")

# The check: the engine must be able to READ BACK what it wrote (the same
# code path a user hits when they clone the repository).
print("\nverifying the export (the engine reads it back) ...")
r = subprocess.run([str(BUILD / "bin" / "ghassan-ai"), "info", "--model", str(OUT)],
                   capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
if r.returncode != 0:
    raise SystemExit("the exported repository does not load back: the export is broken")

# ---- the upload ------------------------------------------------------------
if not REPO_ID:
    print("\nREPO_ID is empty -> the model stays local (in /kaggle/working).")
    print("Set it (and add the HF token to the Kaggle secrets) to upload.")
else:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=not PUBLIC, exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=REPO_ID, repo_type="model")
    print("\nuploaded to https://huggingface.co/" + REPO_ID)
