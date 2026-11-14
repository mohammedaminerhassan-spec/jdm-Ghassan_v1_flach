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

# The Granite checkpoint: where it is (the working directory or the
# read-only /kaggle/input - the dataset does NOT use the 20 GB of
# /kaggle/working, which matters: the full-precision GGUF is 13 GB and it
# has to be written somewhere).
def find_granite():
    for root in [WORK / "granite", WORK / "hf"]:
        if (root / "config.json").exists():
            return root
    for p in pathlib.Path("/kaggle/input").rglob("config.json"):
        return p.parent
    return None

GRANITE = find_granite()
if GRANITE:
    print("the Granite checkpoint:", GRANITE)

# The tokenizer: the Granite one (the vocabulary of the model). The
# DIRECTORY is used when there is one (the export copies the tokenizer
# files: the vocab, the merges, the config...), otherwise the file.
TOKENIZER = ""
TOKENIZER_DIR = ""
for cand in ([GRANITE] if GRANITE else []) + [WORK / "hf", WORK / "granite"]:
    if cand and (cand / "tokenizer.json").exists():
        TOKENIZER = str(cand / "tokenizer.json")
        TOKENIZER_DIR = str(cand)
        break
if not TOKENIZER:
    for p in pathlib.Path("/kaggle/input").rglob("tokenizer.json"):
        TOKENIZER = str(p)
        TOKENIZER_DIR = str(p.parent)
        break
if not TOKENIZER:
    raise SystemExit("no tokenizer.json: run cell 0 or cell 3 first")
print("the tokenizer:", TOKENIZER)

ckpt = pathlib.Path(CKPT)
if not ckpt.exists():
    cands = sorted((WORK / "artifacts" / "checkpoints").rglob("last.ckpt"))
    if cands:
        ckpt = cands[-1]
        print("using the checkpoint:", ckpt)

# The disk: the full-precision GGUF is 13 GB.
try:
    st = os.statvfs(str(WORK))
    free = st.f_bavail * st.f_frsize / (1024 ** 3)
    print(f"the free space on /kaggle/working: {free:.1f} GB")
except Exception:
    free = 0.0

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

# The safetensors export: only when there is a TRAINED checkpoint (the
# training artifact). The Granite checkpoint itself is already in the
# HuggingFace format.
if ckpt.exists():
    cmd = [str(BUILD / "bin" / "ghassan-ai"), "export",
           "--checkpoint", str(ckpt),
           "--tokenizer", TOKENIZER_DIR or TOKENIZER,   # the directory: the tokenizer files
           "--out", str(OUT),
           "--format", "safetensors",
           "--dtype", os.environ.get("DTYPE", "f16"),
           "--model-name", MODEL_NAME]
    print("\n$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit("the export failed")

    print("\nthe exported repository:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size/1024/1024:.1f} MB)")
else:
    print("\nno trained checkpoint: the safetensors export is skipped "
          "(the Granite checkpoint is already in the HuggingFace format)")

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

# =============================================================================
# The GGUF (the deliverable the user opens)
# =============================================================================
# The profile is fp16: NO quantization. The weights keep the precision they
# have (the Granite checkpoint is bf16, the GGUF is fp16: two bytes per
# weight, the same size as the source).
#
# The metadata:
#   general.name          = "Ghassan v1 Flash"        <- the brand
#   general.base_model.*  = ibm-granite/granite-4.0-h <- the Granite identity
#   the geometry          = Granite, untouched
GGUF = pathlib.Path(os.environ.get("GGUF_OUT", str(WORK / "ghassan-v1-flash.gguf")))

# Where do the weights come from? The trained checkpoint's safetensors (the
# default), else the Granite checkpoint itself (the model IS Granite until
# it is trained further).
gguf_src = ""
if (OUT / "config.json").exists():
    gguf_src = str(OUT)
elif GRANITE:
    gguf_src = str(GRANITE)
if not gguf_src:
    raise SystemExit("no model to export: run cell 0 (the weights) or the training first")

# The disk check: the full-precision GGUF is as big as the model (13 GB for
# Granite) and /kaggle/working has 20 GB in total.
need = 13.5
if free and free < need and os.environ.get("FORCE", "0") != "1":
    print(f"\nWARNING: the GGUF needs about {need:.0f} GB and there is only "
          f"{free:.1f} GB free on /kaggle/working.")
    print("         The Granite weights in /kaggle/working take 13 GB: move them")
    print("         to a Kaggle DATASET (read-only, it does not count) and point")
    print("         MODEL_DIR at it, or free the space, or use a quantized")
    print("         profile (GGUF_PROFILE=q8_0).")
    print("         (FORCE=1 to try anyway.)")

cmd = [str(BUILD / "bin" / "ghassan-ai"), "quantize",
       "--model", gguf_src,
       "--out", str(GGUF),
       "--profile", os.environ.get("GGUF_PROFILE", "fp16"),
       "--model-name", MODEL_NAME,
       "--compat", "native"]
print("\n$", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-4000:])
if r.returncode != 0:
    raise SystemExit("the GGUF export failed")
print("the deliverable:", GGUF, f"({GGUF.stat().st_size/1024/1024:.1f} MB)")

# ---- the X-ray: the metadata must say what it must say --------------------
xray = SRC / "scripts" / "gguf_xray.py"
if xray.exists():
    print("\n$ python", xray, GGUF)
    r = subprocess.run([sys.executable, str(xray), str(GGUF)],
                       capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("the GGUF metadata check failed (see the report above)")
    print("the metadata is right: Ghassan v1 Flash, based on Granite")
