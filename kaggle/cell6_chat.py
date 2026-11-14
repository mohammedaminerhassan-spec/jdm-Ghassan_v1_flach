# =============================================================================
# CELL 6 - "HERE I PUT THE TEXT I WANT"
# =============================================================================
# Put your text in the PROMPT below (or in a file) and the engine answers
# with the model. The Granite weights are used (the ones cell 0 downloaded).
#
# The device:
#   * auto  -> the T4 if there is one (the default),
#   * cpu   -> the CPU (slow: the Granite decode on the host is ~10x slower),
#   * cuda  -> the T4 (fails if there is none).
#
# The model:
#   MODEL_DIR  the Granite checkpoint (cell 0),
#   GGUF       the exported GGUF (cell 5) - it is used instead when it exists.

import os, sys, subprocess, pathlib, time, shutil

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
BUILD = WORK / "build"
BIN  = BUILD / "bin" / "ghassan-ai"

MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", str(WORK / "granite")))
GGUF      = pathlib.Path(os.environ.get("GGUF_OUT", str(WORK / "ghassan-v1-flash.gguf")))

# =============================================================================
# >>> HERE I PUT THE TEXT I WANT <<<
# =============================================================================
PROMPT = """Please list one IBM Research laboratory located in the United States."""

# The chat mode uses Granite's role-tag template (the model was trained with
# it); the plain mode just continues the text.
MODE = "chat"          # "chat" or "text"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "96"))
CTX = int(os.environ.get("CTX", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
SEED = int(os.environ.get("SEED", "42"))
# =============================================================================

if not BIN.exists():
    raise SystemExit("the engine is not built: run cell 2 first")

# The GGUF wins when it is there (it is the deliverable), else the
# Granite checkpoint.
if GGUF.exists():
    model = str(GGUF)
    print("the model: the exported GGUF ->", GGUF)
elif (MODEL_DIR / "config.json").exists():
    model = str(MODEL_DIR)
    print("the model: the Granite checkpoint ->", MODEL_DIR)
else:
    raise SystemExit("no model: run cell 0 (the Granite weights) or cell 5 (the export) first")

# ---- the device: the T4 if there is one --------------------------------------
def pick_device():
    forced = os.environ.get("DEVICE", "auto").lower()
    if forced in ("cpu", "cuda"):
        return forced
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        print("the GPUs:")
        for line in r.stdout.strip().splitlines():
            print("   ", line)
        return "cuda"
    print("no GPU here: the CPU it is (slow, but it works)")
    return "cpu"

device = pick_device()
print("the device:", device)

cmd = [str(BIN), "generate", "--model", model, "--device", device,
       "--ctx", str(CTX), "--max-tokens", str(MAX_TOKENS),
       "--temp", str(TEMPERATURE), "--top-p", str(TOP_P),
       "--seed", str(SEED), "--no-stream",
       "--prompt", PROMPT]

print("\n" + "=" * 72)
print(PROMPT)
print("=" * 72)
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout)
if r.stderr.strip():
    print(r.stderr[-3000:])
print(f"\n({time.time()-t0:.0f}s)")
if r.returncode != 0:
    raise SystemExit("the generation failed")

# ---- the chat mode (Granite's role-tag template) ---------------------------
if MODE == "chat":
    print("\n" + "=" * 72)
    print("the chat mode (the role-tag template Granite was trained with):")
    print("=" * 72)
    chat = [str(BIN), "chat", "--model", model, "--device", device,
            "--ctx", str(CTX), "--max-tokens", str(MAX_TOKENS),
            "--temp", str(TEMPERATURE), "--top-p", str(TOP_P),
            "--seed", str(SEED), "--no-stream"]
    t0 = time.time()
    r = subprocess.run(chat, input=PROMPT + "\n/quit\n", capture_output=True, text=True)
    print(r.stdout)
    if r.stderr.strip():
        print(r.stderr[-2000:])
    print(f"\n({time.time()-t0:.0f}s)")
