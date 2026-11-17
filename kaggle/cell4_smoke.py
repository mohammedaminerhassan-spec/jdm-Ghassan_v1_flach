# =============================================================================
# CELL 4 - the smoke training (the real validation)
# =============================================================================
# Twenty steps on a small dense model. It validates, in one go:
#
#   * the CUDA build (cell 2),
#   * the GEMM chunking on the GPU (the LoRA case is the one that used to
#     make cuBLAS refuse the call),
#   * the attention path, the optimizer, the loss scaler,
#   * the checkpointing and the resume,
#   * the safetensors export (cell 5).
#
# The loss MUST go down. A flat loss means the gradients are not reaching
# the weights (the audit found three ways for that to happen).

import os, sys, subprocess, pathlib, json, time, re

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
BUILD = WORK / "build"
HF   = WORK / "hf"
GAIT = BUILD / "bin" / "gai_train"

def find_tokenizer():
    """The Granite tokenizer (the vocabulary the model was trained with)."""
    for cand in (WORK / "hf" / "tokenizer.json",
                 WORK / "granite" / "tokenizer.json"):
        if cand.exists():
            return str(cand)
    # Anything else that looks like one.
    for p in pathlib.Path("/kaggle/input").rglob("tokenizer.json"):
        return str(p)
    raise SystemExit("no tokenizer.json: run cell 0 or cell 3 first")

TOKENIZER = find_tokenizer()
print("the tokenizer:", TOKENIZER)

def run(cmd, env=None, tail=4000):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    print(out[-tail:])
    print(f"  -> exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
    return r.returncode, out

# ---- 1. the CPU tests (they run everywhere) --------------------------------
rc, out = run(["ctest", "--test-dir", str(BUILD), "--output-on-failure"], tail=6000)
if rc != 0:
    raise SystemExit("the tests failed: fix them before training")

# ---- 2. the CUDA parity (the CPU <-> GPU gradients) -----------------------
# This is the test that validates the GEMM chunking ON THE GPU.
rc, out = run([str(BUILD / "bin" / "test_cuda_parity")], tail=4000)
if rc != 0:
    raise SystemExit("the CPU/CUDA parity failed: the GPU computes something else")

# ---- 3. the Granite verification (the real checkpoint) ---------------------
# The Granite weights are where cell 0 put them (/kaggle/working/granite) or
# in a Kaggle dataset (/kaggle/input/...). The engine must read them.
GRANITE_CANDS = [WORK / "granite", WORK / "hf"]
for root in pathlib.Path("/kaggle/input").glob("*"):
    if root.is_dir():
        GRANITE_CANDS.append(root)
        for sub in root.glob("*"):
            if sub.is_dir():
                GRANITE_CANDS.append(sub)

MODEL = ""
for cand in GRANITE_CANDS:
    if (cand / "config.json").exists():
        MODEL = str(cand)
        break

if MODEL:
    print("\nthe Granite checkpoint:", MODEL)
    r = subprocess.run([str(BUILD / "bin" / "ghassan-ai"), "info", "--model", MODEL],
                       capture_output=True, text=True)
    print(r.stdout[-6000:]); print(r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("the engine cannot load the Granite weights")
    # The parameter count must be the checkpoint's (6,939,037,248).
    if "6939037248" not in r.stdout.replace(",", ""):
        print("(note: the parameter count is not the Granite one - this is not "
              "the Granite checkpoint)")
else:
    print("\nno Granite checkpoint found: run cell 0 (the weights) first "
          "(the Granite verification is skipped)")

# ---- 4. the smoke training -------------------------------------------------
SHARDS = WORK / "artifacts" / "shards"
HAVE_DATA = SHARDS.exists() and any(SHARDS.glob("*.gbin"))

if not HAVE_DATA:
    print("\n(no shards: the dataset was not prepared, so the smoke TRAINING is "
          "skipped - the tests above are the ones that matter here. Point "
          "DATA_INPUT at a Kaggle dataset and re-run cell 3 to train.)")
    losses = []
    out = ""
    out_g = ""
else:
    # ---- 4. the smoke training ---------------------------------------------
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")   # one GPU for the smoke test
    rc, out = run([
        str(GAIT),
        "--config", str(SRC / "configs" / "ghassan_v1_flash_smoke.yaml"),
        "--tokenizer", TOKENIZER,
        "--data", str(SHARDS),
        "--checkpoint-dir", str(WORK / "artifacts" / "checkpoints" / "smoke"),
        "--max-steps", "20",
        "--log-every", "1",
        "--resume", "none",
    ], env=env, tail=12000)
    if rc != 0:
        raise SystemExit("the smoke training failed")

    # ---- 5. the Granite-shaped smoke training ------------------------------
    # The dense smoke test above never touches the Mamba-2 block. This one
    # runs the whole Granite-4.0 shape (the Mamba-2 hybrid + the MoE with a
    # shared expert + the multipliers) on the GPU: forward, backward, optimizer
    # and checkpoint. The Mamba-2 backward is the part of the engine that had
    # never run on a GPU (it crashed on a null pointer and used the wrong
    # recurrent state), so it has to be exercised here, not only in the parity
    # test.
    rc, out_g = run([
        str(GAIT),
        "--config", str(SRC / "configs" / "ghassan_v1_flash_smoke_granite.yaml"),
        "--tokenizer", TOKENIZER,
        "--data", str(SHARDS),
        "--checkpoint-dir", str(WORK / "artifacts" / "checkpoints" / "smoke_granite"),
        "--max-steps", "20",
        "--warmup", "5",
        "--log-every", "1",
        "--resume", "none",
    ], env=env, tail=12000)
    if rc != 0:
        raise SystemExit("the Granite-shaped smoke training failed (the Mamba-2 path)")

# ---- 6. did the loss go down? ----------------------------------------------
losses = [float(m) for m in re.findall(r"loss\s+([0-9]+\.[0-9]+)", out)]
print("\nthe loss curve (dense):", losses)
if len(losses) >= 4:
    first = sum(losses[:2]) / 2.0
    last = sum(losses[-2:]) / 2.0
    print(f"  first {first:.4f} -> last {last:.4f}")
    if last >= first:
        raise SystemExit("the loss did NOT go down: the gradients are not reaching the weights")
    print("  the loss is going down: the backward is alive")
else:
    print("  (not enough loss lines to judge)")

# The same for the Granite shape (the Mamba-2 path).
glosses = [float(m) for m in re.findall(r"loss\s+([0-9]+\.[0-9]+)", out_g)]
print("\nthe loss curve (granite hybrid):", glosses)
if len(glosses) >= 4:
    gfirst = sum(glosses[:2]) / 2.0
    glast = sum(glosses[-2:]) / 2.0
    print(f"  first {gfirst:.4f} -> last {glast:.4f}")
    if glast >= gfirst:
        raise SystemExit("the Granite-shaped loss did NOT go down: the Mamba-2 backward is broken")
    print("  the loss is going down: the Mamba-2 backward is alive")
else:
    print("  (not enough loss lines to judge)")
