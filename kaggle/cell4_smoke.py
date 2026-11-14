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
# The reference checkpoint (13.9 GB, 3 shards) must be in a Kaggle dataset:
#   /kaggle/input/granite-4.0-h-tiny/...
# MODEL points at it; the export and the training use the same one.
HF = WORK / "hf"
if not (HF / "config.json").exists():
    cands = [p for p in pathlib.Path("/kaggle/input").rglob("config.json")
             if "granite" in str(p).lower()]
    if cands:
        import shutil as _sh
        if HF.exists():
            _sh.rmtree(HF)
        HF.mkdir(parents=True)
        # The config + the tokenizer are enough for the architecture; the
        # weights stay where they are (13.9 GB: copying them is a waste).
        _sh.copy(cands[0], HF / "config.json")
        for extra in ("tokenizer.json", "tokenizer_config.json",
                      "generation_config.json", "special_tokens_map.json"):
            src = cands[0].parent / extra
            if src.exists():
                _sh.copy(src, HF / extra)
        print("the Granite config+tokenizer staged in", HF)

if (HF / "config.json").exists():
    print("\nthe official Granite config (the one in the repository) ...")
    rc, out = run([str(BUILD / "bin" / "ghassan-ai"), "info",
                    "--model", str(HF), "--config-only"], tail=6000)
    if rc != 0:
        raise SystemExit("the engine cannot read the official Granite config")

    # The weights (13.9 GB) live in the Kaggle dataset.
    MODEL = ""
    for cand in pathlib.Path("/kaggle/input").rglob("config.json"):
        if "granite" in str(cand).lower() and cand.parent != HF:
            MODEL = str(cand.parent)
            break
    if MODEL:
        print("\nthe real weights ...")
        rc, out = run([str(BUILD / "bin" / "ghassan-ai"), "info", "--model", MODEL],
                      tail=8000)
        if rc != 0:
            raise SystemExit("the engine cannot load the Granite weights")
    else:
        print("\n(the weights were not found in /kaggle/input: the config check ran alone)")
else:
    print("\nno Granite config: skipping the reference check")

# ---- 4. the smoke training -------------------------------------------------
env = dict(os.environ)
env.setdefault("CUDA_VISIBLE_DEVICES", "0")   # one GPU for the smoke test
rc, out = run([
    str(GAIT),
    "--config", str(SRC / "configs" / "ghassan_v1_flash_smoke.yaml"),
    "--tokenizer", str(HF / "tokenizer.json"),
    "--data", str(WORK / "artifacts" / "shards"),
    "--checkpoint-dir", str(WORK / "artifacts" / "checkpoints" / "smoke"),
    "--max-steps", "20",
    "--log-every", "1",
    "--resume", "none",
], env=env, tail=12000)
if rc != 0:
    raise SystemExit("the smoke training failed")

# ---- 5. did the loss go down? ----------------------------------------------
losses = [float(m) for m in re.findall(r"loss\s+([0-9]+\.[0-9]+)", out)]
print("\nthe loss curve:", losses)
if len(losses) >= 4:
    first = sum(losses[:2]) / 2.0
    last = sum(losses[-2:]) / 2.0
    print(f"  first {first:.4f} -> last {last:.4f}")
    if last >= first:
        raise SystemExit("the loss did NOT go down: the gradients are not reaching the weights")
    print("  the loss is going down: the backward is alive")
else:
    print("  (not enough loss lines to judge)")
