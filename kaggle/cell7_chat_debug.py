# =============================================================================
# CELL 7 - the chat diagnosis (why the chat mode answers with nothing)
# =============================================================================
# The plain "generate" answers correctly; the chat mode (the role-tag
# template) returns an empty reply. This cell says which one it is:
#
#   * the rendered chat prompt + its token ids (the tokenizer alone, no
#     13 GB of weights),
#   * the first generated token of the chat (is it the end of the turn?),
#   * the same with the repetition penalty off and with greedy sampling.
#
# It prints a verdict at the end.

import os, sys, subprocess, pathlib, time

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
BIN  = WORK / "build" / "bin" / "ghassan-ai"
Q    = "Please list one IBM Research laboratory located in the United States."


def find_weights():
    home = pathlib.Path(os.path.expanduser("~"))
    for p in (home / ".cache" / "huggingface" / "hub").glob("models--*/snapshots/*/config.json"):
        if list(p.parent.glob("*.safetensors")):
            return p.parent
    for p in pathlib.Path("/kaggle/input").rglob("config.json"):
        if list(p.parent.glob("*.safetensors")):
            return p.parent
    for root in (WORK / "granite", WORK / "hf"):
        if (root / "config.json").exists():
            return root
    return None


MODEL = find_weights()
if not MODEL:
    raise SystemExit("no Granite weights: run cell 0 first")
print("the model:", MODEL)

# ---- 1. the prompt the model actually receives (the tokenizer only) --------
print("\n" + "=" * 72)
print("the rendered chat prompt")
print("=" * 72)
r = subprocess.run([str(BIN), "tokenizer-info", "--tokenizer", str(MODEL),
                    "--chat-text", Q], capture_output=True, text=True)
print(r.stdout[-4000:])
if r.returncode != 0:
    print(r.stderr[-2000:])

# ---- 2. what the model generates in the chat mode --------------------------
print("\n" + "=" * 72)
print("the chat mode, with and without the repetition penalty")
print("=" * 72)
print("(the ids of the generated tokens are printed: if the FIRST one is "
      "100257 (<|end_of_text|>) the model ends the turn immediately)")
for label, extra in (("the default", []),
                     ("no repetition penalty", ["--repeat", "1.0"]),
                     ("greedy", ["--greedy", "--repeat", "1.0"])):
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    r = subprocess.run([str(BIN), "chat", "--model", str(MODEL), "--device", "cuda",
                        "--ctx", "1024", "--max-tokens", "24", "--no-stream",
                        "--print-ids"] + extra,
                       input=Q + "\n/quit\n", capture_output=True, text=True)
    print(r.stdout[-2500:])
    print(r.stderr[-1500:])
    print(f"({time.time()-t0:.0f}s)")
