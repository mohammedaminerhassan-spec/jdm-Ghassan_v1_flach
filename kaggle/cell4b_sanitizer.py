# =============================================================================
# CELL 4b - the GPU memory checker (compute-sanitizer)
# =============================================================================
# When a kernel touches memory it does not own, the error surfaces FAR away
# from the call (usually at the next synchronization point), which makes it
# very hard to find. compute-sanitizer points at the exact kernel and the
# exact line.
#
# It is slow (10-50x), so it runs on the SMALL parity test only.
#
# If it reports something like:
#     Invalid __global__ read of size 4
#     Address 0x... is out of bounds
#     in k_attn_... (attention.cu:123)
# -> that is the bug, with its location.

import os, sys, subprocess, pathlib, shutil, time

WORK = pathlib.Path("/kaggle/working")
BUILD = WORK / "build"
TEST = BUILD / "bin" / "test_cuda_parity"

if not TEST.exists():
    raise SystemExit("the engine is not built: run cell 2 first")

SAN = shutil.which("compute-sanitizer")
if not SAN:
    for cand in ("/usr/local/cuda/bin/compute-sanitizer",
                 "/usr/local/cuda-12.8/bin/compute-sanitizer"):
        if os.path.exists(cand):
            SAN = cand
            break
if not SAN:
    raise SystemExit("compute-sanitizer was not found (it ships with the CUDA toolkit)")

print("the tool:", SAN)
print("this takes a few minutes ...")

env = dict(os.environ)
# The sanitizer is much slower: the small tests only.
t0 = time.time()
r = subprocess.run([SAN, "--tool", "memcheck", "--error-exitcode", "9",
                    str(TEST)],
                   capture_output=True, text=True, env=env)
out = r.stdout + r.stderr
print(out[-12000:])
print(f"\n({time.time()-t0:.0f}s, exit {r.returncode})")

if r.returncode == 9:
    print("=" * 72)
    print("compute-sanitizer found a memory error: the lines above say WHERE.")
    print("=" * 72)
    raise SystemExit("the GPU memory check failed")
if "ERROR SUMMARY: 0 errors" in out:
    print("no memory error: the GPU accesses are all legal.")
