# =============================================================================
# CELL 1 - the setup (run it first)
# =============================================================================
# It clones the engine from GitHub into /kaggle/working/ghassan-ai and prints
# the GPU / the toolchain.
#
# The clone is SHALLOW (--depth 1): the repo carries no LFS and no history we
# need, and a shallow clone is ~10x faster on Kaggle's network.

import os, sys, subprocess, pathlib, shutil, time

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"

REPO_URL    = "https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_flach.git"
REPO_BRANCH = "main"

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("=" * 72)
print("gpu    :", sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader").stdout.strip())
print("python :", sys.version.split()[0])
print("cuda   :", sh("nvcc --version 2>/dev/null | tail -1").stdout.strip() or "(no nvcc on PATH)")
print("gcc    :", sh("g++ --version | head -1").stdout.strip())
print("disk   :", sh("df -h /kaggle/working | tail -1").stdout.strip())
print("=" * 72)

def clone_fresh():
    """A clean clone (the shallow one: 10x faster, no history we need)."""
    if SRC.exists():
        shutil.rmtree(SRC, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(SRC)])
    print(f"cloned in {time.time()-t0:.0f}s")
    return r.returncode == 0


if (SRC / "CMakeLists.txt").exists():
    # The source is already there: bring it up to date. A shallow clone can
    # refuse to fast-forward (it lacks the history), so a failed pull falls
    # back to a fresh clone.
    print("the source is already there -> pulling the latest")
    r = sh(f"git -C {SRC} pull --ff-only origin {REPO_BRANCH}")
    if r.returncode != 0:
        print("the pull failed (a shallow clone cannot fast-forward) -> cloning again")
        if not clone_fresh():
            raise SystemExit("the clone failed: check the repository URL and the network")
else:
    if not clone_fresh():
        raise SystemExit("the clone failed: check the repository URL and the network")

if not (SRC / "CMakeLists.txt").exists():
    raise SystemExit("the clone did not bring the engine: " + str(SRC))

# The two packages the data preparation needs (pyarrow reads the parquet,
# huggingface_hub downloads the reference checkpoint).
sh(f"{sys.executable} -m pip -q install pyarrow huggingface_hub")

# The commit that is running: it is what the log has to name.
commit = sh(f"git -C {SRC} rev-parse --short HEAD").stdout.strip()
print("\nthe source :", SRC)
print("the commit:", commit)
# The fixes that MUST be in the tree (the dates are the commit dates): the
# CUDA fix of the attention (the illegal memory access) and the Mamba-2
# backward. If the commit is older, the notebook is running the old code.
try:
    dates = sh(f"git -C {SRC} log -1 --format=%cd --date=short").stdout.strip()
    print("the date  :", dates)
except Exception:
    pass
print("the files :")
for p in sorted(SRC.iterdir()):
    print("   ", p.name + ("/" if p.is_dir() else ""))
