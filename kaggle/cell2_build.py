# =============================================================================
# CELL 2 - the CUDA build (the risky one: the first nvcc compile)
# =============================================================================
# This is the cell that validates the fixes of the audit:
#   * the GEMM chunking (the cuBLAS leading-dimension fix) in
#     cuda/kernels.cu and cuda/attention.cu,
#   * the dead CUDA kernels are gone,
#   * the phase timers.
#
# The T4 is sm_75: only that architecture is compiled (the default in
# CMakeLists.txt is 75;80;90, which would triple the compile time for
# nothing on a T4).

import os, subprocess, pathlib, time, shutil

WORK = pathlib.Path("/kaggle/working")
SRC  = WORK / "ghassan-ai"
BUILD = WORK / "build"

if not (SRC / "CMakeLists.txt").exists():
    raise SystemExit("the source is missing: run cell 1 first")

# A clean build: a stale object file from another CUDA version is the classic
# "undefined symbol" afternoon.
if BUILD.exists():
    shutil.rmtree(BUILD)
BUILD.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env["CMAKE_BUILD_TYPE"] = "Release"
# The T4 (sm_75) only. The A100/T4x2 of Kaggle are both sm_75.
env["CMAKE_CUDA_ARCHITECTURES"] = "75"
# The build is long: give ninja all the cores.
jobs = max(1, (os.cpu_count() or 2))

t0 = time.time()
print("=" * 70)
print("configuring ...", flush=True)
r = subprocess.run(
    ["cmake", "-S", str(SRC), "-B", str(BUILD), "-G", "Ninja",
     "-DCMAKE_BUILD_TYPE=Release",
     "-DCMAKE_CUDA_ARCHITECTURES=75",
     "-DGAI_ENABLE_CUDA=ON",
     "-DGAI_BUILD_TESTS=ON"],
    capture_output=True, text=True, env=env)
print(r.stdout[-4000:])
if r.returncode != 0:
    print(r.stderr[-6000:])
    raise SystemExit("the CMake configuration failed")

print("\nbuilding with", jobs, "jobs ...", flush=True)
r = subprocess.run(["cmake", "--build", str(BUILD), "-j", str(jobs)],
                   capture_output=True, text=True, env=env)
out = r.stdout + r.stderr
# The warnings are the interesting part: the engine builds with -Werror on
# the C++ side, and the CUDA side is where the audit's changes are.
warn = [l for l in out.splitlines() if "warning" in l.lower()]
print(out[-6000:])
print("=" * 70)
print(f"build: {'OK' if r.returncode == 0 else 'FAILED'} in {time.time()-t0:.0f}s")
if warn:
    print(f"{len(warn)} warning(s):")
    for l in warn[:40]:
        print("  ", l)
if r.returncode != 0:
    raise SystemExit("the build failed")

binaries = sorted((BUILD / "bin").glob("*"))
print("the binaries:", [b.name for b in binaries])
for b in binaries:
    b.chmod(0o755)
