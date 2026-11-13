"""Makes the repository self-verifying.

It copies the Granite fixture generator and the independent numpy reference
into tests/python/ (with repository-relative paths) and adds a runner that:

  1. builds the tiny Granite fixture - with the REAL checkpoint's 3-D MoE
     layout ([ne, 2E, d] and [ne, d, E]), the same layer interleaving as
     granite-4.0-h-tiny and the same multipliers,
  2. runs the engine's logits on it,
  3. runs the independent reference (written from the reference modelling
     code, not from the engine) and compares them.

Usage (from the repository root):
    python tests/python/test_granite_forward.py [--exe build/bin/ghassan-ai]
"""
import io
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = r"C:\Users\Ghassan PC\Documents\ghassan-verify"
DST = os.path.join(ROOT, "tests", "python")

FILES = {
    "make_granite_tiny.py": "make_granite_fixture.py",
    "check_granite_ref.py": "granite_reference_forward.py",
}


def copy_and_relativise():
    os.makedirs(DST, exist_ok=True)
    for src_name, dst_name in FILES.items():
        src = os.path.join(SRC, src_name)
        if not os.path.exists(src):
            print("skipping %s (not found at %s)" % (src_name, src))
            continue
        with io.open(src, "r", encoding="utf-8") as f:
            body = f.read()
        # The scripts had hardcoded absolute paths; make them repo-relative.
        body = body.replace(
            'ROOT = r"C:\\Users\\Ghassan PC\\Documents\\Projects\\Ghassan Ai model"',
            'ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))')
        body = body.replace(
            'ROOT = r"C:\\Users\\Ghassan PC\\Documents\\Projects\\Ghassan Ai model"',
            'ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))')
        body = body.replace(
            'SRC = os.path.join(ROOT, "granite-documentation", "granit filles confiques")',
            'SRC = os.path.join(ROOT, "granite-documentation", "granit filles confiques")')
        with io.open(os.path.join(DST, dst_name), "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        print("wrote", os.path.join(DST, dst_name))


def main():
    copy_and_relativise()
    exe = "build/bin/ghassan-ai"
    if "--exe" in sys.argv:
        exe = sys.argv[sys.argv.index("--exe") + 1]
    exe_path = exe if os.path.isabs(exe) else os.path.join(ROOT, exe)
    if not os.path.exists(exe_path):
        exe_path = os.path.join(ROOT, "build", "bin", "ghassan-ai" + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(exe_path):
        print("the engine was not found; build it first (cmake -S . -B build && cmake --build build)")
        return 2

    print("== building the Granite fixture (the real 3-D MoE layout) ==")
    subprocess.run([sys.executable, os.path.join(DST, "make_granite_fixture.py")], check=True)

    fixture = os.path.join(ROOT, "data", "granite_tiny")
    ids = "791 6864 315 9822 374"
    with io.open(os.path.join(fixture, "ids.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(ids)
    out = os.path.join(ROOT, "data", "granite_tiny_logits.f32")
    print("== the engine's logits ==")
    subprocess.run([exe_path, "logits", "--model", fixture, "--ids", ids, "--out", out], check=True)
    print("== the independent reference ==")
    rc = subprocess.run([sys.executable, os.path.join(DST, "granite_reference_forward.py"),
                         fixture, out]).returncode
    print("\n%s" % ("PASS: the engine matches the independent reference"
                    if rc == 0 else "FAIL: the engine does NOT match the reference"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
