#!/usr/bin/env python3
"""Compare two logits dumps (.f32, little endian) row by row.

Used as the acceptance gate for a re-exported Ghassan v1 Flash GGUF:
compare the safetensors logits with the GGUF-reloaded logits on the same
token ids. A correct re-export agrees on (almost) every position's argmax
with only small fp differences; a broken one disagrees structurally.

Usage:
    python3 scripts/gguf_parity.py A.f32 B.f32 [--rows N]
"""
import sys

import numpy as np

def main():
    a = np.fromfile(sys.argv[1], dtype="<f4")
    b = np.fromfile(sys.argv[2], dtype="<f4")
    rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else -1
    if a.shape != b.shape:
        print("FAIL: element count differs: %d vs %d" % (a.size, b.size))
        return 1
    # vocab size is not in the file; infer it from the largest plausible value
    # by trying to see whether a.size is a multiple of the known vocab.
    for V in (100352,):
        if a.size % V == 0:
            ar = a.reshape(-1, V)
            br = b.reshape(-1, V)
            break
    else:
        ar, br = a.reshape(-1, a.size), b.reshape(-1, b.size)
    n = ar.shape[0] if rows < 0 else min(rows, ar.shape[0])
    agree = 0
    worst = 0.0
    for i in range(n):
        ai, bi = int(ar[i].argmax()), int(br[i].argmax())
        if ai == bi:
            agree += 1
        d = float(np.abs(ar[i] - br[i]).max())
        worst = max(worst, d)
        print("  row %3d  argmax A=%6d  argmax B=%6d  max|A-B|=%.4f" % (i, ai, bi, d))
    print("\nargmax agreement: %d/%d rows" % (agree, n))
    print("max abs diff overall: %.6g" % worst)
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main())
