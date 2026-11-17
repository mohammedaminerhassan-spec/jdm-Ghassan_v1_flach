"""Pack the Granite checkpoint into ONE zip for a Kaggle dataset.

STORED (no compression): the weights are bf16 - they do not compress - and
compressing 13 GB would burn the CPU for an hour. Kaggle extracts a zip
dataset by itself, so the notebook finds the shards under /kaggle/input.
"""
import pathlib, sys, time, zipfile

SRC = pathlib.Path(r"C:\Users\Ghassan PC\Documents\granit filles confiques")
OUT = pathlib.Path(r"C:\Users\Ghassan PC\Documents\granite-4.0-h-tiny-kaggle.zip")

SKIP = {".sig"}          # the cosign signature: nothing needs it

files = sorted(p for p in SRC.iterdir()
               if p.is_file() and p.suffix not in SKIP)
total = sum(p.stat().st_size for p in files)
print(f"{len(files)} file(s), {total/1024/1024/1024:.2f} GB")
for p in files:
    print(f"  {p.name}  ({p.stat().st_size/1024/1024:.1f} MB)")

t0 = time.time()
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
    for p in files:
        z.write(p, arcname=p.name)
        done = z.fp.tell()
        print(f"  + {p.name}  ({done/1024/1024/1024:.2f} GB, {time.time()-t0:.0f}s)",
              flush=True)

print(f"\n{OUT}  ({OUT.stat().st_size/1024/1024/1024:.2f} GB) in {time.time()-t0:.0f}s")
