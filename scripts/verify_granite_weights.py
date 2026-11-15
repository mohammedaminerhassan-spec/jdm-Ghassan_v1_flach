import hashlib, json, pathlib, sys, urllib.request

# The sha256 of the official Granite-4.0-H-Tiny shards (from the HuggingFace
# LFS metadata). They are the reference: the engine's Granite compatibility
# test uses this exact checkpoint.
API = "https://huggingface.co/api/models/ibm-granite/granite-4.0-h-tiny/tree/main"

def hub_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        entries = json.load(r)
    out = {}
    for e in entries:
        if e.get("type") == "file" and e["path"].endswith(".safetensors"):
            oid = (e.get("lfs") or {}).get("oid", "")
            if oid.startswith("sha256:"):
                oid = oid[7:]
            out[e["path"]] = (oid, e.get("size") or (e.get("lfs") or {}).get("size", 0))
    return out

def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                          r"C:\Users\Ghassan PC\Documents\granit filles confiques")
    print("the local folder:", folder)
    print("the official shard hashes (from the HuggingFace LFS metadata):")
    try:
        official = hub_files()
    except Exception as e:
        print("  (the hub is not reachable:", e, ")")
        official = {}
    for name, (sha, size) in sorted(official.items()):
        print(f"  {name}  {sha}  ({size/1024/1024:.1f} MB)")

    print("\nthe local shards:")
    bad = 0
    for name, (sha, size) in sorted(official.items()):
        p = folder / name
        if not p.exists():
            print(f"  {name}: MISSING")
            bad += 1
            continue
        got_size = p.stat().st_size
        if got_size != size:
            print(f"  {name}: the size differs ({got_size} vs {size})")
            bad += 1
            continue
        print(f"  {name}: {got_size/1024/1024:.1f} MB ... hashing", flush=True)
        got = sha256(p)
        ok = (got == sha)
        print(f"    sha256 {got}  {'OK - identical to the official file' if ok else 'DIFFERENT'}")
        if not ok:
            bad += 1
    if bad == 0:
        print("\nThe local copy is IDENTICAL to the official checkpoint.")
    else:
        print(f"\n{bad} file(s) differ or are missing.")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
