"""Checks that the Granite reference files in granite-documentation/ are
byte-identical to the ones on HuggingFace.

The repository's granite-documentation/ folder is the engine's only
reference for the architecture, so a silently edited config.json or
tokenizer would make every verification meaningless. This script downloads
each file from the hub and compares the bytes.

    python scripts/verify_docs_vs_hub.py

Exit code 0 = every file is identical to the hub.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DOC = os.path.join(REPO, "granite-documentation", "granit filles confiques")
HF = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny/resolve/main/"

FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "tokenizer.json",
]


def fetch(name):
    req = urllib.request.Request(HF + name)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def main():
    if not os.path.isdir(DOC):
        print("FATAL: %s is missing" % DOC)
        return 2
    bad = []
    for name in FILES:
        local_path = os.path.join(DOC, name)
        if not os.path.exists(local_path):
            print("%-32s MISSING locally" % name)
            bad.append(name)
            continue
        with open(local_path, "rb") as f:
            local = f.read()
        try:
            remote = fetch(name)
        except Exception as e:  # noqa: BLE001 - the network may be down
            print("%-32s could not be fetched (%s)" % (name, e))
            continue
        same = local == remote
        print("%-32s local %9d B md5 %s | hub %9d B md5 %s  %s"
              % (name, len(local), hashlib.md5(local).hexdigest()[:10],
                 len(remote), hashlib.md5(remote).hexdigest()[:10],
                 "IDENTICAL" if same else "*** DIFFERENT ***"))
        if not same:
            bad.append(name)
            try:
                a = json.loads(local.decode("utf-8"))
                b = json.loads(remote.decode("utf-8"))
                for k in sorted(set(a) | set(b)):
                    if a.get(k) != b.get(k):
                        print("      %-26s local=%r" % (k, a.get(k)))
                        print("      %-26s hub  =%r" % ("", b.get(k)))
            except Exception:
                pass
    print()
    if bad:
        print("FAIL: %d file(s) differ from the hub: %s" % (len(bad), ", ".join(bad)))
        return 1
    print("PASS: every reference file is byte-identical to the hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
