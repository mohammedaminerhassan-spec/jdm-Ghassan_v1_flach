#!/usr/bin/env python3
"""Download a HuggingFace checkpoint for the Gai engine.

Only the files the converter needs are fetched (weights, tokenizer, config);
READMEs, licences and any pre-built GGUF are skipped.

    python tools/py/download_model.py --repo Qwen/Qwen3-0.6B \
        --out models/qwen3-0.6b
"""
import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
    raise


WANTED = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "model.safetensors",
    "model.safetensors.index.json",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HuggingFace repo id")
    ap.add_argument("--out", required=True, help="local directory")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--include-gguf", action="store_true",
                    help="also fetch pre-built GGUF files (big)")
    args = ap.parse_args()

    patterns = list(WANTED) + ["*.safetensors", "*.safetensors.index.json"]
    if args.include_gguf:
        patterns.append("*.gguf")

    print(f"downloading {args.repo} -> {args.out}")
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.out,
        revision=args.revision,
        allow_patterns=patterns,
        max_workers=2,
    )
    print(f"done: {path}")
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
    print(f"  {total / 2**20:.1f} MiB on disk")


if __name__ == "__main__":
    main()
