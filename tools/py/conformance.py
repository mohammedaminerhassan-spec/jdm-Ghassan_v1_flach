#!/usr/bin/env python3
"""Conformance harness: HuggingFace tokenizer -> Gai engine expectations.

The Gai C++ engine encodes text itself (it never calls into Python), so the
only thing that keeps training and serving consistent is that both sides
produce the SAME token ids for the same text. This script uses the reference
HuggingFace tokenizer to produce the ground truth; the C++ engine replays the
same fixtures and must match byte for byte.

  python tools/py/conformance.py --tokenizer-dir safetensors \
      --out tests/fixtures/tokenizer_conformance.json

  # round trip against the real binary
  python tools/py/conformance.py --tokenizer-dir safetensors \
      --out tests/fixtures/tokenizer_conformance.json --check \
      --binary build_msys/bin/ghassan-ai.exe \
      --model artifacts/Ghassan-v1-pro-360m-q4_0.gguf
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

TEXTS = [
    "salam bekri",
    "What is the capital of France?",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "don't stop believing",
    "IT'S a TEST, isn't it?",
    "Numbers: 123 4567 3.14 1,000",
    "  leading and trailing spaces  ",
    "multiple\n\nnewlines\nhere",
    "punctuation!!! ??? ...---",
    "Mixed CASE and CamelCase and snake_case_words",
    "email@example.com https://example.org/path?q=1&r=2",
    "tab\tseparated\tvalues",
    "emoji: \U0001f600 \U0001f680 done",
    "accented: caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fcber",
    "arabic: \u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "arabizi: mrhba bzaf 3labi",
    "a",
    " ",
    "\n",
    "",
    "0123456789",
    "x" * 200,
    "word " * 50,
]

# The engine skips empty system messages; the reference template injects a
# default one, so the fixtures always carry an explicit system message.
MESSAGES = [
    [{"role": "system", "content": "You are a helpful assistant."}],
    [{"role": "user", "content": "What is 2+2?"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Tell me a joke"},
    ],
]


def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir)


def encode_ids(tok, text):
    out = tok.encode(text, add_special_tokens=False)
    if isinstance(out, dict):
        return out["input_ids"]
    return out


def load_template(tok, tokenizer_dir):
    cfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("chat_template"):
            return cfg["chat_template"]
    return None


def build_fixtures(tok, template):
    fixtures = {"texts": [], "chats": []}
    for t in TEXTS:
        fixtures["texts"].append({"text": t, "ids": encode_ids(tok, t)})

    for msgs in MESSAGES:
        if template is not None:
            try:
                rendered = tok.apply_chat_template(msgs, tokenize=False)
            except Exception:
                rendered = None
        else:
            rendered = None
        ids = encode_ids(tok, rendered) if rendered is not None else []
        fixtures["chats"].append({
            "messages": msgs,
            "rendered": rendered,
            "ids": ids,
        })
    return fixtures


def binary_check(fixtures, binary, model, workdir):
    """Replay the fixtures through the real Gai binary.

    The binary is invoked once in batch mode so the model is loaded a single
    time instead of once per sample.
    """
    import tempfile

    paths = []
    for case in fixtures["texts"]:
        paths.append(case["text"])

    batch = os.path.join(workdir, "conformance_batch.bin")
    with open(batch, "wb") as f:
        for t in paths:
            raw = t.encode("utf-8")
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)

    proc = subprocess.run(
        [binary, "tokenize", "--model", model, "--batch", batch],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    got_lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not got_lines:
        print(f"NO OUTPUT: {proc.stderr.strip()[:400]}")
        return len(paths)

    failures = 0
    for case, line in zip(fixtures["texts"], got_lines):
        got = json.loads(line)
        if got["ids"] != case["ids"]:
            failures += 1
            print(f"MISMATCH {case['text']!r}")
            print(f"  want {case['ids']}")
            print(f"  got  {got['ids']}")
    if len(got_lines) != len(paths):
        failures += 1
        print(f"expected {len(paths)} results, got {len(got_lines)}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True,
                    help="directory with tokenizer.json (+ tokenizer_config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored fixture instead of writing it")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    template = load_template(tok, args.tokenizer_dir)
    fixtures = build_fixtures(tok, template)

    if not args.check:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out} ({len(fixtures['texts'])} texts, {len(fixtures['chats'])} chats)")
        return 0

    failures = 0
    if args.binary and args.model:
        with tempfile.TemporaryDirectory() as workdir:
            failures += binary_check(fixtures, args.binary, args.model, workdir)
        print(f"binary round trip: {failures} mismatch(es)")
    else:
        with open(args.out, "r", encoding="utf-8") as f:
            want = json.load(f)
        for got, exp in zip(fixtures["texts"], want["texts"]):
            if got["ids"] != exp["ids"]:
                failures += 1
                print(f"MISMATCH text {got['text']!r}")
                print(f"  want {exp['ids']}")
                print(f"  got  {got['ids']}")
        for got, exp in zip(fixtures["chats"], want["chats"]):
            if got["rendered"] != exp["rendered"]:
                failures += 1
                print(f"MISMATCH chat render {got['messages']!r}")
                print(f"  want {exp['rendered']!r}")
                print(f"  got  {got['rendered']!r}")
        print(f"conformance: {failures} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
