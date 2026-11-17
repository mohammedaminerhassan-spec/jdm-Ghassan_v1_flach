#!/usr/bin/env python3
"""prepare_dataset.py - the dataset preparation (the dumb half).

It ONLY converts the format:

    Parquet / JSONL / text  ->  .txt (one document per line)  [+  .jsonl for SFT]

It NEVER tokenizes. The tokenization is done by the ENGINE
(`gai_train --build-shards`), with the engine's own tokenizer, because the
engine implements the byte-level pre-tokenizer by hand (the Granite digit
grouping, the Qwen regex, ...) and a Python tokenizer would disagree with it
on exactly the cases that matter: the numbers. A shard built with another
tokenizer is a corpus the model was never trained on, and it fails silently.

The English filter lives here too: the corpus of Ghassan v1 Flash is pure
English, and a document that is not (CJK, Arabic, Cyrillic, Devanagari...) is
dropped BEFORE it reaches the shards.

Usage:
    python tools/py/prepare_dataset.py \
        --input  english_parquet/ \
        --output artifacts/shards \
        --mode  pretrain \
        --val-fraction 0.005 \
        --max-tokens 200000000
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# the English filter
# ---------------------------------------------------------------------------

# The scripts that disqualify a document: they are not "a bit of foreign
# text", they are a document in another language.
_FOREIGN_RANGES = (
    (0x0400, 0x04FF),   # Cyrillic
    (0x0530, 0x058F),   # Armenian
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),   # Syriac
    (0x0750, 0x077F),   # Arabic supplement
    (0x0900, 0x097F),   # Devanagari
    (0x0980, 0x09FF),   # Bengali
    (0x0A00, 0x0A7F),   # Gurmukhi
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0B00, 0x0B7F),   # Oriya
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
    (0x0E00, 0x0E7F),   # Thai
    (0x0F00, 0x0FFF),   # Tibetan
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3100, 0x312F),   # Bopomofo
    (0x3130, 0x318F),   # Hangul compatibility
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified
    (0xA000, 0xA48F),   # Yi
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility
    (0xFF00, 0xFFEF),   # halfwidth/fullwidth forms
    (0x20000, 0x2A6DF), # CJK ext B
)

# The Latin letters we accept (ASCII + Latin-1 + Latin extended).
_LATIN_RANGES = (
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF),
)


def _in_ranges(cp, ranges):
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def english_stats(text):
    """Returns (latin_ratio, foreign_ratio) over the LETTERS of the text."""
    latin = 0
    foreign = 0
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        if _in_ranges(cp, _LATIN_RANGES):
            latin += 1
        elif _in_ranges(cp, _FOREIGN_RANGES):
            foreign += 1
    if letters == 0:
        return 0.0, 0.0
    return latin / letters, foreign / letters


def is_english(text, min_latin=0.92, max_foreign=0.02, min_letters=64):
    """A document is English when its letters are (almost) all Latin."""
    if not text:
        return False
    latin, foreign = english_stats(text)
    letters = sum(1 for c in text if c.isalpha())
    if letters < min_letters:
        return False
    return latin >= min_latin and foreign <= max_foreign


def clean_text(text):
    """Normalises the text: no control characters, no lone surrogates."""
    if not text:
        return ""
    # The replacement character: a broken UTF-8 in the source.
    text = text.replace("\ufffd", " ")
    out = []
    for ch in text:
        o = ord(ch)
        if o < 0x20 and ch not in "\n\t":
            out.append(" ")
            continue
        if 0xD800 <= o <= 0xDFFF:      # a lone surrogate
            out.append(" ")
            continue
        out.append(ch)
    text = "".join(out)
    # The whitespace runs: the tokenizer handles them, but a 10k space run is
    # just noise.
    text = re.sub(r"[ \t]{4,}", "   ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# the readers
# ---------------------------------------------------------------------------

def _iter_parquet(paths, text_column):
    import pyarrow.parquet as pq
    for path in paths:
        try:
            table = pq.read_table(path, columns=None)
        except Exception as exc:                      # pragma: no cover
            print(f"[prep] cannot read {path}: {exc}", file=sys.stderr)
            continue
        cols = table.column_names
        col = text_column if text_column in cols else (
            "text" if "text" in cols else (cols[0] if cols else None))
        if col is None:
            print(f"[prep] {path}: no column", file=sys.stderr)
            continue
        for value in table.column(col).to_pylist():
            if isinstance(value, str):
                yield value
            elif value is not None:
                yield str(value)


def _iter_jsonl(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, str):
                    yield obj
                elif isinstance(obj, dict):
                    for key in ("text", "content", "document", "body", "raw_content"):
                        if isinstance(obj.get(key), str):
                            yield obj[key]
                            break


def _iter_txt(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def collect_inputs(inputs, mode):
    """Returns the list of files to read, per input path."""
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for ext in (".parquet", ".jsonl", ".json", ".txt"):
                files.extend(sorted(str(x) for x in p.rglob(f"*{ext}")))
        elif p.is_file():
            files.append(str(p))
        else:
            print(f"[prep] no such input: {item}", file=sys.stderr)
    return files


def read_documents(files, text_column):
    parquet = [f for f in files if f.endswith(".parquet")]
    jsonl = [f for f in files if f.endswith((".jsonl", ".json"))]
    txt = [f for f in files if f.endswith(".txt")]
    for value in _iter_parquet(parquet, text_column):
        yield value
    for value in _iter_jsonl(jsonl):
        yield value
    for value in _iter_txt(txt):
        yield value


def split_document(text, max_chars):
    """Splits a long document into the pieces that become the lines."""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    cur = []
    length = 0
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if length + len(para) + 1 > max_chars and cur:
            pieces.append("\n".join(cur))
            cur, length = [], 0
        cur.append(para)
        length += len(para) + 1
    if cur:
        pieces.append("\n".join(cur))
    return pieces


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="the Parquet/JSONL/text files or directories")
    ap.add_argument("--output", required=True, help="the output directory")
    ap.add_argument("--text-column", default="text",
                    help="the column that holds the text (Parquet)")
    ap.add_argument("--mode", choices=("pretrain", "sft"), default="pretrain")
    ap.add_argument("--min-latin", type=float, default=0.92,
                    help="the minimum ratio of Latin letters (the English filter)")
    ap.add_argument("--max-foreign", type=float, default=0.02,
                    help="the maximum ratio of the other scripts")
    ap.add_argument("--max-doc-chars", type=int, default=24000,
                    help="a document longer than this is split")
    ap.add_argument("--val-fraction", type=float, default=0.005,
                    help="the fraction of the documents held out for the validation")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="stop after this many tokens (0 = no limit)")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="stop after this many documents (0 = no limit)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(args.input, args.mode)
    print(f"[prep] {len(files)} input file(s)", file=sys.stderr)

    train_f = open(out / "train.txt", "w", encoding="utf-8")
    val_f = open(out / "val.txt", "w", encoding="utf-8")

    kept = 0
    dropped = 0
    tokens = 0            # an approximation: the characters / 4
    docs_written = 0
    val_docs = 0

    for raw in read_documents(files, args.text_column):
        if args.max_docs and kept >= args.max_docs:
            break
        text = clean_text(raw)
        if not text:
            continue
        if not is_english(text, args.min_latin, args.max_foreign):
            dropped += 1
            continue
        kept += 1
        for piece in split_document(text, args.max_doc_chars):
            # The line must not contain a newline: one document per line.
            piece = piece.replace("\n", " ").strip()
            if len(piece) < 64:
                continue
            if args.max_tokens and tokens >= args.max_tokens:
                break
            is_val = rng.random() < args.val_fraction
            (val_f if is_val else train_f).write(piece + "\n")
            docs_written += 1
            tokens += len(piece) // 4
            if is_val:
                val_docs += 1

    train_f.close()
    val_f.close()

    print(f"[prep] kept {kept} document(s), dropped {dropped} (not English)", file=sys.stderr)
    print(f"[prep] wrote {docs_written} line(s) ({val_docs} for the validation) "
          f"~{tokens} tokens -> {out}", file=sys.stderr)
    if docs_written == 0:
        print("[prep] NOTHING was written: check --input and the filter", file=sys.stderr)
        return 1
    print("[prep] next: gai_train --build-shards <dir> --tokenizer <hf-dir> "
          "--shard-prefix train", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
