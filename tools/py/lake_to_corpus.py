#!/usr/bin/env python3
"""Dump a parquet lake into a plain text corpus.

The BPE tokenizer trainer reads one document per line, so the lake has to
be flattened first. This is the replacement for the C++ `data_pipeline
parquet-corpus` step that went away with the `dataset/` module.

    python tools/py/lake_to_corpus.py \
        --lake /kaggle/input/english_parquet \
        --out artifacts/corpus/corpus_en.txt \
        --limit 400000

Every row is written as one line: a chat row becomes
"user: ...\\nassistant: ...", an instruction row becomes the instruction
and its answer, and a plain text row becomes its text. The output is
deterministic (the rows are read in file order), so a rerun produces the
same corpus and the same tokenizer.
"""
import argparse
import glob
import json
import os
import sys


def _rows(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    cols = table.column_names
    for batch in table.to_batches():
        data = {c: batch.column(c).to_pylist() for c in cols}
        for i in range(batch.num_rows):
            yield {c: data[c][i] for c in cols}


def _get(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return str(v)
    return None


def row_to_text(row):
    """One row -> one line (newlines collapsed)."""
    q = _get(row, "question", "prompt", "instruction", "query", "user", "human")
    a = _get(row, "answer", "response", "output", "completion", "assistant", "gpt", "bot")

    msgs = row.get("messages_json") or row.get("messages")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = None
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if isinstance(m, dict) and "role" in m and "content" in m:
                parts.append(f"{m['role']}: {m['content']}")
        if parts:
            return "\n".join(parts)

    if q and a:
        return f"user: {q}\nassistant: {a}"

    skip = {"messages_json", "messages", "conversation", "uuid", "id", "source", "category"}
    chunks = []
    for k in sorted(row.keys()):
        if k.startswith("_") or k in skip:
            continue
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            chunks.append(v.strip())
    if chunks:
        return " / ".join(chunks)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", required=True, help="directory with the parquet lake")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max lines (0 = no limit)")
    ap.add_argument("--min-chars", type=int, default=8,
                    help="drop the lines shorter than this")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.lake, "**", "*.parquet"), recursive=True))
    if not files:
        print(f"no parquet file under {args.lake}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written = 0
    skipped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path in files:
            for row in _rows(path):
                text = row_to_text(row)
                text = " ".join(text.split())
                if len(text) < args.min_chars:
                    skipped += 1
                    continue
                f.write(text + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    print(f"=== done: {written} lines (limit reached) -> {args.out}")
                    print(f"  files: {len(files)} | skipped (too short): {skipped}")
                    return 0

    print(f"=== done: {written} lines -> {args.out}")
    print(f"  files: {len(files)} | skipped (too short): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
