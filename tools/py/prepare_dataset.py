#!/usr/bin/env python3
"""Build training shards from a corpus, in Python.

This replaces the C++ `data_pipeline` tool (removed with the `dataset/`
module) and is what the Kaggle cells call. It reads every input format the
project ever supported:

    parquet   the "lake": english_parquet/english_chat_part*.parquet,
              english_parquet/english_instruction_part*.parquet
    jsonl     {"question": ..., "answer": ...} or {"messages": [...]}
    json      a list of the above, or {"data": [...]}
    txt       one document per line

and writes

    <out>/train_<domain>_NNNN.gbin     tokenized shards (Gai shard format)
    <out>/val_<domain>_NNNN.gbin
    <out>/train.jsonl                  the same data for a PyTorch trainer
    <out>/prepare_report.json           the statistics

The shard format is byte compatible with training/dataloader.cpp, so the
C++ trainer, the perplexity tool and the evaluation suite can read what
this writes, and the Python trainer can read train.jsonl.

    python tools/py/prepare_dataset.py \
        --lake /kaggle/input/english_parquet \
        --tokenizer /kaggle/input/tokenizer \
        --out artifacts/shards_en \
        --val-ratio 0.005
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import sys
from collections import Counter

GBIN_MAGIC = 0x4E494247
GBIN_VERSION = 1
GBIN_FLAG_MASK = 1


# ---------------------------------------------------------------------------
# shard writing (byte compatible with training/dataloader.cpp)
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, path, vocab_size, with_mask=True):
        self.path = path
        self.vocab_size = vocab_size
        self.with_mask = with_mask
        self.u16 = vocab_size <= 65535
        self.tokens = []
        self.mask = []
        self.doc_offsets = []
        self.closed = False

    def add_document(self, tokens, mask=None):
        if not tokens:
            return
        if self.with_mask:
            if mask is None:
                mask = [1] * len(tokens)
            if len(mask) != len(tokens):
                raise ValueError("loss mask length != token length")
            for m in mask:
                if m not in (0, 1):
                    raise ValueError(f"loss mask must be 0/1, got {m}")
        for t in tokens:
            if t < 0 or t >= self.vocab_size:
                raise ValueError(f"token id {t} outside the vocabulary [0,{self.vocab_size})")
        self.doc_offsets.append(len(self.tokens))
        self.tokens.extend(tokens)
        if self.with_mask:
            self.mask.extend(mask if mask is not None else [1] * len(tokens))

    def close(self):
        if self.closed:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        header = struct.pack("<IIIIQQ", GBIN_MAGIC, GBIN_VERSION,
                             0 if self.u16 else 1,
                             GBIN_FLAG_MASK if self.with_mask else 0,
                             len(self.tokens), len(self.doc_offsets))
        with open(self.path, "wb") as f:
            f.write(header)
            if self.u16:
                f.write(struct.pack(f"<{len(self.tokens)}H", *self.tokens))
            else:
                f.write(struct.pack(f"<{len(self.tokens)}I", *self.tokens))
            f.write(struct.pack(f"<{len(self.doc_offsets)}Q", *self.doc_offsets))
            if self.with_mask:
                f.write(bytes(self.mask))
        self.closed = True


# ---------------------------------------------------------------------------
# corpus reading
# ---------------------------------------------------------------------------

def _rows_from_parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required for parquet input: pip install pyarrow", file=sys.stderr)
        raise
    table = pq.read_table(path)
    cols = table.column_names
    for batch in table.to_batches():
        data = {c: batch.column(c).to_pylist() for c in cols}
        for i in range(batch.num_rows):
            yield {c: data[c][i] for c in cols}


def _doc_from_row(row):
    """Turns a table row into a list of chat messages or a text document.

    Returns ("chat", [(role, content), ...]) or ("text", str).
    """
    def get(*names):
        for n in names:
            if n in row and row[n] not in (None, ""):
                return row[n]
        return None

    q = get("question", "prompt", "instruction", "query")
    a = get("answer", "response", "output", "completion")
    if q and a:
        return "chat", [("user", str(q)), ("assistant", str(a))]

    msgs = get("messages_json", "messages", "conversation")
    if msgs:
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                msgs = None
        if isinstance(msgs, list):
            out = []
            for m in msgs:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    out.append((str(m["role"]), str(m["content"])))
            if out:
                return "chat", out

    u = get("user", "human")
    asst = get("assistant", "gpt", "bot")
    if u and asst:
        return "chat", [("user", str(u)), ("assistant", str(asst))]

    sys_msg = get("system")
    parts = []
    if sys_msg:
        parts.append(("system", str(sys_msg)))
    if q:
        parts.append(("user", str(q)))
    if a:
        parts.append(("assistant", str(a)))
    if parts:
        return "chat", parts

    # generic text: join the non-empty columns
    skip = {"messages_json", "messages", "conversation", "uuid", "id", "source", "category"}
    chunks = []
    for k in sorted(row.keys()):
        if k.startswith("_") or k in skip:
            continue
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            chunks.append(v.strip())
    if chunks:
        return "text", " / ".join(chunks)
    return None, None


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, list):
                for o in obj:
                    yield o
            else:
                yield obj


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ("data", "rows", "examples", "items"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            obj = [obj]
    for o in obj:
        yield o


def _obj_to_doc(obj):
    if isinstance(obj, str):
        return "text", obj
    if not isinstance(obj, dict):
        return None, None
    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = []
        for m in obj["messages"]:
            if isinstance(m, dict) and "role" in m:
                msgs.append((str(m["role"]), str(m.get("content", ""))))
        if msgs:
            return "chat", msgs
    if "question" in obj and "answer" in obj:
        return "chat", [("user", str(obj["question"])), ("assistant", str(obj["answer"]))]
    if "text" in obj:
        return "text", str(obj["text"])
    if "content" in obj:
        return "text", str(obj["content"])
    return _doc_from_row(obj)


def collect_docs(args):
    """Yields (domain, kind, payload)."""
    sources = []

    if args.lake:
        for pattern in ("*chat*.parquet", "*instruction*.parquet", "*.parquet"):
            for p in sorted(glob.glob(os.path.join(args.lake, "**", pattern), recursive=True)):
                if p not in sources:
                    sources.append(("lake", p))
    if args.input:
        for p in args.input:
            if os.path.isdir(p):
                for ext in ("*.parquet", "*.jsonl", "*.json", "*.txt"):
                    for f in sorted(glob.glob(os.path.join(p, "**", ext), recursive=True)):
                        sources.append(("input", f))
            else:
                sources.append(("input", p))

    for kind, path in sources:
        name = os.path.basename(path)
        if path.endswith(".parquet"):
            domain = "chat" if "chat" in name else ("instruction" if "instruction" in name else "text")
            for row in _rows_from_parquet(path):
                d = _doc_from_row(row)
                if d[0]:
                    yield domain, d[0], d[1]
        elif path.endswith(".jsonl"):
            domain = "qa" if "qa" in name else "chat"
            for obj in read_jsonl(path):
                d = _obj_to_doc(obj)
                if d[0]:
                    yield domain, d[0], d[1]
        elif path.endswith(".json"):
            domain = "qa" if "qa" in name else "chat"
            for obj in read_json(path):
                d = _obj_to_doc(obj)
                if d[0]:
                    yield domain, d[0], d[1]
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield "text", "text", line


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------

class TokenizerBundle:
    """Wraps the HuggingFace tokenizer and the loss mask computation."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.vocab_size = len(self.tok)
        self.template = None
        cfg_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.template = json.load(f).get("chat_template")

    def encode_text(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def render(self, msgs):
        return self.tok.apply_chat_template(msgs, tokenize=False)

    def encode_chat(self, messages):
        """Returns (ids, mask). The mask is 1 on the assistant spans.

        The chat template is rendered with the reference tokenizer, then the
        assistant character spans are mapped to token spans through the
        offset mapping. That keeps the mask aligned with whatever template
        the checkpoint declares.
        """
        msgs = [{"role": r, "content": c} for r, c in messages]
        text = self.render(msgs)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc.get("offset_mapping", [])
        mask = [0] * len(ids)
        if not offsets:
            return ids, [1] * len(ids)

        # The character span of every assistant message, found by rendering
        # the conversation prefix by prefix.
        spans = []
        before = ""
        for i, (role, content) in enumerate(messages):
            if role == "assistant":
                try:
                    upto = self.render(msgs[:i + 1])
                except Exception:
                    upto = before
                start = upto.rfind(content, before) if content else -1
                if start >= 0:
                    spans.append((start, start + len(content)))
            try:
                before = self.render(msgs[:i + 1])
            except Exception:
                pass

        for (a, b) in spans:
            for t, (s, e) in enumerate(offsets):
                if e > s and s >= a and e <= b:
                    mask[t] = 1
        if not any(mask):
            # No span could be located: supervise everything rather than
            # training on nothing.
            mask = [1] * len(ids)
        return ids, mask


def split_key(text):
    """Deterministic train/val split key (hash of the normalised text)."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def is_val(key, val_ratio):
    return (int(key[:16], 16) % 10000) < int(val_ratio * 10000)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default=None, help="directory with the parquet lake")
    ap.add_argument("--input", action="append", default=[],
                    help="a corpus file or directory (repeatable)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.005)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--domain", default=None, help="force the domain name")
    ap.add_argument("--jsonl-out", default=None,
                    help="also write the tokenized documents as jsonl "
                         "(default: <out>/train.jsonl)")
    args = ap.parse_args()

    if not 0.0 <= args.val_ratio <= 1.0:
        raise SystemExit("--val-ratio must be in [0,1]")

    print("=== tokenizer ===")
    tb = TokenizerBundle(args.tokenizer)
    print(f"  vocab {tb.vocab_size} | chat template: "
          f"{'yes' if tb.template else 'no (default chat template)'}")

    os.makedirs(args.out, exist_ok=True)
    jsonl_path = args.jsonl_out or os.path.join(args.out, "train.jsonl")

    writers = {}
    stats = Counter()
    per_domain = Counter()
    jsonl = open(jsonl_path, "w", encoding="utf-8")

    def writer_for(domain, split):
        key = (domain, split)
        if key in writers:
            return writers[key]
        idx = sum(1 for k in writers if k[0] == domain and k[1] == split)
        path = os.path.join(args.out, f"{split}_{domain}_{idx:04d}.gbin")
        w = ShardWriter(path, tb.vocab_size, with_mask=True)
        writers[key] = w
        return w

    print("=== reading the corpus ===")
    n_docs = 0
    for domain, kind, payload in collect_docs(args):
        if args.max_docs and n_docs >= args.max_docs:
            break
        if args.domain:
            domain = args.domain

        if kind == "chat":
            ids, mask = tb.encode_chat(payload)
            text = "".join(payload)
        else:
            ids = tb.encode_text(payload)
            mask = [1] * len(ids)
            text = payload

        if not ids:
            continue

        split = "val" if is_val(split_key(text), args.val_ratio) else "train"
        writer_for(domain, split).add_document(ids, mask)
        stats[f"{split}_docs"] += 1
        stats[f"{split}_tokens"] += len(ids)
        per_domain[domain] += 1
        n_docs += 1

        jsonl.write(json.dumps({
            "domain": domain,
            "split": split,
            "ids": ids,
            "mask": mask,
        }, ensure_ascii=False) + "\n")

        # Rotate the shard when it grows past the limit.
        for (d, s), w2 in list(writers.items()):
            if len(w2.tokens) >= args.shard_tokens:
                w2.close()
                stats["shards"] += 1
                idx = sum(1 for k in writers if k[0] == d and k[1] == s)
                writers[(d, s)] = ShardWriter(
                    os.path.join(args.out, f"{s}_{d}_{idx:04d}.gbin"),
                    tb.vocab_size, with_mask=True)

        if n_docs % 20000 == 0:
            print(f"  {n_docs} docs, {stats['train_tokens']} train tokens", flush=True)

    jsonl.close()
    for w in writers.values():
        w.close()
        stats["shards"] += 1

    report = {
        "documents": n_docs,
        "by_domain": dict(per_domain),
        "train_docs": stats["train_docs"],
        "val_docs": stats["val_docs"],
        "train_tokens": stats["train_tokens"],
        "val_tokens": stats["val_tokens"],
        "shards": stats["shards"],
        "vocab_size": tb.vocab_size,
        "val_ratio": args.val_ratio,
    }
    with open(os.path.join(args.out, "prepare_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== done ===")
    print(f"  documents        : {report['documents']}  {report['by_domain']}")
    print(f"  train            : {report['train_docs']} docs / {report['train_tokens']} tokens")
    print(f"  val              : {report['val_docs']} docs / {report['val_tokens']} tokens")
    print(f"  shards           : {report['shards']}")
    print(f"  tokenized jsonl  : {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
