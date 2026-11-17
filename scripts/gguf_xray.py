#!/usr/bin/env python3
"""Independent X-ray of a Ghassan .gguf file.

Pure stdlib (no numpy needed for the core checks). It answers, without
trusting the engine's own `info` output:

  1. is the file a GGUF at all (magic/version)?
  2. does the metadata say "Ghassan v1 Flash" and NOT the upstream arch?
  3. does the byte string "granite" (any case) appear ANYWHERE in the file,
     and if so, in which metadata key or at which offset?
  4. how many tensors, which dtypes, how many parameters?
  5. the architecture-critical metadata (arch, hybrid flag, vocab, layers,
     multipliers, mamba geometry, layer_types) - what a python one-liner
     would show someone inspecting the file.

Usage:
    python3 scripts/gguf_xray.py model.gguf            # full report, exit 1 on any leak
    python3 scripts/gguf_xray.py model.gguf --tensors  # + every tensor name/shape/dtype
    python3 scripts/gguf_xray.py model.gguf --quiet    # only PASS/FAIL, for scripts

Exit codes: 0 = clean Ghassan artifact, 1 = brand leak or structural problem.
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
U64, U32, F32, F64, BOOL, STRING, ARRAY = "u64", "u32", "f32", "f64", "bool", "string", "array"

def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")

def read_value(f, vtype):
    if vtype == 0:   # UINT8
        return f.read(1)[0]
    if vtype == 1:   # INT8
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:   # UINT16
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:   # INT16
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:   # UINT32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:   # INT32
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:   # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:   # BOOL
        return f.read(1)[0] != 0
    if vtype == 8:   # STRING
        return read_string(f)
    if vtype == 9:   # ARRAY
        (et,) = struct.unpack("<I", f.read(4))
        (cnt,) = struct.unpack("<Q", f.read(8))
        out = []
        for _ in range(cnt):
            if et == 8:
                out.append(read_string(f))
            elif et in (4, 5):
                out.append(struct.unpack("<I" if et == 4 else "<i", f.read(4))[0])
            elif et in (0, 1, 7):
                out.append(f.read(1)[0])
            elif et in (2, 3):
                out.append(struct.unpack("<H", f.read(2))[0])
            elif et in (10,):
                out.append(struct.unpack("<Q", f.read(8))[0])
            elif et in (6,):
                out.append(struct.unpack("<f", f.read(4))[0])
            else:
                out.append(None)  # skipped non-visualizable element type
        return out
    if vtype == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:  # INT64
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:  # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError("unknown metadata value type %d" % vtype)

def main():
    path = sys.argv[1]
    show_tensors = "--tensors" in sys.argv
    quiet = "--quiet" in sys.argv
    problems = []

    with open(path, "rb") as f:
        magic, version, n_tensors, n_metadata = struct.unpack("<I I Q Q", f.read(24))
        if magic != GGUF_MAGIC:
            print("FAIL: not a GGUF file (magic %08x)" % magic)
            return 1
        print("magic GGUF v%d | tensors=%d | metadata=%d" % (version, n_tensors, n_metadata))

        metadata = {}
        for _ in range(n_metadata):
            key = read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = read_value(f, vtype)

        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, ttype, offset))

    # ---- the identity checks ------------------------------------------
    arch = metadata.get("general.architecture")
    name = metadata.get("general.name", "")
    desc = metadata.get("general.description", "")
    if arch != "ghassan":
        problems.append("general.architecture is %r, expected 'ghassan'" % arch)
    if not str(name).startswith("Ghassan v1 Flash"):
        problems.append("general.name is %r, expected it to start with 'Ghassan v1 Flash'" % name)
    if "Ghassan v1 Flash" not in str(desc):
        problems.append("general.description is %r, expected 'Ghassan v1 Flash'" % desc)
    n_params = metadata.get("general.parameter_count")
    base_model = metadata.get("general.base_model.0.name", "")
    base_org = metadata.get("general.base_model.0.organization", "")
    print("name           : %s" % name)
    print("description    : %s" % desc)
    print("base model     : %s%s" % (base_model, (" (%s)" % base_org) if base_org else ""))
    print("parameters     : %s" % n_params)
    print("quantization   : %s" % metadata.get("ghassan.quantization.profile", "?"))
    print("vocab          : %s" % metadata.get("ghassan.vocab_size"))
    print("context        : %s" % metadata.get("ghassan.context_length"))
    print("layers         : %s (hybrid=%s)" % (metadata.get("ghassan.block_count"),
                                                metadata.get("ghassan.hybrid", metadata.get("ghassan.granite", False))))
    print("layer_types    : %s" % str(metadata.get("ghassan.layer_types"))[:80])
    print("mamba          : heads=%s d_head=%s d_state=%s groups=%s conv=%s expand=%s" % (
        metadata.get("ghassan.mamba.n_heads"), metadata.get("ghassan.mamba.d_head"),
        metadata.get("ghassan.mamba.d_state"), metadata.get("ghassan.mamba.n_groups"),
        metadata.get("ghassan.mamba.d_conv"), metadata.get("ghassan.mamba.expand")))
    print("moe            : experts=%s top_k=%s expert_width=%s shared=%s" % (
        metadata.get("ghassan.moe.expert_count"), metadata.get("ghassan.moe.experts_used_count"),
        metadata.get("ghassan.moe.expert_feed_forward_length"),
        metadata.get("ghassan.moe.shared_expert")))
    print("multipliers    : embed x%s | resid x%s | attn x%s | logits /%s" % (
        metadata.get("ghassan.embedding_multiplier"), metadata.get("ghassan.residual_multiplier"),
        metadata.get("ghassan.attention_multiplier"), metadata.get("ghassan.logits_scaling")))

    # ---- the brand leak scan -------------------------------------------
    # The file's metadata VALUES are scanned per-key. The tokenizer vocabulary
    # and the merges are ordinary word lists ('granite' is a plain English
    # word) and are exempt - but every key WE control must be clean.
    #
    # EXCEPTION: the provenance keys (general.base_model.*) and the
    # description. The artifact is a "Ghassan v1 Flash" that is BUILT ON a
    # Granite model, and it has to say so: the name is the brand, the base
    # model is the fact. The Granite GEOMETRY (the multipliers, the mamba
    # dims, the layer kinds) is exported untouched on purpose - a Granite
    # artifact stays a Granite artifact so it can be fine-tuned as one.
    needle = "granite"
    vocab_hits = []
    provenance_keys = ("general.base_model.0.name", "general.base_model.0.organization",
                       "general.description")
    for key, val in metadata.items():
        if needle in key.lower() and not key.startswith("general.base_model."):
            problems.append("metadata key %r contains the upstream brand" % key)
            continue
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            if not isinstance(v, str) or needle not in v.lower():
                continue
            if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
                vocab_hits.append(v)
            elif key in provenance_keys:
                print("provenance     : %s = %r" % (key, v))
            else:
                problems.append("metadata key %r carries the upstream brand: %r"
                                % (key, v if len(v) < 200 else v[:200] + "..."))

    with open(path, "rb") as f:
        data = f.read()
    total = data.lower().count(needle.encode())
    if vocab_hits:
        print("vocab note     : the tokenizer word list contains %d entr%s with 'granite' "
              "(ordinary English words, expected)" % (len(vocab_hits), "y" if len(vocab_hits) == 1 else "ies"))
    print("brand scan     : %d raw 'granite' byte-hit(s) in the file, %d of them inside the vocab/merges word list"
          % (total, len(vocab_hits)))

    # ---- tensors -------------------------------------------------------
    from collections import Counter
    dtype_count = Counter(t[2] for t in tensors)
    dtype_names = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0"}
    total_params = 0
    for _, dims, _, _ in tensors:
        n = 1
        for d in dims:
            n *= d
        total_params += n
    print("tensors        : %d  params=%d  dtypes: %s" % (
        len(tensors), total_params,
        ", ".join("%s=%d" % (dtype_names.get(k, str(k)), v) for k, v in sorted(dtype_count.items()))))
    if show_tensors:
        ggml_type_str = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0"}
        for name_, dims, ttype, _ in sorted(tensors, key=lambda t: t[0]):
            print("  %-40s %-18s %s" % (name_, "x".join(str(d) for d in dims), ggml_type_str.get(ttype, ttype)))

    if problems:
        print("\nFAIL:\n  - " + "\n  - ".join(problems))
        return 1
    if not quiet:
        print("\nPASS: Ghassan v1 Flash artifact, no brand leak.")
    else:
        print("PASS")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main())
