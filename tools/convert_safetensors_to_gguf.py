#!/usr/bin/env python3
"""Convert a HuggingFace safetensors checkpoint into a GGUF the Gai engine
can serve.

The Gai engine reads GGUF and nothing else, so this is the bridge between
"a strong brain from HuggingFace" and "the body that serves it".

What matters here (and what a naive converter gets wrong):

* the tokenizer must be reconstructible by the C++ engine, so the byte-level
  alphabet and the control token ids are written explicitly;
* MoE expert tensors must be 3D with llama.cpp's exact layout, otherwise
  the engine reads them in the wrong order and silently produces garbage;
* the architecture description must be complete (expert count, expert feed
  forward length, ...) or the engine sees a dense model.

Usage:
    python tools/convert_safetensors_to_gguf.py \
        --model-dir safetensors \
        --out artifacts/ghassan-base-f16.gguf
"""
import argparse
import json
import os
import re
import struct
import sys

try:
    import numpy as np
except ImportError:
    print("numpy is required", file=sys.stderr)
    raise

try:
    import gguf
except ImportError:
    print("the 'gguf' package is required: pip install gguf", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# safetensors reading
# ---------------------------------------------------------------------------

def read_safetensors_raw(path):
    """Returns {name: (raw_bytes, shape, dtype)} without numpy."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        tensors = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            begin, end = v["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            tensors[k] = (raw, v["shape"], v["dtype"])
        return tensors


def load_checkpoint(model_dir):
    """Loads every safetensors shard of a checkpoint directory."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = [n for n in sorted(os.listdir(model_dir))
                  if n.endswith(".safetensors")]
        if not shards:
            raise SystemExit(f"no .safetensors found in {model_dir}")

    tensors = {}
    for shard in shards:
        path = shard if os.path.isabs(shard) else os.path.join(model_dir, shard)
        print(f"  reading {os.path.basename(path)}")
        for name, (raw, shape, dtype) in read_safetensors_raw(path).items():
            tensors[name] = (raw, shape, dtype)
    return tensors


# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------

class Arch:
    def __init__(self, cfg):
        self.hidden = cfg.get("hidden_size", 960)
        self.layers = cfg.get("num_hidden_layers", 32)
        self.heads = cfg.get("num_attention_heads", 15)
        self.kv_heads = cfg.get("num_key_value_heads", self.heads)
        self.vocab = cfg.get("vocab_size", 49152)
        self.inter = cfg.get("intermediate_size", 2560)
        self.ctx = cfg.get("max_position_embeddings", 8192)
        self.eps = cfg.get("rms_norm_eps", 1e-5)
        self.theta = float(cfg.get("rope_theta", 10000.0))
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        self.head_dim = self.hidden // self.heads if self.heads else 64

        # MoE (Mixtral / Qwen3 style)
        self.experts = 0
        self.top_k = 0
        self.expert_dim = 0
        if "num_local_experts" in cfg:            # Mixtral
            self.experts = cfg["num_local_experts"]
            self.top_k = cfg.get("num_experts_per_tok", 2)
            self.expert_dim = self.inter
        elif "num_experts" in cfg:                # Qwen3 MoE
            self.experts = cfg["num_experts"]
            self.top_k = cfg.get("num_experts_per_tok", 8)
            self.expert_dim = cfg.get("moe_intermediate_size",
                                      cfg.get("intermediate_size", 1408))

        self.qk_norm = False


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def read_tokenizer(model_dir):
    path = os.path.join(model_dir, "tokenizer.json")
    with open(path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]

    # The vocab dict is token -> id; json objects have no order guarantee, so
    # build the list by id.
    ordered = [None] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        ordered[idx] = tok
    if any(t is None for t in ordered):
        raise SystemExit("tokenizer vocabulary has holes")

    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    return ordered, merges, cfg, tj.get("added_tokens", [])


def byte_map_for(ordered):
    """byte -> token id for a byte-level vocabulary."""
    enc = bytes_to_unicode()
    index = {t: i for i, t in enumerate(ordered)}
    out = []
    missing = []
    for b in range(256):
        tid = index.get(enc[b])
        if tid is None:
            missing.append(b)
            out.append(0xFFFFFFFF)
        else:
            out.append(tid)
    return out, missing


def _common_suffix_len(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[len(a) - 1 - n] == b[len(b) - 1 - n]:
        n += 1
    return n


def extract_default_system(tok):
    """Extract the system message a chat template injects by itself.

    SmolLM2, Qwen and friends start the prompt with a fixed system message
    when the conversation does not carry one. The engine reproduces the
    prompt, so it needs to know that message.
    """
    try:
        with_sys = tok.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            tokenize=False)
        without = tok.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False)
    except Exception:
        return ""
    if not isinstance(with_sys, str) or not isinstance(without, str):
        return ""
    n = _common_suffix_len(with_sys, without)
    prefix_without = without[:len(without) - n]
    prefix_with = with_sys[:len(with_sys) - n]
    if prefix_without == prefix_with:
        return ""
    i = prefix_without.find("system")
    if i < 0:
        return ""
    j = prefix_without.find("\n", i)
    if j < 0:
        return ""
    content = prefix_without[j + 1:]
    k = content.rfind("<|im_end|>")
    if k >= 0:
        content = content[:k]
    return content


# ---------------------------------------------------------------------------
# tensor mapping
# ---------------------------------------------------------------------------

DENSE_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

QK_NORM_MAP = {
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
}

MOE_ROUTER_MAP = {
    "block_sparse_moe.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
}

# HF expert leaf -> llama.cpp expert tensor
EXPERT_LEAF = {
    "w1": "ffn_gate_exps.weight",
    "gate_proj": "ffn_gate_exps.weight",
    "w3": "ffn_up_exps.weight",
    "up_proj": "ffn_up_exps.weight",
    "w2": "ffn_down_exps.weight",
    "down_proj": "ffn_down_exps.weight",
}

EXPERT_RE = re.compile(
    r"^(?:block_sparse_moe|mlp)\.experts\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)\.weight$")


def plan_tensors(tensors):
    """Returns {gguf_name: [(hf_name, kind), ...]}.

    kind is "plain" for a single tensor and "expert:<id>" for one block of a
    3D expert tensor.
    """
    plan = {}

    def add(gguf_name, hf_name, kind):
        plan.setdefault(gguf_name, []).append((hf_name, kind))

    for name in tensors:
        if name == "model.embed_tokens.weight":
            add("token_embd.weight", name, "plain")
            continue
        if name == "model.norm.weight":
            add("output_norm.weight", name, "plain")
            continue
        if name == "lm_head.weight":
            add("output.weight", name, "plain")
            continue
        if not name.startswith("model.layers."):
            continue
        rest = name[len("model.layers."):]
        layer, _, sub = rest.partition(".")
        blk = f"blk.{layer}."
        if sub in DENSE_MAP:
            add(blk + DENSE_MAP[sub], name, "plain")
            continue
        if sub in QK_NORM_MAP:
            add(blk + QK_NORM_MAP[sub], name, "plain")
            continue
        if sub in MOE_ROUTER_MAP:
            add(blk + MOE_ROUTER_MAP[sub], name, "plain")
            continue
        m = EXPERT_RE.match(sub)
        if m:
            eid = int(m.group(1))
            add(blk + EXPERT_LEAF[m.group(2)], name, f"expert:{eid}")
            continue
        print(f"  [warn] unmapped tensor {name}")

    return plan


def to_array(raw, shape, dtype, is_norm):
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return arr if is_norm else arr.astype(np.float16)
    if dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        return arr.astype(np.float32) if is_norm else arr
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return arr if is_norm else arr.astype(np.float16)
    raise SystemExit(f"unsupported dtype {dtype}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    arch = Arch(cfg)

    print("=== reading checkpoint ===")
    tensors = load_checkpoint(args.model_dir)
    print(f"  {len(tensors)} tensors")
    arch.qk_norm = any(".self_attn.q_norm.weight" in n for n in tensors)

    ordered, merges, tokcfg, added = read_tokenizer(args.model_dir)
    bmap, missing_bytes = byte_map_for(ordered)
    print("=== tokenizer ===")
    print(f"  vocab {len(ordered)} | merges {len(merges)} | "
          f"missing byte-level entries {len(missing_bytes)}")

    writer = gguf.GGUFWriter(args.out, arch="llama")
    writer.add_name(args.name or "Ghassan")
    writer.add_description("Ghassan language model (converted from safetensors)")
    writer.add_author("Ghassan AI Studio")
    writer.add_file_type(1)  # F16

    writer.add_context_length(arch.ctx)
    writer.add_embedding_length(arch.hidden)
    writer.add_block_count(arch.layers)
    writer.add_feed_forward_length(arch.expert_dim if arch.experts else arch.inter)
    writer.add_head_count(arch.heads)
    writer.add_head_count_kv(arch.kv_heads)
    writer.add_layer_norm_rms_eps(arch.eps)
    writer.add_rope_freq_base(arch.theta)
    writer.add_rope_dimension_count(arch.head_dim)
    writer.add_uint32("llama.vocab_size", arch.vocab)
    writer.add_bool("llama.tie_word_embeddings", arch.tied)

    if arch.experts:
        writer.add_uint32("llama.expert_count", arch.experts)
        writer.add_uint32("llama.expert_used_count", arch.top_k)
        writer.add_uint32("llama.expert_feed_forward_length", arch.expert_dim)
        print(f"=== MoE: {arch.experts} experts, top-{arch.top_k}, E={arch.expert_dim} ===")

    # ---- tokenizer -------------------------------------------------------
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list([t.encode("utf-8") for t in ordered])
    writer.add_token_scores([0.0] * len(ordered))

    types = [gguf.TokenType.NORMAL] * len(ordered)
    for t in added:
        tid = t.get("id")
        if tid is not None and 0 <= tid < len(types):
            types[tid] = gguf.TokenType.CONTROL
    writer.add_token_types(types)

    def esc(piece):
        return piece.replace("\\", "\\\\").replace(" ", "\\x20")

    writer.add_token_merges([f"{esc(a)} {esc(b)}" for a, b in merges])

    bos = tokcfg.get("bos_token_id")
    eos = tokcfg.get("eos_token_id")
    unk = tokcfg.get("unk_token_id")
    pad = tokcfg.get("pad_token_id", eos)
    writer.add_bos_token_id(bos if bos is not None else 0)
    writer.add_eos_token_id(eos if eos is not None else 0)
    writer.add_unk_token_id(unk if unk is not None else 0)
    writer.add_pad_token_id(pad if pad is not None else 0)
    writer.add_bool("tokenizer.ggml.add_bos_token",
                    bool(tokcfg.get("add_bos_token", False)))
    writer.add_bool("tokenizer.ggml.add_eos_token",
                    bool(tokcfg.get("add_eos_token", False)))

    # The engine needs the byte-level alphabet to encode against. It can
    # rebuild it from the vocabulary, but writing it explicitly makes the
    # contract explicit and lets the engine validate the file.
    writer.add_array("ghassan.byte_map", bmap)

    chat_template = tokcfg.get("chat_template")
    if chat_template:
        writer.add_chat_template(chat_template)

    # The system message the template injects by itself.
    default_system = ""
    try:
        from transformers import AutoTokenizer
        probe = AutoTokenizer.from_pretrained(args.model_dir)
        default_system = extract_default_system(probe)
    except Exception as exc:
        print(f"  [warn] could not probe the default system message: {exc}")
    if default_system:
        print(f"  default system message: {default_system[:60]!r}")
    writer.add_string("ghassan.default_system", default_system)

    # ---- tensors ---------------------------------------------------------
    print("=== writing tensors ===")
    plan = plan_tensors(tensors)

    for gname, entries in plan.items():
        if entries[0][1].startswith("expert:"):
            continue
        hf_name = entries[0][0]
        raw, shape, dtype = tensors[hf_name]
        arr = to_array(raw, shape, dtype, "norm" in hf_name)
        writer.add_tensor(gname, arr)

    # Stack the expert tensors into 3D blocks (llama.cpp layout).
    for gname, entries in plan.items():
        if not entries[0][1].startswith("expert:"):
            continue
        entries.sort(key=lambda kv: int(kv[1].split(":")[1]))
        blocks = []
        for hf_name, kind in entries:
            raw, shape, dtype = tensors[hf_name]
            blocks.append(to_array(raw, shape, dtype, False))
        if len(blocks) != arch.experts:
            raise SystemExit(f"{gname}: expected {arch.experts} experts, found {len(blocks)}")
        # per expert: gate/up = [n_ff, n_embd], down = [n_embd, n_ff].
        # Stacking gives [n_expert, ...]; the GGUF writer reverses the dims,
        # which is exactly llama.cpp's [n_embd, n_ff, n_expert] layout.
        stacked = np.stack(blocks, axis=0)
        writer.add_tensor(gname, stacked)
        print(f"  {gname}: {len(blocks)} experts -> {stacked.shape}")

    # Tied embeddings: llama.cpp expects output.weight to exist.
    if arch.tied and "output.weight" not in plan:
        raw, shape, dtype = tensors["model.embed_tokens.weight"]
        writer.add_tensor("output.weight", to_array(raw, shape, dtype, False))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"SUCCESS: wrote {args.out}")

    if not args.no_validate:
        validate(args.out, expected_tensors=len(plan))


def validate(path, expected_tensors=0):
    print("=== validating ===")
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        if magic != 0x46554747:
            raise SystemExit("bad magic in the written GGUF")
        tc, mc = struct.unpack("<QQ", f.read(16))
        print(f"  tensors={tc} metadata={mc}")
        if expected_tensors and tc != expected_tensors:
            print(f"  WARNING: expected {expected_tensors} tensors, file has {tc}")
    print(f"  size {os.path.getsize(path) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
