#!/usr/bin/env python3
"""Export a GGUF model back to safetensors so it can be fine-tuned.

The engine serves GGUF; PyTorch, PEFT and QLoRA want safetensors. This is
the bridge in the other direction: it reads a GGUF file and writes a
HuggingFace style checkpoint (safetensors + config.json + the tokenizer
files) that `transformers`, `peft` and `trl` can load directly.

    python tools/py/gguf_to_safetensors.py \
        --gguf artifacts/my-model.gguf \
        --out models/my-model-hf

Mixture-of-experts checkpoints are supported: the 3D expert tensors are
split back into the per-expert 2D tensors HuggingFace expects, so QLoRA on
a MoE model works with the usual `target_modules` names.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


# ---------------------------------------------------------------------------
# GGUF tensor name -> HuggingFace tensor name
# ---------------------------------------------------------------------------

def dense_map(layer, rest):
    m = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }
    if rest not in m:
        return None
    return f"model.layers.{layer}.{m[rest]}"


EXPERT_MAP = {
    "ffn_gate_exps.weight": "gate_proj.weight",
    "ffn_up_exps.weight": "up_proj.weight",
    "ffn_down_exps.weight": "down_proj.weight",
}


def map_name(name):
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name.startswith("blk."):
        parts = name.split(".")
        layer = parts[1]
        rest = ".".join(parts[2:])
        if rest in EXPERT_MAP:
            return ("expert", layer, rest)
        out = dense_map(layer, rest)
        if out:
            return out
    return None


# ---------------------------------------------------------------------------
# GGUF reading
# ---------------------------------------------------------------------------

def read_gguf(path):
    import gguf

    reader = gguf.GGUFReader(path)
    fields = reader.fields

    def get(key, default=None):
        f = fields.get(key)
        if f is None:
            return default
        if isinstance(f, list):
            return f
        try:
            return f.contents[-1]
        except Exception:
            return default

    meta = {
        "hidden_size": int(get("llama.embedding_length", 0) or 0),
        "layers": int(get("llama.block_count", 0) or 0),
        "heads": int(get("llama.attention.head_count", 0) or 0),
        "kv_heads": int(get("llama.attention.head_count_kv", 0) or 0),
        "vocab_size": int(get("llama.vocab_size", 0) or 0),
        "intermediate": int(get("llama.feed_forward_length", 0) or 0),
        "ctx": int(get("llama.context_length", 0) or 0),
        "eps": float(get("llama.attention.layer_norm_rms_epsilon", 1e-5) or 1e-5),
        "rope_theta": float(get("llama.rope.freq_base", 10000.0) or 10000.0),
        "tied": bool(get("llama.tie_word_embeddings", False)),
        "experts": int(get("llama.expert_count", 0) or 0),
        "top_k": int(get("llama.expert_used_count", 0) or 0),
        "expert_dim": int(get("llama.expert_feed_forward_length", 0) or 0),
    }
    tensors = {t.name: t for t in reader.tensors}
    return meta, tensors


def tensor_to_numpy(t):
    """Returns a float32 numpy array for a GGUF tensor."""
    import gguf
    data = t.data
    ttype = int(getattr(t, "tensor_type", 0))
    if ttype == 0:      # F32
        return np.asarray(data, dtype=np.float32)
    if ttype == 1:      # F16
        return np.asarray(data, dtype=np.float16).astype(np.float32)
    # quantized: use the gguf dequantizer when it is available
    try:
        from gguf.quants import dequantize
        out = dequantize(np.asarray(data), ttype)
        return np.asarray(out, dtype=np.float32)
    except Exception as exc:
        raise SystemExit(
            f"cannot dequantize {t.name} (ggml type {ttype}): {exc}\n"
            "Convert the model to f16/f32 first, or install a gguf build with "
            "the quant kernels.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="directory with tokenizer.json to copy next to the weights")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    from safetensors.numpy import save_file

    print("=== reading the GGUF ===")
    meta, tensors = read_gguf(args.gguf)
    print(f"  {len(tensors)} tensors | d={meta['hidden_size']} L={meta['layers']} "
          f"H={meta['heads']} KV={meta['kv_heads']} V={meta['vocab_size']} "
          f"experts={meta['experts']}")

    def to_store(arr):
        if args.dtype == "bfloat16":
            u = arr.astype(np.float32).view(np.uint32)
            return ((u >> 16) & 0xFFFF).astype(np.uint16)
        if args.dtype == "float16":
            return arr.astype(np.float16)
        return arr.astype(np.float32)

    out_tensors = {}
    expert_blocks = {}

    for name, t in tensors.items():
        mapped = map_name(name)
        if mapped is None:
            print(f"  [warn] unmapped tensor {name}")
            continue
        arr = tensor_to_numpy(t)
        if isinstance(mapped, tuple):
            _, layer, rest = mapped
            key = (f"model.layers.{layer}.mlp.experts", rest)
            expert_blocks.setdefault(key, []).append(arr)
            continue
        out_tensors[mapped] = to_store(arr)

    # Split the 3D expert tensors into the per-expert 2D tensors.
    for (prefix, rest), blocks in expert_blocks.items():
        stacked = np.stack(blocks, axis=0)
        leaf = EXPERT_MAP[rest]
        for e in range(stacked.shape[0]):
            out_tensors[f"{prefix}.{e}.{leaf}"] = to_store(stacked[e])

    if meta["tied"] and "lm_head.weight" not in out_tensors:
        out_tensors["lm_head.weight"] = out_tensors["model.embed_tokens.weight"]

    os.makedirs(args.out, exist_ok=True)
    save_file(out_tensors, os.path.join(args.out, "model.safetensors"))

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": meta["hidden_size"],
        "intermediate_size": meta["intermediate"],
        "num_hidden_layers": meta["layers"],
        "num_attention_heads": meta["heads"],
        "num_key_value_heads": meta["kv_heads"],
        "vocab_size": meta["vocab_size"],
        "max_position_embeddings": meta["ctx"],
        "rms_norm_eps": meta["eps"],
        "rope_theta": meta["rope_theta"],
        "tie_word_embeddings": meta["tied"],
        "torch_dtype": args.dtype,
    }
    if meta["experts"]:
        cfg["num_experts"] = meta["experts"]
        cfg["num_experts_per_tok"] = meta["top_k"]
        cfg["moe_intermediate_size"] = meta["expert_dim"]
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if args.tokenizer:
        for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json", "generation_config.json"):
            src = os.path.join(args.tokenizer, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, fn))

    total = sum(v.nbytes for v in out_tensors.values())
    print("=== done ===")
    print(f"  {len(out_tensors)} tensors -> {args.out}")
    print(f"  {total / 2**20:.1f} MiB ({args.dtype})")
    print("  next: python tools/py/qlora_finetune.py --model "
          f"{args.out} --data <dataset> --out <run>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
