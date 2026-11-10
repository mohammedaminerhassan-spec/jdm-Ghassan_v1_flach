#!/usr/bin/env python3
"""Build a tiny HuggingFace-style checkpoint to exercise the whole
safetensors -> GGUF -> Gai engine path.

The real checkpoints are multi gigabyte, so the pipeline is verified on a
model that fits in a few megabytes:

    python tools/py/make_test_model.py --out artifacts/test_model \
        --moe --experts 4 --top-k 2

The generated checkpoint has the same layout as a real one: a config.json, a
tokenizer.json with a byte-level BPE vocabulary and a chat template, and
safetensors weights with HuggingFace tensor names.
"""
import argparse
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file


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


def build_tokenizer(vocab_size, chat_template):
    enc = bytes_to_unicode()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {}
    for i, s in enumerate(specials):
        vocab[s] = i
    # The byte-level alphabet occupies the next ids; the rest of the
    # vocabulary is filled with harmless two-character pieces.
    n = len(specials)
    for b in range(256):
        vocab[enc[b]] = n
        n += 1
    i = 0
    while n < vocab_size:
        vocab[f"z{i}"] = n
        n += 1
        i += 1
    merges = []
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": i, "content": s, "special": True,
             "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False}
            for i, s in enumerate(specials)
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": True},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=272)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expert-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = args.hidden
    hd = d // args.heads
    v = args.vocab
    L = args.layers

    def randn(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    tensors = {}
    tensors["model.embed_tokens.weight"] = randn(v, d)
    tensors["model.norm.weight"] = np.ones(d, dtype=np.float32)

    for i in range(L):
        p = f"model.layers.{i}."
        tensors[p + "self_attn.q_proj.weight"] = randn(args.heads * hd, d)
        tensors[p + "self_attn.k_proj.weight"] = randn(args.kv_heads * hd, d)
        tensors[p + "self_attn.v_proj.weight"] = randn(args.kv_heads * hd, d)
        tensors[p + "self_attn.o_proj.weight"] = randn(d, args.heads * hd)
        tensors[p + "input_layernorm.weight"] = np.ones(d, dtype=np.float32)
        tensors[p + "post_attention_layernorm.weight"] = np.ones(d, dtype=np.float32)
        if args.moe:
            tensors[p + "mlp.gate.weight"] = randn(args.experts, d)
            for e in range(args.experts):
                tensors[p + f"mlp.experts.{e}.gate_proj.weight"] = randn(args.expert_dim, d)
                tensors[p + f"mlp.experts.{e}.up_proj.weight"] = randn(args.expert_dim, d)
                tensors[p + f"mlp.experts.{e}.down_proj.weight"] = randn(d, args.expert_dim)
        else:
            tensors[p + "mlp.gate_proj.weight"] = randn(args.ffn, d)
            tensors[p + "mlp.up_proj.weight"] = randn(args.ffn, d)
            tensors[p + "mlp.down_proj.weight"] = randn(d, args.ffn)

    save_file(tensors, os.path.join(args.out, "model.safetensors"))

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": d,
        "intermediate_size": args.ffn,
        "num_hidden_layers": L,
        "num_attention_heads": args.heads,
        "num_key_value_heads": args.kv_heads,
        "vocab_size": v,
        "max_position_embeddings": args.ctx,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    if args.moe:
        cfg["num_experts"] = args.experts
        cfg["num_experts_per_tok"] = args.top_k
        cfg["moe_intermediate_size"] = args.expert_dim
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    chat_template = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\nYou are a tiny test model.<|im_end|>\n' }}"
        "{% endif %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tok = build_tokenizer(v, chat_template)
    with open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)

    with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "add_bos_token": False,
            "add_eos_token": False,
            "bos_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|im_end|>",
            "unk_token": "<|endoftext|>",
            "chat_template": chat_template,
            "model_max_length": args.ctx,
        }, f, indent=2)

    print(f"wrote {args.out}: {len(tensors)} tensors, vocab {v}, "
          f"{L} layers, {'MoE ' + str(args.experts) + ' experts' if args.moe else 'dense'}")


if __name__ == "__main__":
    main()
