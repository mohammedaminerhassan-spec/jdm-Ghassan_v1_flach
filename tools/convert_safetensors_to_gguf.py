#!/usr/bin/env python3
import os
import json
import struct
import numpy as np
import gguf

CKPT_DIR = r"C:\Users\Ghassan PC\Documents\Projects\Ghassan Ai model\safetensors"
OUT_GGUF = r"C:\Users\Ghassan PC\Documents\Projects\Ghassan Ai model\artifacts\Ghassan-v1-pro-1.7B-f16.gguf"

def read_safetensors_raw(path):
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_len
        tensors = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            begin, end = v["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            shape = v["shape"]
            dtype = v["dtype"]
            tensors[k] = (raw, shape, dtype)
        return tensors

def main():
    print("=== Converting Ghassan v1 pro to GGUF (FP16) ===")
    os.makedirs(os.path.dirname(OUT_GGUF), exist_ok=True)
    
    # 1. Config
    with open(os.path.join(CKPT_DIR, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    hidden_size = cfg.get("hidden_size", 960)
    intermediate_size = cfg.get("intermediate_size", 2560)
    num_layers = cfg.get("num_hidden_layers", 32)
    num_heads = cfg.get("num_attention_heads", 15)
    num_kv_heads = cfg.get("num_key_value_heads", 5)
    vocab_size = cfg.get("vocab_size", 49152)
    max_seq_len = cfg.get("max_position_embeddings", 8192)
    rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
    rope_theta = float(cfg.get("rope_theta", 100000.0))
    head_dim = hidden_size // num_heads

    # 2. GGUF Writer
    writer = gguf.GGUFWriter(OUT_GGUF, arch="llama")
    writer.add_name("Ghassan v1 pro")
    writer.add_description("Ghassan v1 pro — High-Performance Autonomous Intelligence by Ghassan AI Studio")
    writer.add_author("Ghassan AI Studio")
    writer.add_file_type(1) # FP16
    
    writer.add_context_length(max_seq_len)
    writer.add_embedding_length(hidden_size)
    writer.add_bool("llama.tie_word_embeddings", True)
    writer.add_block_count(num_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_rope_dimension_count(head_dim)
    writer.add_uint32("llama.vocab_size", vocab_size)
    
    # 3. Tokenizer
    with open(os.path.join(CKPT_DIR, "tokenizer.json"), "r", encoding="utf-8") as f:
        tok_data = json.load(f)
        
    vocab_dict = tok_data["model"]["vocab"]
    tokens = [None] * len(vocab_dict)
    for tok, idx in vocab_dict.items():
        if idx < len(tokens):
            tokens[idx] = tok.encode("utf-8")
        else:
            tokens.append(tok.encode("utf-8"))
            
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    
    tok_types = [gguf.TokenType.NORMAL] * len(tokens)
    tok_types[0] = gguf.TokenType.CONTROL # <|endoftext|>
    if len(tokens) > 1: tok_types[1] = gguf.TokenType.CONTROL # <|im_start|>
    if len(tokens) > 2: tok_types[2] = gguf.TokenType.CONTROL # <|im_end|>
    for i in range(3, min(17, len(tokens))):
        tok_types[i] = gguf.TokenType.CONTROL
    writer.add_token_types(tok_types)
    
    writer.add_token_merges(tok_data["model"]["merges"])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(0)
    writer.add_unk_token_id(0)
    
    chat_template = (
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
    )
    writer.add_chat_template(chat_template)

    # 4. Tensors
    sft_path = os.path.join(CKPT_DIR, "model.safetensors")
    print("Reading safetensors file directly...")
    tensors = read_safetensors_raw(sft_path)
    print(f"Read {len(tensors)} tensors successfully.")
    
    for name, (raw, shape, dtype) in tensors.items():
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            f32 = u32.view(np.float32).reshape(shape)
            # In GGUF and llama.cpp, RMSNorm weights must be FP32 to prevent binary_op mismatch
            if "norm" in name:
                arr = f32
            else:
                arr = f32.astype(np.float16)
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
            if "norm" in name:
                arr = arr.astype(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            if "norm" not in name:
                arr = arr.astype(np.float16)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        gguf_name = None
        if name == "model.embed_tokens.weight":
            gguf_name = "token_embd.weight"
        elif name == "model.norm.weight":
            gguf_name = "output_norm.weight"
        elif name == "lm_head.weight":
            gguf_name = "output.weight"
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer_idx = parts[2]
            sub = ".".join(parts[3:])
            if sub == "input_layernorm.weight":
                gguf_name = f"blk.{layer_idx}.attn_norm.weight"
            elif sub == "post_attention_layernorm.weight":
                gguf_name = f"blk.{layer_idx}.ffn_norm.weight"
            elif sub == "self_attn.q_proj.weight":
                gguf_name = f"blk.{layer_idx}.attn_q.weight"
            elif sub == "self_attn.k_proj.weight":
                gguf_name = f"blk.{layer_idx}.attn_k.weight"
            elif sub == "self_attn.v_proj.weight":
                gguf_name = f"blk.{layer_idx}.attn_v.weight"
            elif sub == "self_attn.o_proj.weight":
                gguf_name = f"blk.{layer_idx}.attn_output.weight"
            elif sub == "mlp.gate_proj.weight":
                gguf_name = f"blk.{layer_idx}.ffn_gate.weight"
            elif sub == "mlp.up_proj.weight":
                gguf_name = f"blk.{layer_idx}.ffn_up.weight"
            elif sub == "mlp.down_proj.weight":
                gguf_name = f"blk.{layer_idx}.ffn_down.weight"

        if gguf_name is not None:
            writer.add_tensor(gguf_name, arr)
            
    print(f"Writing GGUF output to {OUT_GGUF}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("SUCCESS: Ghassan v1 pro GGUF created successfully!")

if __name__ == "__main__":
    main()
