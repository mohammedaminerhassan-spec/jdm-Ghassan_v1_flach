# Granite adaptation plan (checkpoint: brain model safetensors/ = IBM granite-4.0-h-tiny)
# Everything here must be done. Checkboxes are filled as we go.

## Files in brain model safetensors/
# config.json            -> GraniteMoeHybridConfig (read, 1344 lines architecture)
# model.safetensors.index.json -> 586 tensors, 6.94 B params, 3 shards
# tokenizer.json          -> byte-level BPE, vocab 100352, digits {1,3}
# vocab.json              -> 100354 entries (100352 + 2 padding?) actually 100354 lines
# merges.txt              -> 100000 merges
# chat_template.jinja     -> granite role tags
# generation_config.json  -> bos=eos=100257 <|end_of_text|>, pad=100256 <|pad|>
# special_tokens_map.json -> bos/eos <|end_of_text|>, pad <|pad|>, unk <|unk|>

## 1) Tokenizer
- [ ] PreTokKind::Granite (digits {1,3}) — refactor pre_tokenize_qwen(digits_max)
- [ ] detect granite pretok in hf_checkpoint (regex \p{N}{1,3})
- [ ] special tokens from added_tokens (100256..100351) + added_ids
- [ ] Granite chat template renderer + encode_granite() with loss mask
- [ ] GGUF pre name "granite" / keep "qwen2"? (decide: export as granite arch)

## 2) Config (ModelConfig extension)
- [ ] arch_identity: "granite_moe_hybrid"
- [ ] layer_types per layer: "mamba" | "attention"
- [ ] multipliers: embedding_multiplier=12, residual_multiplier=0.22,
      attention_multiplier=1/128, logits_scaling=6
- [ ] mamba dims: expand=2 (d_inner=3072), n_heads=48, d_head=64, d_state=128,
      n_groups=1, d_conv=4, chunk=256, conv_bias=true, proj_bias=false
- [ ] NoRoPE (position_embedding_type=nope) -> rope off entirely
- [ ] moe: 64 experts top-6, shared 1024 (router: softmax, no aux loss)
- [ ] vocab 100352, tied, max_pos 131072 (1M capable? engine caps 1M ok)

## 3) Weights mapping (HF safetensors -> engine params)
- [ ] model.embed_tokens.weight -> tok_embeddings
- [ ] model.norm.weight -> final_norm
- [ ] model.layers.N.input_layernorm.weight -> attn_norm (rmsnorm)
- [ ] model.layers.N.post_attention_layernorm.weight -> ffn_norm
- [ ] attention layers (5,15,25,35): q/k/v/o_proj -> wq/wk/wv/wo
- [ ] mamba layers: mamba.in_proj.weight [6448, 1536] -> mamba_in_proj
- [ ] mamba.conv1d.{weight,bias} -> mamba_conv1d_{weight,bias}
- [ ] mamba.A_log, mamba.D, mamba.dt_bias -> mamba_{A_log,D,dt_bias}
- [ ] mamba.norm.weight (gated rmsnorm) -> mamba_norm
- [ ] mamba.out_proj.weight -> mamba_out_proj
- [ ] block_sparse_moe.router.layer.weight [64, 1536] -> moe_router
- [ ] block_sparse_moe.input_linear.weight -> packed experts (read shape!)
- [ ] block_sparse_moe.output_linear.weight -> packed experts
- [ ] shared_mlp.input_linear.weight [2048, 1536] -> shared gate+up
- [ ] shared_mlp.output_linear.weight [1536, 1024] -> shared down
- [ ] shape-driven loader: handle 2D [E*2I,H] or 3D [E,2I,H] layouts

## 4) Forward (inference)
- [ ] inputs_embeds *= embedding_multiplier (12)
- [ ] attention layers: standard GQA path, attention_multiplier scaling, NoRoPE
- [ ] mamba layers:
      in_proj -> split (gate 3072, bc 3328, dt 48)
      causal conv1d (depthwise, k=4) + silu
      split bc -> x (3072), B (128), C (128)
      recurrent SSM scan: state [48, 64, 128]; dt = softplus(dt + dt_bias);
        dA = exp(-exp(A_log) * dt)
        state = dA[:,None,None] * state + (dt * B)[:,None,None... broadcast] * x
        y = einsum("hds,ds->hd", C, state) + D * x
      gated rmsnorm (norm_weight * silu(gate))
      out_proj
- [ ] residual * residual_multiplier (0.22) on both residual adds
- [ ] MoE: softmax top-6 router + experts + shared expert
- [ ] logits / logits_scaling (6)
- [ ] lm_head = tok_embeddings (tied)

## 5) Export / precision (user: fp16 consistent with the code)
- [ ] fp16 compute everywhere (gemm_fp16) + fp16 export allowed
- [ ] GGUF keeps fp16 + Q8_0 options; bf16 not needed (fp32 masters)
- [ ] drop "8-bit max" for this model; fp16 becomes first-class

## 6) Verification (synthetic, no real weights)
- [ ] tiny granite config (hidden 32, 2 layers: 1 mamba + 1 attention,
      2 experts) + random safetensors -> forward logits finite
- [ ] tokenizer: 11/11 conformance vs HF transformers (if installed)
- [ ] build 0 warnings

## 7) Kaggle checklist (user runs)
- upload brain model safetensors folder as-is
- gai_train / ghassan-ai with the granite config
- expect vocab 100352 (index total 6.94B params)

## Known risks / what we cannot verify locally
- exact shapes of block_sparse_moe.input_linear.weight (need the header;
  loader reads the shape -> split accordingly)
- bf16 weights: cast to fp32 at load (engine has fp32 masters)
- 131072 max_pos: rope disabled (nope) but cache is still fp16/fp32
