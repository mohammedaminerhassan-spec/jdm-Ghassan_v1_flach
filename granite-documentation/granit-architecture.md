Hna l jawab mfassal 3la architecture dyal `granite-4.0-h-tiny`, m3a l 3dad dyal kolchi. Kolchi jay mn `config.json` w `model.safetensors.index.json` dyal repo officiel, ila ma kant chi note khra.

## 1. Précision dyal l weights: **bf16 (bfloat16)**, machi fp16 wla fp32
- F `config.json`: `"torch_dtype": "bfloat16"` ([config.json](https://huggingface.co/ibm-granite/granite-4.0-h-tiny/raw/main/config.json)).
- Total size dyal l safetensors = **13,878,074,496 bytes** (13.88 GB) w total parameters = **6,939,037,248** (~6.94B). Hadi kat3ni 2 bytes l kol parameter = bf16 ✅
- Bf16 w fp16 kaydo l 2 bytes, walakin bf16 3ando range dyal fp32 (exponent 8 bits) w mantissa aqal (7 bits). Ila bghiti tkhdem b fp16 khassk tdir `dtype=torch.float16` explicitly, w ghadi ykon f_flat aw overflow f ba3d l layers.
- Ila bghiti l code dyalk y9ra l weights b bf16: `torch_dtype=torch.bfloat16` f `from_pretrained`.

## 2. Architecture (hybrid Mamba2 + Attention + MoE)
- `model_type`: `granitemoehybrid`, `architectures`: `GraniteMoeHybridForCausalLM` ([config](https://huggingface.co/ibm-granite/granite-4.0-h-tiny/raw/main/config.json)).
- **40 layers**: 36 Mamba2 + 4 attention. Attention f layers **5, 15, 25, 35** (0-indexed); ba9i kolchi Mamba. Hadi kayn f `layer_types`.
- **hidden_size = 1536**
- **vocab_size = 100352**, **tie_word_embeddings = true** (ma kayn-ch `lm_head.weight` f safetensors, kay-reuse `embed_tokens`).
- **max_position_embeddings = 131072** (128K).
- **Attention**: 12 heads, **4 KV heads** (GQA), `attention_bias: false`. Head dim = 1536/12 = 128.
- **Position embedding: `"nope"`** = ma kayn-ch RoPE! Ila code dyalk fih RoPE, ghadi ykon ghalat. `rope_theta` ma kay-ssta3mlch.
- **Mamba2 block**: `mamba_d_state=128`, `mamba_d_head=64`, `mamba_n_heads=48` (48×64 = 3072 = 2×1536, ya3ni `mamba_expand=2`), `mamba_d_conv=4`, `mamba_n_groups=1`, `mamba_chunk_size=256`, `mamba_conv_bias=true`, `mamba_proj_bias=false`.
- **MoE**: kull layer fih **64 experts**, **top-6** f kol token, `intermediate_size=512` l kol expert. Kayn zedma **shared MLP** b `shared_intermediate_size=1024`.
- **Activation**: SiLU (`silu`). **Norm**: RMSNorm, eps = 1e-5.
- **Multipliers** (khessak tkhdemhom, ila ma-dirhomch ghadi tkoun outputs ghalta):
  - `embedding_multiplier = 12` (embeddings kay-ttmultiplaw b 12)
  - `attention_multiplier = 0.0078125` (= 1/128 = 1/head_dim, machi 1/√d)
  - `residual_multiplier = 0.22` (kay-ttmultiplaw m3a output dyal kull branch qbel residual add)
  - `logits_scaling = 6` (logits kay-ttqssmou 3la 6 9bel softmax)

## 3. Tensor names f safetensors (3 shards)
Total 3 files: `model-00001-of-00003`, `-00002-`, `-00003` ([index](https://huggingface.co/ibm-granite/granite-4.0-h-tiny/raw/main/model.safetensors.index.json)).

Kolshi f kull layer:
- `model.layers.N.input_layernorm.weight`
- `model.layers.N.post_attention_layernorm.weight`
- `model.layers.N.mamba.{A_log, D, dt_bias, conv1d.weight, conv1d.bias, in_proj.weight, norm.weight, out_proj.weight}` (f Mamba layers)
- `model.layers.N.self_attn.{q_proj, k_proj, v_proj, o_proj}.weight` (f attention layers: 5, 15, 25, 35)
- `model.layers.N.block_sparse_moe.router.layer.weight`
- `model.layers.N.block_sparse_moe.input_linear.weight` w `output_linear.weight` (experts m-jma3in f tensor wahd, machi key l kol expert)
- `model.layers.N.shared_mlp.input_linear.weight` w `output_linear.weight`

Global:
- `model.embed_tokens.weight`
- `model.norm.weight`

Shapes ma-tkhdmtch b tool, walakin 7ssabt mn config (ma-ynfaqch ma3a l 3dad, khessk tchecki b `safetensors`):
- `in_proj` (Mamba) = 2×3072 + 2×128 + 48 = **6448** x 1536 (mahsoub)
- Expert `input_linear` mzyan ykon 64 × (2×512) × 1536, w `output_linear` 64 × 1536 × 512 (inferred, khass verification).

## 4. Taille l GGUF (quantization) — kayn gher f l-format
Ma-tbdelch l 3dad dyal parameters, ghir taille dyal fichier w RAM/VRAM:
- **BF16**: ~13.88 GB ([llmrun](https://llmrun.dev/model/ibm-granite-granite-4-0-h-tiny))
- **Q8_0**: ~7.4 GB
- **Q6_K**: ~6.2 GB
- **Q4_K_M** (recommended): ~4.2–4.6 GB ([llmrun](https://llmrun.dev/model/ibm-granite-granite-4-0-h-tiny))
- Kayn b IQ2_XXS (~1.9 GB) walakin qualité kat-nqss bezzaf.

GGUF kayn f bzzaf dyal repos m3a quantizations m3ahom (bartowski, unsloth, Myric...). Ila l code dyalk kayqra GGUF, khassk tkoun 3ndk support dyal `granitemoehybrid` f llama.cpp, w hadi kat-kheddem f versions recent.

## 5. Ce que khassek tdir f l code dyal `safetensors`
1. **Khass tkoun Mamba2 kernel**: l model hybrid, mamba layers mhtajin Mamba2 implementation. Ila ma-3ndakch `mamba_ssm` w `causal_conv1d`, kayn fallback slow f transformers, walakin bzzaf 3la CPU/GPU ma-kaybanch.
2. **Ma-tdirch RoPE** (nope). Ila ana 3ndek code Llama-style, 3ndek 2 khatar: RoPE zayda, w ma-t-applyich multipliers.
3. **Multipliers** 4 (embedding, attention, residual, logits) mohimin bzzaf.
4. **Attention scale** = `attention_multiplier` (0.0078125), machi `1/sqrt(head_dim)`.
5. **Tied embeddings**: `lm_head` = `embed_tokens.T`.
6. **MoE routing**: top-6 mn 64 experts, w shared MLP kayt-zad 3la output dyal MoE.
7. **Tokenizer**: BOS/EOS = 100257, PAD = 100256. Chat template kayn f `chat_template.jinja`.

**Recommendation**: ila bghiti tkhdem b code dyalk w ma-bghitich tkteb kolchi mn zero, khdem b `transformers` (`AutoModelForCausalLM`, `trust_remote_code` ma-khassouch 3la hasab version), w 3ayyt l code dyal layers w attention b l code dyalk. Hadi kat-khfef l risk dyal bugs f multipliers w mamba.

## 6. 3la l hardware (l weights only)
- BF16: ~14.3 GB VRAM
- Q4_K_M: ~4.6 GB VRAM
- Context kamel 131K ka-yzid ~10.6 GB KV cache

Ila bghiti n-kteb lik had l-info f file `.md` wla n-tbeddel l code dyalk b l-architecture hadi, goulli.