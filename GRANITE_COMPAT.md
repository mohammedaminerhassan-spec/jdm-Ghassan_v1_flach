# Granite-4.0-H ↔ Ghassan AI Engine — تقرير المطابقة

المرجع: `ibm-granite/granite-4.0-h-tiny` (transformers 4.56.0، `GraniteMoeHybridForCausalLM`).
المصدر المرجعي: `modeling_granitemoehybrid.py` من transformers v4.56.0.

**النتيجة: المحرّك يحسب نفس الدالة بالضبط، والـ geometry مطابق لِلـ element.**

---

## 1. الـ geometry (من `config.json` المرجعي)

| الحقل | القيمة | المحرّك |
|---|---|---|
| `hidden_size` | 1536 | ✅ |
| `num_hidden_layers` | 40 (36 mamba + 4 attention) | ✅ `layer_types` |
| `num_attention_heads` / `num_key_value_heads` | 12 / 4 (head_dim = 128) | ✅ |
| `intermediate_size` (عرض الـ routed expert) | 512 | ✅ `moe_expert_dim` |
| `shared_intermediate_size` (الـ shared expert) | **1024** (مش 512!) | ✅ `shared_expert_intermediate_size` |
| `num_local_experts` / `num_experts_per_tok` | 64 / 6 | ✅ |
| `vocab_size` | 100352 | ✅ |
| `max_position_embeddings` | 131072 | ✅ |
| `position_embedding_type` | `"nope"` | ✅ `position_embedding = 1` (مفيش RoPE) |
| `embedding_multiplier` | 12 | ✅ |
| `residual_multiplier` | 0.22 | ✅ |
| `attention_multiplier` | 0.0078125 (مش `1/sqrt(head_dim)`!) | ✅ |
| `logits_scaling` | 6 | ✅ (بيقسّم الـ logits) |
| `tie_word_embeddings` | true | ✅ (مفيش `lm_head` في الـ checkpoint) |
| `attention_bias` | false | ✅ |
| `mamba_*` | n_heads 48, d_head 64, d_state 128, n_groups 1, d_conv 4, expand 2, chunk 256 | ✅ |
| `mamba_proj_bias` | false | ✅ |

**الـ geometry المحسوب:** `d_inner = 3072`, `conv_dim = 3328`, `proj = 6448`.

## 2. عدد البارامترات — الدليل القاطع

```
the engine expects   6939037248 parameters (6.939 B)
the checkpoint has   6939037248 parameters (6.939 B)
the difference       0
```

الفرق = **صفر** بالـ element. مفيش أي بارامتر زيادة أو ناقصة.

## 3. أسماء الـ tensors — الـ 21 pattern كلها مغطّاة

| الـ checkpoint | المحرّك |
|---|---|
| `model.embed_tokens.weight` | `tok_embeddings` ✅ |
| `model.norm.weight` | `final_norm` ✅ |
| `model.layers.N.input_layernorm.weight` | `attn_norm` ✅ |
| `model.layers.N.post_attention_layernorm.weight` | `ffn_norm` ✅ |
| `model.layers.N.self_attn.{q,k,v,o}_proj.weight` | `wq/wk/wv/wo` ✅ |
| `model.layers.N.mamba.{in_proj,out_proj}.weight` | `mamba.*` ✅ |
| `model.layers.N.mamba.conv1d.{weight,bias}` | `mamba.conv1d.*` ✅ |
| `model.layers.N.mamba.{dt_bias,A_log,D}` | `mamba.*` ✅ |
| `model.layers.N.mamba.norm.weight` | `mamba.norm.weight` ✅ |
| `model.layers.N.block_sparse_moe.router.layer.weight` | `moe_router` ✅ |
| `model.layers.N.block_sparse_moe.{input_linear,output_linear}.weight` | الـ experts **مجمّعة** ✅ |
| `model.layers.N.shared_mlp.{input_linear,output_linear}.weight` | الـ shared **مجمّع** ✅ |

## 4. الـ packing — التحقق من الترتيب (الأخطر: لو معكوس = garbage صامت)

**المرجع** (`GraniteMoeHybridParallelExperts`):
```python
self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
# input_linear  = [ne, 2E, d]      output_linear = [ne, d, E]
hidden_states = self.input_linear(expert_inputs, expert_size)   # [n, 2E]
chunked = hidden_states.chunk(2, dim=-1)   # <-- على البُعد الأخير!
hidden_states = act(chunked[0]) * chunked[1]
```
→ لكل expert: **الـ gate في الأول (E صف)** والـ up بعده (E صف) ✓

**المحرّك** (`format/hf_checkpoint.cpp`):
```cpp
const float* src = ti.f32() + e * 2 * block;      // block = E*d
memcpy(dst + e*block, leaf == "moe_gate" ? src : src + block, block*4);
```
→ نفس الترتيب بالظبط ✅

**الـ shared MLP** (`GraniteMoeHybridMLP`): `input_linear` = `[2*SH, d]` و
`chunk(2, dim=-1)` → الـ gate الأول والـ up بعده ✅ (المحرّك نفس الشيء ✅).

**الـ down**: الـ checkpoint `[ne, d, E]` →Layout C في المحرّك (نسخة مباشرة) ✅.

## 5. الرياضيات — مقارنة سطر بسطر

| الجزء | المرجع | المحرّك | ✓ |
|---|---|---|---|
| بنية الـ layer | `residual + attn/mamba(x) * m` | `x + m*proj` | ✅ |
| بنية الـ FFN | `residual + (moe + shared_mlp(x)) * m` | `x + m*(routed + shared)` | ✅ |
| الـ norms | `input_layernorm` قبل attention، `post_attention_layernorm` قبل MoE | `attn_norm` / `ffn_norm` | ✅ |
| الـ embedding | `inputs_embeds * 12` | `scale_inplace(embedding_mult)` | ✅ |
| الـ logits | `logits / 6` | `scale_inplace(logits, 1/6)` | ✅ |
| الـ attention scale | `scaling = config.attention_multiplier` | نفس القيمة (مش `1/sqrt(hd)`) | ✅ |
| الـ routing | `softmax(top_k_logits)` | softmax كامل → top-K → إعادة تطبيع | ✅ **مكافئ رياضياً** |
| الـ conv1d | depthwise, causal, SiLU **بعد** الـ conv | نفس الشيء | ✅ |
| الـ SSM scan | `s = exp(A·dt)·s + (dt·B)·x` ، `y = C·s + D·x` | نفس الشيء | ✅ |
| الـ gated norm | `RMSNorm(y · silu(gate)) · w` | نفس الشيء | ✅ |
| الـ RoPE | مفيش (`nope` → `rotary_emb = None`) | بيتخطّى | ✅ |

## 6. الأخطاء اللي كانت بتبطل التطابق (ومتصلّحة)

| # | الخطأ | الأثر على Granite |
|---|---|---|
| **BUG-01** | الـ FFN في الـ backward ناقصه `residual_multiplier` | **تدرّج الـ FFN كان 4.5× كبير** (1/0.22) ✗✗✗ |
| **BUG-02** | الـ shared expert في الـ backward (CPU) بيستعمل عرض الـ routed | الـ shared (1024) كان بيتعامل معاه كأنه 512 ✗ |
| **BUG-11/12** | الـ loader ما بيحمّلش الـ shared expert / `shared_intermediate_size` بيتجاهَل | **أي MoE غير Granite ما بيحمّلش أصلاً** ✗ |

> **BUG-01 كان أخطرهم على Granite بالذات**: م=0.22 يعني إن التدرّج كان بيكبر 4.5×
> (مش 2×). أي تدريب Granite قبل الإصلاح كان **ماشي بالاتجاه الصح بس ب-step أكبر 4.5×** ✗

## 7. الاختبار

```bash
ctest -R granite_compat --output-on-failure
```

`tests/test_granite_compat.cpp` بيبني الـ ModelConfig من **الـ config.json المرجعي نفسه**
(مدمج حرفياً في الملف) وبيتحقق من:
- الـ geometry (40 طبقة، 36 mamba + 4 attention، العرض، الـ MoE، الـ NoPE).
- **عدد البارامترات = عدد الـ checkpoint بالظبط**.
- كل بارامتر عنده اسم HuggingFace (ما عدا الـ packed ones).
- أحجام الـ tensors المجمّعة (input_linear / output_linear / shared).

**الخطوة الجاية على Kaggle:** حمّل الـ shards (13.9 GB) واعمل
`ghassan-ai info --model <dir>` → بيفتح الـ checkpoint ويطبع تقرير البارامترات.
لو الأرقام طلع 6.939 B والـ geometry مطابق → الـ تحميل سليم 100% ✅
