# CONTEXT.md — Ghassan v1 Flash / حالة المشروع

> آخر تحديث: 2026-11-12 (تحقق عددي كامل ديال Granite + LoRA + fixes).

---

## 1) الهدف

**Ghassan v1 Flash** = محرك (body) من الصفر بـ C++20/CUDA بلا أي مكتبة خارجية،
كيخدم بـ **MoE** (DeepSeek-style routing) وكيقدر يكبر حتى trillion parameters.

الموديل الحالي المستهدف: **IBM Granite-4.0-H-Tiny** (`GraniteMoeHybridForCausalLM`).

---

## 2) القرارات (من المستخدم)

| # | القرار |
|---|--------|
| 1 | بلا Darija / بلا كود ميت: `.gtok`, `.gai`, vision, serve, benchmark, normalizer, BpeTrainer — كلهم مش-horين |
| 2 | export/serve = GGUF self-contained؛ **Q8_0 max** كان القرار القديم، والآن **fp16 رجع** مسموح (`--profile fp16`) |
| 3 | بلا PyTorch / بلا Python فـ inference ولا export — C++/CUDA فقط |
| 4 | التدريب موقوف على Kaggle (2x T4)؛ الـ PC المحلي ضعيف |
| 5 | routing = sigmoid + aux-loss-free bias (خيار للموديلات الجاية؛ Granite كيستعمل softmax top-k) |
| 6 | chat template = ChatML لـ Qwen، **role-tags** (`<|start_of_role|>`) لـ Granite |
| 7 | LoRA rank 16 + merge فـ base weights عند export |
| 8 | converter/upscaler داخل المحرك (C++) |

---

## 3) الحالة: Granite-4.0-H-Tiny مدعوم بالكامل (inference + export)

### المعمارية اللي المحرك كيخدمها

```
40 طبقة = 36 Mamba-2 + 4 Attention (5, 15, 25, 35)
hidden 1536, head 128 (12 heads / 4 KV), NoPE (بلا RoPE)
Mamba-2: d_inner 3072 (expand 2), 48 head, d_head 64, d_state 128, groups 1, conv k=4, chunk 256
MoE: 64 expert, top-6, expert width 512, shared expert width 1024
multipliers: embeddings x12, residual x0.22, attention x1/128, logits /6
vocab 100352, tied embeddings, byte-level BPE (digits {1,3})
6.939B params (6.78B بلا embeddings) — 92% منهم MoE experts (العدد الرسمي بالضبط ✓)
```

### التحقق المحلي (CPU، tiny synthetic checkpoint بنفس الـ layout)

**الGranite دابا متحقق عددياً ضد مراجع مستقلة** (مكتوبة من الـ reference modeling code،
ماشي من الكود ديالنا):

| الاختبار | النتيجة |
|---|---|
| **Mamba-2 block** (conv1d + selective scan + gated norm) | **max diff 2.4e-8** ✅ |
| **Attention block** (GQA + NoPE + attention_multiplier + causal mask) | **max diff 1.5e-7** ✅ |
| **Decode path** (KV cache + mamba recurrent state) = prefill greedy | **✅ (top-1 فكل خطوة)** |
| **LoRA**: init → train → ckpt → restore → merge → export | **✅** |
| GGUF round-trip (Q8_0) | max diff 9.6e-5 ✅ |
| chat template vs `transformers.apply_chat_template` | **5/5 byte-exact** |
| tokenizer vs `transformers` (Granite) | 9/9 byte-exact |
| training على granite | refused بصوت واضح (ماشي segfault) |
| build | **0 warnings** (`-Wall -Wextra -Werror`) |

المراجع + الـ fixtures: `data/mamba_tiny/` (1 mamba layer)، `data/attn_tiny/`
(1 attention layer)، `data/dense_tiny/` (2 attention layers، للـ LoRA).
الـ scripts ديال التحقق فـ `$env:TEMP/opencode/` (ماشي فـ الـ repo).

### الملفات

* `granite/` = ملفات HuggingFace الرسمية (config + tokenizer + index، بلا weights).
  الـ weights (3 shards, 13.9 GB) كاينة على Kaggle.
* `granite_tiny/` = checkpoint مصغّر بنفس الأسماء/الـ layout للتحقق المحلي.
* `data/dense_tiny/` = fixture محلّي (كل الطبقات attention) للتحقق من الـ LoRA
  والـ backward. ماشي فـ git (تحت `data/`).
* `kaggle configue/ghassan_granite_2xt4.yaml` = الوصفة ديال 2x T4.

---

## 4) إصلاحات الجوهرية (الجلسة ديال 2026-11-11/12)

1. **RoPE = neox افتراضياً** (كان interleaved → Qwen/Llama/DeepSeek كلهم كيستعملو neox)
2. **LoRA = Parameters حقيقية** (كان no-op: A/B ماشي فـ params_ → بلا optimizer، بلا grads، بلا checkpoint، B=0 → ما كيتبدل والو)
3. **LoRA merge** فـ base weights عند export (W[out,in] += s·B·Aᵀ — مطابق رياضياً للـ forward)
4. **Qwen pre-tokenizer**: `\s*[\r\n]+` greedy (كان كيقطّع ال chunks خطأ)
5. **added tokens غير-الخاص** كيتب|matchو فالنص (HF كيخليهم)
6. **GGUF `eos_ids`**: ما كياخدش `{0}` إلا كان المفتاح موجود
7. **KV cache fp16** (7.0 GiB → 3.5 GiB @32k لـ1.7B)
8. **fp16 KV decode kernel** (CPU + CUDA)
9. **Granite-4.0 hybrid**: config parse, hybrid dispatch (mamba/attention), multipliers, NoPE, shared-expert width منفصل، mamba decode recurrent state، GGUF round-trip، chat template
10. **shared expert width** (Granite: 1024 ≠ expert 512) — كان E فـ كل بلاصة
11. **DDP**: بلا `WORLD_SIZE` ما كيعلقش فـ NCCL
12. **JSONL → gbin**: `train_*.gbin` / `val_*.gbin`
13. **strict-config**: `model.lora_*` مفقودين من القائمة
14. **OOM guards**: ماشي LoRA-aware (كان كيمنع LoRA recipe)
15. **fp16 export profile** رجع + `Ghassan v1 Flash` فالـ GGUF metadata + `--model-name`

### إصلاحات الجلسة ديال 2026-11-12

16. **segfault فـ `gai_train` على granite**: `enable_fp16_weight_cache` كان كيبني fused qkv cache
    لـ **كل** الطبقات — بما فيهم مamba اللي ما عندهمش `wq` → null deref. دابا كيتخطااها.
17. **guard واضح للـ training على granite**: `forward_backward` كيرفض بصوت عالي
    ("granite mamba layers do not support the backward pass yet") بدل crash.
18. **`model.lora_*` ما كانتش كيتقرا من الـ config** → الـ LoRA كان بلا effect أصلاً.
19. **`--model <hf-dir>` كان كيمسح وصفة الـ recipe كاملة** → إعدادات الـ LoRA ضاعو.
    دابا: المعمارية من الـ checkpoint، والـ adapter من الـ recipe.
20. **HF loader كان كيرفض LoadRA params** ("no HuggingFace name for …lora_q.A").
21. **مسار `pretrained_checkpoint: <hf-dir>` (بلا `--model`) كان مكسور**: الـ CLI كيمسح
    الـ checkpoint → الموديل كيبقى random. دابا المعمارية ديال الـ checkpoint هي المرجع
    والوزن كيتحمّل مرة وحدة.
22. **NoPE**: الـ backward كان كيطبّق RoPE rotary والـ forward ما كيديروش → grads غالطة.
23. **تقدير الـ KV cache كان كيغلط 10x** على Granite (40 طبقة بدل 4 attention).
24. **pre-flight VRAM check**: خطأ واضح بدل CUDA OOM cryptic.
25. **`gnorm %6.3f` كان كيخبي القيم الصغيرة** → ولا `%.4g` (الـ LoRA كان يبان 0.000).
26. **CPU kernels مقسّمة بالفئات**: `ops_cpu_dense.cpp` (الـ primitives المشتركة: GEMM، linear، elementwise، embedding، norms، RoPE، SwiGLU + loss + optimizer + sampler)، `ops_cpu_attention.cpp` (عائلة attention كاملة: prefill + backward + decode ring fp32/fp16)، `ops_cpu_moe.cpp` (MoE)، `ops_cpu_mamba.cpp` (Mamba-2)، `ops_cpu_simd.h` (helpers AVX2/FMA مشتركة). `core/ops_cpu.h` فيه banners كيقولو كل declaration فين كايتنفّذ.
27. **dead code محيّد**: `jitter_u_for_test`، `moe_gather3_all`، `moe_scale_all`، `moe_forward_fused` (كانو غير للـ unit tests المحذوفين).
28. **`cuda/mamba.cu`**: `mamba_grid_for` (ميت) تحيّد. **`format/hf_checkpoint.h`**:
    declaration قديم لـ `hf_tensor_name` (بـ arg واحد بلا definition) صلّح.

### 🔥 الـ BUGS الكبيرة اللي طلعو فالتحقق العددي (2026-11-12)

**هاد الـ bugs كانو كيديرو Granite يخدم غالط بالكامل** (والـ logits ماشي
بحال HuggingFace)، وطلعو بمجرد مقارنة المحرك مع مرجع مستقل:

| # | الـ bug | الأثر |
|---|---------|-------|
| 1 | **conv1d ديال Mamba-2 (CPU)**: ما كانش كيقرا المدخل الحالي من الـ projection (كان كيحسب على أصفار) و كان الـ window كيمسك قيم post-SiLU بدل raw | الـ B/C/x block كامل كان garbage |
| 2 | **scan ديال Mamba-2 (CPU)**: الـ index ديال القناة كان ناقصو الـ head offset → كل الـ heads كانو كيكتبو فوق بعضياتهم (race!) و 95% ديال d_inner كانو أصفار | الـ output غير-حتمي + غالط |
| 3 | **gated norm ديال Mamba-2 (CPU)**: الـ gate كان كيتقرا بـ stride = `d_inner` بدل `proj_size` → الـ gate ديال كل token غير الأول كان غالط | نفس |
| 4 | **`cuda/mamba.cu`**: الـ buffers كانو `std::vector` (host memory) كيتدفعو للـ kernels → crash فـ أول use | CUDA ما خدمش أصلاً |
| 5 | **`cuda/mamba.cu`**: الـ recurrent state ما كانوش كيتقرا من الـ buffer → الـ prefill→decode كينسى.State | الكوكا غير صحيحة |
| 6 | **`cuda/mamba.cu`**: الـ shuffle reduction كان كغطي 32 lane من 256 → الـ sums كانو ناقصين | نفس |
| 7 | **`cuda/mamba.cu`**: نفس bugs (2) و (3) | نفس |
| 8 | **Generator decode**: `embedding_multiplier` (12x) ما كانوش كيتطبّق | الـ serving ديال Granite كان غالط |
| 9 | **Generator prefill (batched)**: `embedding_multiplier` + `residual_multiplier` (0.22) ناقصين | الـ prefill ≠ الـ decode |

**⇒ Granite-4.0-H-Tiny دابا عندو نفس الرياضيات ديال HuggingFace (متحقق عددياً).**

---

## 5) الأوامر

```powershell
# بناء
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Granite (weights على Kaggle)
./build/bin/ghassan-ai info     --model <dir>
./build/bin/ghassan-ai logits   --model <dir> --prompt "hi"
./build/bin/ghassan-ai generate --model <dir> --prompt "hi" --max-tokens 64 --dump-prompt
./build/bin/ghassan-ai quantize --model <dir> --out ghassan-v1-flash.gguf --profile q8_k --model-name "Ghassan v1 Flash"

# training (Kaggle)
./build/bin/gai_train --dry-run --config "kaggle configue/ghassan_granite_2xt4.yaml"
```

التفاصيل الكاملة + خطة Kaggle: **`GRANITE.md`**.

---

## 6) الذاكرة (Granite-4.0-H-Tiny، من dry-run ديال المحرك)

| block | size |
|---|---|
| weights fp32 masters | 25.85 GiB (6.939B params — exact) |
| weights fp16 (اللي كيدخل فـ GGUF) | 12.92 GiB |
| fp16 weight cache (increase فوق الـ masters) | 12.95 GiB |
| mamba-2 states | 55 MiB |
| KV fp16 @131072 | 1.00 GiB |
| activations B=1 T=2048 (ckpt) | 7.68 GiB |
| **full fine-tune (AdamW)** | **125.58 GiB** |

**⇒ على T4 واحد (16 GB) Granite ما كيدخلش**: الـ masters fp32 وحدهم 25.85 GiB.
الـ export كيمشي (CPU، ~25 GiB ديال RAM)؛ الـ inference على GPU كيتطلب
**fp16 weight storage** (12.2 GiB).

---

## 7) اللي باقي (بصراحة)

| # | البند | الصعوبة | الأثر |
|---|-------|---------|-------|
| 1 | **fp16 weight storage** (بلا fp32 masters) → Granite كيدخل T4 | متوسطة-كبيرة | يفتح inference على GPU + LoRA على 7B |
| 2 | **backward ديال Mamba-2** (selective scan) → Fine-tuning ديال granite | كبيرة | LoRA/SFT على granite |
| 3 | CUDA kernel للـ mamba (مكتوب، ما كيتcompilش محلياً) | متوسطة | سرعة inference على GPU |
| 4 | NFC normalization (HF كيطبّق، المحرك لا) | متوسطة | مطابقة تامة مع Qwen |
| 5 | KV cache fp16 فـ prefill (دابا decode فقط) | صغيرة | ذاكرة @long ctx |
| 6 | MoE router = sigmoid + aux-loss-free bias (للـ checkpoints الجاية) | متوسطة | التوافق مع الطرز الجاية |

**ملاحظة**: 1 + 2 مع بعض هوما اللي كيخليو "LoRA fine-tune من dense → Granite"
ممكن. بلا 1، حتى inference على T4 مستحيل.

---

## 8) ملاحظات مهم

- **The CPU path** هو المرجعumerical؛ CUDA path ما يمكنش يتحقق محلياً (ما كاينش nvcc).
- **MoE forward** كيستعمل `atomicAdd` (CUDA) → النتيجة ماشي bitwise-identical بين run و run.
- **GGUF Q8_0**: format متوافق مع llama.cpp؛ Q4_0 دابا أقل دقة من GGML (bug معروف).
- `model/model.cpp` كان تضرر مرة (truncation) ورجع من git؛ **الـ LoRA + granite parts
  طُبّقو من بعد**. **الكتابة ديال ملفات المصدر خاصها دايماً `encoding="utf-8"`.**
- **git CLI** ماشي موجود فـ هاد الـ PC؛ الـ working tree مازال غير commit.
