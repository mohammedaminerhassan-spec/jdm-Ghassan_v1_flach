# Ghassan AI Engine - تقرير الأخطاء (code review)

مراجعة كاملة للكود (~14,000 سطر / 70 ملف).
كل خطأ مذكور بصيغة: **الموقع** <- **السبب** <- **الأثر** <- **الإصلاح**.

**الحالة: كل الأخطاء العشرة اتصلّحت + اتضاف خطآن جديدان (BUG-11 / BUG-12) اتلقوا اختبارات.**

| # | الخطأ | الخطورة | الملف | الحالة |
|---|--------|---------|--------|--------|
| BUG-01 | الـ FFN في الـ backward ناقصه `residual_multiplier` | حرجة | `model/model.cpp` | اتصلّح |
| BUG-02 | MoE backward (CPU) يستعمل عرض الـ routed بدل الـ shared | حرجة | `core/ops_cpu_moe.cpp` | اتصلّح |
| BUG-03 | قيد cuBLAS على الـ leading dimension | حرجة | `cuda/kernels.cu`, `cuda/attention.cu` | اتصلّح |
| BUG-04 | مسار الـ sampling السريع يقرأ logits قديمة | عالية | `inference/generator.cpp` | اتصلّح |
| BUG-05 | الـ checkpoint ما بيحفظش `min_ns_dim` بتاع Muon | متوسطة | `training/checkpoint.cpp` | اتصلّح |
| BUG-06 | الـ fp16 mirror للـ dS buffer كامل | متوسطة | `cuda/attention.cu` | اتصلّح |
| BUG-07 | كيرنلات attention الـ tiled ميتة | متوسطة | `cuda/attention.cu`, `core/ops*` | اتحذفت |
| BUG-08 | خاتم phase timers بينزحلق | متوسطة | `cuda/cuda_utils.cu` | اتصلّح |
| BUG-09 | `front_drop` في الـ prefill بيعمل فجوة في المواضع | كامنة | `inference/generator.cpp` | اتصلّح |
| BUG-10 | `merge_lora` بطيء جدًا (حلقة host بـ O(out*in*r)) | أداء | `model/model.cpp` | اتصلّح (GEMM على الـ device) |
| BUG-11 | **جديد**: الـ loader ما بيحمّلش الـ shared expert في الموديلات non-Granite | حرجة | `format/hf_checkpoint.cpp` | اتصلّح |
| BUG-12 | **جديد**: `shared_expert_intermediate_size` كان بيتجاهَل في الـ config | حرجة | `format/hf_checkpoint.cpp` | اتصلّح |

**الاختبارات الجديدة** (الـ `CMakeLists.txt` كان بيبني `tests/test_*.cpp` من زمان والمجلد كان فاضي):

| الاختبار | إيه بيعمل |
|---|---|
| `tests/test_gradient_check.cpp` | finite-difference على التدرّجات: بيمسك BUG-01 و BUG-02 فورًا |
| `tests/test_safetensors_roundtrip.cpp` | يكتب safetensors -> يقرأها -> يقارن (dense + MoE + Granite) |

```bash
cmake -S . -B build -G Ninja -DGAI_ENABLE_CUDA=OFF -DGAI_BUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

---

## BUG-01 🔴 — الـ FFN في الـ backward ناقصه عامل `residual_multiplier`

**الموقع:** `model/model.cpp` — `Model::forward_backward()`، فرع الـ FFN (الأسطر ~2430–2494)

### السبب

الـ forward بيطبّع الـ FFN قبل الـ residual add:

```cpp
// model.cpp:2287-2290  (forward — صحيح)
if (cfg_.residual_multiplier != 1.0f) {
    ops::scale_inplace(dev, act.ffn_out.f32(), cfg_.residual_multiplier, N * d);
}
ops::add_inplace(dev, act.x.f32(), act.ffn_out.f32(), N * d);
//  =>  x_out = x_mid + m * FFN(x_mid)
```

falسه `∂L/∂(FFN output)` لازم يساوي `m · ∂L/∂x_out`.

الـ backward عنده **فرعين صح** بيطبّقوا العامل:

```cpp
// model.cpp:2508-2510  (فرع Mamba — صحيح)
const float* dy_mamba = act.dx.f32();
if (cfg_.residual_multiplier != 1.0f) {
    ops::copy(dev, act.dtmp.f32(), act.dx.f32(), N * d);
    ops::scale_inplace(dev, act.dtmp.f32(), cfg_.residual_multiplier, N * d);
    dy_mamba = act.dtmp.f32();
}

// model.cpp:2554-2556  (فرع Attention — صحيح)
const float* dy_attn = act.dx.f32();
if (cfg_.residual_multiplier != 1.0f) {
    ops::copy(dev, act.dtmp.f32(), act.dx.f32(), N * d);
    ops::scale_inplace(dev, act.dtmp.f32(), cfg_.residual_multiplier, N * d);
    dy_attn = act.dtmp.f32();
}
```

لكن **فرع الـ FFN ناقص**:

```cpp
// model.cpp ~2432-2494 — مافيش أي scaling لـ act.dx
if (L.has_moe()) {
    ops::moe_backward(dev, act.saved_xb2[sl], ..., act.dx.f32(), act.dxb.f32(), ...);
    //                                                             ^^^^^ بدون m
} else {
    ops::linear_backward(dev, act.saved_act[sl], L.w_down.w.f32(),
                         act.dx.f32(), act.dact.f32(), N, F, d);
    //                                    ^^^^^ بدون m
}
```

الـ FFN path كاملة (dense و MoE) بتستقبل `dL/dx_out` بدل `m · dL/dx_out`.

### الأثر

- `∂L/∂(FFN params)` ناقصة العامل `m` → التدرّج **أكبر بـ `1/m`**.
- في Granite-4.0 (`residual_multiplier ≈ 0.13` لـ h-tiny) الـ FFN بيتدرّب **~7–8× أقوى** من الـ attention.
- **مفيش NaN ولا error** — صامت تمامًا: الـ loss بينزل لكن النموذج مائل.
- `residual_multiplier == 1.0` (كل الموديلات غير Granite) → مفيش أثر.

### الإصلاح

```cpp
// في بداية فرع الـ FFN داخل الحلقة العكسية للطبقات:
const float* dy_ffn = act.dx.f32();
if (cfg_.residual_multiplier != 1.0f) {
    ops::copy(dev, act.dtmp.f32(), act.dx.f32(), N * d);
    ops::scale_inplace(dev, act.dtmp.f32(), cfg_.residual_multiplier, N * d);
    dy_ffn = act.dtmp.f32();
}
// ثم استعمل dy_ffn بدل act.dx في:
//   - moe_backward(..., dy_ffn, act.dxb.f32(), ...)
//   - linear_backward(..., dy_ffn, act.dact.f32(), ...)
```

⚠️ **انتبه:** `act.dtmp` مستعمل في فرعي Attention و Mamba أيضًا. لازم نتأكد إن ترتيب
الاستعمالات لا يعيد استخدام `dtmp` قبل ما يخلص. الأنظف: خد buffer جديد `act.dxb2_ffn`
أو خزّن الـ dy في متغيّر محلي وقسّم الـ buffer.

### كيف تتحقق

- Gradient check عددي (finite differences) على موديل صغير بـ `residual_multiplier = 0.25`:
  لازم الـ FFN grad ينزل بـ factor 4.
- أو: قارن تدرّج FFN على CPU مقابل CUDA — لازم يتطابقوا (BUG-02 بيخليهم يفترقوا كمان).

---

## BUG-02 🔴 — MoE backward (CPU) يستعمل عرض الـ routed expert بدل الـ shared expert

**الموقع:** `core/ops_cpu_moe.cpp` — `moe_backward()`، الأسطر **356–368**

### السبب

الـ forward بيستعمل `SH` (عرض الـ shared expert) صح:

```cpp
// ops_cpu_moe.cpp:166-170  (moe_forward — صحيح)
if (sh_g && sh_u && sh_d) {
    linear_forward(xt, sh_g, sc.sg.data(), 1, d, SH);
    linear_forward(xt, sh_u, sc.su.data(), 1, d, SH);
    swiglu_forward(sc.sg.data(), sc.su.data(), sc.sa.data(), SH);
    linear_forward(sc.sa.data(), sh_d, outt, 1, SH, d);
}
```

لكن الـ backward بيستعمل **`E`** (عرض الـ routed expert):

```cpp
// ops_cpu_moe.cpp:356-368  (moe_backward — خاطئ)
if (sh_g && sh_u && sh_d && dsh_g && dsh_u && dsh_d) {
    linear_forward(xt, sh_g, sc.g.data(), 1, d, E);            // ❌ E → SH
    linear_forward(xt, sh_u, sc.u.data(), 1, d, E);            // ❌ E → SH
    swiglu_forward(sc.g.data(), sc.u.data(), sc.a.data(), E);  // ❌ E → SH
    std::fill(sc.da.begin(), sc.da.begin() + E, 0.0f);         // ❌
    linear_backward(sc.a.data(), sh_d, dt, sc.da.data(), dsh_d, 1, E, d);   // ❌ E → SH + ldb غلط
    std::fill(sc.dg.begin(), sc.dg.begin() + E, 0.0f);         // ❌
    std::fill(sc.du.begin(), sc.du.begin() + E, 0.0f);         // ❌
    swiglu_backward(sc.g.data(), sc.u.data(), sc.da.data(), sc.dg.data(), sc.du.data(), E); // ❌
    linear_backward(xt, sh_g, sc.dg.data(), sc.dx.data(), dsh_g, 1, d, E);   // ❌ E → SH
    linear_backward(xt, sh_u, sc.du.data(), sc.dx.data(), dsh_u, 1, d, E);   // ❌ E → SH
}
```

**مشكلة الـ leading dimension:** `linear_backward(x, w, dy, dx, dw, M, K, N)` بتنادي
`gemm(false,false,M,K,N, dy, N, w, K, ...)` — يعني `w` بيتعامل على إنّه `[K,N]` بـ `ld = K`.
لكن `sh_d` هو فعليًا `[d, SH]` (الـ ld الصحيح = `SH`). لما نمرّر `K = E`:

- الـ GEMM بيقرأ `sh_d` بـ stride `E` بدل `SH` → **قيم خاطئة** (مش out-of-bounds لأن `E ≤ SH`، لكن غلط رياضيًا).
- `dsh_d` (تدرّج الـ shared expert) بينكتب بـ layout `[E, d]` بدل `[d, SH]` → **تخريب الـ buffer** بتدرّجات الـ shared expert.

**ملاحظة:** الـ `BwdScratch` (ops_cpu_moe.cpp:62-80) فيه مصفوفات `sg/su/sa/dsg/dsu/dsa` بحجم `SH`
موجودة و**ما بتستعملش خالص** — الكود بيستعمل `g/u/a/dg/du/da` (حجم `E`).
الدمج بين الكود والـ scratch يوضّح إن الفكرة الأصلية كانت `SH` والـ refactor نسي يحدّثها.

### الأثر

- **Granite-4.0** (`SH=1024`, `E=512`): الـ shared expert بيتدرّب بقيم خاطئة + الـ buffer بتاع تدرّجه بيتخرب.
- **انقسام CPU/GPU**: المسار على CUDA (`cuda/moe.cu:974-987`) **صحيح** (بيستعمل `SH` في كل مكان).
  → نفس الـ config بيدّي نموذجين مختلفين على CPU مقابل GPU.
- الموديلات اللي `SH == E` (Qwen3-MoE وغيره) → مفيش أثر.

### الإصلاح

```cpp
if (sh_g && sh_u && sh_d && dsh_g && dsh_u && dsh_d) {
    linear_forward(xt, sh_g, sc.sg.data(), 1, d, SH);
    linear_forward(xt, sh_u, sc.su.data(), 1, d, SH);
    swiglu_forward(sc.sg.data(), sc.su.data(), sc.sa.data(), SH);
    std::fill(sc.dsa.begin(), sc.dsa.begin() + SH, 0.0f);
    linear_backward(sc.sa.data(), sh_d, dt, sc.dsa.data(), dsh_d, 1, SH, d);
    std::fill(sc.dsg.begin(), sc.dsg.begin() + SH, 0.0f);
    std::fill(sc.dsu.begin(), sc.dsu.begin() + SH, 0.0f);
    swiglu_backward(sc.sg.data(), sc.su.data(), sc.dsa.data(),
                    sc.dsg.data(), sc.dsu.data(), SH);
    linear_backward(xt, sh_g, sc.dsg.data(), sc.dx.data(), dsh_g, 1, d, SH);
    linear_backward(xt, sh_u, sc.dsu.data(), sc.dx.data(), dsh_u, 1, d, SH);
}
```

### كيف تتحقق

- Gradient check عددي على موديل MoE صغير بـ `SH != E` (مثلاً `moe_expert_dim=32`, `shared_expert_intermediate_size=64`).
- أو: نفس الـ seed على CPU و CUDA → لازم الـ gradients يطابقوا.

---

## BUG-03 🔴 — قيد cuBLAS على الـ leading dimension (يرمي INVALID_VALUE)

**الموقع:**
- `cuda/kernels.cu` — `gemm()` (سطر 252، الـ `cublasSgemm` سطر 285) و `gemm_fp16()` (سطر 122، الـ `cublasGemmEx` سطر 136)
- `cuda/attention.cu` — `attn_sbgemm()` (سطر 326، الـ `cublasGemmGroupedBatchedEx` سطر 369/383)
- `training/optimizer.cpp` — `Muon::orthogonalize()` (الأسطر 326-327) ← **أصرح مثال**

### السبب

المحرّك بيحوّل row-major → cuBLAS column-major وبيعطِى الـ leading dimension مباشرة:

```cpp
// cuda/kernels.cu:285-286
cublasStatus_t s = cublasSgemm(h, opA, opB, N, M, K,
                               &alpha, B, ldb, A, lda, &beta, C, ldc);
//                                          ^^^^^     ^^^^
//   cublas B-operand = engine's A, ldb = engine's lda
```

قواعد cuBLAS (مستنداتها):
- الـ B-operand مع `CUBLAS_OP_N`: لازم `ldb >= n` (= `M` بتاع المحرّك).
- الـ B-operand مع `CUBLAS_OP_T`: لازم `ldb >= k` (= `K` بتاع المحرّك).

النتيجة، حسب المسار:

| الاستدعاء | الشرط اللازم | تكسر متى |
|---|---|---|
| `linear_forward(x[M,K], w[N,K])` | `K >= M` | `hidden_size < عدد التوكنز` |
| `linear_backward` → dx | `out_dim >= M` | `out_dim < عدد التوكنز` |
| `linear_backward` → dw | `out_dim >= M` | نفس |
| مسار attention GEMM | `H*hd >= T` | `T > H*hd` |
| **Muon Newton-Schulz** | **`cols >= rows`** | **`hidden < FFN_width`** |

**أصرح مثال — Muon:**

```cpp
// training/optimizer.cpp:326-327
ops::gemm(dev, true,  false, cols, cols, rows, 1.0f, O, cols, O, cols, 0.0f, A, cols);
ops::gemm(dev, false, false, rows, cols, cols, 1.0f, O, cols, A, cols, 0.0f, T, cols);
```

`trans_a=1` → الـ B-operand بـ `OP_T` → لازم `ldb(=cols) >= k(=rows)` → **`cols >= rows`**.
يعني لازم **hidden_size >= FFN_width**. نموذجيًا `FFN_width > hidden_size`، مثلاً:

| الموديل | hidden | FFN | الشرط |
|---|---|---|---|
| Qwen3-1.7B | 2048 | 8960 | ✗ **مخالفة** |
| Qwen3-0.6B | 1024 | 3072 | ✗ **مخالفة** |
| Llama-3.2-1B | 2048 | 8192 | ✗ **مخالفة** |

يعني `training.optimizer: muon` **بيقع على GPU** بـ `CUBLAS_STATUS_INVALID_VALUE`.
و**الـ CPU عنده مفيش القيد ده** (`core/ops_cpu_dense.cpp` بيتعامل مع أي stride) →
**انقسام سلوك**: Muon بيشتغل على CPU وبيقع على GPU.

**ومثال تاني — طول السياق:** الـ default في `TrainerConfig` هو `batch_size=8, seq_len=1024`
= 8192 توكن. موديل صغير بـ `hidden=768` → `768 < 8192` → **أول `linear_forward` يطلع INVALID_VALUE**.
الـ guard الموجود في `training/trainer.cpp:460-471` بيغطي الموديلات > 800M بس.

### الأثر

- `cublasSgemm failed: status=1 (INVALID_VALUE)` → Training بيقع (مش صامت) لكن الرسالة
  مضلّلة (بتفتح على "host-side shape bug" بدل ما توضّح القيد الحقيقي).
- Muon معطّل فعليًا على GPU لأي موديل FFN عادي.
- أي recipe بطول تسلسل > hidden_size (أو > H·hd) بتفشل.

### الإصلاح المقترح

الخيار الأنضف: **تبديل ترتيب الـ operands** بحيث يبقى الـ ld مطابق لشرط cuBLAS دايمًا،
أو **حزم الـ matrices** بـ leading dimension صحيح (lda = عدد الصفوف الفعلي) قبل cuBLAS.

عمليًا (في `gemm` و `attn_sbgemm`): بدل تمرير `lda`/`ldb` الخام، استعمل
`cublasSgemmStridedBatchedEx` مع `strideA/strideB`، أو اعمل pack للـ matrix بـ ld = عدد
الأعمدة الحقيقي (نفس أسلوب `gemm_fp16` بالظبط: `k_f32_to_f16_strided` بيحزم بـ stride ويجعل ld متسق).

**حل وسط سريع (لو مش عايز تعيد كتابة الـ GEMM):** أضف check صريح في `ops::gemm` قبل
النداء، **وقبله** الـ trainer:

```cpp
// في ModelConfig::validate() أو في trainer: أضف الحارس
GAI_CHECK(static_cast<i64>(hidden_size) >= static_cast<i64>(batch_size) * seq_len,
          "tokens per micro-batch must be <= hidden_size (cuBLAS leading-dimension limit)");
```

لكن هذا **يحدّ الـ recipes بشكل قسري**، فالحل الصح هو إصلاح الـ wrapper.

### كيف تتحقق

- `cublasSgemm` بيرجّع `CUBLAS_STATUS_INVALID_VALUE` (status=1) — ابدأ بأقصر حالة:
  موديل `hidden=64`, `hidden FFN=128`, batch يتجاوز 64 توكن.
- قارن نتيجة `ops::gemm` بين CPU و CUDA على أبعاد صغيرة ومتوسطة.

---

## BUG-04 🟠 — مسار الـ sampling السريع يقرأ logits قديمة

**الموقع:** `inference/generator.cpp` — `Generator::generate()` (الأسطر 829–903)

### السبب

قرار استخدام المسار السريع (على الـ GPU) بيعتمد على حجم الـ history:

```cpp
// generator.cpp:829-835
auto use_fast_for = [&](int hist_size) {
    const int w = sc.repetition_window > 0 ? sc.repetition_window : hist_size;
    const bool topk_ok = want_greedy || (K > 0 && V >= K && V <= 262144);
    return sc.gpu_fast_sample && dev == Device::CUDA && V >= 512 &&
           w <= FAST_HIST_MAX && sc.no_repeat_ngram == 0 && topk_ok;
};
```

لما `sampling.repetition_window == 0` (قيمة مسموحة، `sampler.cpp:64` بيت normalizing سالب بس)
الـ `w` = حجم الـ history الحالي. أول ما الـ history يتعدّى **2048** (`FAST_HIST_MAX`)،
`use_fast` بيولي **false** وسط التوليد، فيدخل المسار البطيء:

```cpp
// generator.cpp:864
next = sampler.sample(logits_host_.data(), V, history_);
```

لكن `logits_host_` **ما بيتحدّثش** في المسار السريع: المسار السريع بينادي
`decode_step_logits()` (سطر 892) وهي **ما بتنسخش** للـ host. النسخ للـ host بيحصل في:
- `decode_step()` (سطر 560) — المسار البطيء بس
- آخر سطر من `generate()` (سطر 900-901) — بعد انتهاء التوليد

يعني `sampler.sample()` بيشتغل على **logit vector من الـ prefill** (قديمة).

### الأثر

- توليد ≥ ~2048 توكن مع `repetition_window: 0` → **توكنز عشوائية/مكرّرة** في آخر التوليد.
- مفيش error ولا warning — صامت.
- الـ default `repetition_window = 128` → `w` ثابتة → المسار ثابت → مفيش أثر (اللي هو الـ default).

### الإصلاح

اختيار المسار **مرة واحدة** في بداية `generate()` بدل إعادة حسابه كل خطوة:

```cpp
const bool use_fast = use_fast_for(static_cast<int>(history_.size()));  // مرة واحدة
for (int step = 0; step < limit; ++step) { ... }
```

أو (أأمن): خلي `decode_step_logits()` تنسخ للـ host دايمًا، أو خلي `logits_host_` يُحدَّث
في المسار السريع قبل `sampler.sample()`.

---

## BUG-05 🟡 — الـ checkpoint ما بيحفظش `min_ns_dim` بتاع Muon

**الموقع:** `training/checkpoint.cpp:333` (`write_optimizer_snapshot`) مقابل
`training/optimizer.cpp:480-522` (`Muon::save_state`) و `537-597` (`Muon::load_state`)

### السبب

في **مسار الـ checkpoint المستعمل فعليًا** (`Checkpoint::save` → `write_optimizer_snapshot`)،
الـ Muon يُكتب بـ `fmt = 1` **بدون** `min_ns_dim`:

```cpp
// checkpoint.cpp:333-334
const u8 fmt = 1;      // نفس الـ fmt لكل الـ optimizers
const i64 step = opt.step;
...
// checkpoint.cpp:372-401  (Muon)
wr(f, opt.muon.lr); wr(f, opt.muon.vec_lr_ratio); wr(f, opt.muon.beta1);
wr(f, opt.muon.beta2); wr(f, opt.muon.eps); wr(f, opt.muon.weight_decay);
wr(f, opt.muon.grad_clip);
{ i32 ns = opt.muon.ns_steps; wr(f, ns); }     // ❌ مافيش min_ns_dim
```

لكن `Muon::save_state` (المسار القديم/البديل) بيكتب `fmt = 2` **معاه**:

```cpp
// optimizer.cpp:484-500
const u8 fmt = 2;      // ✅ فيه min_ns_dim
...
{ i32 ns = ...; wr(f, ns); i32 min_ns = ...; wr(f, min_ns); }
```

و`Muon::load_state` بيتوقع:
```cpp
// optimizer.cpp:552-554
if (fmt == 2) { if (!is.read(&saved_min_ns, 4)) return false; }
// optimizer.cpp:571-580
if (fmt == 2 && saved_min_ns != cfg_.min_ns_dim) { log_error(...); return false; }
if (fmt == 1 && cfg_.min_ns_dim != 0) { log_warn("...cannot verify muon min_ns_dim..."); }
```

### الأثر

- `min_ns_dim` **مفقود من كل checkpoint جديد** → في وضع `exact` الـ check بيتخطّى
  (FORMAT 1) والـ warning بيشتغل **دايمًا**.
- `min_ns_dim` بيحدد أي الـ matrices بتاخد Newton-Schulz وأيها بتاخد Lion/AdamW.
  تغييره بين الـ run والـ resume → **تغيير في قاعدة التحديث** من غير ما الـ exact resume يحس.
- ده بيكسر بالظبط العقد اللي الكود عايز يحميه ("the NS/vec branch assignment would change").

### الإصلاح

خلّي الـ fmt بتاع الـ Muon في `write_optimizer_snapshot` يبقى `2` (أو `1 + min_ns_dim`).
لاحظ إن `fmt` مشترك في الدالة، فالأفضل تفريعه حسب الـ kind:

```cpp
// checkpoint.cpp:333 — بدل const u8 fmt = 1;
const u8 fmt_adamw = 1, fmt_lion = 1, fmt_muon = 2;
// وفي حالة Muon:
wr(f, fmt_muon);
...
{ i32 ns = opt.muon.ns_steps; wr(f, ns);
  i32 min_ns = opt.muon.min_ns_dim; wr(f, min_ns); }
```

**تحذير:** تغيير الـ format لازم يتوافق مع `Muon::load_state` (اللي دلوقتي بيقبل `fmt==1 || fmt==2`).
لو في checkpoints موجودة بـ fmt=1، لازم يفضلوا يقرؤوا.

---

## BUG-06 🟡 — attention backward بيعمل mirror بـ fp16 للـ dS buffer كامل

**الموقع:** `cuda/attention.cu:326-398` (`attn_sbgemm`، الفرع الـ fp16)

### السبب

في الفرع الـ fp16، الكود بيعمل "mirror-convert" للنطاقات **الكاملة**:

```cpp
// attention.cu:354-361
if (use_fp16) {
    attn_pool_ensure(Aspan + Bspan, 0, 0);
    attn_cvt(Brange, g_attn_mir, Bspan);
    attn_cvt(Arange, g_attn_mir + Bspan, Aspan);   // ← Aspan قد تكون BH*T*T
    ...
}
```

وفي `attention_backward_ex`، الاستدعاءات هي:

```cpp
// attention.cu:899-902  (dq = dS * K)
attn_sbgemm(false, false, T, hd, T, 1.0f,
            As.data(), T, dS, (size_t)BH * T * T,   // ← Aspan = BH*T*T
            Bs.data(), KV * hd, k, (size_t)B * T * KV * hd, ...);
// attention.cu:920-923  (dk = dS^T * Q)  → نفس Aspan
// attention.cu:925-928  (dv = P^T * dO)  → Aspan = BH*T*T (probs)
```

يعني **buffer الـ dS/probs كامل** (حجم `B·H·T·T`) بيتحوّل لـ fp16 **في كل طبقة وكل خطوة**،
و**3 مرات** في الـ backward.

### الأثر

- مثال B=1, H=16, T=1024 → `BH*T*T` = 16.7M floats = 67 MB → mirror بـ 33 MB،
  يتحوّل 3× في كل طبقة (26 طبقة = 78 تحويل لـ 67 MB = ~5 GB قراءة/كتابة إضافية لكل خطوة).
- ذاكرة إضافية دائمة (33 MB) + ضغط على الـ T4 (16 GB).
- slowdown واضح في الـ step time على T4.

### الإصلاح

- خفّض عتبة الـ fp16 (`ops::gemm_fp16_mnk_threshold`) لمسار الـ attention تحديدًا، أو
- استعمل نسخة **fp32** للـ dS (المصفوفة صغيرة نسبيًا مقارنة بـ dq/dk/dv) وخلّي الـ fp16
  للـ operands الصغيرة بس، أو
- حوّل الـ dS **مرة واحدة** للـ fp16 وشاركه بين الـ dq/dk/dv بدل 3 تحويلات.

---

## BUG-07 🟡 — كيرنلات attention الـ tiled ميتة (بلا اختبار parity)

**الموقع:** `cuda/attention.cu`
- `k_attn_fwd` (سطر 59) + `attention_forward` (سطر 161)
- `k_attn_fwd_swa` (سطر 178)
- `k_attn_bwd_dq` (سطر 595) + `k_attn_bwd_dkv` (سطر 651) + `attention_backward` (سطر 728)

### السبب

الموديل **دايمًا** بيستدعي `attention_forward_ex` / `attention_backward_ex`
(اللي بيستخدموا مسار GEMM). الدوال `attention_forward` / `attention_backward`
(غير الـ ex) **ما بتتنادىش** من أي مكان في مسار التدريب أو الاستدلال.

### الأثر

- ~400 سطر من كيرنلات CUDA **بلا أي تنفيذ فعلي** → أي bug جواها مش هيتكشّف.
- **الملاحظة:** `k_attn_bwd_dq` فيه buffer مشترك اسمه `dots` بين الكيرنلتين — لو الـ launch order
  اتغيّر، الـ dots ممكن تتقرا قبل ما تُكتب.
- **الخطر:** لو حد محتاج في المستقبل استعمل `attention_forward` (مثلاً ل faster path)، هو مجرّب.

### الإصلاح

- **احذف** الكيرنلات الميتة (فحص الـ `grep` الأول: `attention_forward(` و `attention_backward(`).
- **أو** استعملها فعليًا وضيف test parity مقابل الـ GEMM path.

---

## BUG-08 🟡 — خاتم phase timers بينزحلق مع start بلا stop

**الموقع:** `cuda/cuda_utils.cu:212-227`

### السبب

```cpp
void phase_start(int ph) {
    phase_pool();
    PhaseSlot& sl = g_slots[g_head];     // نفس الـ slot
    sl.ph = ph; sl.used = true;
    cudaEventRecord(sl.s, 0);
}
void phase_stop(int ph) {
    phase_pool();
    PhaseSlot& sl = g_slots[g_head];     // نفس الـ slot
    cudaEventRecord(sl.e, 0);
    g_head = (g_head + 1) % kPhaseRing;  // التقديم هنا فقط
}
```

`phase_start` و `phase_stop` بيشاركوا نفس الـ `g_head`. لو:
- `phase_start` بلا `phase_stop`-matching → الـ slot بيتكتب فوقه والـ `g_head` ما بيتقدّمش.
- `phase_start` مرتين متتاليتين → الـ event الخاص بالـ start الأول بيتداس.

كمان `phase_reset()` (سطر 228) بيعمل `used=false` لكل الـ slots **بدون** ما يصفّر `g_head`.

### الأثر

- **كامن حاليًا**: الـ trainer بيستعمل phase الـ Load بس، و`ops::phase_start/stop` بيعودوا
  قبل ما يوصلوا لـ CUDA في حالة الـ Load (`core/ops.cpp:134-160`).
- لو حد فعّل phases تانية (Attn/Moe) → التوقيتات بتبوظ بصمت، و`phase_report()` هيتبني على
  events غير متطابقة.

### الإصلاح

```cpp
void phase_start(int ph) {
    phase_pool();
    PhaseSlot& sl = g_slots[g_head];
    if (sl.used) { /* الـ slot السابق غير المغلق: اقفله */ }
    sl.ph = ph; sl.used = true;
    cudaEventRecord(sl.s, 0);
}
```
أو: خلي `phase_start` يحجز slot (`g_head++`) و`phase_stop` يقفل آخر slot مفتوح
(ـ stack بدل ring).

---

## BUG-09 ⚪ — `front_drop` في الـ prefill بيعمل فجوة في المواضع (كامنة)

**الموقع:** `inference/generator.cpp:783-793`

### السبب

```cpp
const int base = drop + front_drop;
decode_step(work[0], 0);                       // ← التوكن الأول موضعه 0
for (size_t i = 1; i < work.size(); ++i) {
    int pos = base + static_cast<int>(i);      // ← الباقي بيبدأ من base+1
    decode_step(work[i], pos);
}
absolute_pos_ = front_drop + tokens.size();
```

مع `front_drop > 0`، المواضع بتبقى `0, front_drop+1, front_drop+2, ...` — **فجوة** بين
التوكن الأول والتاني (inober مقطع RoPE متّصل).

### الأثر

- **كامن**: المسار دلوقتي ما بينادىش بـ `front_drop != 0` (النداءات في
  `generator.cpp:917` و `955` بيستعملوا 0).
- لو الـ multimodal front-end استعمل `front_drop` → صدى RoPE مكسور.

### الإصلاح

```cpp
const int base = drop + front_drop;
for (size_t i = 0; i < work.size(); ++i) {
    int pos = (i == 0) ? 0 : base + static_cast<int>(i);
    decode_step(work[i], pos);
}
```
أو الأوضح: `base = drop + front_drop` والتوكن الأول موضعه `base` (وليس 0).

---

## BUG-10 ⚪ — `merge_lora` بطيء جدًا

**الموقع:** `model/model.cpp:1257-1283`

### السبب

```cpp
for (int o = 0; o < out_d; ++o) {
    const float* brow = &Bc[o * r];
    float* wrow = &Wc[o * in_d];
    for (int rr = 0; rr < r; ++rr) {
        const float b = brow[rr];
        if (b == 0.0f) continue;
        const float f = sc * b;
        const float* arow = &Ac[rr * in_d];
        for (int i = 0; i < in_d; ++i) wrow[i] += f * arow[i];  // O(out·in·r) على الـ host
    }
}
```

حلقة ثلاثية على الـ host بـ float، single-threaded. لـ موديل 1B (`out=2048, in=2048, r=16`)
= 67 مليون mult-add → ثواني. لـ موديل 6.9B (Granite) → **دقائق**.

كمان الـ copy الكامل للـ host (`Wc`) لـ 6.9B = 27 GB → **OOM على الـ host**.

### الأثر

- `-export` بعد LoRA training بيبقى بطيء جدًا أو بيقع ب OOM على الموديلات الكبيرة.
- نفس الملاحظة لـ `Model::save_raw` (`model.cpp:2677`) — بيكتب كل weights بـ float على الـ host.

### الإصلاح

استعمل `ops::gemm` على الـ device:
```cpp
// W[out,in] += scale * B[out,r] @ A[r,in]
ops::gemm(device_, false, false, out_d, in_d, r, sc, B_ptr, r, A_ptr, in_d, 1.0f, W_ptr, in_d);
```
مع التأكد إن `B` و `A` متاحين على الـ device (حاليًا الـ `device_copy` بينزّلهم للـ host).

---

## ملاحظات إضافية (مش أخطاء، للمراجعة)

| # | ملاحظة | المكان |
|---|---------|--------|
| N-01 | `ops::linear_forward_fp16` (CPU) بيحلّل كل عنصر بـ `fp16_to_fp32` → O(M·N·K) بطيء جدًا. محتاج vectorization. | `core/ops_cpu_dense.cpp:121` |
| N-02 | `DataLoader::fill_from_shard` بيحطّ `segment_ids = 0` لكل الصف → الـ segment mask ما بيعمل حاجة في الوضع غير-packed. صحيح (الصف جوه مستند واحد)، لكن الـ padding بتاع الـ token id 0 بيفضل موجود. | `training/dataloader.cpp:531` |
| N-03 | `Trainer::run_pretrain` و `run_sft` **مكرّران حرفيًا** (نفس الـ 90 سطر). أي إصلاح لازم يتطبّق في الاتنين. | `training/trainer.cpp:1519-1825` |
| N-04 | `g_moe_jitter` بيُطبَّق على logits الـ router قبل الـ softmax في الـ CPU (`ops_cpu_moe.cpp:145`) وفي الـ CUDA (`k_route`) — متسق ✓، لكن مافيش test بيأكّد تطابق الأرقام بين الباكين. | `core/ops_cpu_moe.cpp`, `cuda/moe.cu` |
| N-05 | `phase_report()` بتعمل `cudaEventElapsedTime` لكل slot في الـ ring (16384) → تكلفة على الـ host كل log line. | `cuda/cuda_utils.cu:240` |
| N-06 | `Model::estimate_activation_bytes_for` بستعمل `sat_add`/`sat_mul` بـ cap = 2^40 — التقدير بيبقى over-estimated في الحالات المتطرفة (وهو Intentional: "the guard always over-estimates"). ✓ مقصود. | `model/model.cpp:1675` |

---

## أشياء اتأكدت منها وطلعت **سليمة** ✅

- `rmsnorm_forward/backward` — الـ backprop صحيح رياضيًا (اتحقق من الـ coef يدويًا).
- `swiglu_backward` — صحيح.
- `softmax_cross_entropy` + z-loss (`scale = 1 + 2·z·logZ`) — صحيح.
- **تدرّج الـ router في MoE** (CPU و CUDA متطابقين): `dL/dp_e = (g_e − ḡ)/S` — صحيح.
- **aux-loss scaling**: `aux_scale = moe_aux_scale · eff_scale · ntok / L` متسق مع
  `loss += moe_aux_scale · aux_total / L` ✓.
- chunked CE + `linear_backward` المتسارع + masked-block skip ✓.
- الـ loss scaler: `grad_scale = 1/(ntok·dscale)` ✓، والـ overflow handling ✓.
- `LrScheduler` (cosine + wsd) ✓.
- resume drift checks (scheduler/arch/tokenizer fingerprint) ✓.
- OOM / RAM / disk guards ✓.
- checkpoint publish (tmp → rename مع backup) ✓.
- BPE encoder (priority queue + merge ranks + byte map) ✓ و GPT-2 byte table ✓.
- MoE forward CUDA: grouping، scatter-add بـ atomicAdd، ترتيب shared قبل routed ✓.
- Mamba-2: conv window + selective scan + layout `[B, n_heads, d_head, d_state]` متسق
  بين CPU training والـ decode ✓.
- `Model::same_architecture_as` + `take_adapter_settings_from` ✓.
- HF checkpoint loader: split الـ expert weights (4 layouts) + shared MLP fused ✓.


---

## BUG-11 (جديد) - الـ loader ما بيحمّلش الـ shared expert في الموديلات non-Granite

**الموقع:** `format/hf_checkpoint.cpp` - `HFCheckpoint::load_model()` (فرع الموديلات non-Granite)

**السبب:** كل الـ MoE leaves (المشترك فيها `moe_sh_gate` / `moe_sh_up` / `moe_sh_down`)
كانت بتدخل في فرع "per-expert" اللي بيقرأ `mlp.experts.<e>.<proj>.weight` - أي إنه
بيقرا الـ experts الموجهين!** مكان الـ shared expert:

```
[hf] expert tensor size mismatch layers.0.moe_sh_gate: file=2048 model=512
```

**الأثر:** أي checkpoint على HuggingFace فيه shared expert (Qwen3-MoE، olmoe، ...)
**ما بيحمّلش أصلاً** - وكتل Sometimes بيكتب تدرّجات الـ shared expert من الـ experts
الموجهين (تلويث صامت).

**الإصلاح:** فرع مستقل للـ shared expert قبل فرع الـ per-expert ✓

---

## BUG-12 (جديد) - `shared_expert_intermediate_size` كان بيتجاهَل في الـ config

**الموقع:** `format/hf_checkpoint.cpp` - `HFCheckpoint::read_config_json()`

**السبب:** القارئ كان بيقرا `moe_expert_dim` بس، ومش بيقرا
`shared_expert_intermediate_size` أصلاً → `shared_width()` كان بيرجع الـ routed width.

**الأثر:** الـ shared expert كان بيتبنى بالعرض الغلط (أصغر من الصح) → الـ load
بيفشل أو (أسوأ) بيبني معمول خاطئ.

**الإصلاح:** القارئ بقى بيقرا الحقل ✓ (وBUG-11 بيقرا الـ tensors الصح).

---

# سجل الإصلاح (إيه اتغيّر فين بالظبط)

| الملف | إيه اتغيّر |
|---|---|
| `model/model.cpp` | BUG-01: `dy_ffn` بيتطبّق فيه `residual_multiplier` قبل دخول الـ FFN (dense + MoE). BUG-10: `merge_lora` بقى GEMM على الـ device. |
| `core/ops_cpu_moe.cpp` | BUG-02: فرع الـ shared expert بقى يستعمل `SH` والـ scratch المخصص (`sg/su/sa/dsg/dsu/dsa`). |
| `cuda/kernels.cu` | BUG-03: `gemm` / `gemm_fp16` / `gemm_bf16` / `linear_forward_fp16` بقى **بيقطّعوا الـ GEMM** على البُعد المخالف للقيد (صفر copy، صفر VRAM إضافي). |
| `cuda/attention.cu` | BUG-03: نفس التقطيع في `attn_sbgemm`. BUG-06: الـ GEMMs اللي الـ A فيها هو مصفوفة `T x T` بقت fp32 (الـ mirror كان 67 MB × 3 × كل طبقة). BUG-07: اتحذفت الكيرنلات الميتة (~430 سطر). |
| `cuda/cuda_utils.cu` | BUG-08: خاتم الـ phase timers بقى يقفل الـ slot المعلّق قبل ما يفتح جديد. |
| `inference/generator.cpp` | BUG-04:قرار المسار السريع بقى يتاخد **مرة واحدة** قبل التوليد. BUG-09: مواضع الـ prefill بقت متتالية. |
| `training/checkpoint.cpp` | BUG-05: Muon بقى يُكتب بـ `fmt=2` مع `min_ns_dim`. |
| `training/trainer.cpp` | `run_pretrain` + `run_sft` اتوحّدوا في `run_stage` (90 سطر مكرر اتشالوا). |
| `training/lora.h` | اتحذف (ميت: مكرر لـ `model::LoraAdapter`). |
| `core/ops.h`, `core/ops.cpp`, `core/ops_cpu.h`, `core/ops_cpu_attention.cpp`, `cuda/cuda_ops.h` | BUG-07: واجهات الـ attention الميتة اتشالت. |
| `format/hf_checkpoint.cpp` | BUG-11 + BUG-12: الـ shared expert (العرض + الـ tensors). |
| `format/hf_writer.{h,cpp}` | **جديد**: الـ safetensors writer (HF export) + `config.json` + الـ sharding. |
| `tools/model_loader.{h,cpp}` | `copy_tokenizer_files()` (نسخ ملفات الـ tokenizer للـ export). |
| `tools/inference_cli.cpp` | `ghassan-ai export --format safetensors --out <dir>`. |
| `tools/training_cli.cpp` | `gai_train --export-format safetensors`. |
| `tests/` | **جديد**: اختبار التدرّجات + اختبار round-trip للـ safetensors. |

## تصميم إصلاح BUG-03 (الأهم)

المحرّك بيحوّل row-major لـ cuBLAS column-major وبيعطِى الـ leading dimension
مباشرة. cuBLAS يشترط:

```
cublasA (engine's B): lda = ldb >= m (=N)  when OP_N
                                   >= k (=K)  when OP_T
cublasB (engine's A): ldb = lda >= n (=M)  when OP_N
                                   >= k (=K)  when OP_T
```

القيد ده كان **بيوقّع LoRA على GPU 100%** (كل GEMM فيه rank=8..64 مع 512 توكن:
`lda = rank < n = tokens`) وبيوقّع Muon على أي موديل `FFN_width > hidden`،
وكل recipe عدد توكناتها > `hidden_size`.

**الحل: تقطيع الـ GEMM على البُعد المخالف** (مش copy ولا packing):

```
trans_a=0 -> يقسّم M (عدد الصفوف) لـ chunks بحجم lda
trans_a=1 -> يقسّم K (بُعد التعاقد) لـ chunks بحجم lda
```

- الـ chunks في M/N بيكتبوا أجزاء **متفرقة** من الـ C (مفيش جمع).
- الـ chunks في K **بيتجمعوا** في نفس الـ C (beta=1 بعد الـ chunk الأول).
- كل operand بيتقدّم بـ **الـ stride الخاص بيه** (مش الكود اللي في الـ C).
- الحالة الشائعة (القيد متحقق) = chunk واحد = **نفس الكود بالظبط** بلا أي تكلفة.

النتيجة: **صفر VRAM إضافي، صفر copy، وتكلفة بتتلعيب في الـ GEMMs الكبيرة.**

---

## BUG-13 (معروف، مش متصلّح) - التدرّجات بتتخصّص للبارامترات المجمّدة كمان

**الموقع:** `model/model.cpp` - `Model::enable_grad()` + الـ backward

**الوضع:** `enable_grad(true)` بيخصّص `p->g` (buffer بـ fp32) لكل بارامتر، بما فيهم
المجمّد (في وضع الـ LoRA: **كل** أوزان الـ base). الـ optimizer بيتجاهلها ✓،
لكن:

1. **الـ VRAM:** لموديل 6.9B = **27 GB** من التدرّجات الفاضية ✗ (لازم 14 GB
   عشان الـ base نفسه بـ fp32). ده بيمنع الـ LoRA على T4 16 GB.
2. **الـ compute:** الـ backward بيحسب الـ dW لكل الأوزان (نص الـ FLOPs) رغم
   إن الـ optimizer رايح يتجاهلها ✗.

**الإصلاح المقترح:** تخصيص الـ `g` للبارامترات القابلة للتدريب فقط + تمرير
`nullptr` للـ `dw` المجمد (الـ `dx` لازم يتحسب دايماً: هو اللي بيوصّل التدرّج
للطبقات اللي قبلها). محتاج null-safety في:
`core/ops_cpu_moe.cpp` + `cuda/moe.cu` (الـ `dge`/`due`/`dde` + الـ shared)
و`embedding_backward` (CPU + CUDA).

**الاختبار:** `tests/test_cuda_parity.cpp` (المقارنة CPU/CUDA) بيغطّي المسار
بعد التعديل.

---

## BUG-14 (معروف، مصمّم) - مسار الـ training بيرفض الـ fp16 weight storage

**الموقع:** `model/model.cpp` - `Model::forward_backward()`

```cpp
GAI_CHECK(!weight_f16_,
          "the training path needs the fp32 masters: turn the fp16 weight storage off");
```

**السبب:** الـ optimizer (AdamW/Lion/Muon) و الـ moments محتاجين masters بـ fp32
(التحديثات الصغيرة بتضيع في الـ half precision).

**الأثر:** موديل Granite-4.0-H كامل (6.9B) **مش بيتدرّب** على T4 16 GB:
الـ weights بـ fp32 = 27 GB ✗. الحل على المدى البعيد: masters بـ fp32 +
moments بـ fp32 بس للبارامترات القابلة للتدريب (الـ LoRA) → يعني
"mixed master weights": الـ base بأوزان fp16 (12.9 GB، للـ inference) + نسخ��
fp32 للـ LoRA بس.

**الأولوية:** متوسطة. موديل "Flash" (أصغر) مش متأثر.

---

## BUG-15 (معروف) - الـ MoE routing بين الـ CPU والـ CUDA ممكن يفرق

**الموقع:** `core/ops_cpu_moe.cpp` (topk_pick) و `cuda/moe.cu` (k_route)

الـ CPU بيختار الـ experts بـ `p[e] > bv` (الأول في الترتيب عند التعادل) والـ CUDA
نفس الـ منطق ✓، لكن **الفرق الصغير في الـ ties** بين الـ routing والـ backward
ممكن يطلع في الـ routing gradients. `tests/test_cuda_parity.cpp` بيقيس ده
(يُشغَّل على الـ GPU).
