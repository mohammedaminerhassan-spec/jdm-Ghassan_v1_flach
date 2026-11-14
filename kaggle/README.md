# دليل الـ Kaggle (Ghassan v1 Flash)

الملف ده بيشرح كيفاش تشغّل المحرك على Kaggle (T4 x2) خطوة بخطوة.
كل الخلايا في `kaggle/cell*.py` — انسخهم بالترتيب.

---

## 0) قبل كل حاجة: ارفع الكود

عندك 3 خيارات (أي واحد فيهم يكفي):

### (أ) ارفعه كـ Kaggle Dataset (الأسهل، مافيش اعتماد على GitHub)

```powershell
# من مجلد المشروع (على جهازك):
tar -czf ghassan-ai-src.tar.gz --exclude=build --exclude=build_cpu --exclude=.git .
```

ثم في الـ notebook: **Add Data → Upload** → ارفع `ghassan-ai-src.tar.gz` كـ dataset.
الخلية الأولى بتلاقيها تلقائياً في `/kaggle/input/` وبتفكّها في `/kaggle/working/ghassan-ai`.

> الملف `ghassan-ai-src.tar.gz` موجود جاهز في جذر المشروع (0.29 MB) ✓

### (ب) ارفعه على GitHub

```powershell
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

وخلّي `REPO_URL` في الخلية الأولى = رابط الـ repo.

### (ج) انسخ الكود يدوي

حمّل `ghassan-ai-src.tar.gz` وفكّه في `/kaggle/working/ghassan-ai` (ارفعه كـ dataset
|Researcher من نوع "Code" أو استخدم `!tar -xzf`).

---

## 1) الخلايا بالترتيب

**الخلايا جوه الـ repo نفسه** — فالـ notebook بيناديها من الـ clone (مفيش copy-paste):

| # | الخلية (في الـ notebook) | إيه بتعمل |
|---|---|---|
| 1 | `!git clone --depth 1 https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_flach.git /kaggle/working/ghassan-ai` | ياخد الكود (أو شغّل `kaggle/cell1_setup.py`) |
| 2 | `!python {SRC}/kaggle/cell2_build.py` | **البناء بـ CUDA** (أول compile بـ nvcc — أخطر خطوة) |
| 3 | `!python {SRC}/kaggle/cell3_data.py` | ينزّل الـ checkpoint المرجعي + يجهّز البيانات + يبني الـ shards |
| 4 | `!python {SRC}/kaggle/cell4_smoke.py` | الاختبارات + **تحقق Granite** (الـ config والوزن الحقيقيين) + 20 خطوة تدريب |
| 5 | `!python {SRC}/kaggle/cell5_export.py` | يصدّر **safetensors** + يرفعه على HuggingFace |

> `{SRC}` = `/kaggle/working/ghassan-ai` (حدده في الخلية الأولى).
> أو خلّي `kaggle/cell1_setup.py` هو الخلية الأولى: بيعمل الـ clone ويطبع الـ commit.

**والـ flow القديم بتاعك لسه شغّال** (بيدمج التحقق + الـ export + الـ checks فـ cell واحد):
```python
!bash "{SRC}/kaggle configue/kaggle_granite_run.sh"
```

---

## 2) متغيرات البيئة (كل واحدة اختيارية)

قبل الخلية الأولى (أو في أول خلية):

```python
import os
os.environ["REPO_URL"]   = ""                                   # (ب) فقط
os.environ["HF_REPO"]     = "ibm-granite/granite-4.0-h-tiny"      # المرجع
os.environ["DATA_INPUT"]  = "/kaggle/input/my-english-parquet"    # الـ بيانات
os.environ["DATA_MAX_TOKENS"] = "200000000"                      # سقف (0 = بلا)
os.environ["MODEL_NAME"]  = "Ghassan v1 Flash"
os.environ["HF_REPO_OUT"] = ""                                   # وجهة الرفع
os.environ["HF_PUBLIC"]   = "0"
```

`DATA_INPUT` لازم يكون **مجلد** فيه ملفات `.parquet` (أو `.jsonl` / `.txt`).
لو ما حطّيتش حاجة، الخلية 3 بتتخطى تجهيز البيانات وبتقولك.

---

## 3) إيه اللي بيتحقّق في كل مرحلة

### الخلية 2 (البناء)
- أول compile بـ nvcc → لو هنا وقع، المشكلة في الـ CUDA.
- **الخطر الأساسي:** التعديلات على `cuda/kernels.cu` و `cuda/attention.cu`
  (تقطيع الـ GEMM). لو الـ build عدّى،Syntax تمام، والـ runtime بيبقى في الخلية 4.

### الخلية 4 (الـ smoke) — الأهم
تشغّل:
1. `ctest` → 7 اختبارات:
   - `test_granite_compat` → **يطابق هندسة Granite-4.0-H بالـ element**
     (عدد البارامترات = عدد الـ checkpoint بالظبط = 6,939,037,248).
     اقرأ `GRANITE_COMPAT.md` — فيها مقارنة سطر بسطر مع مرجع transformers 4.56.
   - `test_gradient_check` → finite-difference على التدرّجات (dense + MoE +
     Granite + Granite بلا MoE عشان الـ routing المتقطع ما يفسدش القياس).
   - `test_mamba_backward` → **طبقة الـ Mamba-2 لوحدها** (loss خطي → إشارة
     نقية): كل بارامتر من بارامتراتها الـ 9 بيتقابل مع العددي.
   - `test_cuda_parity` → **مقارنة تدرّجات CPU مع CUDA** (فيها حالة Granite
     هجين: دي اللي بتثبت إن الـ backward بتاع الـ Mamba صح على الـ GPU).
   - `test_safetensors_roundtrip` → الكتابة والقراءة والمقارنة.
   - `test_gguf_eos` / `test_gguf_quant` → الـ GGUF.
2. `test_cuda_parity` → **مقارنة تدرّجات CPU مع CUDA**. دي اللي بتثبت إن
   التقطيع صح على الـ GPU. لو يفشل: الـ GPU بتحسب غير الـ CPU ✗.
3. **20 خطوة تدريب على موديل dense** + **فحص إن الـ loss نزل**.
4. **20 خطوة تدريب على موديل Granite shape** (Mamba-2 + MoE + الـ
   multipliers، `configs/ghassan_v1_flash_smoke_granite.yaml`) + فحص إن الـ
   loss نزل — دي_cell الـ Mamba-2 path على الـ GPU.

### الخلية 5 (التصدير)
- `ghassan-ai export --format safetensors` → بيكتب `model.safetensors` +
  `config.json` + ملفات الـ tokenizer.
- الـ `config.json` بيطلع **Granite-compatible بالظبط** (الـ `model_type` =
  `granitemoehybrid`، الـ `layer_types`، الـ `intermediate_size` = عرض الـ
  routed expert، الـ `shared_intermediate_size` = عرض الـ shared expert،
  الـ multipliers، الـ NoPE) → فالموديل بيفتح تاني في أي Granite loader.
- **ثم** `ghassan-ai info --model <out>` → بيثبت إن المحرك بيقرأ اللي كتبه
  (نفس مسار الكود اللي المستخدم هيشغله بعد الـ clone).

### الـ GGUF (الـ deliverable)
```python
!{WORK}/build/bin/ghassan-ai quantize --model {HF} --out {WORK}/ghassan-v1-flash.gguf \
    --profile fp16 --model-name "Ghassan v1 Flash" --compat native
```
الـ metadata بتبقى:
- `general.name` = **"Ghassan v1 Flash"** (البراند بتاعك)
- `general.base_model.0.name` = **`ibm-granite/granite-4.0-h`** (إنه مبني
  على Granite — عشان يفضل معروف إنه Granite لما تفينتунه بعدين)
- `general.description` = "Ghassan v1 Flash - based on ibm-granite/granite-4.0-h"
- هندسة Granite كما هي (الـ hybrid، الـ mamba geometry، الـ layer types،
  الـ multipliers) — **ما اتغيرش حاجة**.

`scripts/gguf_xray.py` بيتحقق من كل ده (وبيقولك لو الـ brand leaks في أي
مفتاح تاني).

---

## 4) التدريب الحقيقي (بعد ما الـ smoke يعدّي)

```python
!{WORK}/build/bin/gai_train \
    --config {SRC}/configs/ghassan_v1_flash_pretrain.yaml \
    --config-from {HF}                 # الـ geometry من الـ config.json (من غير weights)
    --tokenizer {HF}/tokenizer.json \
    --data {WORK}/artifacts/shards \
    --max-steps 20000
```

- `--config-from` بيقرا الـ **architecture** من `config.json` من غير الـ weights →
  نفس دالة Granite-4.0-H بأوزان جديدة ✓.
- الـ MoE + الـ Mamba-2 + الـ multipliers + الـ NoPE كلهم بيجوا من المرجع ✓.

### الـ LoRA (على الـ checkpoint نفسه)
```python
!{WORK}/build/bin/gai_train \
    --config {SRC}/configs/granite_4_0_h_lora.yaml \
    --model {HF}                        # الـ config + الـ weights
    --tokenizer {HF}/tokenizer.json \
    --data {WORK}/artifacts/shards
```
⚠️ Granite-4.0-H كامل ~6.9B → 27 GB بـ fp32، **مش داخلش على T4 16 GB**.
الخيارات: tier أصغر (H-Tiny)، أو `--weight-f16 1` (12.9 GB) — بس مسار
الـ training بيطلب fp32 حالياً (انظر BUG-14 في BUGS.md).

---

## 5) الـ T4 x2 (DDP)

الـ notebook بيشوف GPU واحد. للـ DDP:

```python
!python -m torch.distributed.run --nproc_per_node=2 ...   # مابيشTorch
```

المحرك بيستخدم **NCCL** (`GAI_ENABLE_NCCL`) وبيقرا `WORLD_SIZE` / `RANK` /
`LOCAL_RANK` من البيئة. الطريقة الأسهل: خلي أول خلية تشغّل:

```python
import subprocess, os, sys
env = dict(os.environ, WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
procs = [subprocess.Popen([sys.executable, "-m", "gai_train", ...], env=dict(env, RANK=str(r), LOCAL_RANK=str(r)))
         for r in range(2)]
```

(الـ engine نفسه مافيش launcher →Cells بتعمل الـ spawn.)

---

## 6) ملاحظات مهمة عن الأداء

- **`ce_chunks`**: الـ logits `[B,T,V]` هي أكبر كتلة في الـ VRAM (4 bytes × T × 49152).
  `ce_chunks: 4` بيقسّمها ✓.
- **`optimizer: lion`**: الـ moments 4 bytes/param بدل 8 → الموديل الهجين
  مش داخلش على T16 GB بـ AdamW.
- **`batch_size: 1, grad_accum: 32`**: الـ throughput من الـ accumulation.
- **الـ fp16 weight storage** (`--weight-f16 1`): بيوفّر 14 GB بس **مسار
  الـ training بي.refuse** (fp32 masters مطلوبة) — انظر BUG-14.

---

## 7) لو حاجة وقعت

| العلامة | السبب المحتمل | الحل |
|---|---|---|
| `cublasSgemm failed: INVALID_VALUE` | الـ GEMM chunking ناقص في حالة ما | أرسل الـ `M/N/K/lda/ldb` من الرسالة |
| `nvcc` مش موجود | صورة Kaggle | `!ls /usr/local/cuda*` |
| `the source is still missing` | الـ dataset ما اتضافش | Add Data → Upload |
| `token id N out of writing-vocab range` | الـ shards اتعملت بـ tokenizer تاني | امسح الـ `.gbin` وأعد البناء |
| الـ loss ما نزلش | الـ gradients مش بيوصلوا | شغّل `test_cuda_parity` وشوف |
