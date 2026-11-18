# تقرير التدقيق البرمجي — Ghassan v1 Flash

- **المستودع:** mohammedaminerhassan-spec/jdm-Ghassan_v1_flach
- **الفرع:** main
- **تاريخ المراجعة:** 18 نوفمبر 2026
- **نطاق اللقطة:** شجرة GitHub recursive غير مبتورة، 84 ملفاً (1,412,217 بايتاً من ملفات blobs).
- **نوع المراجعة:** قراءة ساكنة كاملة للمصدر، مع تتبّع استدعاءات ومسارات البيانات. لم تُعدّل ملفات المصدر.

> **تنبيه منهجي:** هذا التقرير لا يدّعي إثبات خلوّ كل مسار من العيوب الرياضية أو التشغيلية، ولا يدّعي نجاح البناء أو الاختبارات. لم أنفّذ CMake build أو CUDA runtime أو اختبارات رقمية/تدريبية في هذه المراجعة. أرقام الأسطر أدناه تخص نسخة الملفات في الفرع main وقت الفحص؛ تحقّق من تطابقها إذا تحرك الفرع لاحقاً.

## 1. الملخص التنفيذي

| الدرجة | عدد الملاحظات |
|---|---:|
| CRITICAL | 0 |
| WARNING | 1 |
| MEDIUM | 4 |
| LOW | 6 |
| **المجموع** | **11** |

أهم نتيجة: توجد مشكلة إعدادات يمكنها تحويل نوع طبقة Granite غير المعروف بصمت إلى Mamba، ومشكلات في التحقق الصارم من JSON، ومعالجة حالة تاريخ المحادثة صفر، وفحص توافق checkpoint، ورصد فشل flush عند حفظ checkpoint. بقية الملاحظات محصورة في تقارير تشخيصية، مخرجات CLI، UTF-8، وتخصيصات غير ضرورية داخل حلقة التوليد.

لم أجد دليلاً يثبت فساداً عاماً في forward/backward أو في فك GGUF Q4_1 الفعلي. **فك Q4_1 المستخدم لتحميل أوزان النموذج صحيح**؛ الملاحظة المتعلقة به تخص حساب تقرير خطأ التكميم فقط، كما هو موضح أدناه.

## 2. النتائج التفصيلية

### A-01 — تحويل أي اسم طبقة غير معروف إلى Mamba بصمت

- **الدرجة:** WARNING
- **المسارات والأسطر:**
  - model/model.cpp:519-530، وبالأخص السطر 527.
  - format/hf_checkpoint.cpp:604-612.
  - format/gguf_codec.cpp:1142-1149.
- **المقتطف:**

~~~cpp
// model/model.cpp
m.layer_types.push_back(tok == "attention" ? 1 : 0);

// format/hf_checkpoint.cpp
const std::string kind = el.as_string("mamba");
const bool attn = (kind == "attention" || kind == "full_attention");
cfg_.layer_types.push_back(attn ? 1 : 0);

// format/gguf_codec.cpp
c.layer_types.push_back(lt[i] == '1' ? 1 : 0);
~~~

- **سبب الملاحظة:** في محلّل YAML/recipe، لا يُميّز الشرط بين mamba وبين أي كلمة أخرى غير attention. وفي قارئ HF، كل نوع غير attention وfull_attention يتحول إلى Mamba، بما في ذلك قيمة غير معروفة أو قيمة ذات نوع JSON غير متوقع تُقرأ بالافتراضي. كما أن قارئ GGUF يحوّل أي رمز غير 1 إلى Mamba.
- **الأثر:** خطأ مطبعي مثل attension أو اسم طبقة غير مدعوم لا يوقف التحميل ولا يوضح الخطأ؛ بل يغيّر بنية الطبقة التي ينشئها المحرك. عند تحميل checkpoint قد يؤدي ذلك إلى اختلاف أسماء/أشكال الأوزان وفشل التحميل، وعند إنشاء نموذج جديد من recipe قد ينتج نموذجاً ببنية غير التي قصدها المستخدم.
- **الإصلاح المقترح:** استخدام تعداد صريح للقيم المدعومة: attention وmamba، وفي HF قبول full_attention وlinear_attention فقط وفق mapping موثّق. ارفض أي قيمة أخرى برسالة تتضمن رقم الطبقة والقيمة. في GGUF تحقّق من طول السلسلة ومحتواها (0 أو 1 فقط) ومن مطابقته لعدد الطبقات قبل بناء ModelConfig.

### A-02 — محلّل JSON يقبل صيغ أرقام غير صالحة ولا يكشف overflow

- **الدرجة:** MEDIUM
- **المسار:** core/json.cpp:239-264، مع معالجة Unicode escapes في core/json.cpp:300-340.
- **المقتطف:**

~~~cpp
if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
// ...
if (i < s.size() && s[i] == '.') {
    is_int = false;
    ++i;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        ++i;
        any = true;
    }
}
// ...
out = Json::integer(std::strtoll(num.c_str(), nullptr, 10));
// ...
out = Json::number(std::strtod(num.c_str(), nullptr));
~~~

- **سبب الملاحظة:** نحو JSON لا يسمح بعلامة + قبل العدد، ولا يسمح بأصفار بادئة مثل 01، ويتطلب رقماً واحداً على الأقل بعد الفاصلة العشرية. التنفيذ يسمح بهذه الحالات. كذلك لا يفحص errno أو نهاية التحويل عند strtoll/strtod؛ فيمكن أن يتحول عدد صحيح خارج مجال int64 إلى قيمة مشبعة، كما يمكن قبول قيمة عشرية غير منتهية ناتجة عن overflow. وفي مسار \uXXXX لا يُرفض surrogate منفرد عند غياب زوج صالح، بل يمكن ترميزه إلى UTF-8 غير صالح.
- **الأثر:** قبول ملفات JSON غير قياسية أو إسقاط أعداد كبيرة إلى حدود النوع بصمت. القارئ مشترك بين إعدادات/ملفات tokenizer وmetadata وSafetensors؛ لذلك قد يتحول خطأ الملف إلى قيمة داخلية خاطئة بدلاً من فشل واضح.
- **الإصلاح المقترح:** تطبيق grammar JSON حرفياً قبل التحويل: صفر مفرد أو رقم يبدأ من 1–9، إلزام digits بعد . وe، منع + في بداية العدد، والتحقق من errno وendptr وstd::isfinite. ارفض surrogate منفرداً، ولا تسمح بإنتاج UTF-8 يحوي surrogate code point.

### A-03 — max_history_turns = 0 يحذف رسالة المستخدم الحالية قبل التوليد

- **الدرجة:** MEDIUM
- **المسار:** inference/chat.cpp:13, 43-70.
- **المقتطف:**

~~~cpp
GAI_CHECK(opts_.max_history_turns >= 0, "chat history turns must be >= 0");

const size_t max_msgs =
    sys + static_cast<size_t>(2 * opts_.max_history_turns);
if (history_.size() <= max_msgs) return;

const size_t keep_from =
    history_.size() - static_cast<size_t>(2 * opts_.max_history_turns);
// ...
reply = gen_.chat(history_, turn_cfg, cb);
~~~

- **سبب الملاحظة:** القيمة صفر مسموحة صراحةً. بعد إضافة رسالة المستخدم، يكون عدد الرسائل أكبر من الحد؛ ثم يصبح keep_from == history_.size()، فلا تُنسخ أي رسالة مستخدم/مساعد إلى التاريخ الجديد. إذا وُجد system prompt تبقى رسالة النظام وحدها، وإذا لم يوجد يصبح التاريخ فارغاً.
- **الأثر:** ChatSession::send() يرسل إلى المولّد system prompt فقط أو قائمة رسائل فارغة، مع أن المستخدم أرسل سؤالاً. ينتج رد غير متعلق بالسؤال أو سلوكاً غير متوقع.
- **الإصلاح المقترح:** إذا كان المقصود صفر أدوار سابقة، احتفظ دائماً بآخر رسالة User الحالية، واحذف فقط الأزواج الأقدم. بديل أبسط: عرّف الحد الأدنى المسموح به على أنه دور واحد، أو افصل max_previous_turns عن رسالة الطلب الحالية. أضف اختباراً لحالتي system prompt موجود/غير موجود عند القيمة صفر.

### A-04 — فحص توافق checkpoint لا يقارن كل حقول المعمارية التي تغيّر دالة النموذج

- **الدرجة:** MEDIUM
- **المسار:** training/checkpoint.cpp:269-295 وموضع استعمال arch_match قرب السطر 612.
- **المقتطف:**

~~~cpp
static bool arch_match(const ModelConfig& a, const ModelConfig& b) {
    if (!(a.vocab_size == b.vocab_size &&
          a.hidden_size == b.hidden_size &&
          a.num_layers == b.num_layers &&
          a.num_heads == b.num_heads &&
          a.num_kv_heads == b.num_kv_heads &&
          a.intermediate_size == b.intermediate_size &&
          /* ... */))
        return false;

    if (a.arch != b.arch ||
        a.shared_expert_intermediate_size != b.shared_expert_intermediate_size ||
        a.position_embedding != b.position_embedding ||
        a.head_dim_override != b.head_dim_override)
        return false;

    // يقارن بعض قيم RoPE/eps/MoE بالتحذير فقط
    return true;
}
~~~

- **سبب الملاحظة:** فحص arch_match لا يقارن embedding_multiplier وresidual_multiplier وattention_multiplier وlogits_scaling، ولا هندسة Mamba كاملة. هذه الحقول تؤثر في الحساب الأمامي، ويمكن أن تختلف من دون تغير عدد عناصر الأوزان أو أسمائها. توجد دالة أدق هي ModelConfig::same_architecture_as() في model/model.cpp:630-659 وتقارن هذه المضاعفات وأبعاد Mamba، لكن Checkpoint::load يعتمد فحصه الخاص.
- **الأثر:** عند استعمال واجهة Checkpoint::load مباشرةً مع نموذج له أوزان متطابقة الأشكال وإعدادات حسابية مختلفة، يمكن قبول checkpoint ثم تنفيذ النموذج بدالة مختلفة عن recipe الأصلية. مسار Trainer يملك فحوص توافق إضافية، لكن ذلك لا يجعل فحص واجهة checkpoint منخفضة المستوى مكتملاً بذاته.
- **الإصلاح المقترح:** اجعل Checkpoint::load يستعمل same_architecture_as() كمرجع موحّد، أو وحّد دوال التحقق. لا تسمح لـstrict=true بتجاوز اختلافات المكونات التي تغيّر forward. أضف اختبارات تغيّر كل multiplier وMamba dimension على حدة وتتحقق أن exact load يرفضها.

### A-05 — فحص نجاح كتابة checkpoint يحدث قبل flush() ولا يُفحص بعده

- **الدرجة:** MEDIUM
- **المسار:** training/checkpoint.cpp:500-547، وبالأخص 545-546.
- **المقتطف:**

~~~cpp
GAI_CHECK(f.good(), "checkpoint write failed");
f.flush();
}
publish_file(path, tmp);
guard.keep = true;
~~~

- **سبب الملاحظة:** التحقق من حالة stream يحدث قبل تفريغ مخزن الكتابة. يمكن أن يفشل flush() بسبب امتلاء القرص أو خطأ I/O، فتتغير حالة stream بعد آخر فحص، ثم ينتقل التنفيذ إلى نشر الملف المؤقت باسم checkpoint النهائي.
- **الأثر:** يمكن نشر checkpoint ناقص أو غير مكتمل على أنه checkpoint مكتوب، ما يضر الاستعادة بعد انقطاع أو فشل تخزين.
- **الإصلاح المقترح:** نفّذ flush() ثم تحقق من f.good() مباشرةً، ثم نفّذ close() صراحةً وتحقق من حالة الملف بعد الإغلاق قبل publish_file. عند الحاجة إلى ضمان durability بعد انقطاع الطاقة، أضف fsync/FlushFileBuffers قبل rename، مع الإبقاء على آلية النشر الذرّي.

### A-06 — فك Q4_1 في تقرير خطأ التكميم يستخدم فهرسة خاطئة (ولا يفسد التحميل الفعلي)

- **الدرجة:** LOW
- **المسار:** format/gguf_codec.cpp:630-641.
- **المقتطف:**

~~~cpp
const size_t i0 = b * 32 + static_cast<size_t>(2 * j);
const size_t i1 = i0 + 1;
if (i0 < n) back[i0] = d * (byte & 0x0F) + m;
if (i1 < n) back[i1] = d * (byte >> 4) + m;
~~~

- **سبب الملاحظة:** صيغة GGML Q4_1 تخزن nibble المنخفض للقيمة j وnibble العالي للقيمة j+16، وليس لقيمتين متجاورتين 2*j و2*j+1. كاتب Q4_1 عند السطور 455-500 يكتب layout الصحيح، وقارئ النموذج الفعلي عند السطور 1374-1390 يفكّه إلى b*32+j وb*32+j+16 بالشكل الصحيح. الخطأ هنا فقط في إعادة فك البيانات لأجل حساب إحصاءات خطأ quantization أثناء التصدير.
- **الأثر:** قيم back المستخدمة في حساب خطأ إعادة البناء غير مطابقة للأوزان التي سينتجها الملف؛ بالتالي قد تكون مؤشرات الخطأ المعلنة لكل tensor غير دقيقة. **لا يوجد في هذا الدليل ما يثبت فساد أوزان Q4_1 المحمّلة إلى النموذج.**
- **الإصلاح المقترح:** استخدم نفس فهرسة القارئ الفعلي: i0=b*32+j، i1=b*32+j+16. أضف round-trip test لكتلة Q4_1 بقيم 32 مختلفة، وتحقق من تقرير الخطأ ومن bytes وفق مرجع GGML.

### A-07 — تقرير عدد المعاملات لا يحصي بنية Granite الهجينة بصورة صحيحة

- **الدرجة:** LOW
- **المسار:** model/model.cpp:1185-1230.
- **المقتطف:**

~~~cpp
add("attn.wq   [q_dim, d]",  qd * d,  L);
add("attn.wk   [kv_dim, d]", kvd * d, L);
add("attn.wv   [kv_dim, d]", kvd * d, L);
add("attn.wo   [d, q_dim]",  d * qd,  L);
if (cfg_.moe_shared) {
    add("moe.shared.gate [E, d]", E * d, L);
    add("moe.shared.up   [E, d]", E * d, L);
    add("moe.shared.down [d, E]", d * E, L);
}
~~~

- **سبب الملاحظة:** الدالة تضيف Attention projections لكل طبقات النموذج، بينما Granite قد يحتوي طبقات Mamba لا تملك Q/K/V/O، ولا تضيف أوزان Mamba-2 المقابلة. كذلك تستخدم moe_expert_dim كعرض للـshared expert بدلاً من cfg_.shared_width()؛ في Granite الذي يختلف فيه عرض shared expert عن routed expert يصبح التقرير غير صحيح.
- **الأثر:** print_parameter_report() يعطي مجموعاً وتفصيلاً غير مطابقين للمعاملات المنشأة فعلياً في Granite. هذا يؤثر في التشخيص والتقدير اليدوي فقط؛ عدّ num_parameters() الذي يجمع الـParameters الفعلية مستقل عن هذا التقرير.
- **الإصلاح المقترح:** ابنِ التقرير من params_ الفعلية، أو فرّع الحساب حسب layer_types وأبعاد Mamba، واستخدم shared_width() للخبير المشترك. أضف اختبار مقارنة بين مجموع التقرير وnum_parameters() لكل من Qwen dense وMoE وGranite hybrid.

### A-08 — أمر ghassan-ai info يحسب حجم Q4_1 على أنه FP32

- **الدرجة:** LOW
- **المسار:** tools/inference_cli.cpp:96-103.
- **المقتطف:**

~~~cpp
static u64 ggml_nbytes_of(GGMLType t, u64 nelems) {
    switch (t) {
        case GGMLType::F32:  return nelems * 4;
        case GGMLType::F16:  return nelems * 2;
        case GGMLType::Q8_0: return ((nelems + 31) / 32) * 34;
        case GGMLType::Q4_0: return ((nelems + 31) / 32) * 18;
        default:             return nelems * 4;
    }
}
~~~

- **سبب الملاحظة:** ggml_type_str() يتعرف على Q4_1، لكن ggml_nbytes_of() لا يحتوي حالة له؛ لذلك يقع Q4_1 في default ويحسب 4 بايت لكل عنصر بدلاً من 20 بايت لكل كتلة من 32 عنصراً.
- **الأثر:** أرقام إجمالي البايتات وbits/parameter في ghassan-ai info تكون خاطئة عند وجود Q4_1، وقد تبدو أكبر كثيراً من الحجم الحقيقي.
- **الإصلاح المقترح:** أضف case GGMLType::Q4_1: return ((nelems + 31) / 32) * 20;، واستخرج حسابات أحجام الكتل إلى helper مشترك مع codec لتجنب اختلاف القارئ عن CLI.

### A-09 — التحقق من UTF-8 يفحص شكل continuation bytes فقط ولا يتحقق من صلاحية code point

- **الدرجة:** LOW
- **المسارات والأسطر:** core/unicode.cpp:7-12, 71-115 وtokenizer/tokenizer.cpp:695-725.
- **المقتطف:**

~~~cpp
bool utf8_is_complete(const std::string& s) {
    // ...
    for (int k = 1; k < len; ++k) {
        if ((static_cast<u8>(s[i + static_cast<size_t>(k)]) & 0xC0) != 0x80)
            return false;
    }
    i += static_cast<size_t>(len);
}
~~~

- **سبب الملاحظة:** لا يُرفض في هذا المسار encoding زائد الطول (overlong)، ولا surrogate UTF-8، ولا بادئات 4-byte المحجوزة فوق U+10FFFF. الدالة utf8_decode() تحتوي فحوص code-point أشد، لكن utf8_from_bytes() وTokenizer::Stream::push() يكتفيان بفحص continuation bits، وقد يمرران البايتات غير الصالحة إلى callback.
- **الأثر:** نص مخرجات tokenizer قد يحتوي UTF-8 غير قانوني عند إدخال تسلسل بايتات غير صالح من token IDs أو بيانات غير متوقعة.
- **الإصلاح المقترح:** اجعل تحقق streaming يستخدم decoder موحداً يعيد حالة valid/incomplete/invalid، ويطبق قيود lead-byte وoverlong وsurrogates وU+10FFFF. لا تُصدر sequence حتى يثبت أنها code point صالح.

### A-10 — نسخ/تخصيص قائمة stop tokens داخل كل خطوة توليد

- **الدرجة:** LOW
- **المسار:** inference/generator.cpp:954-955.
- **المقتطف:**

~~~cpp
for (int step = 0; step < limit; ++step) {
    // ...
    std::vector<i32> stops = cfg.stop_tokens;
    if (stops.empty()) stops.push_back(tok_.eos_id());
    bool is_stop = std::find(stops.begin(), stops.end(), next) != stops.end();
}
~~~

- **سبب الملاحظة:** تُنسخ قائمة التوقف في كل token iteration، وإذا كانت فارغة ينشأ عنصر جديد في vector في كل مرة. القائمة لا تتغير أثناء الاستدعاء.
- **الأثر:** heap allocation/copy غير ضروري في hot loop؛ يتكرر حتى مليون مرة وفق حد CLI، ويضيف ضجيجاً زمنياً وتكلفة CPU صغيرة لكنها قابلة للإزالة.
- **الإصلاح المقترح:** كوّن stop-token vector مرة واحدة قبل الحلقة، أضف EOS الافتراضي إليها عند الحاجة، ثم استخدمها دون نسخ في الحلقة.

### A-11 — فشل كتابة generation_config.json لا يوقف تصدير Safetensors

- **الدرجة:** LOW
- **المسار:** format/hf_writer.cpp:588-598، وبالأخص السطر 592.
- **المقتطف:**

~~~cpp
write_file((fs::path(dir) / "generation_config.json").string(),
           g.dump_pretty());

log_info(strfmt("[hf ] exported ...");
return true;
~~~

- **سبب الملاحظة:** تُتجاهل القيمة المعادة من write_file في حين أن كتابة config.json والـshards تُفحص وتؤدي إلى return false عند الفشل.
- **الأثر:** قد يعلن exporter نجاح العملية مع غياب generation_config.json أو عدم اكتماله، وتفقد الحزمة ملف إعدادات التوليد/stop-token الذي يقصده المصدر.
- **الإصلاح المقترح:** افحص نتيجة write_file مثل فحص config.json، وأعد false مع اسم الملف ورسالة الخطأ عند الفشل. أضف اختباراً يحاكي فشل الكتابة إلى هذا الملف.

## 3. الشيفرة الميتة والواجهات الوهمية والميزات غير المكتملة

### ما تم التحقق منه

- لم أعثر أثناء تتبع الاستدعاءات على دالة رئيسية في forward/backward أو sampler أو GGUF أو checkpoint يمكن الجزم بأنها ميتة بالكامل. لا أضع وسم dead code لمجرد أن الدالة غير ظاهرة في CLI؛ توجد واجهات مكتبية يستعملها مسار التدريب/الاختبارات أو التحميل.
- CPU operators في core/ تنفذ عمليات حسابية فعلية، وCUDA files kernels.cu وattention.cu وmamba.cu وmoe.cu تحتوي kernels فعلية، لا مجرد أسماء/واجهات تعيد قيماً ثابتة.
- مسارات MoE تحتوي توجيه خبراء وحسابات forward/backward فعلية. ومسارات Mamba-2 تحتوي scan/step وعمليات state فعلية.
- لا أعدّ مسار Q4_1 في قارئ GGUF واجهة وهمية؛ القارئ الفعلي يفك توزيع nibble الصحيح. الملاحظة الوحيدة المثبتة عنه تخص أداة قياس quantization الواردة في A-06.

### قيود معلنة يجب عدم تسويقها كدعم أصلي

- في tools/model_loader.cpp، اختيار --device metal|vulkan|tpu في inference يطبع تحذيراً ثم يعود إلى CPU؛ لا توجد backends native لهذه الأجهزة. وفي tools/training_cli.cpp يرفض التدريب هذه الخيارات صراحةً. هذا fallback معلن وليس CUDA/Metal/Vulkan implementation.
- الاسم q8_k في profile لا يعني وجود GGML K-quant حقيقي؛ README يصرّح بأن codec المنفذ Q8_0. والأسماء/aliases الأخرى يجب قراءتها حسب mapping الفعلي في gguf_profile_for()، لا حسب اسم K-quant وحده.
- توجد واجهة embed_override لحقن embeddings محسوبة مسبقاً عند مواقع محددة في prefill، لكنها ليست image encoder أو vision pipeline. لم يظهر في شجرة هذا المستودع frontend يعالج صوراً أو فيديو/صوتاً ويحوّلها إلى embeddings.
- copy_tokenizer_files() وملفات HF writer أدوات تحويل/تصدير حقيقية، لكن لا ينبغي افتراض أن تصدير safetensors يضمن كل ملفات tokenizer/generation عند فشل I/O؛ راجع A-11.

## 4. الأداء واستهلاك الموارد — نتائج مؤكدة وحدود الاستنتاج

- **Granite prefill:** المسار في inference/generator.cpp يعالج Granite token-by-token لأن الحالة التكرارية لـMamba-2 لا تستخدم attention-only batched prefill. هذا قيد أداء بنيوي في المسار الحالي؛ ليس عيب صحة بحد ذاته.
- **Mamba CUDA decode:** توجد دالة decode_mamba_layer() تختار mamba2_step أو mamba2_step_half عندما تكون حالة CUDA موجودة. لذلك لم أعمّم مسار CPU fallback على كل تشغيل CUDA.
- **CUDA Attention/MoE/kernels:** الملفات تحتوي تنفيذات فعلية. لم أثبت من القراءة وحدها خطأً حسابياً محدداً في GEMM أو masked attention أو MoE backward يمكن تصنيفه كفساد أوزان. يلزم numerical equivalence tests مقابل CPU/reference قبل اعتمادها تجارياً.
- **تخصيصات hot loop:** الملاحظة المثبتة هي نسخ stop tokens في كل خطوة (A-10). لم أدرج ملاحظات أداء تخمينية من أسماء الدوال.
- **VRAM/CPU/RAM:** لم أشغّل benchmark أو Nsight أو profiler، لذلك لا أورد قيماً أو claims عن throughput/VRAM الفعلي.

## 5. الاختبارات والتغطية

- شجرة المستودع التي فُحصت تحتوي 84 ملفاً ولا تحتوي مجلد tests/.
- CMakeLists.txt:212-228 يضبط GAI_BUILD_TESTS افتراضياً على ON، لكنه يبحث عن tests/test_*.cpp، وإذا لم يجد ملفات يطبع أنه لن ينشئ test targets. هذا موثق أيضاً في README.
- لم أشغّل build أو CTest أو اختبارات CUDA في هذه المراجعة. لذلك لا يوجد ادعاء بأن البناء ناجح أو أن kernels تطابق CPU رقمياً.
- الاختبارات التي ينبغي إضافتها أولاً: parser conformance/overflow، layer-type parsing، chat history zero، checkpoint I/O failure injection، checkpoint architecture mismatch، Q4_1 round-trip + diagnostic، UTF-8 invalid scalar sequences، parameter report مقابل تعداد المعاملات الفعلي.

## 6. الملفات التي لم تسجل فيها ملاحظة مثبتة

قُرئت الملفات التالية كاملة، ولم أسجل فيها عيباً مؤكداً ضمن الفحوص الساكنة التي أُجريت. هذا لا يعني شهادة خلوّ مطلق من كل bug؛ بل يعني أنني لم أجد دليلاً كافياً لإدراج ملاحظة محددة فيها.

**الجذر**
- .gitattributes
- .gitignore
- CMakeLists.txt — مع ملاحظة غياب test sources الموضحة في القسم 5.
- README.md

**core/**
- core/attn_cpu.cpp
- core/common.cpp
- core/common.h
- core/config.cpp
- core/config.h
- core/device.cpp
- core/device.h
- core/dtype.cpp
- core/dtype.h
- core/json.h
- core/mamba_cpu.cpp
- core/math_cpu.cpp
- core/mmap.cpp
- core/mmap.h
- core/moe_cpu.cpp
- core/ops.cpp
- core/ops.h
- core/ops_cpu.h
- core/quant.h
- core/rng.h
- core/signals.cpp
- core/signals.h
- core/simd_cpu.h
- core/tensor.cpp
- core/tensor.h
- core/unicode.h

**cuda/**
- cuda/attention.cu
- cuda/cuda_ops.h
- cuda/cuda_utils.cu
- cuda/cuda_utils.h
- cuda/kernels.cu
- cuda/mamba.cu
- cuda/moe.cu

**evaluation/**
- evaluation/perplexity.cpp
- evaluation/perplexity.h

**format/**
- format/gguf_codec.h
- format/hf_checkpoint.h
- format/hf_writer.h

**inference/**
- inference/chat.h
- inference/generator.h
- inference/kv_cache.cpp
- inference/kv_cache.h
- inference/sampler.cpp
- inference/sampler.h

**model/**
- model/model.h

**quantization/**
- quantization/quantize.cpp
- quantization/quantize.h

**tokenizer/**
- tokenizer/chat_template.cpp
- tokenizer/chat_template.h
- tokenizer/tokenizer.h

**tools/**
- tools/cli_common.cpp
- tools/cli_common.h
- tools/model_loader.cpp
- tools/model_loader.h
- tools/training_cli.cpp

**training/**
- training/checkpoint.h
- training/dataloader.cpp
- training/dataloader.h
- training/distributed.cpp
- training/distributed.h
- training/jsonl_loader.cpp
- training/jsonl_loader.h
- training/optimizer.cpp
- training/optimizer.h
- training/scheduler.h
- training/shard_builder.cpp
- training/shard_builder.h
- training/trainer.cpp
- training/trainer.h

**عدد الملفات بلا ملاحظة مثبتة في هذا التدقيق: 73 ملفاً.** أما الملفات الـ11 التي تحتوي على ملاحظات فهي: core/json.cpp, core/unicode.cpp, format/gguf_codec.cpp, format/hf_checkpoint.cpp, format/hf_writer.cpp, inference/chat.cpp, inference/generator.cpp, model/model.cpp, tokenizer/tokenizer.cpp, tools/inference_cli.cpp, training/checkpoint.cpp.

## 7. أولويات الإصلاح

1. **أولاً — A-01:** أوقف التحويل الصامت لأنواع طبقات Granite غير المعروفة. هذا يحمي إعداد المعمارية من التحول إلى شبكة مختلفة.
2. **ثانياً — A-04 وA-05:** وحّد فحص توافق checkpoints، واجعل فشل flush/close يمنع نشر الملف. هذان الأمران يحميان استعادة التدريب وصحة checkpoint.
3. **ثالثاً — A-02 وA-03:** شدد JSON parser، وأصلح trimming للمحادثة عند history=0.
4. **رابعاً — A-06 وA-07 وA-08:** صحح حسابات التشخيص: Q4_1 error report، parameter report، وحجم Q4_1 في أمر info.
5. **خامساً — A-09 إلى A-11:** أصلح تحقق UTF-8، وانقل إعداد stop tokens خارج حلقة التوليد، وافحص نتيجة كتابة generation_config.
6. **بعد الإصلاحات:** أضف tests/ فعلية واختبارات CPU/CUDA numerical comparison، ثم شغّل build وCTest واختبارات load → forward → backward → save/resume → export/reload على checkpoints معروفة.

## 8. الملفات التي أُنشئها هذا التدقيق

هذا التقرير هو الملف الوحيد المطلوب إنشاؤه. لم تُعدّل أية ملفات مصدر أو إعدادات للمشروع خلال المراجعة.
