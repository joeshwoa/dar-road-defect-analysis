# اكتشاف عيوب الطرق وتحليل حالة الطريق من فيديو الداش كام

> 🇬🇧 **English README:** [README.md](README.md) &nbsp;|&nbsp; 📖 **رحلة البناء خطوة بخطوة:** [docs/JOURNEY.ar.md](docs/JOURNEY.ar.md) &nbsp;|&nbsp; 🛰️ **ملاحظات Jetson:** [docs/JETSON.md](docs/JETSON.md)

نظام بيتفرّج على فيديو طريق وبيسجّل **كل عيب فعلي مرة واحدة بس**: حفر، شروخ (طولية / عرضية / جلد التمساح)، حفر مليانة مية،
وبالوعات - مع وقت أول ظهور، أعلى ثقة، صورة، ومكانه على خريطة مسار تقريبية ثنائية الأبعاد. كمان بيقول الطريق مسفلت ولا ترابي،
فيه رصيف ولا لأ، السرعة كانت كام، عدّى على أنهي عربيات، وبيعلّم **"مشاكل محتملة"** في دهان الحارات وعبور المشاة واللوحات
وإشارات المرور. القرارات **ثابتة** (من غير وميض) وبتعتمد على **الإطارات السابقة بس** زي أي نظام حقيقي منشور.

> ⚠️ الفيديوهين الأصليين ملك العميل و**مش** في الريبو ده. الفيديوهات المرسوم عليها في `results/` فيها نفس اللقطات -
> خلّي الريبو **Private**.

{run_note_ar}

<p align="center">
  <img src="docs/images/gallery_clipB.jpg" width="49%"/>
  <img src="docs/images/gallery_clipA.jpg" width="49%"/>
</p>

## 1. المطلوب ← فين تلاقيه

| المخرَج المطلوب | الملف |
|---|---|
| فيديو مرسوم عليه | [`results/clipA/annotated.mp4`](results/clipA/annotated.mp4)، [`results/clipB/annotated.mp4`](results/clipB/annotated.mp4) |
| جدول العيوب: رقم التراك، النوع، أول ظهور، أعلى ثقة، الصورة | `results/clip*/defects.csv` و`defects.json` (+ `images/`، `crops/`) |
| الـ FPS والهاردوير | القسم 3 تحت، `results/clip*/run_info.json`، `results/benchmarks/` |
| README من صفحة: الطريقة، الأخطاء، التحسين، Jetson | الأقسام 4-8 تحت، [`docs/JETSON.md`](docs/JETSON.md) |
| الكود | `roadscan/` (الحزمة)، `scripts/`، `notebooks/`، `tests/` |
| *إضافات من عندنا* | نتايج السطح والرصيف، سجل العربيات، خريطة المسار، جرد معالم الطريق، طبقة الثبات |

## 2. النتايج

### clipA - شارع مسفلت داخل المدينة وقت الغروب/الليل (59.4 ث، 496×368، 36 إطار/ث)

{tabA}

* الطريق: **مسفلت** 100% من المسار، رصيف على جنب واحد على الأقل {sw:.0f}% من الوقت.
* معالم الطريق اللي اتفحصت: **{n_sign} لوحة، {n_light} إشارة مرور، {n_cw} عبور مشاة**؛ علامات الأرض ظاهرة في {mkvis:.0f}% من الإطارات؛
  **{n_prob}** عنصر اتعلّم "مشكلة محتملة" (حل تقريبي، القسم 5).
* المسار حوالي {lenA:.0f} م بمتوسط ~{spdA:.0f} كم/س (تقدير من كاميرا واحدة، شوف القيود).
* التتبع: {rawA} تراك خام ← **{repA} اتسجّلوا** بعد الدمج والفلترة.

![timeline clipA](docs/images/fig_timeline_clipA.png)

### clipB - طريق ترابي ريفي (15 ث، 848×478، 25 إطار/ث)

{tabB}

* الطريق: **ترابي** 100%، من غير رصيف، مية واقفة ≈ {water:.0f}% من الطريق الظاهر.
* المسار حوالي {lenB:.0f} م بسرعة ~{spdB:.0f} كم/س. التتبع: {rawB} تراك خام ← **{repB} اتسجّلوا**.

![timeline clipB](docs/images/fig_timeline_clipB.png)

![counts](docs/images/fig_defects_by_type.png)

## 3. السرعة والهاردوير

| | القيمة |
|---|---|
| الـ GPU المستخدم | **{gpu}** (Kaggle)، CPU {cpu}، PyTorch {torch}، CUDA {cuda} |
| الـ detector لوحده، PyTorch FP16 | 78.6 FPS (12.7 ms/إطار) |
| الـ detector لوحده، **TensorRT FP16** | **177.3 FPS** (5.6 ms/إطار) |
| detect + فلتر الطريق + التتبع ("النواة") | **{fpsA:.1f} FPS** (clipA) / {fpsB:.1f} FPS (clipB) - Real-time على T4 |
| *كل* الموديلات (السطح، open-vocabulary، Mask2Former، المشهد، odometry) | {fpsA_all:.1f} FPS (clipA) / {fpsB_all:.1f} FPS (clipB) - وضع تحليل offline |
| قياس ONNX | 8 FPS - **مش حد الموديل**: نسخة onnxruntime-gpu ناقصها `libcublasLt` فاشتغلت على CPU من غير ما تقول |

![fps](docs/images/fig_fps.png)
![stage timing](docs/images/fig_stage_timing.png)

*القراءة*: الـ detector مش هو عنق الزجاجة. الموديلات الثقيلة الاختيارية (OWLv2 ≈ 150 ms/إطار، Mask2Former ≈ 50-70 ms/إطار، بتشتغل
كل 3 / 4 إطارات) هي اللي بتبطّأ. للـ real-time اقفلها أو قلّل تكرارها - النواة بتشتغل {fpsA:.0f}+ FPS.

## 4. الطريقة

![pipeline](docs/images/fig_pipeline.png)

1. **Detector العيوب** - YOLO26s متدرّب (fine-tuned) على RDD2022 (4 كلاسات)، حجم الصورة {imgsz}، وثقة منخفضة ({conf}) عن قصد: التتبع وفلتر التقرير هما اللي بيقرروا مين يعدّي.
   اختياريًا بيشتغل كمان على نسخة CLAHE من الإطارات الغامقة ويتدمج بـ NMS (`detect.enhance=auto`).
2. **قناع الطريق** (SegFormer-B0 / Cityscapes) - أي اكتشاف مش على الطريق بيتشال.
3. **السطح والرصيف** (SegFormer-B0 / ADE20K) - مسفلت / ترابي / مية / رصيف لكل إطار ← جدول مقاطع ونتايج بلغة بسيطة؛ المية الواقفة على أرض تالفة بتبقى مرشّح *حفرة مليانة مية*.
4. **البالوعات وفتحات الصرف** (OWLv2 بـ prompts نصية على منطقة الطريق)؛ الحفر اللي على بالوعة بتتمتص فيها.
5. **معالم الطريق** (Mask2Former / Mapillary) - دهان الحارات، عبور المشاة، اللوحات، إشارات المرور؛ مؤشرات الحالة بتتصوّت عليها عبر الزمن.
6. **التتبع و"سجّل مرة واحدة"** - BoT-SORT بتعويض حركة الكاميرا، وبعدها مرحلة دمج (قطع، تكرار، تصويت الكلاس، حد أدنى إطارات/ثقة).
7. **Visual odometry** (Lucas-Kanade على نقاط الأرض + نموذج أرض مسطّحة) ← السرعة والمسار والخريطة ومكان كل عيب.
8. **طبقة الثبات** (كلها past-only) ← القسم 6.

ليه عدة موديلات صغيرة بدل واحد كبير: [docs/JOURNEY.ar.md](docs/JOURNEY.ar.md) - كل نقطة ضعف في الـ detector الأساسي مشكلة مختلفة ومكانش فيه labels نعيد بيها التدريب.

## 5. الأخطاء اللي شفناها (ومعناها)

* **الأسفلت بالليل / الدقة القليلة (clipA)**: RDD2022 صور نهارية، فالـ recall على الشروخ الرفيعة والحفر الضحلة أقل من clipB. تجميعة CLAHE زوّدت الاكتشافات الخام ~44% في اختبار 250 إطار، لكن من غير labels مانقدرش نقول كام منها حقيقي.
* **البالوعات بتشبه الحفر في الضلمة** ← بنمتص الحفر اللي على تراك بالوعة/صرف.
* **العدّ**: العدد الفريد بيعتمد على قاعدة الدمج؛ clipB حقل حفر متداخلة، فـ "{nB} عيب فريد" تقدير مش حقيقة مطلقة. مفيش مرجع معنّون يدويًا، فـ**مش بندّعي precision/recall**.
* **أحكام اللوحات/الإشارات/الدهان حلول تقريبية** (ميل، لون اللمبة، تباين)، وبتتسجّل كـ *مشكلة محتملة*. CLIP على قصاصات اللوحات الصغيرة ضوضاء واتعطّل تحت 40 بكسل.
* **الـ odometry من كاميرا واحدة**: السرعة والمسافات بتتناسب مع ارتفاع كاميرا مفترض (1.4 م) وزاوية رؤية (75°)؛ اعتبرها تقريبية. أول ثواني الفيديو مفيهاش تاريخ حركة.
* **الـ smoothing السببي بيكلّف تأخير**: تغيّر الحالة بيظهر متأخر ~1-2 ث - تمن إننا ما بنبصّش للمستقبل.

## 6. الثبات ("من غير وميض، الماضي بس")

![speed](docs/images/fig_speed_stability.png)

* العيب **بيترسم بعد ما يتأكد بس** (درجة EMA بـ hysteresis، ≥ لمستين)، وبيكمّل كام إطار لو الاكتشاف فات، وإحداثيات المربع ونص الثقة متنعّمين - `DisplaySmoother`.
* التراكات المؤكدة السابقة **بترجع تغذّي خطوة الاكتشاف الجاية** (`TemporalPrior`: مزج ثقة وتصويت كلاس جوه مجموعة الشروخ؛ {boosted} اكتشاف اتعزّز في clipA) - ده "الـ attention window على الماضي بس" اللي كان مطلوب.
* **السرعة** بتعدّي على متوسط مقصوص سببي ← EMA ← محدّد تسارع/تباطؤ مع حماية بداية التشغيل، فمستحيل ننزل 40 ← 0 ← 40؛ حالات السطح والرصيف مابتتغيّرش إلا بعد ما تثبت 1.5 ث.
* قبل / بعد (التشغيل السابق مقابل ده):

![before after B](docs/images/before_after_clipB.jpg)

**مفتاح الألوان** (مرسوم كمان على الفيديو): المربعات الملونة = كلاس العيب (شوف الـ legend)، تلوين برتقالي فاتح = أرض ترابية، مربع أخضر رفيع = لوحة/إشارة/عبور مشاة اتحكم عليها سليمة، أحمر = *مشكلة محتملة*، طبقة حمرا على دهان الحارة = باهت (بتظهر بس وقت ما يبقى باهت).

## 7. إزاي نحسّنه

1. تعنين يدوي ~100 إطار لكل فيديو ← precision/recall حقيقي وضبط thresholds لكل كلاس.
2. Fine-tune على بيانات ليل / دقة قليلة / ترابي وعلى الحفر المليانة مية (أكبر فجوة).
3. استبدال حلول اللوحات/الإشارات بمصنّف صغير متدرّب على بضع مئات قصاصة.
4. مقياس حقيقي من عرض الحارة (أو GPS/IMU) بدل ارتفاع كاميرا مفترض.
5. تجربة RF-DETR-S (جاهز، notebook 02) كمنافس.
6. ضبط التكرار: السطح كل 3-5 إطارات، OWLv2 بس لما السطح مسفلت، Mask2Former كل 8 إطارات.

## 8. التشغيل على Jetson

TensorRT FP16 على الجهاز (`yolo export format=engine half=True`)، قناع الطريق كل 3-5 إطارات، SAM على أفضل إطار لكل عيب بس، فك تشفير بـ GStreamer/NVDEC، ووضع headless بيكتب CSV/JSON والقصاصات بس. الدليل الكامل: [docs/JETSON.md](docs/JETSON.md). *قِسنا T4 بس؛ أرقام Jetson في الدليل قيم مرجعية مش قياساتنا.*

## 9. إعادة الإنتاج

```bash
pip install -r requirements.txt
# 1) أوزان الـ detector: درّب بـ scripts/train_yolo.py (notebooks/01_train_yolo26_kaggle.ipynb) أو استخدم أوزانك
python scripts/run_video.py my_video.mp4 --weights weights/yolo26s_rdd.pt --out outputs \
       --scene --openvocab --furniture            # --sam للأقنعة، --no-temporal للمقارنة
python scripts/make_figures.py --results outputs --out docs/images
python -m pytest -q tests                          # 25 اختبار وحدة (المنطق الزمني، دمج التراكات، المعالم)
```

على Kaggle: `notebooks/03_infer_videos_kaggle.ipynb` (استنتاج)، `01_train_yolo26_kaggle.ipynb` / `04_finetune_yolo26_kaggle.ipynb` (تدريب)، والـ runbook في [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md).

## 10. خريطة الريبو

```
roadscan/            detectors.py  tracking.py  temporal.py  furniture.py  surface.py  openvocab.py  odometry.py
                     mapping.py  vehicles.py  road_mask.py  render.py  export.py  pipeline.py  config.py
scripts/             run_video.py  benchmark.py  sweep.py  train_yolo.py  prepare_data.py  make_figures.py  make_gallery.py
notebooks/           01 تدريب · 02 RF-DETR (منافس) · 03 استنتاج · 04 fine-tune (بتتولّد من make_notebooks.py)
tests/               25 اختبار وحدة
results/             المخرجات النهائية لكل فيديو + benchmarks + تاريخ التشغيلات السابقة (results/history)
docs/                JOURNEY(.ar).md · JETSON.md · KAGGLE_RUNBOOK.md · images/
```

## 11. شكر وتراخيص

Ultralytics YOLO26 (AGPL-3.0)، RDD2022، SegFormer (NVIDIA)، Mask2Former (Meta / أوزان Mapillary Vistas)، OWLv2 (Google)، BoT-SORT، SAM 2 (Meta)، OpenCV. راجع ترخيص كل موديل قبل أي استخدام تجاري.
