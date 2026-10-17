# اكتشاف عيوب الطرق وتحليل حالة الطريق من فيديو الداش كام

> 🇬🇧 **English README:** [README.md](README.md) &nbsp;|&nbsp; 📖 **رحلة البناء خطوة بخطوة:** [docs/JOURNEY.ar.md](docs/JOURNEY.ar.md) &nbsp;|&nbsp; 🛰️ **ملاحظات Jetson:** [docs/JETSON.md](docs/JETSON.md)

نظام بيتفرّج على فيديو طريق وبيسجّل **كل عيب فعلي مرة واحدة بس**: حفر، شروخ (طولية / عرضية / جلد التمساح)، حفر مليانة مية،
وبالوعات - مع وقت أول ظهور، أعلى ثقة، صورة، ومكانه على خريطة مسار تقريبية ثنائية الأبعاد. كمان بيقول الطريق مسفلت ولا ترابي،
فيه رصيف ولا لأ، السرعة كانت كام، عدّى على أنهي عربيات، وبيعلّم **"مشاكل محتملة"** في دهان الحارات وعبور المشاة واللوحات
وإشارات المرور. القرارات **ثابتة** (من غير وميض) وبتعتمد على **الإطارات السابقة بس** زي أي نظام حقيقي منشور.

> ⚠️ الفيديوهين الأصليين ملك العميل و**مش** في الريبو ده. الفيديوهات المرسوم عليها في `results/` فيها نفس اللقطات -
> خلّي الريبو **Private**.

> **حالة النتايج في الفولدر ده:** كل اللي تحت `clipA/` و`clipB/` (الفيديو المُعلَّم، CSV / JSON، القصاصات، الصور، خريطة المسار، بيانات كل إطار في `frames/`)
> طالع من **تشغيل واحد لكود الريبو ده (Kaggle notebook "Version 10"، T4 x2، 28 دقيقة)** - تشغيل المرجع بكل الإطارات + أول تشغيل بوضع Jetson
> (`jetson_10fps/` ومعاه الإطارات الأصلية النضيفة). `mode_comparison.md` بيقارن الاتنين. نتايج الأقدم محفوظة في `history/`.
> بروفايلات Jetson بسقوف تانية (24 / 18 / 12 إطار/ث) بتتضاف من تشغيل لاحق وبتظهر في `mode_comparison.md` لما تكون موجودة.

<p align="center">
  <img src="docs/images/gallery_clipB.jpg" width="49%"/>
  <img src="docs/images/gallery_clipA.jpg" width="49%"/>
</p>

## 1. المطلوب ← فين تلاقيه

| المخرَج المطلوب | الملف |
|---|---|
| فيديو مرسوم عليه | [`results/clipA/annotated.mp4`](results/clipA/annotated.mp4)، [`results/clipB/annotated.mp4`](results/clipB/annotated.mp4) |
| جدول العيوب: رقم التراك، النوع، أول ظهور، أعلى ثقة، الصورة | `results/clip*/defects.csv` و`defects.json` (+ `images/`، `crops/`) |
| **تفاصيل كل إطار** (كل frame له سجل خاص) | `results/clip*/frames/frame_summary.csv` (صف لكل إطار)، `frames/detections.csv` (صف لكل اكتشاف في كل إطار)، `frames/frames.jsonl` (سجل كامل لكل إطار)، `frames/img/frame_NNNNN.jpg` (الإطار نفسه عليه الرسم) |
| الـ FPS والهاردوير | القسم 3 تحت، `results/clip*/run_info.json`، `results/benchmarks/` |
| README من صفحة: الطريقة، الأخطاء، التحسين، Jetson | الأقسام 4-8 تحت، [`docs/JETSON.md`](docs/JETSON.md) |
| الكود | `roadscan/` (الحزمة)، `scripts/`، `notebooks/`، `tests/` |
| *إضافات من عندنا* | نتايج السطح والرصيف، سجل العربيات، خريطة المسار، جرد معالم الطريق، طبقة الثبات |

## 2. النتايج

### clipA - شارع مسفلت داخل المدينة وقت الغروب/الليل (59.4 ث، 496×368، 36 إطار/ث)

| clipA defect type | unique defects | mean best conf |
|---|---:|---:|
| longitudinal_crack | 21 | 0.59 |
| alligator_crack | 13 | 0.69 |
| pothole | 12 | 0.56 |
| manhole_or_drain | 3 | 0.59 |
| transverse_crack | 3 | 0.49 |
| traffic_light_fault | 1 | 1.00 |
| **total** | **53** | |

* الطريق: **مسفلت** 100% من المسار، رصيف على جنب واحد على الأقل 57% من الوقت.
* معالم الطريق اللي اتفحصت: **33 لوحة، 62 إشارة مرور، 2 عبور مشاة**؛ علامات الأرض ظاهرة في 95% من الإطارات؛
  **1** عنصر اتعلّم "مشكلة محتملة" (حل تقريبي، القسم 5).
* المسار حوالي 474 م بمتوسط ~29 كم/س (تقدير من كاميرا واحدة، شوف القيود).
* التتبع: 82 تراك خام ← **53 اتسجّلوا** بعد الدمج والفلترة.

![timeline clipA](docs/images/fig_timeline_clipA.png)

### clipB - طريق ترابي ريفي (15 ث، 848×478، 25 إطار/ث)

| clipB defect type | unique defects | mean best conf |
|---|---:|---:|
| water_filled_pothole | 61 | 0.89 |
| pothole | 38 | 0.69 |
| alligator_crack | 3 | 0.60 |
| **total** | **102** | |

* الطريق: **ترابي** 100%، من غير رصيف، مية واقفة ≈ 20% من الطريق الظاهر.
* المسار حوالي 24 م بسرعة ~6 كم/س. التتبع: 239 تراك خام ← **102 اتسجّلوا**.

![timeline clipB](docs/images/fig_timeline_clipB.png)

![counts](docs/images/fig_defects_by_type.png)

## 3. السرعة والهاردوير

| | القيمة |
|---|---|
| الـ GPU المستخدم | **Tesla T4 x2** (Kaggle)، CPU Intel(R) Xeon(R) CPU @ 2.00GHz، PyTorch 2.11.0+cu128، CUDA 12.8 |
| الـ detector لوحده، PyTorch FP16 | 78.6 FPS (12.7 ms/إطار) |
| الـ detector لوحده، **TensorRT FP16** | **177.3 FPS** (5.6 ms/إطار) |
| detect + فلتر الطريق + التتبع ("النواة") | **17.8 FPS** (clipA) / 17.6 FPS (clipB) - Real-time على T4 |
| *كل* الموديلات (السطح، open-vocabulary، Mask2Former، المشهد، odometry) | 3.1 FPS (clipA) / 2.6 FPS (clipB) - وضع تحليل offline |
| قياس ONNX | 8 FPS - **مش حد الموديل**: نسخة onnxruntime-gpu ناقصها `libcublasLt` فاشتغلت على CPU من غير ما تقول |

![fps](docs/images/fig_fps.png)
![stage timing](docs/images/fig_stage_timing.png)

*القراءة*: الـ detector مش هو عنق الزجاجة. الموديلات الثقيلة الاختيارية (OWLv2 ≈ 150 ms/إطار، Mask2Former ≈ 50-70 ms/إطار، بتشتغل
كل 3 / 4 إطارات) هي اللي بتبطّأ. للـ real-time اقفلها أو قلّل تكرارها - النواة بتشتغل 18+ FPS.

### أنهي frame rate؟ (وإيه اللي يشتغل على Jetson)

نفس الكود اتشغل على **وضعين** (`scripts/run_video.py`، الخيار `--deploy-fps 24,18,12`):

* **وضع المرجع (offline)** - كل إطار: clipA 36 إطار/ث، clipB 25 إطار/ث. منه طلع `annotated.mp4` والجداول الأساسية.
* **وضع Jetson** - الموديلات بتشوف **حد أقصى N إطار/ث** (سقف؛ `target_fps` بيختار أصغر stride تحت السقف ده، وكل بارامتر معدود بالإطارات - ومعاه قاعدة "اتشاف في ≥ N إطار" - بيتعدّل على المعدل ده). كل إطار متحلل ليه سجل وصورة **الإطار الأصلي** عليها البوكسات بس. السقوف المتجرّبة: 10 إطار/ث. لسه مفيش بروفايل متجرّب وصل 90% recall و precision في الكليبين؛ أحسن recall هو `jetson_10fps` (clipA: stride 4 = **9.004 إطار/ث**، clipB: stride 2 = **12.5 إطار/ث**)؛ وإطاراته في `results/clip*/jetson_10fps/frames/`.

| clip | profile (cap) | analysis rate | stride | defects (reference) | defects (Jetson mode) | matched | only reference | only Jetson | recall | precision | core FPS (T4) | all-models FPS (T4) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clipA | 10 FPS | 9.004 FPS of 36.015 | 4 | 53 | 11 | 11 | 42 | 0 | 21% | 100% | 17.3 | 3.04 |
| clipB | 10 FPS | 12.5 FPS of 25.0 | 2 | 102 | 55 | 54 | 48 | 1 | 53% | 98% | 17.18 | 2.66 |

**No tested profile reaches 90 % recall and precision on both clips; best recall: `jetson_10fps`** (worst-clip recall 21%, precision 98%).

*إزاي تقرا الجدول*: **recall** = نسبة عيوب المرجع اللي وضع Jetson لقاها برضه، و**precision** = نسبة عيوبه اللي موجودة في المرجع (من `scripts/compare_modes.py`؛ التطابق = نفس الكلاس ووقت ظهور متداخل ±0.75 ث). ده قياس *اتساق* مع تشغيلنا بكل الإطارات، مش مع تعنين بشري. العيب بيظهر في الكادر لحظات قصيرة (وسيط ~0.3 ث في clipA و~1 ث في clipB)، فكل ما المعدل قل ضاع منهم أكتر - سقف 10 إطار/ث (stride 4 في clipA) كل اللي طلّعه صح (precision ~100%) لكنه لقى حوالي خُمس عيوب clipA بس، عشان كده سقف 10 **مش** كفاية. فالـ detector والتتبع (الخفيفين) يشتغلوا بأعلى معدل يستحمله الجهاز، والموديلات التقيلة (manholes، furniture، surface) بمعدل أقل وثابت. على T4 البروفايل الموصى بيه بيشغل النواة على 17.3 إطار/ث (clipA) / 17.18 (clipB) و*كل* الموديلات على 3.04 / 2.66. الـ Jetson أبطأ من T4 - شوف [docs/JETSON.md](docs/JETSON.md)؛ إحنا **ما قسناش** على Jetson فعلياً.

### مخرجات كل إطار

`results/clip*/frames/` (المرجع، كل إطار: بيانات بس) و `results/clip*/jetson_10fps/frames/` (الإطارات المتحللة: بيانات **وصور**):

| الملف | المحتوى |
|---|---|
| `frame_summary.csv` | صف لكل إطار: الوقت، نوع الطريق، الرصيف، حالة دهان الحارة / علامة الاختفاء، السرعة، الموقع، عدد العيوب / مستخدمي الطريق / اللافتات / الإشارات / العبور |
| `detections.csv` | صف لكل كائن في كل إطار: النوع، رقم التراك، الكلاس، الثقة، البوكس (x1,y1,x2,y2) |
| `frames.jsonl` | السجل الكامل المتداخل لكل إطار (عيوب، مستخدمو طريق، معالم، حالة الطريق، حركة العربية) |
| `img/frame_NNNNN.jpg` | **الإطار الأصلي** وعليه بوكسات كل اللي بنتتبعه (من غير HUD ولا legend ولا خريطة) |

كل القيم هي اللي نظام real-time يعرفها *في الإطار ده* (من الماضي بس). الخريطة الحية في `annotated.mp4` بتعرض بس المسار المتذكّر (آخر 120 متر) والعيوب اللي اتشافت لحد دلوقتي - مش باقي الطريق أبداً؛ و`route_map.png` ملخص بعد التشغيل للتقرير.

## 4. الطريقة

![pipeline](docs/images/fig_pipeline.png)

1. **Detector العيوب** - YOLO26s متدرّب (fine-tuned) على RDD2022 (4 كلاسات)، حجم الصورة 800، وثقة منخفضة (0.15) عن قصد: التتبع وفلتر التقرير هما اللي بيقرروا مين يعدّي.
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
* **العدّ**: العدد الفريد بيعتمد على قاعدة الدمج؛ clipB حقل حفر متداخلة، فـ "102 عيب فريد" تقدير مش حقيقة مطلقة. مفيش مرجع معنّون يدويًا، فـ**مش بندّعي precision/recall**.
* **أحكام اللوحات/الإشارات/الدهان حلول تقريبية** (ميل، لون اللمبة، تباين)، وبتتسجّل كـ *مشكلة محتملة*. CLIP على قصاصات اللوحات الصغيرة ضوضاء واتعطّل تحت 40 بكسل.
* **الـ odometry من كاميرا واحدة**: السرعة والمسافات بتتناسب مع ارتفاع كاميرا مفترض (1.4 م) وزاوية رؤية (75°)؛ اعتبرها تقريبية. أول ثواني الفيديو مفيهاش تاريخ حركة.
* **الـ smoothing السببي بيكلّف تأخير**: تغيّر الحالة بيظهر متأخر ~1-2 ث - تمن إننا ما بنبصّش للمستقبل.

## 6. الثبات ("من غير وميض، الماضي بس")

![speed](docs/images/fig_speed_stability.png)

* العيب **بيترسم بعد ما يتأكد بس** (درجة EMA بـ hysteresis، ≥ لمستين)، وبيكمّل كام إطار لو الاكتشاف فات، وإحداثيات المربع ونص الثقة متنعّمين - `DisplaySmoother`.
* التراكات المؤكدة السابقة **بترجع تغذّي خطوة الاكتشاف الجاية** (`TemporalPrior`: مزج ثقة وتصويت كلاس جوه مجموعة الشروخ؛ 421 اكتشاف اتعزّز في clipA) - ده "الـ attention window على الماضي بس" اللي كان مطلوب.
* **السرعة** بتعدّي على متوسط مقصوص سببي ← EMA ← محدّد تسارع/تباطؤ مع حماية بداية التشغيل، فمستحيل ننزل 40 ← 0 ← 40؛ حالات السطح والرصيف مابتتغيّرش إلا بعد ما تثبت 1.5 ث.
* قبل / بعد (التشغيل السابق مقابل ده):

![before after B](docs/images/before_after_clipB.jpg)

**مفتاح الألوان** (مرسوم كمان على الفيديو): المربعات الملونة = كلاس العيب (شوف الـ legend)، تلوين برتقالي فاتح = أرض ترابية، مربع أخضر رفيع = لوحة/إشارة/عبور مشاة اتحكم عليها سليمة، أحمر = *مشكلة محتملة*، **دهان الحارة بيتلوّن أخضر لما يكون ظاهر وسليم، وأحمر لما يبقى باهت/متآكل**، والـ HUD بيكتب سطر أحمر *MISSING - possible problem* لو الطريق مسفلت ومفيش دهان ظاهر لمدة 3 ثواني متواصلة (من الماضي بس).

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
python -m pytest -q tests                          # 28 اختبار وحدة (المنطق الزمني، دمج التراكات، المعالم)
```

على Kaggle: `notebooks/03_infer_videos_kaggle.ipynb` (استنتاج)، `01_train_yolo26_kaggle.ipynb` / `04_finetune_yolo26_kaggle.ipynb` (تدريب)، والـ runbook في [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md).

## 10. خريطة الريبو

```
roadscan/            detectors.py  tracking.py  temporal.py  furniture.py  surface.py  openvocab.py  odometry.py
                     mapping.py  vehicles.py  road_mask.py  render.py  export.py  pipeline.py  config.py
scripts/             run_video.py  benchmark.py  sweep.py  train_yolo.py  prepare_data.py  make_figures.py  make_gallery.py
notebooks/           01 تدريب · 02 RF-DETR (منافس) · 03 استنتاج · 04 fine-tune (بتتولّد من make_notebooks.py)
tests/               28 اختبار وحدة
results/             المخرجات النهائية لكل فيديو + benchmarks + تاريخ التشغيلات السابقة (results/history)
docs/                JOURNEY(.ar).md · JETSON.md · KAGGLE_RUNBOOK.md · images/
```

## 11. شكر وتراخيص

Ultralytics YOLO26 (AGPL-3.0)، RDD2022، SegFormer (NVIDIA)، Mask2Former (Meta / أوزان Mapillary Vistas)، OWLv2 (Google)، BoT-SORT، SAM 2 (Meta)، OpenCV. راجع ترخيص كل موديل قبل أي استخدام تجاري.
