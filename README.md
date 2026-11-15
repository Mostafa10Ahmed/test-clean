# بوت التبييض (Discord → gofile)

بوت ديسكورد بياخد فصل (zip فيه صور، أو لينك)، **يبيّضه بنفسه**، ويرفع النتيجة على gofile ويرجعلك الرابط.
مش محتاج أي سيرفر تاني: موديل كشف الكلام (CTD) بيتحمّل أوتوماتيك أول تشغيل، وبيشتغل على CPU عادي.

## إزاي بيبيّض

1. موديل **CTD** (comic-text-detector) بيحدد مكان الكلام في الصفحة. الصفحات الطويلة (ويبتون) بتتقسم عشان الكلام ما يصغرش.
2. لكل جملة بيبص على اللي حواليها:
   - **فقاعة سادة** (بيضا أو لون واحد): بيملاها بنفس اللون، فالتبييض بيطلع نضيف 100% والإطار بيفضل زي ما هو.
   - **كلام فوق رسمة**: بيعمل inpaint. بروفايل `perfect` بيستخدم LaMa لو `simple-lama-inpainting` متسطّبة، وإلا بيستخدم OpenCV.

## المكونات

| ملف | الوظيفة |
|---|---|
| `bot.py` | أوامر ديسكورد: `/clean` و `/queue` و `/cancel` |
| `local_cleaner.py` | **محرك التبييض المدمج** (الافتراضي) |
| `queue_manager.py` | الطابور (FIFO، إلغاء، حد أقصى للانتظار، حالة كل جوب) |
| `sources.py` | تحميل الـ zip من لينك (Google Drive، gofile، رابط مباشر) |
| `cli_cleaner.py` | محرك اختياري: manga-translator-ui متسطّب على نفس الجهاز |
| `remote_cleaner.py` | محرك اختياري: سيرفر GPU عن بعد (Colab) |
| `worker_server.py` | سيرفر الـ GPU نفسه (بيشتغل على Colab) |
| `whitening.py` | فك الـ zip، ترتيب الصفحات طبيعياً، تجميع الخرج |
| `profiles.py` | بروفايلات `perfect` و `fast` |
| `gofile.py` | رفع الـ zip على gofile |

## التشغيل

### 1) إعداد البوت

```bash
pip install -r requirements.txt
copy .env.example .env
```

عبّي `DISCORD_TOKEN` في `.env` (وممكن `GOFILE_TOKEN`). كده البوت جاهز يبيّض.

اختياري، لجودة أعلى للكلام اللي فوق الرسومات (محتاج torch، تقيلة):

```bash
pip install simple-lama-inpainting
```

### 2) شغّل البوت

```bash
python bot.py
```

## فولدر Google Drive

`/clean link:<لينك فولدر>` بيحمّل الصور اللي جوه الفولدر (وفولدراته الفرعية لحد مستويين) ويبيّضها. لازم:

1. تعمل **API Key**: Google Cloud Console ← مشروع جديد ← فعّل **Google Drive API** ← Credentials ← Create API key.
2. تحط `GOOGLE_API_KEY` في Railway Variables.
3. الفولدر يكون **Anyone with the link** (Viewer).

لو الفولدر فيه ملف zip واحد بس من غير صور، بيحمّله ويبيّضه.

## الأوامر

- `/clean chapter:<ملف zip>` أو `/clean link:<لينك>` + `profile:<perfect|fast>` — بيضيف الفصل للطابور ويرجعلك رابط gofile لما يخلص.
- `/queue` — الجاري والمنتظر.
- `/cancel job_id:<رقم>` — يلغي فصل لسه في الانتظار (صاحبه أو أدمن بس).

## البروفايلات

| | الكلام في الفقاعات | الكلام فوق الرسومات |
|---|---|---|
| `perfect` | ملء بلون الخلفية | LaMa (لو متسطّبة) وإلا OpenCV |
| `fast` | ملء بلون الخلفية | OpenCV (أسرع) |

## محركات تانية (اختياري)

`CLEAN_ENGINE` في `.env`:
- `builtin` (الافتراضي): البوت بيبيّض بنفسه.
- `remote`: يبعت لسيرفر GPU على Colab. شوف [SETUP.md](SETUP.md).
- `cli`: يشغّل [manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui) متسطّب على نفس الجهاز.

## الإعدادات (`.env`)

شوف `.env.example` للقائمة كاملة. أهمها:

- `PAGE_CONCURRENCY` — عدد الصفحات اللي بتتبيّض مع بعض (المحرك المدمج). الافتراضي 1 (أقل رامات).
- `CV_THREADS` — عدد threads الحساب (فاضي = أنوية الحاوية).
- `TILE_RATIO` — الصفحات الطويلة: `1.5` أسرع، `1` أدق.
- `QUEUE_WORKERS` — عدد الفصول اللي بتتعالج مع بعض. خليه 1 مع GPU واحد.
- `MAX_PENDING` — أقصى عدد فصول في الانتظار.
- `ALLOWED_CHANNEL_IDS` — الرومات المسموح فيها بالأوامر.
- `ADMIN_ROLE_ID` — رتبة تقدر تلغي أي جوب.

## قيود معروفة

- الطابور في الذاكرة: لو البوت اتقفل، الفصول اللي في الانتظار بتضيع.
- الملف المرفوع على ديسكورد لازم يكون ≤ 25MB؛ للأكبر استخدم لينك.

## الاختبارات (سيرفر الـ worker)

```bash
pip install fastapi httpx python-multipart
python -m pytest -q test_worker.py
```
