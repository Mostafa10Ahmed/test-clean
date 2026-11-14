# تشغيل سيرفر التبييض على Colab

الـ Colab بيشغّل الـ GPU، والبوت على Railway بيبعت له الفصول عن طريق لينك عام (Cloudflare Tunnel).

> ملاحظات:
> - الجلسة بتتقفل بعد فترة، ولازم تعيد تشغيل الخلايا.
> - اللينك بيتغيّر كل مرة، فلازم تحدّث `WORKER_URL` في Railway.
> - اتأكد من شروط استخدام Colab قبل ما تعتمد عليه كسيرفر دايم.

## 1) فعّل الـ GPU
من القائمة: **Runtime → Change runtime type → T4 GPU**.

## 2) خلية 1: حط توكن السيرفر في Secrets
من أيقونة المفتاح 🔑 في الشريط الجانبي، اعمل secret اسمه `WORKER_TOKEN` بقيمة طويلة عشوائية (مثلاً من `openssl rand -hex 24`)، وفعّل **Notebook access**. نفس القيمة لازم تتحط في `WORKER_TOKEN` على Railway.

## 3) خلية 2: حمّل ريبو manga-translator-ui
```python
!git clone --depth 1 https://github.com/hgmzhn/manga-translator-ui /content/manga-translator-ui
```

## 4) خلية 3: ركّب المكتبات
Colab فيه torch وCUDA جاهزين، فبنستبعدهم من القايمة:
```python
!pip install -q uv
!cd /content/manga-translator-ui && uv export --frozen --no-hashes --no-emit-project 2>/dev/null \
  | grep -v -E '^(torch|torchvision|triton|nvidia-|cuda-|pyqt6|customtkinter)' > /content/req.txt
!pip install -q -r /content/req.txt
!pip install -q fastapi "uvicorn[standard]" python-multipart
```
لو ظهر تعارض في الإصدارات، ابعت الخطأ كامل وأنا أعدّل القايمة.

## 5) خلية 4: ارفع ملفات الـ worker
من الشريط الجانبي (📁 Files) ارفع الملفات دي (الريبو خاص، فمش بنسحبه من GitHub):
- `worker_server.py`
- `cli_cleaner.py`
- `whitening.py`
- `profiles.py`

## 6) خلية 5: شغّل الـ worker
```python
import os
from google.colab import userdata
os.environ["WORKER_TOKEN"] = userdata.get("WORKER_TOKEN")
os.environ["TRANSLATOR_REPO"] = "/content/manga-translator-ui"
os.environ["WORK_ROOT"] = "/content/whiten-work"
!nohup uvicorn worker_server:app --host 127.0.0.1 --port 8001 > /content/worker.log 2>&1 &
```

## 7) خلية 6: افتح لينك عام بـ Cloudflare Tunnel
```python
!wget -q -O /content/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i /content/cloudflared.deb > /dev/null
!nohup cloudflared tunnel --url http://127.0.0.1:8001 > /content/tunnel.log 2>&1 &
import time, re
time.sleep(8)
print(re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/tunnel.log").read())[:1])
```
الرابط اللي هيطلع (`https://....trycloudflare.com`) هو `WORKER_URL`.

## 8) تأكد إن السيرفر شغال
```python
!curl -s http://127.0.0.1:8001/health
```
المفروض يرجع `{"ok":true}`.

## 9) على Railway
في **Variables** للبوت، حط:
- `WORKER_URL` = اللينك من الخلية 6
- `WORKER_TOKEN` = نفس القيمة اللي في Colab Secrets
- باقي المتغيرات زي `DISCORD_TOKEN` وغيرها من `.env.example`

لو الـ worker مش شغال، البوت هيرد برسالة واضحة إن السيرفر مش متاح.
