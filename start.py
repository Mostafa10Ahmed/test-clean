"""نقطة التشغيل الوحيدة على Railway.

- بدون متغيرات إضافية: بيشغّل البوت (bot.py).
- WORKER_MODE=1: بيشغّل سيرفر التبييض (worker_server) من غير ما يحتاج DISCORD_TOKEN.
"""
import os
import runpy

if os.getenv("WORKER_MODE", "").strip().lower() in ("1", "true", "yes"):
    import uvicorn

    uvicorn.run(
        "worker_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
else:
    runpy.run_path("bot.py", run_name="__main__")
