"""نقطة التشغيل الموحدة على Railway.

- بدون WORKER_MODE (ومع وجود DISCORD_TOKEN): بيشغّل البوت (bot.py).
- مع WORKER_MODE=1 (أو لو مفيش DISCORD_TOKEN وفيه WORKER_TOKEN): بيشغّل سيرفر التبييض (worker_server).
"""
import os

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("CV_THREADS", "4"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

worker_mode = os.getenv("WORKER_MODE", "").strip().lower() in ("1", "true", "yes")
has_discord = bool(os.getenv("DISCORD_TOKEN", "").strip())
has_worker_token = bool(os.getenv("WORKER_TOKEN", "").strip())

if worker_mode or (not has_discord and has_worker_token):
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    print(f"[start.py] Starting worker_server on 0.0.0.0:{port} ...", flush=True)
    uvicorn.run(
        "worker_server:app",
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=120,
        limit_concurrency=10,
    )
else:
    import runpy

    print("[start.py] Starting Discord bot (bot.py) ...", flush=True)
    runpy.run_path("bot.py", run_name="__main__")
