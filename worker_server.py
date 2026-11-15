"""سيرفر التبييض اللي بيشتغل على Colab (GPU): بيستقبل zip فصل، ويرجع zip متبيّض.

التشغيل (من نوتبوك Colab، شوف colab/SETUP.md):
    WORKER_TOKEN=... TRANSLATOR_REPO=/content/manga-translator-ui \
    uvicorn worker_server:app --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

import asyncio
import hmac
import os
import sys

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from profiles import PROFILE_CHOICES

WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
TRANSLATOR_REPO = os.environ.get("TRANSLATOR_REPO", "/content/manga-translator-ui")
WORK_ROOT = os.environ.get("WORK_ROOT", "/content/whiten-work")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PAGE_CONCURRENCY = int(os.environ.get("PAGE_CONCURRENCY", "1") or 1)
# cli = manga-translator-ui على Colab (GPU) | builtin = المحرك المدمج (CPU) على أي سيرفر عادي
WORKER_ENGINE = (
    os.environ.get("WORKER_ENGINE") or ("cli" if os.path.isdir(TRANSLATOR_REPO) else "builtin")
).strip().lower()

app = FastAPI(title="whiten-worker")
# GPU واحد: فصل واحد في نفس الوقت
_gpu_lock = asyncio.Lock()


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "engine": WORKER_ENGINE}


@app.post("/clean")
async def clean(
    file: UploadFile = File(...),
    profile: str = Form("fast"),
    x_worker_token: str = Header(default=""),
) -> Response:
    if not WORKER_TOKEN:
        raise HTTPException(503, detail="WORKER_TOKEN is not set on the worker")
    if not hmac.compare_digest(x_worker_token, WORKER_TOKEN):
        raise HTTPException(401, detail="bad token")
    if profile not in PROFILE_CHOICES:
        raise HTTPException(400, detail=f"unknown profile: {profile}")

    payload = await file.read()
    async with _gpu_lock:
        try:
            if WORKER_ENGINE == "builtin":
                from local_cleaner import LocalCleanError, clean_chapter_local

                cleaned = await clean_chapter_local(
                    payload, profile=profile, model_dir=MODEL_DIR, concurrency=PAGE_CONCURRENCY
                )
            else:
                from cli_cleaner import CliCleanError, clean_chapter_cli

                cleaned = await clean_chapter_cli(
                    payload,
                    profile=profile,
                    repo_dir=TRANSLATOR_REPO,
                    python_exe=sys.executable,
                    work_root=WORK_ROOT,
                )
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)[:500]) from exc
        except RuntimeError as exc:  # LocalCleanError و CliCleanError كلهم RuntimeError
            raise HTTPException(422, detail=str(exc)[:500]) from exc
    return Response(content=cleaned, media_type="application/zip")
