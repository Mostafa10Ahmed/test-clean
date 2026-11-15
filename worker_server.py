"""سيرفر التبييض (Worker): يشتغل على Railway أو Hugging Face Spaces أو Colab.

يدعم طريقتين:
1. غير متزامن (Async Polling) عبر `/jobs` لتفادي خطأ Railway HTTP 502 (Application failed to respond) مع الفصول الطويلة والـ stitched.
2. مباشر عبر `/clean` للتوافق مع الاختبارات والفصول الصغيرة.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from profiles import PROFILE_CHOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("whiten-worker")

WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
TRANSLATOR_REPO = os.environ.get("TRANSLATOR_REPO", "/content/manga-translator-ui")
WORK_ROOT = os.environ.get("WORK_ROOT", "/tmp/whiten-work")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PAGE_CONCURRENCY = int(os.environ.get("PAGE_CONCURRENCY", "1") or 1)
# cli = manga-translator-ui على Colab (GPU) | builtin = المحرك المدمج (CPU) على أي سيرفر عادي
WORKER_ENGINE = (
    os.environ.get("WORKER_ENGINE") or ("cli" if os.path.isdir(TRANSLATOR_REPO) else "builtin")
).strip().lower()


@dataclass
class WorkerJob:
    job_id: str
    profile: str
    work_dir: Path
    in_zip: Path
    out_zip: Path
    status: str = "pending"  # pending | running | done | failed
    done_pages: int = 0
    total_pages: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


_jobs: dict[str, WorkerJob] = {}
_gpu_lock = asyncio.Lock()


async def _warmup_model() -> None:
    if WORKER_ENGINE == "builtin":
        try:
            from local_cleaner import ensure_model

            await ensure_model(MODEL_DIR)
        except Exception:
            log.exception("Warmup model download failed (will retry on first job)")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(_warmup_model())
    yield
    task.cancel()


app = FastAPI(title="whiten-worker", lifespan=lifespan)


def _check_auth(x_worker_token: str) -> None:
    if not WORKER_TOKEN:
        raise HTTPException(503, detail="WORKER_TOKEN is not set on the worker")
    if not hmac.compare_digest(x_worker_token, WORKER_TOKEN):
        raise HTTPException(401, detail="bad token")


async def _save_upload_to_disk(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def _cleanup_job(job_id: str) -> None:
    job = _jobs.pop(job_id, None)
    if job is not None:
        shutil.rmtree(job.work_dir, ignore_errors=True)


def _cleanup_stale_jobs(max_age_s: float = 7200) -> None:
    now = time.monotonic()
    stale = [jid for jid, j in _jobs.items() if now - j.created_at > max_age_s]
    for jid in stale:
        _cleanup_job(jid)


async def _run_worker_job(job: WorkerJob) -> None:
    async with _gpu_lock:
        job.status = "running"
        try:
            if WORKER_ENGINE == "builtin":
                from local_cleaner import clean_chapter_local_file

                def _on_progress(done: int, total: int) -> None:
                    job.done_pages = done
                    job.total_pages = total

                await clean_chapter_local_file(
                    job.in_zip,
                    job.out_zip,
                    profile=job.profile,
                    model_dir=MODEL_DIR,
                    concurrency=PAGE_CONCURRENCY,
                    progress_cb=_on_progress,
                )
            else:
                from cli_cleaner import clean_chapter_cli

                payload = job.in_zip.read_bytes()
                cleaned = await clean_chapter_cli(
                    payload,
                    profile=job.profile,
                    repo_dir=TRANSLATOR_REPO,
                    python_exe=sys.executable,
                    work_root=WORK_ROOT,
                )
                job.out_zip.write_bytes(cleaned)
                del payload, cleaned
            job.in_zip.unlink(missing_ok=True)
            job.status = "done"
        except Exception as exc:
            log.exception("Worker job %s failed", job.job_id)
            job.status = "failed"
            job.error = str(exc)[:500] or exc.__class__.__name__
            job.in_zip.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "engine": WORKER_ENGINE}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    profile: str = Form("fast"),
    x_worker_token: str = Header(default=""),
) -> dict:
    """يستقبل الفصل ويبدأ تبييضه في الخلفية فوراً، ويرد في أقل من ثانية لتجنب Railway 502."""
    _check_auth(x_worker_token)
    if profile not in PROFILE_CHOICES:
        raise HTTPException(400, detail=f"unknown profile: {profile}")

    _cleanup_stale_jobs()
    Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="wjob-", dir=WORK_ROOT))
    in_zip = work_dir / "input.zip"
    out_zip = work_dir / "output.zip"
    await _save_upload_to_disk(file, in_zip)

    job_id = uuid.uuid4().hex
    job = WorkerJob(
        job_id=job_id,
        profile=profile,
        work_dir=work_dir,
        in_zip=in_zip,
        out_zip=out_zip,
    )
    _jobs[job_id] = job
    asyncio.create_task(_run_worker_job(job))
    return {"job_id": job_id, "status": job.status}


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_worker_token: str = Header(default=""),
) -> dict:
    _check_auth(x_worker_token)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail="job not found or expired")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "done_pages": job.done_pages,
        "total_pages": job.total_pages,
        "error": job.error,
    }


@app.get("/jobs/{job_id}/result")
async def get_job_result(
    job_id: str,
    background_tasks: BackgroundTasks,
    x_worker_token: str = Header(default=""),
) -> FileResponse:
    _check_auth(x_worker_token)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail="job not found or expired")
    if job.status != "done" or not job.out_zip.exists():
        raise HTTPException(409, detail=f"job is not ready (status={job.status})")
    background_tasks.add_task(_cleanup_job, job_id)
    return FileResponse(
        path=job.out_zip,
        media_type="application/zip",
        filename="cleaned.zip",
    )


@app.post("/clean")
async def clean(
    file: UploadFile = File(...),
    profile: str = Form("fast"),
    x_worker_token: str = Header(default=""),
) -> Response:
    _check_auth(x_worker_token)
    if profile not in PROFILE_CHOICES:
        raise HTTPException(400, detail=f"unknown profile: {profile}")

    payload = await file.read()
    async with _gpu_lock:
        try:
            if WORKER_ENGINE == "builtin":
                from local_cleaner import clean_chapter_local

                cleaned = await clean_chapter_local(
                    payload, profile=profile, model_dir=MODEL_DIR, concurrency=PAGE_CONCURRENCY
                )
            else:
                from cli_cleaner import clean_chapter_cli

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
