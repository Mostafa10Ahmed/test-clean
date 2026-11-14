"""طابور الفصول: جوب واحد بالتتابع افتراضياً عشان الـ GPU ما يتخنقش، ويقدر يتزود بعدد workers."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


class QueueFullError(Exception):
    pass


@dataclass
class Job:
    id: int
    owner_id: int
    owner_name: str
    channel_id: int
    name: str
    profile: str
    payload: bytes = field(repr=False)
    status: str = PENDING
    result_url: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


JobHandler = Callable[[Job], Awaitable[str]]
JobListener = Callable[[Job], Awaitable[None]]


class JobQueue:
    def __init__(
        self,
        handler: JobHandler,
        *,
        workers: int = 1,
        max_pending: int = 20,
        listener: Optional[JobListener] = None,
    ):
        self._handler = handler
        self._workers_count = max(1, workers)
        self._max_pending = max(1, max_pending)
        self._listener = listener
        self._pending: deque[Job] = deque()
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._cond = asyncio.Condition()
        self._tasks: list[asyncio.Task] = []

    # ---------- دورة الحياة ----------
    def start(self) -> None:
        for _ in range(self._workers_count):
            self._tasks.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # ---------- الإضافة والاستعلام ----------
    async def submit(
        self,
        *,
        owner_id: int,
        owner_name: str,
        channel_id: int,
        name: str,
        profile: str,
        payload: bytes,
    ) -> Job:
        async with self._cond:
            if len(self._pending) >= self._max_pending:
                raise QueueFullError(f"The queue is full ({self._max_pending} chapters). Please wait a bit.")
            job = Job(
                id=self._next_id,
                owner_id=owner_id,
                owner_name=owner_name,
                channel_id=channel_id,
                name=name,
                profile=profile,
                payload=payload,
            )
            self._next_id += 1
            self._jobs[job.id] = job
            self._pending.append(job)
            self._cond.notify()
            return job

    async def position(self, job: Job) -> Optional[int]:
        """ترتيب الجوب في الانتظار (1 = التالي)، أو None لو مش في الانتظار."""
        async with self._cond:
            for index, pending in enumerate(self._pending):
                if pending.id == job.id:
                    return index + 1
            return None

    def get(self, job_id: int) -> Optional[Job]:
        return self._jobs.get(job_id)

    def snapshot(self) -> tuple[list[Job], list[Job]]:
        """(الجاري، المنتظر بالترتيب)."""
        running = [j for j in self._jobs.values() if j.status == RUNNING]
        return running, list(self._pending)

    async def cancel(self, job_id: int, requester_id: int, is_admin: bool = False) -> Job:
        async with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"No job with ID #{job_id}")
            if job.owner_id != requester_id and not is_admin:
                raise PermissionError("You can only cancel your own jobs")
            if job.status != PENDING:
                raise ValueError(f"Job #{job_id} is `{job.status}` and cannot be cancelled")
            self._pending.remove(job)
            job.status = CANCELLED
            job.payload = b""
            job.finished_at = time.monotonic()
        await self._notify(job)
        return job

    # ---------- الـ worker ----------
    async def _next_job(self) -> Job:
        async with self._cond:
            while not self._pending:
                await self._cond.wait()
            return self._pending.popleft()

    async def _worker(self) -> None:
        while True:
            job = await self._next_job()
            job.status = RUNNING
            job.started_at = time.monotonic()
            await self._notify(job)
            try:
                job.result_url = await self._handler(job)
                job.status = DONE
            except asyncio.CancelledError:
                job.status = FAILED
                job.error = "The bot was stopped while processing"
                raise
            except Exception as exc:
                log.exception("Job #%s failed", job.id)
                job.status = FAILED
                job.error = str(exc) or exc.__class__.__name__
            finally:
                job.finished_at = time.monotonic()
                job.payload = b""  # نفضي الذاكرة
                await self._notify(job)

    async def _notify(self, job: Job) -> None:
        if self._listener is None:
            return
        try:
            await self._listener(job)
        except Exception:
            log.exception("Queue listener failed for job #%s", job.id)
