"""يبعت الفصل لسيرفر التبييض (worker_server.py) بنظام المهام غير المتزامنة (Polling) لتجنب خطأ HTTP 502 على Railway."""
from __future__ import annotations

import asyncio
import logging

import aiohttp

from cli_cleaner import CliCleanError

log = logging.getLogger("whiten-bot.remote")


async def clean_chapter_remote(
    zip_bytes: bytes,
    *,
    profile: str,
    worker_url: str,
    token: str,
    timeout_s: int = 3600,
    poll_interval_s: float = 3.0,
) -> bytes:
    base_url = worker_url.rstrip("/")
    headers = {"X-Worker-Token": token}
    timeout = aiohttp.ClientTimeout(total=timeout_s, connect=30, sock_read=300)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1) نحاول أولاً إرسال الجوب لـ /jobs (الوضع غير المتزامن لمنع 502 timeout)
            form = aiohttp.FormData()
            form.add_field("file", zip_bytes, filename="chapter.zip", content_type="application/zip")
            form.add_field("profile", profile)

            async with session.post(f"{base_url}/jobs", data=form, headers=headers) as resp:
                if resp.status == 404:
                    # لو السيرفر قديم مش عليه /jobs نرجع لـ /clean المباشر
                    return await _clean_legacy_sync(session, base_url, zip_bytes, profile, headers)
                if resp.status != 200:
                    detail = (await resp.read())[:300].decode("utf-8", "replace")
                    raise CliCleanError(f"worker returned HTTP {resp.status} on /jobs: {detail}")
                data = await resp.json(content_type=None)
                job_id = data["job_id"]

            log.info("Submitted remote job %s to %s", job_id, base_url)

            # 2) متابعة حالة الجوب كل بضع ثوانٍ (كل طلب خفيف جداً < 10ms فلا ينقطع اتصال Railway أبداً)
            started = asyncio.get_running_loop().time()
            transient_errors = 0
            while True:
                if asyncio.get_running_loop().time() - started > timeout_s:
                    raise CliCleanError("remote worker timed out while whitening chapter")
                await asyncio.sleep(poll_interval_s)
                try:
                    async with session.get(f"{base_url}/jobs/{job_id}", headers=headers) as poll_resp:
                        if poll_resp.status in (502, 503, 504):
                            transient_errors += 1
                            if transient_errors > 10:
                                body = (await poll_resp.read())[:300].decode("utf-8", "replace")
                                raise CliCleanError(
                                    f"worker crashed or stopped responding (HTTP {poll_resp.status}, likely OOM on Railway): {body}"
                                )
                            continue
                        if poll_resp.status != 200:
                            body = (await poll_resp.read())[:300].decode("utf-8", "replace")
                            raise CliCleanError(f"worker status check returned HTTP {poll_resp.status}: {body}")
                        transient_errors = 0
                        status_data = await poll_resp.json(content_type=None)
                except aiohttp.ClientError as exc:
                    transient_errors += 1
                    if transient_errors > 10:
                        raise CliCleanError(f"lost connection to worker while polling: {exc}") from exc
                    continue

                status = status_data.get("status")
                if status == "done":
                    break
                if status == "failed":
                    err = status_data.get("error") or "unknown worker error"
                    raise CliCleanError(f"worker failed: {err}")

            # 3) تحميل الـ zip الجاهز بعد انتهاء التبييض
            async with session.get(f"{base_url}/jobs/{job_id}/result", headers=headers) as res_resp:
                body = await res_resp.read()
                if res_resp.status != 200:
                    detail = body[:300].decode("utf-8", "replace")
                    raise CliCleanError(f"worker result download returned HTTP {res_resp.status}: {detail}")
                return body

    except aiohttp.ClientError as exc:
        raise CliCleanError(
            f"the worker is unreachable (check WORKER_URL and port settings): {exc}"
        ) from exc


async def _clean_legacy_sync(
    session: aiohttp.ClientSession,
    base_url: str,
    zip_bytes: bytes,
    profile: str,
    headers: dict[str, str],
) -> bytes:
    form = aiohttp.FormData()
    form.add_field("file", zip_bytes, filename="chapter.zip", content_type="application/zip")
    form.add_field("profile", profile)
    async with session.post(f"{base_url}/clean", data=form, headers=headers) as resp:
        body = await resp.read()
        if resp.status != 200:
            detail = body[:300].decode("utf-8", "replace")
            raise CliCleanError(f"worker returned HTTP {resp.status}: {detail}")
        return body
