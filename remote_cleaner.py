"""يبعت الفصل لسيرفر التبييض على Colab (worker_server.py) ويرجع الـ zip المتبيّض."""
from __future__ import annotations

import aiohttp

from cli_cleaner import CliCleanError


async def clean_chapter_remote(
    zip_bytes: bytes,
    *,
    profile: str,
    worker_url: str,
    token: str,
    timeout_s: int = 3600,
) -> bytes:
    form = aiohttp.FormData()
    form.add_field("file", zip_bytes, filename="chapter.zip", content_type="application/zip")
    form.add_field("profile", profile)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{worker_url.rstrip('/')}/clean",
                data=form,
                headers={"X-Worker-Token": token},
            ) as resp:
                body = await resp.read()
                if resp.status != 200:
                    detail = body[:300].decode("utf-8", "replace")
                    raise CliCleanError(f"worker returned HTTP {resp.status}: {detail}")
                return body
    except aiohttp.ClientError as exc:
        raise CliCleanError(
            f"the GPU worker is unreachable (is the Colab notebook running?): {exc}"
        ) from exc
