"""رفع الملفات على gofile وإرجاع رابط التحميل."""
from __future__ import annotations

from typing import Optional

import aiohttp

GOFILE_UPLOAD_URL = "https://upload.gofile.io/uploadfile"


class GofileError(Exception):
    pass


async def upload_to_gofile(
    session: aiohttp.ClientSession,
    data: bytes,
    filename: str,
    *,
    token: Optional[str] = None,
    upload_url: str = GOFILE_UPLOAD_URL,
) -> str:
    form = aiohttp.FormData()
    form.add_field("file", data, filename=filename, content_type="application/zip")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with session.post(upload_url, data=form, headers=headers) as resp:
            payload = await resp.json(content_type=None)
    except aiohttp.ClientError as exc:
        raise GofileError(f"Failed to connect to gofile: {exc}") from exc

    if resp.status != 200 or payload.get("status") != "ok":
        raise GofileError(f"gofile rejected the upload: {payload}")

    link = payload.get("data", {}).get("downloadPage")
    if not link:
        raise GofileError(f"gofile did not return a download link: {payload}")
    return link
