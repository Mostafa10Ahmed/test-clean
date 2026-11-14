"""تحميل الـ zip من لينك: Google Drive، gofile، أو أي رابط مباشر لملف zip."""
from __future__ import annotations

import asyncio
import io
import ipaddress
import re
import socket
import zipfile
from typing import Optional
from urllib.parse import ParseResult, urlencode, urljoin, urlparse

import aiohttp

MAX_REDIRECTS = 5
_DRIVE_FILE_RE = re.compile(r"/file/d/([\w-]+)")
_DRIVE_OPEN_RE = re.compile(r"[?&]id=([\w-]+)")
_DRIVE_FOLDER_RE = re.compile(r"/folders/([\w-]+)")
_DRIVE_API = "https://www.googleapis.com/drive/v3/files"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_FOLDER_MIME = "application/vnd.google-apps.folder"
_FOLDER_MAX_DEPTH = 2
_FOLDER_MAX_FILES = 400
_FOLDER_PARALLEL = 4
_GOFILE_RE = re.compile(r"gofile\.io/d/(\w+)")
_CONFIRM_RE = re.compile(r'confirm=([0-9A-Za-z_-]+)')
_FORM_ACTION_RE = re.compile(r'<form[^>]+action="([^"]+)"', re.I)
_FORM_INPUT_RE = re.compile(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', re.I)
_DRIVE_DL = "https://drive.usercontent.google.com/download"


class SourceError(Exception):
    pass


def _is_public_host(host: str) -> bool:
    """يمنع الروابط اللي بتشاور على الشبكة الداخلية أو الـ localhost (حماية من SSRF)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def validate_url(url: str) -> ParseResult:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SourceError("The link must use https")
    if not parsed.hostname:
        raise SourceError("Invalid link")
    if not _is_public_host(parsed.hostname):
        raise SourceError("This link is not allowed (internal network)")
    return parsed


def classify_link(url: str) -> tuple[str, str]:
    """يرجّع (النوع, الرابط المحول). النوع: drive / gofile / direct."""
    url = url.strip()
    if "drive.google.com" in url or "drive.usercontent.google.com" in url or "docs.google.com" in url:
        folder = _DRIVE_FOLDER_RE.search(url)
        if folder:
            return "drive_folder", folder.group(1)
        match = _DRIVE_FILE_RE.search(url) or _DRIVE_OPEN_RE.search(url)
        if not match:
            raise SourceError("Could not find the file ID in the Google Drive link. Use the file's Share > Copy link")
        return "drive", f"{_DRIVE_DL}?id={match.group(1)}&export=download&confirm=t"
    match = _GOFILE_RE.search(url)
    if match:
        return "gofile", match.group(1)
    return "direct", url


def _api_error(status: int) -> SourceError:
    if status in (400, 403):
        return SourceError(
            "Google rejected the API key. Check GOOGLE_API_KEY, that Google Drive API is enabled for it, "
            "and that it has no restrictions blocking this server"
        )
    if status == 404:
        return SourceError("Folder not found. Set the folder to 'Anyone with the link' (Viewer)")
    return SourceError(f"Google Drive API returned HTTP {status}")


async def _drive_get_json(session: aiohttp.ClientSession, url: str, params: dict) -> dict:
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise _api_error(resp.status)
        return await resp.json(content_type=None)


async def _list_folder(session, folder_id: str, api_key: str, prefix: str, depth: int) -> list[tuple[str, str, str, int]]:
    """يرجّع (المسار, id, mimeType, الحجم) لكل ملف في الفولدر، ويدخل الفولدرات الفرعية لحد مستوى معين."""
    items: list[tuple[str, str, str, int]] = []
    page_token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,size)",
            "pageSize": "1000",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        data = await _drive_get_json(session, _DRIVE_API, params)
        for f in data.get("files", []):
            name = f.get("name", "")
            if f.get("mimeType") == _FOLDER_MIME:
                if depth < _FOLDER_MAX_DEPTH:
                    items += await _list_folder(session, f["id"], api_key, f"{prefix}{name}/", depth + 1)
                continue
            items.append((f"{prefix}{name}", f["id"], f.get("mimeType", ""), int(f.get("size") or 0)))
        page_token = data.get("nextPageToken")
        if not page_token:
            return items


async def _drive_download(session, file_id: str, api_key: str) -> bytes:
    async with session.get(f"{_DRIVE_API}/{file_id}", params={"alt": "media", "key": api_key, "supportsAllDrives": "true"}) as resp:
        if resp.status != 200:
            raise _api_error(resp.status)
        return await resp.read()


async def _download_drive_folder(session, folder_id: str, api_key: Optional[str], max_bytes: int) -> tuple[str, bytes]:
    if not api_key:
        raise SourceError(
            "Folder links need GOOGLE_API_KEY on the server. Add it, or send a single zip file instead"
        )
    meta = await _drive_get_json(session, f"{_DRIVE_API}/{folder_id}", {"fields": "name", "supportsAllDrives": "true", "key": api_key})
    folder_name = re.sub(r'[\\/:*?"<>|]+', "_", meta.get("name") or "chapter").strip() or "chapter"

    files = await _list_folder(session, folder_id, api_key, "", 0)
    images = [f for f in files if f[0].lower().endswith(_IMAGE_EXTS)]

    if not images:
        zips = [f for f in files if f[0].lower().endswith(".zip")]
        if len(zips) == 1:
            if zips[0][3] > max_bytes:
                raise SourceError("The file is larger than the allowed limit")
            data = await _drive_download(session, zips[0][1], api_key)
            if not data.startswith(b"PK"):
                raise SourceError("The downloaded file is not a zip")
            return zips[0][0].rsplit("/", 1)[-1], data
        raise SourceError("No images found in this folder")

    if len(images) > _FOLDER_MAX_FILES:
        raise SourceError(f"The folder has {len(images)} images, more than the limit ({_FOLDER_MAX_FILES})")
    if sum(f[3] for f in images) > max_bytes:
        raise SourceError("The folder is larger than the allowed limit")

    sem = asyncio.Semaphore(_FOLDER_PARALLEL)

    async def fetch(item):
        async with sem:
            return item[0], await _drive_download(session, item[1], api_key)

    downloaded = await asyncio.gather(*(fetch(i) for i in images))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for path, data in downloaded:
            zf.writestr(path, data)
    return f"{folder_name}.zip", buf.getvalue()


async def _fetch_gofile_zip_url(session: aiohttp.ClientSession, code: str, token: Optional[str]) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with session.get(f"https://api.gofile.io/contents/{code}", headers=headers) as resp:
        payload = await resp.json(content_type=None)
    if payload.get("status") != "ok":
        raise SourceError(f"gofile rejected the request: {payload.get('status')}")
    children = payload.get("data", {}).get("children", {}) or {}
    for child in children.values():
        if child.get("type") == "file" and child.get("name", "").lower().endswith(".zip"):
            return child["link"]
    raise SourceError("No zip file found in this gofile link")


async def download_zip(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_bytes: int,
    gofile_token: Optional[str] = None,
    drive_api_key: Optional[str] = None,
) -> tuple[str, bytes]:
    """يرجّع (اسم الملف, البيانات). بيقرا بحد أقصى للحجم عشان ما يملاش الذاكرة."""
    kind, target = classify_link(url)
    if kind == "drive_folder":
        return await _download_drive_folder(session, target, drive_api_key, max_bytes)
    if kind == "gofile":
        target = await _fetch_gofile_zip_url(session, target, gofile_token)
    validate_url(target)

    current = target
    confirm_tries = 0
    for _ in range(MAX_REDIRECTS + 4):
        async with session.get(current, allow_redirects=False) as resp:
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location", "")
                current = urljoin(current, location)
                validate_url(current)
                continue
            if resp.status != 200:
                raise SourceError(f"The link returned HTTP {resp.status}")

            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype:
                # صفحة تأكيد Google Drive للملفات الكبيرة
                html = (await resp.text())[:200_000]
                if kind != "drive":
                    raise SourceError("The link returned a web page, not a file")
                if "accounts.google.com" in current or "ServiceLogin" in html:
                    raise SourceError(
                        "The file is not public. On Drive: Share > General access > Anyone with the link"
                    )
                form = _FORM_ACTION_RE.search(html)
                if form and confirm_tries < 2:
                    confirm_tries += 1
                    params = {k: v.replace("&amp;", "&") for k, v in _FORM_INPUT_RE.findall(html)}
                    action = form.group(1).replace("&amp;", "&")
                    current = f"{action}{'&' if '?' in action else '?'}{urlencode(params)}"
                    validate_url(current)
                    continue
                match = _CONFIRM_RE.search(html)
                if match and confirm_tries < 2:
                    confirm_tries += 1
                    current = f"{target}&confirm={match.group(1)}"
                    continue
                raise SourceError(
                    "Drive returned a web page instead of the file. Make sure the file is shared "
                    "as 'Anyone with the link' and is a real .zip"
                )

            name = _filename_from(resp.headers.get("Content-Disposition", ""), current)
            declared = int(resp.headers.get("Content-Length") or 0)
            if declared > max_bytes:
                raise SourceError("The file is larger than the allowed limit")
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.content.iter_chunked(1 << 20):
                total += len(chunk)
                if total > max_bytes:
                    raise SourceError("The file is larger than the allowed limit")
                chunks.append(chunk)
            data = b"".join(chunks)
            if not data.startswith(b"PK"):
                raise SourceError("The downloaded file is not a zip")
            return name, data
    raise SourceError("Too many redirects")


def _filename_from(disposition: str, url: str) -> str:
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        return match.group(1)
    name = urlparse(url).path.rsplit("/", 1)[-1]
    return name or "chapter.zip"
