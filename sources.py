"""تحميل الـ zip من لينك: Google Drive، gofile، أو أي رابط مباشر لملف zip."""
from __future__ import annotations

import ipaddress
import re
import socket
from typing import Optional
from urllib.parse import ParseResult, urljoin, urlparse

import aiohttp

MAX_REDIRECTS = 5
_DRIVE_FILE_RE = re.compile(r"/file/d/([\w-]+)")
_DRIVE_OPEN_RE = re.compile(r"[?&]id=([\w-]+)")
_GOFILE_RE = re.compile(r"gofile\.io/d/(\w+)")
_CONFIRM_RE = re.compile(r'confirm=([0-9A-Za-z_-]+)')


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
        if "/folders/" in url:
            raise SourceError("This is a Drive folder link. Share the chapter as a single zip file instead")
        match = _DRIVE_FILE_RE.search(url) or _DRIVE_OPEN_RE.search(url)
        if not match:
            raise SourceError("Could not find the file ID in the Google Drive link. Use the file's Share > Copy link")
        return "drive", f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    match = _GOFILE_RE.search(url)
    if match:
        return "gofile", match.group(1)
    return "direct", url


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
) -> tuple[str, bytes]:
    """يرجّع (اسم الملف, البيانات). بيقرا بحد أقصى للحجم عشان ما يملاش الذاكرة."""
    kind, target = classify_link(url)
    if kind == "gofile":
        target = await _fetch_gofile_zip_url(session, target, gofile_token)
    validate_url(target)

    current = target
    for _ in range(MAX_REDIRECTS + 1):
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
                match = _CONFIRM_RE.search(html)
                if kind != "drive" or not match:
                    raise SourceError("The link returned a web page, not a file")
                current = f"{target}&confirm={match.group(1)}"
                continue

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
