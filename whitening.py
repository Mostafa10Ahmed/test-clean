"""أدوات الفصل: فك الـ zip وترتيب الصفحات طبيعياً، وتجميع الصفحات المتبيّضة في zip جديد."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import PurePosixPath

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_PAGES = 400
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024  # 1 GB: حماية من zip bomb


def _natural_key(name: str) -> list:
    # 2.png قبل 10.png
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def extract_pages(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """يرجّع قائمة (اسم_الملف, بيانات) للصور الموجودة في الـ zip مرتبة طبيعياً."""
    pages: list[tuple[str, bytes]] = []
    total = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infos = [
            info for info in zf.infolist()
            if not info.is_dir()
            and not info.filename.startswith("__MACOSX")
            and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTS
        ]
        if len(infos) > MAX_PAGES:
            raise ValueError(f"The zip has {len(infos)} pages, more than the limit ({MAX_PAGES})")
        infos.sort(key=lambda info: _natural_key(info.filename))
        for info in infos:
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("The extracted size is larger than the limit")
            pages.append((PurePosixPath(info.filename).name, zf.read(info)))

    if not pages:
        raise ValueError("No images found in the zip")
    return pages


def output_names(names: list[str]) -> list[str]:
    """أسماء الخرج بصيغة PNG، ومن غير تكرار لو فيه نفس الاسم في فولدرات مختلفة."""
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        stem = PurePosixPath(name).stem or "page"
        candidate = f"{stem}.png"
        counter = 2
        while candidate.lower() in seen:
            candidate = f"{stem}_{counter}.png"
            counter += 1
        seen.add(candidate.lower())
        result.append(candidate)
    return result


def build_zip(pages: list[tuple[str, bytes]]) -> bytes:
    # PNG مضغوط أصلاً، فالـ STORED أسرع بكتير ومفيش فايدة من ضغط تاني
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in pages:
            zf.writestr(name, data)
    return buffer.getvalue()
