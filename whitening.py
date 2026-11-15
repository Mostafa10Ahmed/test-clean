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


def list_zip_page_infos(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """يرجّع قائمة ZipInfo للصور داخل الأرشيف مرتبة طبيعياً بدون تحميل الصور كلها في الرامات."""
    infos = [
        info for info in zf.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX")
        and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTS
    ]
    if len(infos) > MAX_PAGES:
        raise ValueError(f"The zip has {len(infos)} pages, more than the limit ({MAX_PAGES})")
    infos.sort(key=lambda info: _natural_key(info.filename))
    total = sum(info.file_size for info in infos)
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("The extracted size is larger than the limit")
    if not infos:
        raise ValueError("No images found in the zip")
    return infos


def extract_pages(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """يرجّع قائمة (اسم_الملف, بيانات) للصور الموجودة في الـ zip مرتبة طبيعياً."""
    pages: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infos = list_zip_page_infos(zf)
        for info in infos:
            pages.append((PurePosixPath(info.filename).name, zf.read(info)))
    return pages


def output_names(names: list[str], ext: str = ".png") -> list[str]:
    """أسماء الخرج بصيغة موحدة، ومن غير تكرار لو فيه نفس الاسم في فولدرات مختلفة."""
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        stem = PurePosixPath(name).stem or "page"
        candidate = f"{stem}{ext}"
        counter = 2
        while candidate.lower() in seen:
            candidate = f"{stem}_{counter}{ext}"
            counter += 1
        seen.add(candidate.lower())
        result.append(candidate)
    return result


def build_zip(pages: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in pages:
            zf.writestr(name, data)
    return buffer.getvalue()
