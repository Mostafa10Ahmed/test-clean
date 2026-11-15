"""التبييض جوه البوت نفسه، من غير أي سيرفر أو مشروع تاني.

الخطوات لكل صفحة:
1. موديل CTD (comic-text-detector، ملف ONNX بيتحمّل أوتوماتيك أول مرة) بيطلع mask لمكان الكلام.
2. كل حتة كلام بنبص على الحلقة اللي حواليها:
   - لو الخلفية سادة (فقاعة بيضا أو لون واحد) بنملاها بنفس اللون، وده تبييض نضيف 100%.
   - لو الكلام فوق رسمة بنعمل inpaint: LaMa لو مكتبة simple-lama-inpainting متسطّبة (بروفايل perfect)،
     وإلا OpenCV (أخف وأسرع بس أقل جودة).
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import aiohttp
import cv2
import numpy as np
from PIL import Image

from whitening import build_zip, extract_pages, output_names

log = logging.getLogger("whiten-bot.local")

MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
MODEL_FILE = "comictextdetector.pt.onnx"
MODEL_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
INPUT_SIZE = 1024  # الموديل متصدّر بمدخل ثابت 1024x1024


def _cpu_limit() -> int:
    """عدد الأنوية المتاحة فعلاً للحاوية (مش عدد أنوية الجهاز الفعلي)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except Exception:
        try:
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                n = min(n, max(1, math.ceil(quota / period)))
        except Exception:
            pass
    return max(1, n)


def _mem_limit_mb() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw.isdigit() and int(raw) < 1 << 50:
                return int(raw) // (1024 * 1024)
        except Exception:
            continue
    return None


def _rss_mb() -> int:
    """الرامات الحالية للعملية (MB)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except Exception:
        return 0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# عدد threads الحساب. الافتراضي = أنوية الحاوية (أهم نقطة للسرعة)، وممكن تتغيّر بـ CV_THREADS
CPU_THREADS = max(1, int(_env_float("CV_THREADS", _cpu_limit())))
cv2.setNumThreads(CPU_THREADS)
# طول الـ tile في الصفحات الطويلة كنسبة من العرض. أكبر = أسرع (tiles أقل) وأقل دقة قليلاً. 1 = الأدق
# المحرك: auto (onnxruntime لو الرامات 2GB+، وإلا cv2.dnn الأخف)، أو ort، أو cv2
DETECTOR = (os.getenv("DETECTOR") or "auto").strip().lower()
# الصفحات الطويلة (ويبتون مدموج) بتتعالج على أشرطة، كل شريط حوالي BAND_PIXELS بكسل، عشان الرامات
# متزيدش مع طول الصفحة. أقل = رامات أقل
BAND_PIXELS = int(_env_float("BAND_PIXELS", 3_000_000))
BAND_MARGIN = 160  # صفوف سياق فوق وتحت كل شريط عشان الكلام اللي على الحدود
TILE_RATIO = min(2.0, max(1.0, _env_float("TILE_RATIO", 1.5)))

# إعدادات البروفايلات للمحرك المدمج
_SETTINGS = {
    "perfect": {"mask_thresh": 0.3, "dilate": 5, "use_lama": True},
    "fast": {"mask_thresh": 0.35, "dilate": 4, "use_lama": False},
}


class LocalCleanError(RuntimeError):
    pass


# ---------- تحميل الموديل ----------
_download_lock = asyncio.Lock()


async def ensure_model(model_dir: str) -> Path:
    path = Path(model_dir) / MODEL_FILE
    if path.exists():
        return path
    async with _download_lock:
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading text detector model (first run only)...")
        tmp = path.with_suffix(".part")
        digest = hashlib.sha256()
        timeout = aiohttp.ClientTimeout(total=1800)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(MODEL_URL) as resp:
                    if resp.status != 200:
                        raise LocalCleanError(f"model download failed: HTTP {resp.status}")
                    with tmp.open("wb") as fh:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            digest.update(chunk)
                            fh.write(chunk)
        except aiohttp.ClientError as exc:
            raise LocalCleanError(f"model download failed: {exc}") from exc
        if digest.hexdigest() != MODEL_SHA256:
            tmp.unlink(missing_ok=True)
            raise LocalCleanError("model download is corrupted (hash mismatch)")
        tmp.replace(path)
        log.info("Model saved to %s", path)
        return path


# ---------- الكشف ----------
_thread_state = threading.local()
_ort_session = None
_ort_failed = False
_ort_lock = threading.Lock()


def _get_ort(model_path: Path):
    """onnxruntime أسرع وأخف في الرامات من cv2.dnn. لو مش متسطّب أو فشل بنرجع لـ cv2.dnn."""
    global _ort_session, _ort_failed
    if _ort_failed:
        return None
    if _ort_session is not None:
        return _ort_session
    with _ort_lock:
        if _ort_session is None and not _ort_failed:
            limit = _mem_limit_mb()
            if DETECTOR == "cv2" or (DETECTOR == "auto" and limit is not None and limit < 2048):
                _ort_failed = True
                log.info("Detector: cv2.dnn (low-memory mode, mem_limit=%sMB, DETECTOR=%s)", limit or "?", DETECTOR)
                return None
            try:
                import onnxruntime as ort  # type: ignore

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = CPU_THREADS
                opts.inter_op_num_threads = 1
                opts.enable_cpu_mem_arena = False  # بيرجّع الرامات بدل ما يحجزها
                _ort_session = ort.InferenceSession(
                    str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                log.info("Detector: onnxruntime (%d threads)", CPU_THREADS)
            except Exception as exc:
                _ort_failed = True
                log.info("onnxruntime not used (%s); falling back to cv2.dnn", exc)
        return _ort_session


def _get_net(model_path: Path) -> "cv2.dnn.Net":
    # cv2.dnn.Net مش thread-safe، فكل thread ليه نسخته
    net = getattr(_thread_state, "net", None)
    if net is None:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        _thread_state.net = net
        _thread_state.outputs = net.getUnconnectedOutLayersNames()
    return net


def _run_detector(canvas: np.ndarray, model_path: Path) -> list:
    global _ort_failed
    sess = _get_ort(model_path)
    if sess is not None:
        try:
            blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            return list(sess.run(None, {sess.get_inputs()[0].name: blob}))
        except Exception as exc:
            _ort_failed = True
            log.warning("onnxruntime failed (%s); falling back to cv2.dnn", exc)
    net = _get_net(model_path)
    net.setInput(cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0))
    return list(net.forward(_thread_state.outputs))


def _detect_tile(rgb: np.ndarray, model_path: Path) -> np.ndarray:
    """احتمالية وجود كلام لكل بكسل (float32 بنفس حجم الـ tile)."""
    h, w = rgb.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized  # padding تحت وعلى اليمين زي letterbox بتاع CTD

    outputs = _run_detector(canvas, model_path)
    # مخرجات CTD: blocks و seg mask (1x1xHxW) و lines map. بناخد الـ seg mask بالشكل
    seg = next(
        (o for o in outputs if o.ndim == 4 and o.shape[1] == 1 and o.shape[2] == INPUT_SIZE),
        None,
    )
    if seg is None:
        raise LocalCleanError(f"unexpected detector outputs: {[o.shape for o in outputs]}")
    prob = seg[0, 0, :nh, :nw].astype(np.float32)
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def _tiles(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def detect_text_mask(rgb: np.ndarray, model_path: Path, thresh: float) -> np.ndarray:
    """mask (0/255) لمكان الكلام. الصفحات الطويلة (ويبتون) بتتقسم مربعات عشان الكلام ما يصغرش."""
    h, w = rgb.shape[:2]
    prob = np.zeros((h, w), dtype=np.float32)
    if h > w * 1.5:
        side = max(w, int(w * TILE_RATIO))
        for y in _tiles(h, side, w // 8):
            part = _detect_tile(rgb[y:y + side], model_path)
            np.maximum(prob[y:y + side], part, out=prob[y:y + side])
    elif w > h * 1.5:
        side = max(h, int(h * TILE_RATIO))
        for x in _tiles(w, side, h // 8):
            part = _detect_tile(rgb[:, x:x + side], model_path)
            np.maximum(prob[:, x:x + side], part, out=prob[:, x:x + side])
    else:
        prob = _detect_tile(rgb, model_path)
    return (prob > thresh).astype(np.uint8) * 255


# ---------- المسح ----------
_lama = None
_lama_lock = threading.Lock()
_lama_checked = False


def _get_lama():
    global _lama, _lama_checked
    with _lama_lock:
        if not _lama_checked:
            _lama_checked = True
            try:
                from simple_lama_inpainting import SimpleLama  # type: ignore

                _lama = SimpleLama()
                log.info("LaMa inpainting enabled")
            except Exception as exc:  # المكتبة اختيارية
                log.info("LaMa not available (%s); using OpenCV inpainting", exc)
        return _lama


def _inpaint_region(rgb: np.ndarray, mask: np.ndarray, use_lama: bool) -> np.ndarray:
    lama = _get_lama() if use_lama else None
    if lama is not None:
        with _lama_lock:
            out = lama(Image.fromarray(rgb), Image.fromarray(mask))
        out = np.asarray(out.convert("RGB"))[: rgb.shape[0], : rgb.shape[1]]
        result = rgb.copy()
        result[mask > 0] = out[mask > 0]
        return result
    return cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)


def clean_image(rgb: np.ndarray, model_path: Path, profile: str) -> np.ndarray:
    settings = _SETTINGS[profile]
    raw = detect_text_mask(rgb, model_path, settings["mask_thresh"])
    if not raw.any():
        return rgb

    k = settings["dilate"]
    mask = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    # نجمّع الحروف القريبة في بلوك واحد عشان نحكم على الخلفية بتاعة الجملة كلها
    groups = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(groups, connectivity=8)

    out = rgb.copy()
    h, w = rgb.shape[:2]
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for label in range(1, count):
        x, y, bw, bh, _ = stats[label]
        pad = 24
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        region_mask = np.where((labels[y0:y1, x0:x1] == label) & (mask[y0:y1, x0:x1] > 0), 255, 0).astype(np.uint8)
        if not region_mask.any():
            continue
        crop = out[y0:y1, x0:x1]

        # الحلقة اللي حوالين الكلام تقول لنا الخلفية سادة ولا رسمة
        ring = cv2.dilate(region_mask, ring_kernel) > 0
        ring &= region_mask == 0
        ring_pixels = crop[ring]
        if len(ring_pixels) >= 20:
            spread = np.percentile(ring_pixels, 90, axis=0) - np.percentile(ring_pixels, 10, axis=0)
            if spread.max() <= 30:
                crop[region_mask > 0] = np.median(ring_pixels, axis=0).astype(np.uint8)
                continue

        out[y0:y1, x0:x1] = _inpaint_region(crop, region_mask, settings["use_lama"])
    return out


def clean_image_banded(rgb: np.ndarray, model_path: Path, profile: str) -> np.ndarray:
    """بيعالج الصفحة الطويلة على أشرطة. بيعدّل المصفوفة نفسها (لازم تكون قابلة للكتابة)."""
    h, w = rgb.shape[:2]
    band = max(512, BAND_PIXELS // max(1, w))
    if h <= band + 2 * BAND_MARGIN:
        return clean_image(rgb, model_path, profile)

    prev_tail = None
    y = 0
    while y < h:
        y1 = min(h, y + band)
        # لو الشريط الأخير قصير جداً نضمّه للي قبله
        if h - y1 < BAND_MARGIN * 2:
            y1 = h
        parts = []
        if prev_tail is not None:
            parts.append(prev_tail)
        top_len = 0 if prev_tail is None else len(prev_tail)
        parts.append(rgb[y:y1])
        if y1 < h:
            parts.append(rgb[y1:min(h, y1 + BAND_MARGIN)])
        crop = np.concatenate(parts) if len(parts) > 1 else rgb[y:y1]
        next_tail = rgb[max(y, y1 - BAND_MARGIN):y1].copy() if y1 < h else None  # الأصل قبل ما نعدّله

        cleaned = clean_image(crop, model_path, profile)
        rgb[y:y1] = cleaned[top_len:top_len + (y1 - y)]
        del crop, cleaned, parts
        prev_tail = next_tail
        y = y1
    return rgb


def clean_page_bytes(data: bytes, model_path: Path, profile: str) -> bytes:
    # cv2.imdecode بيفك الصورة في نسخة واحدة بس (PIL بيعمل كذا نسخة وده بيفرق في الصفحات الطويلة)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        with Image.open(io.BytesIO(data)) as im:
            bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=bgr)  # بدون نسخة جديدة
    cleaned = clean_image_banded(rgb, model_path, profile)
    out = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR, dst=cleaned)
    ok, png = cv2.imencode(".png", out)
    if not ok:
        raise LocalCleanError("failed to encode PNG")
    return png.tobytes()


# ---------- الفصل كامل ----------
async def clean_chapter_local(
    zip_bytes: bytes,
    *,
    profile: str,
    model_dir: str,
    concurrency: int = 2,
) -> bytes:
    if profile not in _SETTINGS:
        raise LocalCleanError(f"Unknown profile: {profile}")
    model_path = await ensure_model(model_dir)
    pages = extract_pages(zip_bytes)
    names = output_names([name for name, _ in pages])

    loop = asyncio.get_running_loop()
    total = len(pages)
    started = time.monotonic()
    done = 0
    log.info(
        "Whitening %d pages (profile=%s, page_concurrency=%d, cpu_threads=%d, mem_limit=%sMB)",
        total, profile, concurrency, CPU_THREADS, _mem_limit_mb() or "?",
    )
    results: list = [b""] * total
    gate = asyncio.Semaphore(max(1, concurrency))
    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="whiten") as pool:
        async def run(index: int) -> None:
            nonlocal done
            async with gate:
                name, data = pages[index]
                pages[index] = (name, b"")  # نحرّر الصورة الأصلية من الذاكرة أول ما تتعالج
                try:
                    results[index] = await loop.run_in_executor(pool, clean_page_bytes, data, model_path, profile)
                except Exception as exc:
                    raise LocalCleanError(f"Failed to whiten page `{name}`: {exc}") from exc
                del data
            done += 1
            log.info(
                "Page %d/%d done (%s) - %.0fs elapsed, rss=%dMB",
                done, total, name, time.monotonic() - started, _rss_mb(),
            )

        await asyncio.gather(*(run(i) for i in range(total)))
    return build_zip(list(zip(names, results)))


__all__ = ["LocalCleanError", "clean_chapter_local", "clean_image", "ensure_model"]
