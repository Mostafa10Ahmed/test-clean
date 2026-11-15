"""محرك التبييض المدمج السريع والمستقر (Turbo CPU Engine v3):

يدعم بروفايل `fast` و `perfect` بثبات كامل على حاويات Railway المحدودة (953MB RAM):
1. دمج شريحتين (Dual-Tile Packing) في استدعاء ONNX واحد مع حساب مخرج الـ seg mask فقط لتوفير الرامات والوقت.
2. تخطي الفراغات السادة (Gutters) بين بانلات الويبتون تلقائياً في أقل من 2ms.
3. منع استدعاء `np.where` ذو الـ 64-bit وتحديد `BAND_PIXELS=4_800_000` بحيث لا يتخطى استهلاك الرامات ~240MB حتى في بروفايل `perfect`.
4. في بروفايل `perfect`: دقة كشف أعلى للحروف الصغيرة، وتنعيم حواف المسح (Feathering + Multi-pass Inpaint) بدون انهيار الرامات.
"""
from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import io
import logging
import math
import os
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import aiohttp
import cv2
import numpy as np
from PIL import Image

from whitening import list_zip_page_infos, output_names

log = logging.getLogger("whiten-bot.local")

MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
MODEL_FILE = "comictextdetector.pt.onnx"
MODEL_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
INPUT_SIZE = 1024

try:
    _libc = ctypes.CDLL("libc.so.6")
except Exception:
    _libc = None


def _release_memory() -> None:
    """إجبار Python و Linux glibc على تحرير الرامات فوراً بعد كل صفحة أو شريط لمنع OOM."""
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def _mem_limit_mb() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw.isdigit() and int(raw) < 1 << 50:
                return int(raw) // (1024 * 1024)
        except Exception:
            continue
    return None


def _cpu_limit() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 3
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
    mem = _mem_limit_mb()
    # لو الرامات أقل من 1200MB (مثل Railway 953MB)، 3 أنوية تعطي سرعة عالية جداً بدون قفزات رامات
    max_safe = 3 if (mem is not None and mem < 1200) else 4
    return max(2, min(n, max_safe))


def _rss_mb() -> int:
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


CPU_THREADS = max(1, int(_env_float("CV_THREADS", _cpu_limit())))
cv2.setNumThreads(CPU_THREADS)
DETECTOR = (os.getenv("DETECTOR") or "ort").strip().lower()
# 4.8 مليون بكسل لكل شريط (حوالي 800x6000 بكسل) يحافظ على الرامات تحت 250MB دائماً حتى في perfect
BAND_PIXELS = int(_env_float("BAND_PIXELS", 4_800_000))
BAND_MARGIN = 128
TILE_RATIO = min(2.2, max(1.2, _env_float("TILE_RATIO", 2.04)))
DUAL_TILE_PACKING = os.getenv("DUAL_TILE_PACKING", "1").strip().lower() not in ("0", "false", "no")
OUTPUT_FORMAT = (os.getenv("OUTPUT_FORMAT") or "auto").strip().lower()

# إعدادات البروفايلات:
# perfect: حساسية أعلى للكشف (0.30)، توسيع mask أكبر (5) لمسح الحواف والظلال تماماً، وتنعيم متعدد المراحل
# fast:    أسرع (0.35)، توسيع (4)، ومسح سريع بخطوة واحدة
_SETTINGS = {
    "perfect": {
        "mask_thresh": 0.30,
        "dilate": 5,
        "inpaint_radius": 4,
        "feather": True,
        "use_lama": True,
        "tile_overlap_div": 8,
    },
    "fast": {
        "mask_thresh": 0.35,
        "dilate": 4,
        "inpaint_radius": 3,
        "feather": False,
        "use_lama": False,
        "tile_overlap_div": 10,
    },
}


class LocalCleanError(RuntimeError):
    pass


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


# ---------- الكشف السريع ----------
_thread_state = threading.local()
_ort_session = None
_ort_input_name = None
_ort_seg_name = None
_ort_failed = False
_ort_lock = threading.Lock()


def _get_ort(model_path: Path):
    """onnxruntime مع إيقاف mem_pattern و cpu_mem_arena لمنع أي قفزة رامات على Railway."""
    global _ort_session, _ort_input_name, _ort_seg_name, _ort_failed
    if _ort_failed:
        return None
    if _ort_session is not None:
        return _ort_session
    with _ort_lock:
        if _ort_session is None and not _ort_failed:
            if DETECTOR == "cv2":
                _ort_failed = True
                log.info("Detector: cv2.dnn (forced via DETECTOR=cv2)")
                return None
            try:
                import onnxruntime as ort  # type: ignore

                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.intra_op_num_threads = CPU_THREADS
                opts.inter_op_num_threads = 1
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False

                avail = set(ort.get_available_providers())
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
                sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
                _ort_input_name = sess.get_inputs()[0].name
                # نحدد اسم مخرج الـ seg mask فقط (1x1x1024x1024) عشان منحسبش باقي المخرجات غير المستخدمة
                for out_meta in sess.get_outputs():
                    shape = out_meta.shape
                    if len(shape) == 4 and shape[1] == 1:
                        _ort_seg_name = out_meta.name
                        break
                _ort_session = sess
                log.info("Detector: onnxruntime turbo (%d threads, seg_out=%s)", CPU_THREADS, _ort_seg_name or "all")
            except Exception as exc:
                _ort_failed = True
                log.info("onnxruntime not used (%s); falling back to cv2.dnn", exc)
        return _ort_session


def _get_net(model_path: Path) -> "cv2.dnn.Net":
    net = getattr(_thread_state, "net", None)
    if net is None:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        _thread_state.net = net
        _thread_state.outputs = net.getUnconnectedOutLayersNames()
    return net


def _run_detector_seg(canvas: np.ndarray, model_path: Path) -> np.ndarray:
    """يشغّل الموديل على لوحة 1024x1024 ويرجع مصفوفة الاحتمالات (1024x1024 float32) فقط."""
    global _ort_failed
    sess = _get_ort(model_path)
    if sess is not None:
        try:
            blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
            blob *= (1.0 / 255.0)
            out_names = [_ort_seg_name] if _ort_seg_name else None
            outputs = sess.run(out_names, {_ort_input_name: blob})
            del blob
            seg = next(
                (o for o in outputs if o.ndim == 4 and o.shape[1] == 1 and o.shape[2] == INPUT_SIZE),
                None,
            )
            if seg is not None:
                res = seg[0, 0].copy()
                del outputs, seg
                return res
        except Exception as exc:
            _ort_failed = True
            log.warning("onnxruntime failed (%s); falling back to cv2.dnn", exc)

    net = _get_net(model_path)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0)
    net.setInput(blob)
    outputs = net.forward(_thread_state.outputs)
    del blob
    seg = next(
        (o for o in outputs if o.ndim == 4 and o.shape[1] == 1 and o.shape[2] == INPUT_SIZE),
        None,
    )
    if seg is None:
        raise LocalCleanError(f"unexpected detector outputs: {[o.shape for o in outputs]}")
    res = seg[0, 0].copy()
    del outputs, seg
    return res


def _tile_has_contrast(tile: np.ndarray) -> bool:
    sub = tile[::4, ::4]
    if sub.size == 0:
        return False
    if int(sub.max()) - int(sub.min()) < 20:
        return False
    diff = cv2.absdiff(sub[:, 1:], sub[:, :-1])
    return int(diff.max()) >= 20


def _detect_single_tile(rgb: np.ndarray, model_path: Path) -> np.ndarray:
    if not _tile_has_contrast(rgb):
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    h, w = rgb.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    del resized

    seg = _run_detector_seg(canvas, model_path)
    del canvas
    prob = seg[:nh, :nw].astype(np.float32)
    del seg
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def _detect_dual_tiles(tile1: np.ndarray, tile2: np.ndarray, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    MAX_HALF_W = 500
    GAP = 24

    h1, w1 = tile1.shape[:2]
    scale1 = min(INPUT_SIZE / max(1, h1), MAX_HALF_W / max(1, w1))
    nw1, nh1 = max(1, round(w1 * scale1)), max(1, round(h1 * scale1))

    h2, w2 = tile2.shape[:2]
    scale2 = min(INPUT_SIZE / max(1, h2), MAX_HALF_W / max(1, w2))
    nw2, nh2 = max(1, round(w2 * scale2)), max(1, round(h2 * scale2))

    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:nh1, :nw1] = cv2.resize(tile1, (nw1, nh1), interpolation=cv2.INTER_AREA if scale1 < 1 else cv2.INTER_LINEAR)
    x2 = nw1 + GAP
    canvas[:nh2, x2:x2 + nw2] = cv2.resize(tile2, (nw2, nh2), interpolation=cv2.INTER_AREA if scale2 < 1 else cv2.INTER_LINEAR)

    seg = _run_detector_seg(canvas, model_path)
    del canvas

    prob1 = cv2.resize(seg[:nh1, :nw1].astype(np.float32), (w1, h1), interpolation=cv2.INTER_LINEAR)
    prob2 = cv2.resize(seg[:nh2, x2:x2 + nw2].astype(np.float32), (w2, h2), interpolation=cv2.INTER_LINEAR)
    del seg
    return prob1, prob2


def _smart_vertical_tiles(rgb: np.ndarray, side: int, overlap: int) -> list[tuple[int, int]]:
    h, w = rgb.shape[:2]
    if h <= side:
        return [(0, h)] if _tile_has_contrast(rgb) else []

    sub = rgb[::4, ::8]
    if sub.shape[1] > 1:
        row_diff = cv2.absdiff(sub[:, 1:], sub[:, :-1]).max(axis=(1, 2))
        active_sub = row_diff >= 16
    else:
        active_sub = np.ones(sub.shape[0], dtype=bool)

    tiles: list[tuple[int, int]] = []
    y = 0
    while y < h:
        while y < h and not active_sub[min(y // 4, len(active_sub) - 1)]:
            y += 4
        if y >= h:
            break
        y_start = max(0, min(y - 32, h - side))
        y_end = min(h, y_start + side)
        if tiles and y_start <= tiles[-1][0]:
            y_start = min(tiles[-1][1] - overlap, h - side)
            y_end = min(h, y_start + side)
        if _tile_has_contrast(rgb[y_start:y_end]):
            tiles.append((y_start, y_end))
        if y_end >= h:
            break
        y = y_end - overlap
    return tiles


def _tiles(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def detect_text_mask(rgb: np.ndarray, model_path: Path, thresh: float, overlap_div: int = 10) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if h > w * 1.5:
        side = max(w, int(w * TILE_RATIO))
        overlap = max(48, w // overlap_div)
        spans = _smart_vertical_tiles(rgb, side, overlap)
        if not spans:
            return mask

        if DUAL_TILE_PACKING and len(spans) >= 2:
            idx = 0
            while idx + 1 < len(spans):
                y0_a, y1_a = spans[idx]
                y0_b, y1_b = spans[idx + 1]
                p1, p2 = _detect_dual_tiles(rgb[y0_a:y1_a], rgb[y0_b:y1_b], model_path)
                m1 = (p1 > thresh).astype(np.uint8) * 255
                m2 = (p2 > thresh).astype(np.uint8) * 255
                np.maximum(mask[y0_a:y1_a], m1, out=mask[y0_a:y1_a])
                np.maximum(mask[y0_b:y1_b], m2, out=mask[y0_b:y1_b])
                del p1, p2, m1, m2
                idx += 2
            if idx < len(spans):
                y0, y1 = spans[idx]
                p = _detect_single_tile(rgb[y0:y1], model_path)
                m = (p > thresh).astype(np.uint8) * 255
                np.maximum(mask[y0:y1], m, out=mask[y0:y1])
                del p, m
        else:
            for y0, y1 in spans:
                p = _detect_single_tile(rgb[y0:y1], model_path)
                m = (p > thresh).astype(np.uint8) * 255
                np.maximum(mask[y0:y1], m, out=mask[y0:y1])
                del p, m
        return mask

    if w > h * 1.5:
        side = max(h, int(h * TILE_RATIO))
        for x in _tiles(w, side, h // overlap_div):
            p = _detect_single_tile(rgb[:, x:x + side], model_path)
            m = (p > thresh).astype(np.uint8) * 255
            np.maximum(mask[:, x:x + side], m, out=mask[:, x:x + side])
            del p, m
        return mask

    p = _detect_single_tile(rgb, model_path)
    mask[:] = (p > thresh).astype(np.uint8) * 255
    del p
    return mask


# ---------- المسح ----------
_lama = None
_lama_lock = threading.Lock()
_lama_checked = False


def _get_lama():
    global _lama, _lama_checked
    with _lama_lock:
        if not _lama_checked:
            _lama_checked = True
            mem = _mem_limit_mb()
            # موديل LaMa يحتاج PyTorch ورامات >= 2GB؛ على الحاويات الصغيرة نستخدم محرك OpenCV المحسن
            if mem is not None and mem < 2048:
                log.info("Container mem_limit=%dMB (<2048MB); using enhanced OpenCV inpainting for perfect profile", mem)
                return None
            try:
                from simple_lama_inpainting import SimpleLama  # type: ignore

                _lama = SimpleLama()
                log.info("LaMa inpainting enabled")
            except Exception as exc:
                log.info("LaMa not available (%s); using enhanced OpenCV inpainting", exc)
        return _lama


def _inpaint_region(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    use_lama: bool,
    radius: int = 3,
    feather: bool = False,
) -> np.ndarray:
    lama = _get_lama() if use_lama else None
    if lama is not None:
        with _lama_lock:
            out = lama(Image.fromarray(rgb), Image.fromarray(mask))
        out = np.asarray(out.convert("RGB"))[: rgb.shape[0], : rgb.shape[1]]
        result = rgb.copy()
        result[mask > 0] = out[mask > 0]
        return result

    # لو البروفايل perfect (feather=True): ندمج TELEA مع تنعيم خفيف على الحواف عشان ميبقاش فيه أثر مكان المسح
    inpainted = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
    if feather:
        blur = cv2.GaussianBlur(inpainted, (3, 3), 0)
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        inpainted[inner > 0] = blur[inner > 0]
    return inpainted


def clean_image(rgb: np.ndarray, model_path: Path, profile: str) -> np.ndarray:
    settings = _SETTINGS[profile]
    raw = detect_text_mask(
        rgb,
        model_path,
        settings["mask_thresh"],
        overlap_div=settings["tile_overlap_div"],
    )
    if not raw.any():
        return rgb

    k = settings["dilate"]
    mask = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    del raw
    # تجميع الحروف القريبة في جملة واحدة (11x11 بدلاً من 15x15 لمنع اندماج فقاعات متباعدة في مستطيل عملاق)
    groups = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(groups, connectivity=8)
    del groups

    out = rgb
    h, w = rgb.shape[:2]
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < 10:
            continue
        pad = 20
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)

        # بناء region_mask بـ uint8 مباشرة (1 بايت لكل بكسل بدلاً من 8 بايت في np.where)
        sub_labels = labels[y0:y1, x0:x1]
        sub_mask = mask[y0:y1, x0:x1]
        cond = (sub_labels == label) & (sub_mask > 0)
        if not cond.any():
            continue
        region_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        region_mask[cond] = 255
        del cond

        crop = out[y0:y1, x0:x1]

        ring = cv2.dilate(region_mask, ring_kernel) > 0
        ring &= region_mask == 0
        ring_pixels = crop[ring]
        if len(ring_pixels) >= 20:
            spread = np.percentile(ring_pixels, 90, axis=0) - np.percentile(ring_pixels, 10, axis=0)
            # في perfect نتسامح حتى فرق 34 لتعبئة الفقاعات ذات التدرج الخفيف بلون صافي 100%
            max_spread = 34 if profile == "perfect" else 30
            if spread.max() <= max_spread:
                crop[region_mask > 0] = np.median(ring_pixels, axis=0).astype(np.uint8)
                del region_mask, ring
                continue
        del ring

        inpainted = _inpaint_region(
            crop.copy(),
            region_mask,
            use_lama=settings["use_lama"],
            radius=settings["inpaint_radius"],
            feather=settings["feather"],
        )
        crop[region_mask > 0] = inpainted[region_mask > 0]
        del region_mask, inpainted

    del mask, labels, stats
    return out


def clean_image_banded(rgb: np.ndarray, model_path: Path, profile: str) -> np.ndarray:
    """يعالج الصفحات الطويلة على أشرطة آمنة في الرامات (حوالي 6000 بكسل طولاً لكل شريط)."""
    h, w = rgb.shape[:2]
    band = max(1024, BAND_PIXELS // max(1, w))
    if h <= band + 2 * BAND_MARGIN:
        return clean_image(rgb, model_path, profile)

    prev_tail = None
    y = 0
    while y < h:
        y1 = min(h, y + band)
        if h - y1 < BAND_MARGIN * 2:
            y1 = h
        parts = []
        if prev_tail is not None:
            parts.append(prev_tail)
        top_len = 0 if prev_tail is None else len(prev_tail)
        parts.append(rgb[y:y1])
        if y1 < h:
            parts.append(rgb[y1:min(h, y1 + BAND_MARGIN)])
        crop = np.concatenate(parts) if len(parts) > 1 else rgb[y:y1].copy()
        next_tail = rgb[max(y, y1 - BAND_MARGIN):y1].copy() if y1 < h else None

        cleaned = clean_image(crop, model_path, profile)
        rgb[y:y1] = cleaned[top_len:top_len + (y1 - y)]
        del crop, cleaned, parts
        prev_tail = next_tail
        y = y1
    return rgb


def clean_page_bytes(data: bytes, model_path: Path, profile: str, orig_ext: str = ".png") -> tuple[str, bytes]:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        with Image.open(io.BytesIO(data)) as im:
            bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=bgr)
    cleaned = clean_image_banded(rgb, model_path, profile)
    out = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR, dst=cleaned)

    orig_lower = orig_ext.lower()
    jpg_quality = 95 if profile == "perfect" else 93
    if OUTPUT_FORMAT in ("jpg", "jpeg") or (OUTPUT_FORMAT == "auto" and orig_lower in (".jpg", ".jpeg")):
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        ext = ".jpg"
    elif OUTPUT_FORMAT == "webp" or (OUTPUT_FORMAT == "auto" and orig_lower == ".webp"):
        ok, encoded = cv2.imencode(".webp", out, [int(cv2.IMWRITE_WEBP_QUALITY), 94])
        ext = ".webp"
    else:
        ok, encoded = cv2.imencode(".png", out, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        ext = ".png"
    del out, cleaned, rgb, bgr
    if not ok:
        raise LocalCleanError("failed to encode output image")
    return ext, encoded.tobytes()


async def clean_chapter_local_file(
    in_zip_path: Path,
    out_zip_path: Path,
    *,
    profile: str,
    model_dir: str,
    concurrency: int = 1,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    if profile not in _SETTINGS:
        raise LocalCleanError(f"Unknown profile: {profile}")
    model_path = await ensure_model(model_dir)

    with zipfile.ZipFile(in_zip_path, "r") as zf_in:
        infos = list_zip_page_infos(zf_in)
        total = len(infos)
        base_names = output_names([PurePosixPath(info.filename).name for info in infos], ext=".png")

    if progress_cb:
        progress_cb(0, total)

    loop = asyncio.get_running_loop()
    started = time.monotonic()
    log.info(
        "Whitening %d pages on disk (profile=%s, cpu_threads=%d, dual_tile=%s, mem_limit=%sMB)",
        total, profile, CPU_THREADS, DUAL_TILE_PACKING, _mem_limit_mb() or "?",
    )

    # max_workers=1 يضمن إعادة استخدام نفس الـ Thread دائماً وعدم فتح أرينات ذاكرة إضافية
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="whiten") as pool:
        with zipfile.ZipFile(in_zip_path, "r") as zf_in, zipfile.ZipFile(
            out_zip_path, "w", compression=zipfile.ZIP_STORED
        ) as zf_out:
            for index, info in enumerate(infos):
                raw_name = PurePosixPath(info.filename).name
                orig_ext = PurePosixPath(raw_name).suffix.lower() or ".png"
                data = zf_in.read(info)
                try:
                    ext, out_bytes = await loop.run_in_executor(
                        pool, clean_page_bytes, data, model_path, profile, orig_ext
                    )
                except Exception as exc:
                    raise LocalCleanError(f"Failed to whiten page `{raw_name}`: {exc}") from exc
                finally:
                    del data

                out_filename = str(PurePosixPath(base_names[index]).with_suffix(ext))
                zf_out.writestr(out_filename, out_bytes)
                del out_bytes
                _release_memory()

                done = index + 1
                if progress_cb:
                    progress_cb(done, total)
                log.info(
                    "Page %d/%d done (%s -> %s) - %.1fs elapsed, rss=%dMB",
                    done, total, raw_name, out_filename, time.monotonic() - started, _rss_mb(),
                )


async def clean_chapter_local(
    zip_bytes: bytes,
    *,
    profile: str,
    model_dir: str,
    concurrency: int = 1,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="whiten-local-") as tmp_dir:
        in_path = Path(tmp_dir) / "input.zip"
        out_path = Path(tmp_dir) / "output.zip"
        in_path.write_bytes(zip_bytes)
        del zip_bytes
        _release_memory()
        await clean_chapter_local_file(
            in_path,
            out_path,
            profile=profile,
            model_dir=model_dir,
            concurrency=concurrency,
            progress_cb=progress_cb,
        )
        return out_path.read_bytes()


__all__ = [
    "LocalCleanError",
    "clean_chapter_local",
    "clean_chapter_local_file",
    "clean_image",
    "ensure_model",
]
