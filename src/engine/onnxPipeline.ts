import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import type * as OrtTypes from 'onnxruntime-node';

let ortPromise: Promise<typeof import('onnxruntime-node')> | null = null;
async function getOrt(): Promise<typeof import('onnxruntime-node')> {
  if (!ortPromise) {
    ortPromise = import('onnxruntime-node');
  }
  return ortPromise;
}
import {
  buildRegionPixelMask,
  cleanAndCompositeRegionInPlace,
  evaluateCleaningQuality,
  isScanlationWatermark,
  rgbToLuma,
  selectCleaningEngine,
  tryCleanFlatOrGradientRing,
} from './cleanerCore';
import { measureSurroundRing, verifyBubbleEnclosure } from './detector';
import { computeCtdBoxMetrics, evaluateNonTextGuard } from './nonTextGuard';
import {
  BoundingBox,
  CleaningSettings,
  DetectedRegion,
  QualityAssessment,
  RegionCategory,
} from './types';

const MODELS_DIR = process.env.MODEL_DIR
  ? path.resolve(process.env.MODEL_DIR)
  : path.resolve(process.cwd(), 'models');
const LOW_RAM_MODE = process.env.LOW_RAM_MODE !== '0' && process.env.LOW_RAM_MODE !== 'false';
const ONNX_THREADS = Math.max(1, Number(process.env.ONNX_THREADS) || (LOW_RAM_MODE ? 1 : 4));
const ENABLE_CTD = process.env.ENABLE_CTD !== '0' && process.env.ENABLE_CTD !== 'false';
const ENABLE_LAMA = process.env.ENABLE_LAMA !== '0' && process.env.ENABLE_LAMA !== 'false';
const RTDETR_MODEL_PATH = path.join(MODELS_DIR, 'detector-v4-s_int8.onnx');
const CTD_MODEL_PATH = path.join(MODELS_DIR, 'comictextdetector.pt.onnx');
const LAMA_MODEL_PATH = path.join(MODELS_DIR, 'lama-manga.onnx');

const RTDETR_MODEL_URL =
  'https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector-v4-s_int8.onnx';
const CTD_MODEL_URL =
  'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx';
const LAMA_MODEL_URL =
  'https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx';

let rtdetrSessionPromise: Promise<OrtTypes.InferenceSession | null> | null = null;
let ctdSessionPromise: Promise<OrtTypes.InferenceSession | null> | null = null;
let lamaSessionPromise: Promise<OrtTypes.InferenceSession | null> | null = null;

// Cache the most recently computed full-page CTD segmentation mask & YOLO text-block detections
let cachedPageCtd: {
  width: number;
  height: number;
  fingerprint: number;
  segMask: Uint8Array;
  blkBoxes: { bbox: BoundingBox; score: number }[];
} | null = null;

function computeRgbaFingerprint(rgba: Uint8Array, width: number, height: number): number {
  let hash = ((width * 73856093) ^ (height * 19349663)) >>> 0;
  const step = Math.max(4, Math.floor(rgba.length / 256));
  for (let i = 0; i < rgba.length; i += step) {
    hash = ((hash * 16777619) ^ rgba[i]) >>> 0;
  }
  return hash;
}

async function ensureModelDownloaded(filePath: string, url: string): Promise<boolean> {
  if (fs.existsSync(filePath) && fs.statSync(filePath).size > 1024 * 1024) {
    return true;
  }
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    const tmpPath = `${filePath}.part_${Date.now()}`;
    console.log(`[ONNX] Downloading ${path.basename(filePath)}...`);
    await new Promise<void>((resolve, reject) => {
      const child = spawn('curl', ['-L', '-sS', '--fail', '-o', tmpPath, url], { stdio: 'ignore' });
      child.on('error', reject);
      child.on('close', (code) =>
        code === 0 ? resolve() : reject(new Error(`curl exited with code ${code}`))
      );
    });
    console.log(`[ONNX] Downloaded ${path.basename(filePath)} (${(fs.statSync(tmpPath).size / 1048576).toFixed(1)} MB)`);
    if (fs.existsSync(tmpPath) && fs.statSync(tmpPath).size > 1024 * 1024) {
      fs.renameSync(tmpPath, filePath);
      return true;
    }
    if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    return false;
  } catch (err) {
    console.error(`Failed to auto-download ONNX model from ${url}:`, err);
    return false;
  }
}

export async function getRtdetrSession(): Promise<OrtTypes.InferenceSession | null> {
  if (!rtdetrSessionPromise) {
    rtdetrSessionPromise = (async () => {
      const ok = await ensureModelDownloaded(RTDETR_MODEL_PATH, RTDETR_MODEL_URL);
      if (!ok) return null;
      const ort = await getOrt();
      return await ort.InferenceSession.create(RTDETR_MODEL_PATH, {
        intraOpNumThreads: ONNX_THREADS,
        interOpNumThreads: 1,
        enableCpuMemArena: !LOW_RAM_MODE,
        enableMemPattern: !LOW_RAM_MODE,
        graphOptimizationLevel: LOW_RAM_MODE ? 'basic' : 'all',
      });
    })().catch((err) => {
      console.error('Failed to load detector-v4-s_int8.onnx:', err);
      rtdetrSessionPromise = null;
      return null;
    });
  }
  return rtdetrSessionPromise;
}

export async function getCtdSession(): Promise<OrtTypes.InferenceSession | null> {
  if (!ctdSessionPromise) {
    ctdSessionPromise = (async () => {
      const ok = await ensureModelDownloaded(CTD_MODEL_PATH, CTD_MODEL_URL);
      if (!ok) return null;
      const ort = await getOrt();
      return await ort.InferenceSession.create(CTD_MODEL_PATH, {
        intraOpNumThreads: ONNX_THREADS,
        interOpNumThreads: 1,
        enableCpuMemArena: !LOW_RAM_MODE,
        enableMemPattern: !LOW_RAM_MODE,
        graphOptimizationLevel: LOW_RAM_MODE ? 'basic' : 'all',
      });
    })().catch((err) => {
      console.error('Failed to load comictextdetector.pt.onnx:', err);
      ctdSessionPromise = null;
      return null;
    });
  }
  return ctdSessionPromise;
}

async function releaseSessionPromise(
  promise: Promise<OrtTypes.InferenceSession | null> | null
): Promise<void> {
  if (!promise) return;
  try {
    const sess = await promise;
    if (sess && typeof (sess as unknown as { release?: () => Promise<void> }).release === 'function') {
      await (sess as unknown as { release: () => Promise<void> }).release();
    }
  } catch {
    // ignore release errors
  }
}

export async function releaseCtdSession(): Promise<void> {
  const p = ctdSessionPromise;
  ctdSessionPromise = null;
  await releaseSessionPromise(p);
}

export async function releaseRtdetrSession(): Promise<void> {
  const p = rtdetrSessionPromise;
  rtdetrSessionPromise = null;
  await releaseSessionPromise(p);
}

export async function releaseLamaSession(): Promise<void> {
  const p = lamaSessionPromise;
  lamaSessionPromise = null;
  await releaseSessionPromise(p);
}

export function clearPageCtdCache(): void {
  cachedPageCtd = null;
}

export function getCachedPageCtd() {
  return cachedPageCtd;
}

export function setCachedPageCtd(data: typeof cachedPageCtd): void {
  cachedPageCtd = data;
}

export async function releaseAllOnnxSessions(): Promise<void> {
  cachedPageCtd = null;
  await releaseCtdSession();
  await releaseRtdetrSession();
  await releaseLamaSession();
}

export async function getLamaSession(): Promise<OrtTypes.InferenceSession | null> {
  if (!ENABLE_LAMA) {
    return null;
  }
  if (LOW_RAM_MODE) {
    await releaseCtdSession();
    await releaseRtdetrSession();
  }
  if (!lamaSessionPromise) {
    lamaSessionPromise = (async () => {
      const ok = await ensureModelDownloaded(LAMA_MODEL_PATH, LAMA_MODEL_URL);
      if (!ok) return null;
      const ort = await getOrt();
      return await ort.InferenceSession.create(LAMA_MODEL_PATH, {
        intraOpNumThreads: ONNX_THREADS,
        interOpNumThreads: 1,
        enableCpuMemArena: !LOW_RAM_MODE,
        enableMemPattern: !LOW_RAM_MODE,
        graphOptimizationLevel: LOW_RAM_MODE ? 'basic' : 'all',
      });
    })().catch((err) => {
      console.error('Failed to load lama-manga.onnx:', err);
      lamaSessionPromise = null;
      return null;
    });
  }
  return lamaSessionPromise;
}

function computeBoxIoU(a: BoundingBox, b: BoundingBox): number {
  const ix0 = Math.max(a.x, b.x);
  const iy0 = Math.max(a.y, b.y);
  const ix1 = Math.min(a.x + a.w, b.x + b.w);
  const iy1 = Math.min(a.y + a.h, b.y + b.h);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  const minArea = Math.min(areaA, areaB);
  const aspectA = a.w / Math.max(1, a.h);
  const aspectB = b.w / Math.max(1, b.h);
  const aspectRatioDiff = Math.max(
    aspectA / Math.max(0.01, aspectB),
    aspectB / Math.max(0.01, aspectA)
  );
  if (minArea > 0 && inter / minArea >= 0.72 && aspectRatioDiff <= 3.2) return 0.85;
  return inter / Math.max(1, areaA + areaB - inter);
}

/**
 * Bilinearly resizes an RGBA region [srcX, srcY, srcW, srcH] from full image into a planar RGB Float32Array [3, dstH, dstW] in 0..1.
 */
function extractAndResizeRgbPlanar(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  srcX: number,
  srcY: number,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Float32Array {
  const out = new Float32Array(3 * dstW * dstH);
  const planeSize = dstW * dstH;
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let dy = 0; dy < dstH; dy++) {
    const sy = Math.min(imgHeight - 1, Math.max(0, srcY + (dy + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(imgHeight - 1, y0 + 1);
    const wy = sy - y0;

    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(imgWidth - 1, Math.max(0, srcX + (dx + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(imgWidth - 1, x0 + 1);
      const wx = sx - x0;

      const p00 = (y0 * imgWidth + x0) * 4;
      const p10 = (y0 * imgWidth + x1) * 4;
      const p01 = (y1 * imgWidth + x0) * 4;
      const p11 = (y1 * imgWidth + x1) * 4;

      const w00 = (1 - wx) * (1 - wy);
      const w10 = wx * (1 - wy);
      const w01 = (1 - wx) * wy;
      const w11 = wx * wy;

      const r = rgba[p00] * w00 + rgba[p10] * w10 + rgba[p01] * w01 + rgba[p11] * w11;
      const g =
        rgba[p00 + 1] * w00 + rgba[p10 + 1] * w10 + rgba[p01 + 1] * w01 + rgba[p11 + 1] * w11;
      const b =
        rgba[p00 + 2] * w00 + rgba[p10 + 2] * w10 + rgba[p01 + 2] * w01 + rgba[p11 + 2] * w11;

      const dstIdx = dy * dstW + dx;
      out[dstIdx] = r / 255.0;
      out[planeSize + dstIdx] = g / 255.0;
      out[2 * planeSize + dstIdx] = b / 255.0;
    }
  }
  return out;
}

/**
 * Computes full-page neural pixel-level text segmentation mask (`0..255`) using
 * `comictextdetector.pt.onnx` (1024x1024) with packed vertical tiles (`tiles_per_call = 2`,
 * matching `test-clean-main/local_cleaner.py` `_detect_text_mask_ctd`).
 */
export async function computePageCtdSegMask(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number
): Promise<Uint8Array | null> {
  const fingerprint = computeRgbaFingerprint(rgba, imgWidth, imgHeight);
  if (
    cachedPageCtd &&
    cachedPageCtd.width === imgWidth &&
    cachedPageCtd.height === imgHeight &&
    cachedPageCtd.fingerprint === fingerprint
  ) {
    return cachedPageCtd.segMask;
  }

  const ctdSession = await getCtdSession();
  if (!ctdSession) return null;

  const inputSize = 1024;
  const planeSize = inputSize * inputSize;
  const tilesPerCall = 2;
  const slotW = Math.floor(inputSize / tilesPerCall); // 512px wide per packed tile slot
  const tileH = Math.min(imgHeight, Math.max(720, Math.round(imgWidth * 1.85)));
  const overlap = Math.min(180, Math.round(tileH * 0.16));
  const step = Math.max(360, tileH - overlap);

  const tiles: { y0: number; curH: number }[] = [];
  for (let yStart = 0; yStart < imgHeight; ) {
    const curH = Math.min(tileH, imgHeight - yStart);

    // Check if tile has any ink transitions
    let transitionCount = 0;
    for (let y = yStart + 8; y < yStart + curH - 8; y += 24) {
      let prevL = -1;
      for (let x = 16; x < imgWidth - 16; x += 16) {
        const p = (y * imgWidth + x) * 4;
        const l = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
        if (prevL >= 0 && Math.abs(l - prevL) > 45) {
          transitionCount++;
        }
        prevL = l;
      }
    }
    if (transitionCount >= 4) {
      tiles.push({ y0: yStart, curH });
    }
    if (yStart + curH >= imgHeight) break;
    yStart += step;
  }

  const fullMask = new Uint8Array(imgWidth * imgHeight);
  const rawBlkBoxes: { bbox: BoundingBox; score: number }[] = [];
  const canvasTensor = new Float32Array(3 * planeSize);

  for (let batchStart = 0; batchStart < tiles.length; batchStart += tilesPerCall) {
    canvasTensor.fill(0);
    const batch = tiles.slice(batchStart, batchStart + tilesPerCall);
    const metas: { y0: number; curH: number; xOff: number; nw: number; nh: number }[] = [];

    for (let idx = 0; idx < batch.length; idx++) {
      const { y0, curH } = batch[idx];
      const scale = Math.min(slotW / imgWidth, inputSize / curH);
      const nw = Math.max(1, Math.min(slotW, Math.round(imgWidth * scale)));
      const nh = Math.max(1, Math.min(inputSize, Math.round(curH * scale)));
      const xOff = idx * slotW;

      const scaleX = imgWidth / nw;
      const scaleY = curH / nh;

      for (let dy = 0; dy < nh; dy++) {
        const sy = Math.min(imgHeight - 1, Math.max(y0, y0 + (dy + 0.5) * scaleY - 0.5));
        const sy0 = Math.floor(sy);
        const sy1 = Math.min(imgHeight - 1, sy0 + 1);
        const wy = sy - sy0;

        for (let dx = 0; dx < nw; dx++) {
          const sx = Math.min(imgWidth - 1, Math.max(0, (dx + 0.5) * scaleX - 0.5));
          const sx0 = Math.floor(sx);
          const sx1 = Math.min(imgWidth - 1, sx0 + 1);
          const wx = sx - sx0;

          const p00 = (sy0 * imgWidth + sx0) * 4;
          const p10 = (sy0 * imgWidth + sx1) * 4;
          const p01 = (sy1 * imgWidth + sx0) * 4;
          const p11 = (sy1 * imgWidth + sx1) * 4;

          const w00 = (1 - wx) * (1 - wy);
          const w10 = wx * (1 - wy);
          const w01 = (1 - wx) * wy;
          const w11 = wx * wy;

          const r = rgba[p00] * w00 + rgba[p10] * w10 + rgba[p01] * w01 + rgba[p11] * w11;
          const g =
            rgba[p00 + 1] * w00 +
            rgba[p10 + 1] * w10 +
            rgba[p01 + 1] * w01 +
            rgba[p11 + 1] * w11;
          const b =
            rgba[p00 + 2] * w00 +
            rgba[p10 + 2] * w10 +
            rgba[p01 + 2] * w01 +
            rgba[p11 + 2] * w11;

          const dstIdx = dy * inputSize + (xOff + dx);
          canvasTensor[dstIdx] = r / 255.0;
          canvasTensor[planeSize + dstIdx] = g / 255.0;
          canvasTensor[2 * planeSize + dstIdx] = b / 255.0;
        }
      }
      metas.push({ y0, curH, xOff, nw, nh });
    }

    const ort = await getOrt();
    const inputName = ctdSession.inputNames[0] || 'images';
    const outputs = await ctdSession.run({
      [inputName]: new ort.Tensor('float32', canvasTensor, [1, 3, inputSize, inputSize]),
    });

    // 'blk' is the [1, 64512, 7] YOLOv5 text-block detector head (`[cx, cy, w, h, obj, cls0, cls1]`)
    const blkTensor = outputs.blk || outputs[ctdSession.outputNames[0]];
    if (blkTensor && blkTensor.data instanceof Float32Array) {
      const blkData = blkTensor.data;
      for (let i = 0; i < blkData.length; i += 7) {
        const obj = blkData[i + 4];
        if (obj < 0.35) continue;
        const score = obj * Math.max(blkData[i + 5], blkData[i + 6]);
        if (score < 0.42) continue;
        const cx = blkData[i];
        const cy = blkData[i + 1];
        const bw = blkData[i + 2];
        const bh = blkData[i + 3];
        if (bw < 6 || bh < 6) continue;

        // Match prediction to its packed tile slot (`metas`)
        for (const { y0, curH, xOff, nw, nh } of metas) {
          if (cx >= xOff - 4 && cx <= xOff + nw + 4 && cy <= nh + 4) {
            const scaleX = imgWidth / nw;
            const scaleY = curH / nh;
            const lx0 = Math.max(0, Math.round((cx - xOff - bw / 2) * scaleX));
            const ly0 = Math.max(y0, Math.round(y0 + (cy - bh / 2) * scaleY));
            const lx1 = Math.min(imgWidth, Math.round((cx - xOff + bw / 2) * scaleX));
            const ly1 = Math.min(imgHeight, Math.round(y0 + (cy + bh / 2) * scaleY));
            if (lx1 - lx0 >= 14 && ly1 - ly0 >= 12) {
              rawBlkBoxes.push({
                bbox: { x: lx0, y: ly0, w: lx1 - lx0, h: ly1 - ly0 },
                score,
              });
            }
            break;
          }
        }
      }
    }

    // 'seg' is the 1x1x1024x1024 pixel text segmentation map
    const segTensor = outputs.seg || outputs[ctdSession.outputNames[1]];
    if (!segTensor) continue;
    const segData = segTensor.data as Float32Array;

    for (const { y0, curH, xOff, nw, nh } of metas) {
      const invScaleX = nw / imgWidth;
      const invScaleY = nh / curH;

      for (let ly = 0; ly < curH; ly++) {
        const gy = y0 + ly;
        const my = Math.min(nh - 1, Math.max(0, Math.floor((ly + 0.5) * invScaleY)));
        const rowOff = my * inputSize + xOff;
        const dstRow = gy * imgWidth;

        for (let lx = 0; lx < imgWidth; lx++) {
          const mx = Math.min(nw - 1, Math.max(0, Math.floor((lx + 0.5) * invScaleX)));
          const val = Math.max(0, Math.min(255, Math.round(segData[rowOff + mx] * 255)));
          if (val > fullMask[dstRow + lx]) {
            fullMask[dstRow + lx] = val;
          }
        }
      }
    }
  }

  // NMS deduplicate CTD YOLO blk text-block boxes across overlapping tiles
  rawBlkBoxes.sort((a, b) => b.score - a.score);
  const blkBoxes: { bbox: BoundingBox; score: number }[] = [];
  for (const cand of rawBlkBoxes) {
    let dup = false;
    for (const kept of blkBoxes) {
      if (computeBoxIoU(cand.bbox, kept.bbox) >= 0.35) {
        dup = true;
        break;
      }
    }
    if (!dup) blkBoxes.push(cand);
  }

  cachedPageCtd = {
    width: imgWidth,
    height: imgHeight,
    fingerprint,
    segMask: fullMask,
    blkBoxes,
  };
  return fullMask;
}

/**
 * Extracts grouped text connected components from the full-page CTD segmentation mask
 * so tilted texts, system banners, SFX, and multi-character bubbles missed by RT-DETR are captured.
 */
function extractCtdComponents(
  ctdSegMask: Uint8Array,
  imgWidth: number,
  imgHeight: number
): { bbox: BoundingBox; pixelCount: number; peakConf: number }[] {
  // Downsample 2x for fast morphological grouping (radius=5 at 2x = 10px grouping radius)
  const gw = Math.ceil(imgWidth / 2);
  const gh = Math.ceil(imgHeight / 2);
  const grid = new Uint8Array(gw * gh);
  const gridConf = new Uint8Array(gw * gh);

  for (let y = 0; y < imgHeight; y++) {
    const gy = y >> 1;
    const rowOff = y * imgWidth;
    const gRow = gy * gw;
    for (let x = 0; x < imgWidth; x++) {
      const v = ctdSegMask[rowOff + x];
      if (v >= 70) {
        const gx = x >> 1;
        grid[gRow + gx] = 1;
        if (v > gridConf[gRow + gx]) gridConf[gRow + gx] = v;
      }
    }
  }

  // Dilate grid by rx=5, ry=4 (groups sibling characters in a line or bubble)
  const dilated = new Uint8Array(gw * gh);
  const rx = 5;
  const ry = 4;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      if (grid[y * gw + x] === 0) continue;
      const y0 = Math.max(0, y - ry);
      const y1 = Math.min(gh - 1, y + ry);
      const x0 = Math.max(0, x - rx);
      const x1 = Math.min(gw - 1, x + rx);
      for (let ny = y0; ny <= y1; ny++) {
        const nRow = ny * gw;
        for (let nx = x0; nx <= x1; nx++) {
          dilated[nRow + nx] = 1;
        }
      }
    }
  }

  const visited = new Uint8Array(gw * gh);
  const q = new Int32Array(gw * gh);
  const comps: { bbox: BoundingBox; pixelCount: number; peakConf: number }[] = [];

  for (let i = 0; i < gw * gh; i++) {
    if (dilated[i] === 0 || visited[i] === 1) continue;
    let head = 0;
    let tail = 0;
    q[tail++] = i;
    visited[i] = 1;

    let minX = gw;
    let minY = gh;
    let maxX = 0;
    let maxY = 0;
    let activeGridCells = 0;
    let maxConf = 0;

    while (head < tail) {
      const curr = q[head++];
      const cy = Math.floor(curr / gw);
      const cx = curr - cy * gw;

      if (grid[curr] === 1) {
        activeGridCells++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        if (gridConf[curr] > maxConf) maxConf = gridConf[curr];
      }

      if (cy > 0 && dilated[curr - gw] === 1 && visited[curr - gw] === 0) {
        visited[curr - gw] = 1;
        q[tail++] = curr - gw;
      }
      if (cy + 1 < gh && dilated[curr + gw] === 1 && visited[curr + gw] === 0) {
        visited[curr + gw] = 1;
        q[tail++] = curr + gw;
      }
      if (cx > 0 && dilated[curr - 1] === 1 && visited[curr - 1] === 0) {
        visited[curr - 1] = 1;
        q[tail++] = curr - 1;
      }
      if (cx + 1 < gw && dilated[curr + 1] === 1 && visited[curr + 1] === 0) {
        visited[curr + 1] = 1;
        q[tail++] = curr + 1;
      }
    }

    const pixelCount = activeGridCells * 4;
    if (pixelCount < 44 || maxConf < 105) continue;

    const x0 = Math.max(0, minX * 2 - 4);
    const y0 = Math.max(0, minY * 2 - 4);
    const x1 = Math.min(imgWidth, (maxX + 1) * 2 + 4);
    const y1 = Math.min(imgHeight, (maxY + 1) * 2 + 4);
    const w = x1 - x0;
    const h = y1 - y0;
    if (w < 12 || h < 12) continue;

    comps.push({
      bbox: { x: x0, y: y0, w, h },
      pixelCount,
      peakConf: maxConf / 255.0,
    });
  }

  return comps;
}

/**
 * Counts how many active CTD segmentation pixels (`>= 65`) lie inside `bbox`.
 */
function countCtdPixelsInBox(
  ctdSegMask: Uint8Array | null,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): number {
  if (!ctdSegMask) return 999;
  const x0 = Math.max(0, bbox.x);
  const y0 = Math.max(0, bbox.y);
  const x1 = Math.min(imgWidth, bbox.x + bbox.w);
  const y1 = Math.min(imgHeight, bbox.y + bbox.h);
  let count = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * imgWidth;
    for (let x = x0; x < x1; x++) {
      if (ctdSegMask[row + x] >= 65) count++;
    }
  }
  return count;
}

/**
 * Runs hybrid neural detection combining:
 * 1. Ogkalu RT-DETR-v2 (`detector-v4-s_int8.onnx`) across near-square 1.15x vertical tiles
 * 2. Comic-Text-Detector (`comictextdetector.pt.onnx`) 1024x1024 pixel-level text segmentation
 *
 * Returns structured `DetectedRegion[]` with exact bubble associations, watermark filtering,
 * and surround statistics.
 */
export async function detectRegionsWithOnnx(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  pageIndex: number,
  settings: CleaningSettings
): Promise<DetectedRegion[]> {
  let session: OrtTypes.InferenceSession | null = null;
  let pageCtdMask: Uint8Array | null = null;
  const precomputedFp = computeRgbaFingerprint(rgba, imgWidth, imgHeight);
  const hasPrecomputedCtd = Boolean(
    cachedPageCtd &&
      cachedPageCtd.width === imgWidth &&
      cachedPageCtd.height === imgHeight &&
      cachedPageCtd.fingerprint === precomputedFp
  );
  if (LOW_RAM_MODE || hasPrecomputedCtd) {
    if (hasPrecomputedCtd && cachedPageCtd) {
      pageCtdMask = cachedPageCtd.segMask;
    } else if (ENABLE_CTD) {
      await releaseLamaSession();
      pageCtdMask = await computePageCtdSegMask(rgba, imgWidth, imgHeight);
      await releaseCtdSession();
    }
    session = await getRtdetrSession();
  } else {
    [session, pageCtdMask] = await Promise.all([
      getRtdetrSession(),
      computePageCtdSegMask(rgba, imgWidth, imgHeight),
    ]);
  }
  if (!session) {
    return [];
  }

  // Near-square vertical tile height (~1.15x width) prevents vertical squashing of small/tilted text
  const tileH = Math.min(imgHeight, Math.max(720, Math.min(1440, Math.round(imgWidth * 1.15))));
  const overlap = Math.min(240, Math.floor(tileH * 0.25));
  const step = Math.max(320, tileH - overlap);

  const rawBubbles: { bbox: BoundingBox; score: number }[] = [];
  const rawTexts: { bbox: BoundingBox; score: number; label: number }[] = [];

  for (let yStart = 0; yStart < imgHeight; ) {
    const curH = Math.min(tileH, imgHeight - yStart);

    // Fast 0.2ms check: skip pure white/black webtoon gutter tiles with no ink transitions
    let transitionCount = 0;
    for (let y = yStart + 8; y < yStart + curH - 8; y += 24) {
      let prevL = -1;
      for (let x = 16; x < imgWidth - 16; x += 16) {
        const p = (y * imgWidth + x) * 4;
        const l = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
        if (prevL >= 0 && Math.abs(l - prevL) > 48) {
          transitionCount++;
        }
        prevL = l;
      }
    }

    if (transitionCount < 4) {
      if (yStart + curH >= imgHeight) break;
      yStart += step;
      continue;
    }

    const inputData = extractAndResizeRgbPlanar(
      rgba,
      imgWidth,
      imgHeight,
      0,
      yStart,
      imgWidth,
      curH,
      640,
      640
    );

    const ort = await getOrt();
    const imagesTensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
    const origSizesTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(imgWidth), BigInt(curH)]),
      [1, 2]
    );

    const outputs = await session.run({
      images: imagesTensor,
      orig_target_sizes: origSizesTensor,
    });

    const scores = outputs.scores.data as Float32Array;
    const labels = outputs.labels.data as BigInt64Array;
    const boxes = outputs.boxes.data as Float32Array;

    for (let i = 0; i < scores.length; i++) {
      const score = scores[i];
      const label = Number(labels[i]);
      const minScore = label === 0 ? 0.40 : label === 1 ? 0.33 : 0.35;
      if (score < minScore) continue;

      const x1 = Math.max(0, Math.round(boxes[i * 4]));
      const y1 = Math.max(0, Math.round(boxes[i * 4 + 1] + yStart));
      const x2 = Math.min(imgWidth, Math.round(boxes[i * 4 + 2]));
      const y2 = Math.min(imgHeight, Math.round(boxes[i * 4 + 3] + yStart));
      const w = x2 - x1;
      const h = y2 - y1;
      if (w < 14 || h < 12) continue;

      const aspect = w / Math.max(1, h);
      // Reject thin panel-border false positives (e.g. 18x665 vertical lines or 420x16 horizontal borders)
      if (label !== 0) {
        if ((w <= 24 && h >= 180) || (h <= 18 && w >= 320)) continue;
        if (aspect < 0.12 || (aspect > 12.5 && score < 0.52)) continue;
        // Reject giant full-page-width illustrated cover title emblems (e.g. 672x566 on 005.jpg)
        if (label === 2 && w > imgWidth * 0.76 && h > imgWidth * 0.52 && score < 0.68) {
          continue;
        }
      }

      const bbox: BoundingBox = { x: x1, y: y1, w, h };
      if (label === 0) {
        // Reject full-panel or full-crop false-positive bubble detections (e.g. a small crop where the entire image is tagged as label=0)
        if (w >= imgWidth * 0.92 && h >= imgHeight * 0.85 && score < 0.75) {
          continue;
        }
        rawBubbles.push({ bbox, score });
      } else if (label === 1 || label === 2) {
        rawTexts.push({ bbox, score, label });
      }
    }

    if (yStart + curH >= imgHeight) break;
    yStart += step;
  }

  // Non-maximum suppression / tile-seam deduplication for bubbles
  rawBubbles.sort((a, b) => b.score - a.score);
  const mergedBubbles: { bbox: BoundingBox; score: number }[] = [];
  for (const cand of rawBubbles) {
    let dup = false;
    for (const kept of mergedBubbles) {
      if (computeBoxIoU(cand.bbox, kept.bbox) >= 0.45) {
        const nx0 = Math.min(kept.bbox.x, cand.bbox.x);
        const ny0 = Math.min(kept.bbox.y, cand.bbox.y);
        const nx1 = Math.max(kept.bbox.x + kept.bbox.w, cand.bbox.x + cand.bbox.w);
        const ny1 = Math.max(kept.bbox.y + kept.bbox.h, cand.bbox.y + cand.bbox.h);
        kept.bbox = { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
        dup = true;
        break;
      }
    }
    if (!dup) mergedBubbles.push(cand);
  }

  // Non-maximum suppression / tile-seam deduplication for text boxes
  rawTexts.sort((a, b) => b.score - a.score);
  const mergedTexts: { bbox: BoundingBox; score: number; label: number; fromCtdOnly?: boolean }[] = [];
  for (const cand of rawTexts) {
    let dup = false;
    for (const kept of mergedTexts) {
      if (computeBoxIoU(cand.bbox, kept.bbox) >= 0.32) {
        const nx0 = Math.min(kept.bbox.x, cand.bbox.x);
        const ny0 = Math.min(kept.bbox.y, cand.bbox.y);
        const nx1 = Math.max(kept.bbox.x + kept.bbox.w, cand.bbox.x + cand.bbox.w);
        const ny1 = Math.max(kept.bbox.y + kept.bbox.h, cand.bbox.y + cand.bbox.h);
        kept.bbox = { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
        if (cand.label === 1 && cand.score >= 0.46) kept.label = 1;
        kept.score = Math.max(kept.score, cand.score);
        dup = true;
        break;
      }
    }
    if (!dup) mergedTexts.push(cand);
  }

  // Fuse Comic-Text-Detector (`pageCtdMask` + YOLO `blkBoxes`) with RT-DETR detections:
  // 1) Expand partial RT-DETR boxes (e.g. `괴물의 숲은 장난으로...` or lavender bubble missing `그`)
  // 2) Add missed text/SFX ONLY when confirmed by CTD's YOLO `blk` text-block detector OR inside a verified bubble!
  //    Never spawn standalone detections from raw `seg` blobs alone (which trigger on eyes, speedlines, and window grilles).
  if (pageCtdMask) {
    const ctdBlkBoxes = cachedPageCtd?.blkBoxes || [];
    const ctdComps = extractCtdComponents(pageCtdMask, imgWidth, imgHeight);
    for (const comp of ctdComps) {
      if (
        comp.bbox.w > imgWidth * 0.76 &&
        comp.bbox.h > imgWidth * 0.52
      ) {
        continue;
      }

      // Check if comp is inside a detected speech bubble that already has an RT-DETR text box
      const compCx = comp.bbox.x + comp.bbox.w / 2;
      const compCy = comp.bbox.y + comp.bbox.h / 2;
      let enclosingBubble: BoundingBox | null = null;
      for (const b of mergedBubbles) {
        if (
          compCx >= b.bbox.x &&
          compCx <= b.bbox.x + b.bbox.w &&
          compCy >= b.bbox.y &&
          compCy <= b.bbox.y + b.bbox.h &&
          b.bbox.w >= comp.bbox.w * 1.15 &&
          b.bbox.h >= comp.bbox.h * 1.15
        ) {
          enclosingBubble = b.bbox;
          break;
        }
      }

      // Check whether CTD's YOLO text-block detector (`blk`) confirms this component as real text
      let matchedBlkScore = 0;
      for (const blk of ctdBlkBoxes) {
        const blkIou = computeBoxIoU(comp.bbox, blk.bbox);
        const centerInsideBlk =
          compCx >= blk.bbox.x &&
          compCx <= blk.bbox.x + blk.bbox.w &&
          compCy >= blk.bbox.y &&
          compCy <= blk.bbox.y + blk.bbox.h;
        if (blkIou >= 0.15 || (centerInsideBlk && comp.bbox.w <= blk.bbox.w * 1.35 && comp.bbox.h <= blk.bbox.h * 1.35)) {
          if (blk.score > matchedBlkScore) matchedBlkScore = blk.score;
        }
      }

      let matchedExisting = false;
      for (const kept of mergedTexts) {
        const iou = computeBoxIoU(comp.bbox, kept.bbox);
        const strictlyInsideBubble =
          enclosingBubble != null &&
          comp.bbox.x >= enclosingBubble.x + 8 &&
          comp.bbox.y >= enclosingBubble.y + 8 &&
          comp.bbox.x + comp.bbox.w <= enclosingBubble.x + enclosingBubble.w - 8 &&
          comp.bbox.y + comp.bbox.h <= enclosingBubble.y + enclosingBubble.h - 8;
        const yGap = Math.max(
          0,
          kept.bbox.y - (comp.bbox.y + comp.bbox.h),
          comp.bbox.y - (kept.bbox.y + kept.bbox.h)
        );
        const sameBubble =
          strictlyInsideBubble &&
          yGap <= 42 &&
          kept.bbox.x + kept.bbox.w / 2 >= enclosingBubble!.x &&
          kept.bbox.x + kept.bbox.w / 2 <= enclosingBubble!.x + enclosingBubble!.w &&
          kept.bbox.y + kept.bbox.h / 2 >= enclosingBubble!.y &&
          kept.bbox.y + kept.bbox.h / 2 <= enclosingBubble!.y + enclosingBubble!.h;

        if (iou >= 0.12 || sameBubble) {
          const nx0 = Math.min(kept.bbox.x, comp.bbox.x);
          const ny0 = Math.min(kept.bbox.y, comp.bbox.y);
          const nx1 = Math.max(kept.bbox.x + kept.bbox.w, comp.bbox.x + comp.bbox.w);
          const ny1 = Math.max(kept.bbox.y + kept.bbox.h, comp.bbox.y + comp.bbox.h);
          kept.bbox = { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
          // Only boost confidence from CTD if YOLO `blk` (or bubble enclosure) confirms it
          if (matchedBlkScore >= 0.45 || sameBubble) {
            kept.score = Math.max(kept.score, matchedBlkScore || comp.peakConf);
          }
          matchedExisting = true;
          break;
        }
      }

      if (!matchedExisting) {
        // Standalone CTD additions MUST be confirmed by either a verified speech bubble OR CTD YOLO `blk` >= 0.48
        if (!enclosingBubble && matchedBlkScore < 0.48) {
          continue;
        }
        if (enclosingBubble && comp.peakConf < 0.50) {
          continue;
        }
        mergedTexts.push({
          bbox: comp.bbox,
          score: Number(
            Math.min(0.95, enclosingBubble ? comp.peakConf : matchedBlkScore).toFixed(2)
          ),
          label: enclosingBubble ? 1 : 2,
          fromCtdOnly: true,
        });
      }
    }
  }

  // Final pass to merge any boxes that became overlapping after CTD expansion
  const consolidatedTexts: { bbox: BoundingBox; score: number; label: number; fromCtdOnly?: boolean }[] = [];
  for (const cand of mergedTexts) {
    // Reject low-confidence label=2 false positives (like blue/yellow curved arrows) when CTD sees zero text pixels
    if (pageCtdMask && cand.label === 2 && cand.score < 0.49) {
      const ctdPixels = countCtdPixelsInBox(pageCtdMask, imgWidth, imgHeight, cand.bbox);
      if (ctdPixels < 14) {
        continue;
      }
    }

    let merged = false;
    for (const kept of consolidatedTexts) {
      if (computeBoxIoU(cand.bbox, kept.bbox) >= 0.30) {
        const nx0 = Math.min(kept.bbox.x, cand.bbox.x);
        const ny0 = Math.min(kept.bbox.y, cand.bbox.y);
        const nx1 = Math.max(kept.bbox.x + kept.bbox.w, cand.bbox.x + cand.bbox.w);
        const ny1 = Math.max(kept.bbox.y + kept.bbox.h, cand.bbox.y + cand.bbox.h);
        kept.bbox = { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
        if (cand.label === 1 && cand.score >= 0.46) kept.label = 1;
        kept.score = Math.max(kept.score, cand.score);
        merged = true;
        break;
      }
    }
    if (!merged) consolidatedTexts.push(cand);
  }

  // Sort top-to-bottom, left-to-right for natural reading order
  consolidatedTexts.sort((a, b) => a.bbox.y - b.bbox.y || a.bbox.x - b.bbox.x);

  const regions: DetectedRegion[] = [];
  let regionCounter = 1;

  for (const item of consolidatedTexts) {
    // Expand text box by 10px safety margin so stroke tips, outlines, and punctuation near box edges are never clipped!
    const padBox = 10;
    const bx = Math.max(0, item.bbox.x - padBox);
    const by = Math.max(0, item.bbox.y - padBox);
    const bw = Math.min(imgWidth - bx, item.bbox.w + padBox * 2);
    const bh = Math.min(imgHeight - by, item.bbox.h + padBox * 2);
    const bbox: BoundingBox = { x: bx, y: by, w: bw, h: bh };

    // Match with enclosing RT-DETR Bubble box (label 0) if present and genuinely enclosing
    const cx = bx + bw / 2;
    const cy = by + bh / 2;
    let matchedBubble: BoundingBox | undefined;
    let matchedBubbleIdx = -1;
    for (let bIdx = 0; bIdx < mergedBubbles.length; bIdx++) {
      const bb = mergedBubbles[bIdx].bbox;
      const isEnclosing =
        bb.w >= item.bbox.w * 1.12 &&
        bb.h >= item.bbox.h * 1.12 &&
        cx >= bb.x &&
        cx <= bb.x + bb.w &&
        cy >= bb.y &&
        cy <= bb.y + bb.h;
      if (isEnclosing) {
        matchedBubble = bb;
        matchedBubbleIdx = bIdx;
        break;
      }
    }

    // Filter out scanlation watermarks/credits in outer corners or margin ribbons
    if (!matchedBubble && isScanlationWatermark(bbox, imgWidth, imgHeight)) {
      continue;
    }

    const surroundStats = measureSurroundRing(rgba, imgWidth, imgHeight, bbox);
    const rayEnclosure = verifyBubbleEnclosure(rgba, imgWidth, imgHeight, bbox);

    const insideBubble =
      (item.label === 1 && item.score >= 0.46) || Boolean(matchedBubble) || rayEnclosure.enclosed;

    // Non-text protection gate: never erase character faces, standalone X/checkmark/circle symbols,
    // low-CTD decorative artwork (ornaments, weapons, clouds), or UI bars / architectural grilles.
    const ctdMetrics = computeCtdBoxMetrics(pageCtdMask, imgWidth, imgHeight, bbox);
    const guardDecision = evaluateNonTextGuard({
      rgba,
      imgWidth,
      imgHeight,
      bbox,
      confidence: item.score,
      insideBubble,
      surroundStats,
      ctdMetrics,
      fromCtdOnly: item.fromCtdOnly,
    });
    if (guardDecision.reject) {
      continue;
    }

    const bubbleBbox = matchedBubble || rayEnclosure.bubbleBbox;

    let category: RegionCategory = 'BUBBLE';
    const relCenterX = (bbox.x + bbox.w / 2) / imgWidth;
    const aspect = bbox.w / Math.max(1, bbox.h);
    const isNearPageSide = relCenterX < 0.18 || relCenterX > 0.82;

    const isGiantStylizedSfx =
      !insideBubble &&
      (bbox.w * bbox.h >= 45000 || (bbox.w >= imgWidth * 0.45 && bbox.h >= 120));

    if (insideBubble) {
      category = 'BUBBLE';
    } else if (
      isGiantStylizedSfx ||
      (!surroundStats.isUniform && (surroundStats.edgeEnergy > 38 || bbox.w * bbox.h > 14000))
    ) {
      category = 'SFX';
    } else if (isNearPageSide || (aspect < 0.48 && bbox.w <= 84)) {
      category = 'ST';
    } else {
      category = 'OT';
    }

    if (category === 'BUBBLE' && !settings.enableBubble) continue;
    if (category === 'OT' && !settings.enableOT) continue;
    if (category === 'ST' && !settings.enableST) continue;
    if (category === 'SFX' && !settings.enableSFX) continue;

    let routedEngine = selectCleaningEngine(category, surroundStats, settings);
    if (isGiantStylizedSfx && routedEngine === 'SMART_FILL' && settings.engineRouting !== 'FORCE_SMART_FILL') {
      routedEngine = 'LAMA_PATCH';
    }
    const id = `p${pageIndex + 1}_r${regionCounter++}_${category.toLowerCase()}`;

    regions.push({
      id,
      pageIndex,
      category,
      bbox,
      bubbleBbox,
      bubbleId:
        matchedBubbleIdx >= 0
          ? `bub_${pageIndex + 1}_${matchedBubbleIdx + 1}`
          : insideBubble
            ? `bub_${pageIndex + 1}_r${regionCounter}`
            : undefined,
      insideBubble,
      confidence: Number(Math.min(0.99, item.score).toFixed(2)),
      sourceScript: category === 'SFX' ? 'SFX_STYLIZED' : 'KO',
      routedEngine,
      bubbleCleanMode: settings.bubbleCleanMode,
      paddingPx: settings.defaultPaddingPx,
      featherPx: settings.defaultFeatherPx,
      fillColorHex: surroundStats.isWhiteBubble
        ? '#FFFFFF'
        : `#${Math.round(surroundStats.meanR).toString(16).padStart(2, '0')}${Math.round(surroundStats.meanG).toString(16).padStart(2, '0')}${Math.round(surroundStats.meanB).toString(16).padStart(2, '0')}`.toUpperCase(),
      inkCoverageRatio: 0.24,
      surroundStats,
      status: 'DETECTED',
      reviewStatus:
        category === 'SFX' && item.score < 0.55 ? 'NEEDS_REVIEW' : 'AUTO_APPROVED',
      manualEdits: [],
    });
  }

  return regions;
}

/**
 * Reflect-101 coordinate helper (OpenCV BORDER_REFLECT_101 equivalent).
 */
function reflect101(idx: number, len: number): number {
  if (len <= 1) return 0;
  let c = idx;
  while (c < 0 || c >= len) {
    if (c < 0) c = -c;
    if (c >= len) c = 2 * len - 2 - c;
  }
  return Math.max(0, Math.min(len - 1, c));
}

/**
 * Extracts the `bw x bh` slice of the full-page CTD segmentation mask for `region.bbox`
 * if available in cache.
 */
function sliceCachedCtdRegionMask(
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): Uint8Array | undefined {
  if (!cachedPageCtd || cachedPageCtd.width !== imgWidth || cachedPageCtd.height !== imgHeight) {
    return undefined;
  }
  const { x: bx, y: by, w: bw, h: bh } = bbox;
  const out = new Uint8Array(bw * bh);
  const full = cachedPageCtd.segMask;
  for (let ly = 0; ly < bh; ly++) {
    const gy = by + ly;
    if (gy < 0 || gy >= imgHeight) continue;
    const srcRow = gy * imgWidth;
    const dstRow = ly * bw;
    for (let lx = 0; lx < bw; lx++) {
      const gx = bx + lx;
      if (gx < 0 || gx >= imgWidth) continue;
      out[dstRow + lx] = full[srcRow + gx];
    }
  }
  return out;
}

/**
 * Cleans a detected region using the 3-tier Manhwa cleaning pipeline:
 * 1. Verified pure-white (`#FFFFFF`) or pure-black (`#000000`) speech bubble whitening
 * 2. Immediate clean-ring Median Fill (`FLAT_RING`) or 2D Linear Plane Least-Squares Fit (`GRADIENT_PLANE`, `a*y + b*x + c`)
 *    for flat tinted bubbles (lavender, grey, gold) and smooth linear gradient bubbles (`헨리에타 마차시...`)
 * 3. Neural `lama-manga.onnx` 512x512 inpainting for all textured backgrounds, wooden planks, system boxes over artwork,
 *    screentone bubbles, red bokeh, palm fronds, ropes, OT, ST, and SFX!
 */
export async function cleanRegionWithOnnxSupport(params: {
  rgba: Uint8Array;
  imgWidth: number;
  imgHeight: number;
  region: DetectedRegion;
  settings: CleaningSettings;
}): Promise<{
  writeMask: Uint8Array;
  strokeSeedMask: Uint8Array;
  quality: QualityAssessment;
}> {
  const { rgba, imgWidth, imgHeight, region, settings } = params;
  const engine = selectCleaningEngine(region.category, region.surroundStats, settings);

  // Fast path ONLY for verified pure-white or pure-black speech bubbles in WHITEN mode when SMART_FILL is selected
  const isVerifiedPureBubble =
    region.category === 'BUBBLE' &&
    engine === 'SMART_FILL' &&
    settings.engineRouting !== 'FORCE_LAMA_PATCH' &&
    (region.surroundStats.isWhiteBubble || region.surroundStats.isDarkBubble);

  if (isVerifiedPureBubble) {
    return cleanAndCompositeRegionInPlace(params);
  }

  const { x: bx, y: by, w: bw, h: bh } = region.bbox;
  const ctdRegionMask = sliceCachedCtdRegionMask(imgWidth, imgHeight, region.bbox);

  const { writeMask, strokeSeedMask, interiorCandidate } = buildRegionPixelMask({
    rgba,
    imgWidth,
    imgHeight,
    region,
    preserveColoredBubbles: settings.preserveColoredBubbles,
    ctdRegionMask,
  });

  // Check if any pixel in writeMask is active
  let activeCount = 0;
  for (let i = 0; i < writeMask.length; i++) {
    if (writeMask[i] > 0) activeCount++;
  }
  if (activeCount === 0) {
    return cleanAndCompositeRegionInPlace(params);
  }

  // Tier 1 & Tier 2: Immediate Clean-Ring Median Fill or 2D Linear Plane Fit (`a*y + b*x + c`)
  // Handles flat tinted bubbles (lavender, grey, beige) and linear gradient bubbles with zero LaMa patches!
  // Skip flat ring fill for large SFX (e.g. cyan "BEEEP" over radar rings) so background artwork is inpainted by LaMa.
  const isLargeSfxOverArt = region.category === 'SFX' && bw * bh >= 30000;
  if (settings.engineRouting !== 'FORCE_LAMA_PATCH' && !isLargeSfxOverArt) {
    const ringTier = tryCleanFlatOrGradientRing({
      rgba,
      imgWidth,
      imgHeight,
      region,
      writeMask,
      interiorCandidate,
    });

    if (ringTier !== null) {
      region.appliedEngine = 'SMART_FILL';
      return {
        writeMask,
        strokeSeedMask,
        quality: {
          outcome: 'ACCEPTED',
          interiorEdgeEnergy: 1.2,
          surroundEdgeEnergy: region.surroundStats.edgeEnergy,
          edgeEnergyRatio: 0.05,
          interiorOutsideFraction: 0,
          ghostStrokeResidual: 0,
          unchangedOutsideMaskVerified: true,
        },
      };
    }

    // Tier 2.5: Tilted / Colored Ribbon Banner Fill (e.g. #TalkedForAnHourAboutOneBurger on red streamer header)
    // When a wide banner region (bw >= 260, aspect >= 2.8) has a consistent saturated ribbon color along its
    // unmasked scanline ring (e.g. red banner r >= 210, g <= 105, b <= 105 with low variance), fill masked pixels
    // inside the ribbon's linear top-edge equation `y >= a*x + b` with the ribbon color, and fill masked outline
    // pixels protruding above the ribbon's top edge (`y < a*x + b`) with the outer background 6px above!
    if (bw >= 260 && bw / Math.max(1, bh) >= 2.8) {
      const ribbonSamples: { r: number; g: number; b: number; x: number; y: number }[] = [];
      const colTopY = new Int32Array(bw).fill(-1);
      for (let ly = 0; ly < bh; ly++) {
        const py = by + ly;
        for (let lx = 0; lx < bw; lx++) {
          const idx = ly * bw + lx;
          if (writeMask[idx] > 0) continue;
          const px = bx + lx;
          const p = (py * imgWidth + px) * 4;
          const r = rgba[p],
            g = rgba[p + 1],
            b = rgba[p + 2];
          const chroma = Math.max(r, g, b) - Math.min(r, g, b);
          const isRibbonColor = chroma >= 75 && Math.abs(r - g) >= 85;
          if (isRibbonColor && colTopY[lx] === -1) {
            colTopY[lx] = ly;
          }
          if ((lx & 1) === 0 && interiorCandidate[idx] > 0 && isRibbonColor) {
            ribbonSamples.push({ r, g, b, x: lx, y: ly });
          }
        }
      }
      if (ribbonSamples.length >= 140) {
        const rs = ribbonSamples.map((s) => s.r).sort((a, b) => a - b);
        const gs = ribbonSamples.map((s) => s.g).sort((a, b) => a - b);
        const bs = ribbonSamples.map((s) => s.b).sort((a, b) => a - b);
        const p15Idx = Math.floor(ribbonSamples.length * 0.15);
        const p85Idx = Math.floor(ribbonSamples.length * 0.85);
        const spread = Math.max(
          rs[p85Idx] - rs[p15Idx],
          gs[p85Idx] - gs[p15Idx],
          bs[p85Idx] - bs[p15Idx]
        );
        if (spread <= 24) {
          const medR = rs[Math.floor(rs.length * 0.5)];
          const medG = gs[Math.floor(gs.length * 0.5)];
          const medB = bs[Math.floor(bs.length * 0.5)];

          // Compute straight tilted ribbon top edge yTop(x) = slope * x + intercept:
          // Since the bottom edge of the ribbon (`colBotY[lx]`) has no protruding letters, we also measure `colBotY[lx]`
          // and fit a linear regression to the unmasked red ribbon band!
          const colBotY = new Int32Array(bw).fill(-1);
          for (let ly = bh - 1; ly >= 0; ly--) {
            const py = by + ly;
            for (let lx = 0; lx < bw; lx++) {
              if (colBotY[lx] !== -1 || writeMask[ly * bw + lx] > 0) continue;
              const p = (py * imgWidth + (bx + lx)) * 4;
              const r = rgba[p],
                g = rgba[p + 1],
                b = rgba[p + 2];
              const chroma = Math.max(r, g, b) - Math.min(r, g, b);
              if (chroma >= 75 && Math.abs(r - g) >= 85) {
                colBotY[lx] = ly;
              }
            }
          }

          // Fit linear least-squares line to colBotY (which is an unbroken straight tilted line at the bottom of the red banner)
          let bSumX = 0,
            bSumY = 0,
            bSumXX = 0,
            bSumXY = 0,
            bCnt = 0;
          for (let lx = 0; lx < bw; lx++) {
            if (colBotY[lx] >= bh * 0.45) {
              bSumX += lx;
              bSumY += colBotY[lx];
              bSumXX += lx * lx;
              bSumXY += lx * colBotY[lx];
              bCnt++;
            }
          }
          let slope = 0;
          let botIntercept = bh - 1;
          if (bCnt >= 20) {
            const denom = bCnt * bSumXX - bSumX * bSumX;
            if (Math.abs(denom) > 1e-3) {
              slope = (bCnt * bSumXY - bSumX * bSumY) / denom;
              botIntercept = (bSumY - slope * bSumX) / bCnt;
            }
          }
          // Ribbon thickness is the 80th percentile of (botLine(x) - colTopY[x]) across unmasked columns
          const thicknesses: number[] = [];
          for (let lx = 0; lx < bw; lx++) {
            if (colTopY[lx] >= 0) {
              const botY = slope * lx + botIntercept;
              const th = botY - colTopY[lx];
              if (th >= 20 && th <= bh) thicknesses.push(th);
            }
          }
          thicknesses.sort((a, b) => a - b);
          const ribbonHeight =
            thicknesses.length > 0
              ? thicknesses[Math.floor(thicknesses.length * 0.8)]
              : bh * 0.72;

          for (let ly = 0; ly < bh; ly++) {
            const py = by + ly;
            for (let lx = 0; lx < bw; lx++) {
              const mAlpha = writeMask[ly * bw + lx];
              if (mAlpha === 0) continue;
              const px = bx + lx;
              const dstIdx = (py * imgWidth + px) * 4;
              const expectedTopY = slope * lx + botIntercept - ribbonHeight;

              let targetR = medR;
              let targetG = medG;
              let targetB = medB;
              if (bCnt >= 20 && ly < expectedTopY + 1.0) {
                // Protruding letter outline above the straight tilted ribbon top edge -> sample outer background at (by - 3)
                const sampleY = Math.max(0, by - 3);
                const sp = (sampleY * imgWidth + px) * 4;
                targetR = rgba[sp];
                targetG = rgba[sp + 1];
                targetB = rgba[sp + 2];
              }

              const wL = mAlpha / 255.0;
              const wS = wL * wL * (3.0 - 2.0 * wL);
              rgba[dstIdx] = Math.round(rgba[dstIdx] * (1 - wS) + targetR * wS);
              rgba[dstIdx + 1] = Math.round(rgba[dstIdx + 1] * (1 - wS) + targetG * wS);
              rgba[dstIdx + 2] = Math.round(rgba[dstIdx + 2] * (1 - wS) + targetB * wS);
            }
          }
          region.appliedEngine = 'SMART_FILL';
          return {
            writeMask,
            strokeSeedMask,
            quality: {
              outcome: 'ACCEPTED',
              interiorEdgeEnergy: 1.2,
              surroundEdgeEnergy: region.surroundStats.edgeEnergy,
              edgeEnergyRatio: 0.05,
              interiorOutsideFraction: 0,
              ghostStrokeResidual: 0,
              unchangedOutsideMaskVerified: true,
            },
          };
        }
      }

      // Tier 2.6: Tilted White Banner with Dark Text (e.g. "New Mickey D's Menu" at bottom of page between dark video bar and yellow arrows)
      // Note: CTD (`comictextdetector.pt.onnx`) often masks the white inter-letter strip inside a white banner as `seg >= 55`.
      // So we check BOTH total pure-white pixels (`r >= 244, g >= 244, b >= 244`) in the bbox AND verify that pure-white pixels
      // exist in the unmasked portion of the banner strip (`unmaskedWhiteCount >= 450`), while rejecting double-outlined white subtitles
      // over dark artwork (where `unmaskedWhiteCount` is near zero!).
      let unmaskedWhiteCount = 0;
      let totalPureWhiteCount = 0;
      let maskedDarkCount = 0;
      for (let ly = 0; ly < bh; ly++) {
        const py = by + ly;
        for (let lx = 0; lx < bw; lx += 2) {
          const idx = ly * bw + lx;
          const p = (py * imgWidth + (bx + lx)) * 4;
          const r = rgba[p],
            g = rgba[p + 1],
            b = rgba[p + 2];
          const isPureWhite = r >= 244 && g >= 244 && b >= 244;
          if (isPureWhite) {
            totalPureWhiteCount++;
            if (writeMask[idx] === 0) unmaskedWhiteCount++;
          }
          if (writeMask[idx] >= 128 && rgbToLuma(r, g, b) <= 95) {
            maskedDarkCount++;
          }
        }
      }
      if (
        unmaskedWhiteCount >= 420 &&
        totalPureWhiteCount >= (bw * bh) / 10 &&
        maskedDarkCount >= 40
      ) {
        // Clean all dark text strokes inside the white banner strip (including rightwards up to the red frame / yellow arrows)
        const scanMaxX = Math.min(imgWidth, bx + Math.round(bw * 1.45));
        for (let ly = 0; ly < bh; ly++) {
          const py = by + ly;
          for (let gx = bx; gx < scanMaxX; gx++) {
            const lx = gx - bx;
            const dstIdx = (py * imgWidth + gx) * 4;
            const r = rgba[dstIdx],
              g = rgba[dstIdx + 1],
              b = rgba[dstIdx + 2];
            // Protect yellow arrows, red borders, or the top dark video player bar
            if (
              (r >= 215 && b <= 155) ||
              (r >= 205 && g <= 105 && b <= 105) ||
              (r <= 48 && g <= 48 && b <= 48 && ly < 28)
            ) {
              continue;
            }
            const mAlpha = lx < bw ? writeMask[ly * bw + lx] : 0;
            const isDarkBannerGlyph =
              ly >= 16 &&
              rgbToLuma(r, g, b) <= 220 &&
              Math.max(r, g, b) - Math.min(r, g, b) <= 35;
            if (mAlpha === 0 && !isDarkBannerGlyph) continue;

            // Verify local 19x19 neighborhood has pure-white banner pixels AND does not touch the black video bar
            let hasLocalWhite = false;
            let touchesTopVideoBar = false;
            for (let wy = Math.max(0, ly - 8); wy <= Math.min(bh - 1, ly + 8); wy += 2) {
              for (
                let wx = Math.max(bx, gx - 8);
                wx <= Math.min(imgWidth - 1, gx + 8);
                wx += 2
              ) {
                const wp = ((by + wy) * imgWidth + wx) * 4;
                if (rgba[wp] >= 244 && rgba[wp + 1] >= 244 && rgba[wp + 2] >= 244) {
                  hasLocalWhite = true;
                }
                if (wy < 22 && rgba[wp] <= 35 && rgba[wp + 1] <= 35 && rgba[wp + 2] <= 35) {
                  touchesTopVideoBar = true;
                }
              }
            }
            if (!hasLocalWhite || touchesTopVideoBar) continue;

            const effAlpha = isDarkBannerGlyph ? 255 : mAlpha;
            const wL = effAlpha / 255.0;
            const wS = wL * wL * (3.0 - 2.0 * wL);
            rgba[dstIdx] = Math.round(r * (1 - wS) + 255 * wS);
            rgba[dstIdx + 1] = Math.round(g * (1 - wS) + 255 * wS);
            rgba[dstIdx + 2] = Math.round(b * (1 - wS) + 255 * wS);
          }
        }
        region.appliedEngine = 'SMART_FILL';
        return {
          writeMask,
          strokeSeedMask,
          quality: {
            outcome: 'ACCEPTED',
            interiorEdgeEnergy: 1.0,
            surroundEdgeEnergy: region.surroundStats.edgeEnergy,
            edgeEnergyRatio: 0.04,
            interiorOutsideFraction: 0,
            ghostStrokeResidual: 0,
            unchangedOutsideMaskVerified: true,
          },
        };
      }
    }
  }

  // Tier 3: Neural `lama-manga.onnx` 512x512 inpainting for all complex textures, screentones,
  // translucent system panels, wooden planks, dark scenes, OT, ST, and SFX
  const lamaSession = await getLamaSession();
  if (!lamaSession) {
    return cleanAndCompositeRegionInPlace(params);
  }

  // Helper to run 512x512 LaMa inference on a crop [cx0..cx1, cy0..cy1] and return a sampler(gx, gy) -> [r, g, b]
  const runLamaOnSubCrop = async (subX0: number, subY0: number, subX1: number, subY1: number) => {
    const cw = subX1 - subX0;
    const ch = subY1 - subY0;

    // Build closedCropMask for LaMa input:
    // Dilate writeMask by r=6 and erode by r=2 (morphological closing + 4px outer envelope)
    // so inter-letter and intra-letter gaps (and outer black outline halos on double-outlined subtitles)
    // are merged into a clean word-envelope mask for LaMa!
    const closedCropMask = new Uint8Array(cw * ch);
    const closeR = 6;
    const closeR2 = closeR * closeR + 1;
    for (let ly = 0; ly < bh; ly++) {
      const gy = by + ly;
      if (gy < subY0 || gy >= subY1) continue;
      for (let lx = 0; lx < bw; lx++) {
        const gx = bx + lx;
        if (gx < subX0 || gx >= subX1) continue;
        if (writeMask[ly * bw + lx] > 0) {
          const cyIdx = gy - subY0;
          const cxIdx = gx - subX0;
          for (let dy = -closeR; dy <= closeR; dy++) {
            const ny = Math.min(ch - 1, Math.max(0, cyIdx + dy));
            for (let dx = -closeR; dx <= closeR; dx++) {
              if (dx * dx + dy * dy > closeR2) continue;
              const nx = Math.min(cw - 1, Math.max(0, cxIdx + dx));
              closedCropMask[ny * cw + nx] = 1;
            }
          }
        }
      }
    }

    const cropMask = new Uint8Array(cw * ch);
    const erodeR = 2;
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        if (closedCropMask[y * cw + x] === 0) continue;
        let allSet = true;
        for (let dy = -erodeR; dy <= erodeR && allSet; dy++) {
          const ny = Math.min(ch - 1, Math.max(0, y + dy));
          for (let dx = -erodeR; dx <= erodeR; dx++) {
            const nx = Math.min(cw - 1, Math.max(0, x + dx));
            if (closedCropMask[ny * cw + nx] === 0) {
              allSet = false;
              break;
            }
          }
        }
        if (allSet) {
          cropMask[y * cw + x] = 1;
        }
      }
    }
    for (let ly = 0; ly < bh; ly++) {
      const gy = by + ly;
      if (gy < subY0 || gy >= subY1) continue;
      for (let lx = 0; lx < bw; lx++) {
        const gx = bx + lx;
        if (gx < subX0 || gx >= subX1) continue;
        if (writeMask[ly * bw + lx] > 0) {
          cropMask[(gy - subY0) * cw + (gx - subX0)] = 1;
        }
      }
    }

    const S = Math.max(cw, ch, 64);
    const padTop = Math.floor((S - ch) / 2);
    const padLeft = Math.floor((S - cw) / 2);

    const modelSize = 512;
    const planeSize = modelSize * modelSize;
    const imgTensorData = new Float32Array(3 * planeSize);
    const maskTensorData = new Float32Array(planeSize);

    const scale = S / modelSize;
    for (let my = 0; my < modelSize; my++) {
      const sySquare = Math.floor((my + 0.5) * scale);
      const cyCrop = reflect101(sySquare - padTop, ch);
      const inPadY = sySquare >= padTop && sySquare < padTop + ch;

      for (let mx = 0; mx < modelSize; mx++) {
        const sxSquare = Math.floor((mx + 0.5) * scale);
        const cxCrop = reflect101(sxSquare - padLeft, cw);
        const inPadX = sxSquare >= padLeft && sxSquare < padLeft + cw;

        const gx = subX0 + cxCrop;
        const gy = subY0 + cyCrop;
        const pIdx = (gy * imgWidth + gx) * 4;
        const mVal = inPadY && inPadX ? cropMask[cyCrop * cw + cxCrop] : 0;

        const dstIdx = my * modelSize + mx;
        imgTensorData[dstIdx] = mVal ? 0 : rgba[pIdx] / 255.0;
        imgTensorData[planeSize + dstIdx] = mVal ? 0 : rgba[pIdx + 1] / 255.0;
        imgTensorData[2 * planeSize + dstIdx] = mVal ? 0 : rgba[pIdx + 2] / 255.0;
        maskTensorData[dstIdx] = mVal ? 1.0 : 0.0;
      }
    }

    const ort = await getOrt();
    const outMap = await lamaSession.run({
      image: new ort.Tensor('float32', imgTensorData, [1, 3, modelSize, modelSize]),
      mask: new ort.Tensor('float32', maskTensorData, [1, 1, modelSize, modelSize]),
    });

    const outData = outMap.output.data as Float32Array;
    let maxOut = 0;
    for (let i = 0; i < Math.min(2048, outData.length); i++) {
      if (outData[i] > maxOut) maxOut = outData[i];
    }
    const mult = maxOut <= 1.5 ? 255.0 : 1.0;

    const sampleLocal = (cxCrop: number, cyCrop: number): [number, number, number] => {
      const u = ((cxCrop + padLeft + 0.5) / S) * modelSize - 0.5;
      const v = ((cyCrop + padTop + 0.5) / S) * modelSize - 0.5;
      const x0 = Math.max(0, Math.min(modelSize - 1, Math.floor(u)));
      const y0 = Math.max(0, Math.min(modelSize - 1, Math.floor(v)));
      const x1 = Math.min(modelSize - 1, x0 + 1);
      const y1 = Math.min(modelSize - 1, y0 + 1);
      const wx = Math.max(0, Math.min(1, u - x0));
      const wy = Math.max(0, Math.min(1, v - y0));

      const i00 = y0 * modelSize + x0;
      const i10 = y0 * modelSize + x1;
      const i01 = y1 * modelSize + x0;
      const i11 = y1 * modelSize + x1;

      const w00 = (1 - wx) * (1 - wy);
      const w10 = wx * (1 - wy);
      const w01 = (1 - wx) * wy;
      const w11 = wx * wy;

      const r =
        (outData[i00] * w00 + outData[i10] * w10 + outData[i01] * w01 + outData[i11] * w11) * mult;
      const g =
        (outData[planeSize + i00] * w00 +
          outData[planeSize + i10] * w10 +
          outData[planeSize + i01] * w01 +
          outData[planeSize + i11] * w11) *
        mult;
      const b =
        (outData[2 * planeSize + i00] * w00 +
          outData[2 * planeSize + i10] * w10 +
          outData[2 * planeSize + i01] * w01 +
          outData[2 * planeSize + i11] * w11) *
        mult;
      return [r, g, b];
    };

    let biasR = 0,
      biasG = 0,
      biasB = 0,
      ringSamples = 0;
    for (let cyIdx = 2; cyIdx < ch - 2; cyIdx += 2) {
      for (let cxIdx = 2; cxIdx < cw - 2; cxIdx += 2) {
        if (cropMask[cyIdx * cw + cxIdx] !== 0) continue;
        const gx = subX0 + cxIdx;
        const gy = subY0 + cyIdx;
        const pIdx = (gy * imgWidth + gx) * 4;
        const [lr, lg, lb] = sampleGlobal(gx, gy, false);
        biasR += rgba[pIdx] - lr;
        biasG += rgba[pIdx + 1] - lg;
        biasB += rgba[pIdx + 2] - lb;
        ringSamples++;
      }
    }
    if (ringSamples > 0) {
      biasR = Math.max(-18, Math.min(18, biasR / ringSamples));
      biasG = Math.max(-18, Math.min(18, biasG / ringSamples));
      biasB = Math.max(-18, Math.min(18, biasB / ringSamples));
    }

    function sampleGlobal(
      gx: number,
      gy: number,
      withBias = true
    ): [number, number, number] {
      const [rawR, rawG, rawB] = sampleLocal(gx - subX0, gy - subY0);
      if (!withBias) return [rawR, rawG, rawB];
      return [
        Math.max(0, Math.min(255, Math.round(rawR + biasR))),
        Math.max(0, Math.min(255, Math.round(rawG + biasG))),
        Math.max(0, Math.min(255, Math.round(rawB + biasB))),
      ];
    }

    return sampleGlobal;
  };

  const pad = 40;
  const cy0 = Math.max(0, by - pad);
  const cy1 = Math.min(imgHeight, by + bh + pad);

  let samplePixelRgb: (gx: number, gy: number) => [number, number, number];

  if (bw >= 500 && bw / Math.max(1, bh) >= 2.8) {
    // Split wide subtitle regions (e.g. 812x142 "Today I'm trying Mickey D's...") into two overlapping
    // horizontal sub-crops so each sub-crop runs through 512x512 LaMa at 1:1 native pixel resolution!
    const midX = bx + Math.floor(bw / 2);
    const overlapHalf = 36;
    const leftX0 = Math.max(0, bx - pad);
    const leftX1 = Math.min(imgWidth, midX + overlapHalf + pad);
    const rightX0 = Math.max(0, midX - overlapHalf - pad);
    const rightX1 = Math.min(imgWidth, bx + bw + pad);

    const sampleLeft = await runLamaOnSubCrop(leftX0, cy0, leftX1, cy1);
    const sampleRight = await runLamaOnSubCrop(rightX0, cy0, rightX1, cy1);

    samplePixelRgb = (gx: number, gy: number): [number, number, number] => {
      if (gx <= midX - overlapHalf) return sampleLeft(gx, gy);
      if (gx >= midX + overlapHalf) return sampleRight(gx, gy);
      const t = (gx - (midX - overlapHalf)) / (2 * overlapHalf);
      const [lr, lg, lb] = sampleLeft(gx, gy);
      const [rr, rg, rb] = sampleRight(gx, gy);
      return [
        Math.round(lr * (1 - t) + rr * t),
        Math.round(lg * (1 - t) + rg * t),
        Math.round(lb * (1 - t) + rb * t),
      ];
    };
  } else {
    const cx0 = Math.max(0, bx - pad);
    const cx1 = Math.min(imgWidth, bx + bw + pad);
    const sampleFull = await runLamaOnSubCrop(cx0, cy0, cx1, cy1);
    samplePixelRgb = (gx: number, gy: number) => sampleFull(gx, gy);
  }

  const candidateRegionRgba = new Uint8Array(bw * bh * 4);

  for (let ly = 0; ly < bh; ly++) {
    const py = by + ly;
    for (let lx = 0; lx < bw; lx++) {
      const px = bx + lx;
      const srcIdx = (py * imgWidth + px) * 4;
      const rIdx = (ly * bw + lx) * 4;

      const mAlpha = writeMask[ly * bw + lx];
      if (mAlpha === 0) {
        candidateRegionRgba[rIdx] = rgba[srcIdx];
        candidateRegionRgba[rIdx + 1] = rgba[srcIdx + 1];
        candidateRegionRgba[rIdx + 2] = rgba[srcIdx + 2];
        candidateRegionRgba[rIdx + 3] = rgba[srcIdx + 3];
        continue;
      }

      const [fillR, fillG, fillB] = samplePixelRgb(px, py);

      const wLinear = mAlpha / 255.0;
      const wSmooth = wLinear * wLinear * (3.0 - 2.0 * wLinear);

      candidateRegionRgba[rIdx] = Math.round(rgba[srcIdx] * (1 - wSmooth) + fillR * wSmooth);
      candidateRegionRgba[rIdx + 1] = Math.round(
        rgba[srcIdx + 1] * (1 - wSmooth) + fillG * wSmooth
      );
      candidateRegionRgba[rIdx + 2] = Math.round(
        rgba[srcIdx + 2] * (1 - wSmooth) + fillB * wSmooth
      );
      candidateRegionRgba[rIdx + 3] = rgba[srcIdx + 3];
    }
  }

  // Commit strictly where writeMask > 0
  for (let ly = 0; ly < bh; ly++) {
    const py = by + ly;
    for (let lx = 0; lx < bw; lx++) {
      if (writeMask[ly * bw + lx] === 0) continue;
      const px = bx + lx;
      const dstIdx = (py * imgWidth + px) * 4;
      const rIdx = (ly * bw + lx) * 4;
      rgba[dstIdx] = candidateRegionRgba[rIdx];
      rgba[dstIdx + 1] = candidateRegionRgba[rIdx + 1];
      rgba[dstIdx + 2] = candidateRegionRgba[rIdx + 2];
    }
  }

  region.appliedEngine = 'LAMA_PATCH';

  // Evaluate quality over region + 16px surround ring
  const qPad = 16;
  const qx0 = Math.max(0, bx - qPad);
  const qy0 = Math.max(0, by - qPad);
  const qx1 = Math.min(imgWidth, bx + bw + qPad);
  const qy1 = Math.min(imgHeight, by + bh + qPad);
  const qw = qx1 - qx0;
  const qh = qy1 - qy0;

  const lumaBuf = new Float32Array(qw * qh);
  const intMask = new Uint8Array(qw * qh);
  const surMask = new Uint8Array(qw * qh);
  const seedExpanded = new Uint8Array(qw * qh);

  for (let y = 0; y < qh; y++) {
    for (let x = 0; x < qw; x++) {
      const gx = qx0 + x;
      const gy = qy0 + y;
      const pIdx = (gy * imgWidth + gx) * 4;
      lumaBuf[y * qw + x] = rgbToLuma(rgba[pIdx], rgba[pIdx + 1], rgba[pIdx + 2]);
      const inBox = gx >= bx && gx < bx + bw && gy >= by && gy < by + bh;
      if (inBox) {
        const mIdx = (gy - by) * bw + (gx - bx);
        if (writeMask[mIdx] > 128) intMask[y * qw + x] = 1;
        if (strokeSeedMask[mIdx] === 1) seedExpanded[y * qw + x] = 1;
      } else {
        surMask[y * qw + x] = 1;
      }
    }
  }

  const quality = evaluateCleaningQuality({
    interiorLumaAfter: lumaBuf,
    surroundLuma: lumaBuf,
    width: qw,
    height: qh,
    interiorMask: intMask,
    surroundMask: surMask,
    strokeSeedMask: seedExpanded,
    isOutlineDominated: region.surroundStats.isOutlineDominated,
    isPeriodicScreentone: region.surroundStats.isPeriodicScreentone,
    unchangedOutsideVerified: true,
  });

  return { writeMask, strokeSeedMask, quality };
}
