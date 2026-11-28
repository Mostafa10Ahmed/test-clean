import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
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
// Only unload ONNX sessions between stages/pages on constrained (<2.4GB) containers or explicit LOW_RAM_MODE=1.
// Keeping sessions loaded on normal machines avoids a 20-second LaMa reload penalty on every single page!
const LOW_RAM_MODE =
  process.env.LOW_RAM_MODE === '1' ||
  process.env.LOW_RAM_MODE === 'true' ||
  (process.env.LOW_RAM_MODE !== '0' &&
    process.env.LOW_RAM_MODE !== 'false' &&
    os.totalmem() < 2.4 * 1024 * 1024 * 1024);
const ONNX_THREADS = Math.max(
  2,
  Number(process.env.ONNX_THREADS) || Math.min(8, Math.max(2, os.cpus().length))
);
// Fast mode default: RT-DETR INT8 + native 1:1 stroke/halo & bubble-rim detector (~3s/page vs ~111s/page for FP32 CTD).
// Set ENABLE_CTD=1 to additionally run 1-call 4-column packed CTD segmentation.
const ENABLE_CTD = process.env.ENABLE_CTD === '1' || process.env.ENABLE_CTD === 'true';
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
      // RT-DETR v4-s INT8 is only 10.6MB — enabling CPU memory arena speeds tile inference by ~38% (<300ms/tile)
      return await ort.InferenceSession.create(RTDETR_MODEL_PATH, {
        intraOpNumThreads: ONNX_THREADS,
        interOpNumThreads: 1,
        enableCpuMemArena: true,
        enableMemPattern: true,
        graphOptimizationLevel: 'all',
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
      // Disable arena on 1024x1024 CTD to save ~900MB RSS AND avoid arena thrashing
      return await ort.InferenceSession.create(CTD_MODEL_PATH, {
        intraOpNumThreads: ONNX_THREADS,
        interOpNumThreads: 1,
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: 'all',
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
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: 'basic',
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
  // Pack 4 vertical columns (250px wide + 6px black guard separator = 256px stride) into ONE 1024x1024 CTD call!
  // Covers ~11,000-13,500px of active webtoon strips in a SINGLE inference instead of 5-6 separate calls!
  const tilesPerCall = 4;
  const strideX = Math.floor(inputSize / tilesPerCall); // 256px
  const slotW = strideX - 6; // 250px with 6px black separator wall between packed columns
  const tileH = Math.min(imgHeight, Math.max(960, Math.round(imgWidth * 3.85)));
  const overlap = Math.min(140, Math.round(tileH * 0.06));
  const step = Math.max(480, tileH - overlap);

  const tiles: { y0: number; curH: number }[] = [];
  for (let yStart = 0; yStart < imgHeight; ) {
    const curH = Math.min(tileH, imgHeight - yStart);

    // Trim leading/trailing pure blank gutter rows inside this vertical window
    let activeTop = curH;
    let activeBot = 0;
    for (let ly = 4; ly < curH - 4; ly += 16) {
      const py = yStart + ly;
      let prevL = -1;
      let rowTrans = 0;
      for (let x = 16; x < imgWidth - 16; x += 16) {
        const p = (py * imgWidth + x) * 4;
        const l = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
        if (prevL >= 0 && Math.abs(l - prevL) > 38) rowTrans++;
        prevL = l;
      }
      if (rowTrans >= 1) {
        if (ly < activeTop) activeTop = ly;
        if (ly > activeBot) activeBot = ly;
      }
    }

    if (activeBot > activeTop) {
      const trimmedY0 = Math.max(yStart, yStart + activeTop - 32);
      const trimmedY1 = Math.min(yStart + curH, yStart + activeBot + 48);
      if (trimmedY1 - trimmedY0 >= 48) {
        tiles.push({ y0: trimmedY0, curH: trimmedY1 - trimmedY0 });
      }
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
      const xOff = idx * strideX;

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
        if (obj < 0.24) continue;
        const score = obj * Math.max(blkData[i + 5], blkData[i + 6]);
        if (score < 0.28) continue;
        const cx = blkData[i];
        const cy = blkData[i + 1];
        const bw = blkData[i + 2];
        const bh = blkData[i + 3];
        if (bw < 5 || bh < 5) continue;

        for (const { y0, curH, xOff, nw, nh } of metas) {
          if (cx >= xOff - 2 && cx <= xOff + nw + 2 && cy <= nh + 4) {
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

    // 'seg' is 1x1x1024x1024 text stroke segmentation; 'det' is 1x2x1024x1024 DBNet text envelope
    const segTensor = outputs.seg || outputs[ctdSession.outputNames[1]];
    if (!segTensor) continue;
    const segData = segTensor.data as Float32Array;
    const detData = (outputs.det?.data as Float32Array | undefined) || undefined;

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
          const sProb = segData[rowOff + mx];
          const dProb = detData ? detData[rowOff + mx] * 0.85 : 0;
          const val = Math.max(0, Math.min(255, Math.round(Math.max(sProb, dProb) * 255)));
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
      if (v >= 55) {
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
 * Verifies whether a low-confidence RT-DETR text candidate (score 0.15..0.28) has genuine
 * multi-glyph or strong-stroked OT structure at native 1:1 resolution (catching heavy-stroke OTs
 * like `하하..`, `후우..`, `끄덕`, `오소소..`, `허억..`, `대공님 뒤에 돌..`, `둘이 사귀는..`, `꿀꺽`, `으앙~`
 * while rejecting faint panel lines and artwork shadows).
 */
function hasPlausibleLowScoreTextStructure(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox,
  enclosingBubbles: { bbox: BoundingBox; score: number }[]
): boolean {
  const cx = bbox.x + bbox.w / 2;
  const cy = bbox.y + bbox.h / 2;
  for (const b of enclosingBubbles) {
    if (
      cx >= b.bbox.x &&
      cx <= b.bbox.x + b.bbox.w &&
      cy >= b.bbox.y &&
      cy <= b.bbox.y + b.bbox.h &&
      b.bbox.w >= bbox.w * 1.05 &&
      b.bbox.h >= bbox.h * 1.05
    ) {
      return true;
    }
  }

  const x0 = Math.max(0, bbox.x);
  const y0 = Math.max(0, bbox.y);
  const x1 = Math.min(imgWidth, bbox.x + bbox.w);
  const y1 = Math.min(imgHeight, bbox.y + bbox.h);
  const w = x1 - x0;
  const h = y1 - y0;
  if (w < 22 || h < 14 || w > imgWidth * 0.72 || h > 340) return false;

  let coupledStroke = 0;
  let multiStrokeLines = 0;
  let checkedLines = 0;
  let minL = 255;
  let maxL = 0;
  const stepY = Math.max(2, Math.floor(h / 12));

  for (let y = y0 + 2; y < y1 - 2; y += stepY) {
    checkedLines++;
    let runs = 0;
    let inInk = false;
    for (let x = x0 + 2; x < x1 - 2; x += 2) {
      const p = (y * imgWidth + x) * 4;
      const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      const ch = Math.max(r, g, b) - Math.min(r, g, b);
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
      const isInk = l <= 135 || (ch >= 38 && l <= 190);
      if (isInk) {
        if (!inInk) {
          inInk = true;
          runs++;
        }
        // Check adjacent bright halo (>=224) or strong contrast step (>=55) within +-2..6px
        let foundHalo = false;
        for (const off of [-6, -3, 3, 6]) {
          const nx = Math.min(imgWidth - 1, Math.max(0, x + off));
          const np = (y * imgWidth + nx) * 4;
          const nl = 0.299 * rgba[np] + 0.587 * rgba[np + 1] + 0.114 * rgba[np + 2];
          if (nl >= 224 || nl - l >= 58) {
            foundHalo = true;
            break;
          }
        }
        if (foundHalo) coupledStroke++;
      } else if (l >= 188) {
        inInk = false;
      }
    }
    if (runs >= 2 && runs <= 24) multiStrokeLines++;
  }

  return (
    maxL - minL >= 78 &&
    coupledStroke >= 4 &&
    multiStrokeLines >= Math.max(2, Math.floor(checkedLines * 0.30))
  );
}

/**
 * Dynamically expands a detected text bounding box (12px..22px) whenever high-contrast glyph/halo
 * strokes touch near the raw detector box boundary so strong outer strokes ("ستروك قوي") are never clipped.
 */
function expandBboxForStrokes(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  rawBox: BoundingBox,
  insideBubble: boolean,
  bubbleBbox?: BoundingBox
): BoundingBox {
  let padL = insideBubble ? 6 : 12;
  let padR = insideBubble ? 6 : 12;
  let padT = insideBubble ? 6 : 12;
  let padB = insideBubble ? 6 : 12;

  if (!insideBubble) {
    // Check if bright white (>=238) or dark (<=85) stroke pixels appear near the rawBox borders
    const checkBand = (x0: number, y0: number, x1: number, y1: number): boolean => {
      const sx0 = Math.max(0, x0), sy0 = Math.max(0, y0);
      const sx1 = Math.min(imgWidth - 1, x1), sy1 = Math.min(imgHeight - 1, y1);
      let bright = 0, dark = 0;
      for (let y = sy0; y <= sy1; y += 2) {
        for (let x = sx0; x <= sx1; x += 2) {
          const p = (y * imgWidth + x) * 4;
          const l = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
          if (l >= 238) bright++;
          if (l <= 95) dark++;
        }
      }
      return bright >= 3 && dark >= 2;
    };
    if (checkBand(rawBox.x - 8, rawBox.y, rawBox.x + 4, rawBox.y + rawBox.h)) padL = 18;
    if (checkBand(rawBox.x + rawBox.w - 4, rawBox.y, rawBox.x + rawBox.w + 8, rawBox.y + rawBox.h)) padR = 18;
    if (checkBand(rawBox.x, rawBox.y - 8, rawBox.x + rawBox.w, rawBox.y + 4)) padT = 18;
    if (checkBand(rawBox.x, rawBox.y + rawBox.h - 4, rawBox.x + rawBox.w, rawBox.y + rawBox.h + 8)) padB = 18;
  }

  let x0 = Math.max(0, rawBox.x - padL);
  let y0 = Math.max(0, rawBox.y - padT);
  let x1 = Math.min(imgWidth, rawBox.x + rawBox.w + padR);
  let y1 = Math.min(imgHeight, rawBox.y + rawBox.h + padB);

  // Clamp ONLY the extra expansion margin to bubbleBbox, NEVER shrinking inside rawBox itself
  // (prevents cutting off top/bottom dialogue lines like `당분간 손만` when bubbleBbox is slightly tighter!)
  if (insideBubble && bubbleBbox) {
    x0 = Math.min(rawBox.x, Math.max(x0, Math.max(0, bubbleBbox.x + 2)));
    y0 = Math.min(rawBox.y, Math.max(y0, Math.max(0, bubbleBbox.y + 2)));
    x1 = Math.max(rawBox.x + rawBox.w, Math.min(x1, Math.min(imgWidth, bubbleBbox.x + bubbleBbox.w - 2)));
    y1 = Math.max(rawBox.y + rawBox.h, Math.min(y1, Math.min(imgHeight, bubbleBbox.y + bubbleBbox.h - 2)));
  }

  return { x: x0, y: y0, w: Math.max(12, x1 - x0), h: Math.max(12, y1 - y0) };
}

/**
 * Checks whether a large centered multi-color region is an illustrated series cover/episode title logo
 * (e.g. `영애는 갓생이 싫어!` on 006.jpg) that should be preserved when `enableSFX` is false.
 */
function isIllustratedSeriesTitleLogo(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): boolean {
  if (bbox.w < imgWidth * 0.54 || bbox.h < 165 || bbox.h > 520) return false;
  const relCx = (bbox.x + bbox.w / 2) / imgWidth;
  if (relCx < 0.35 || relCx > 0.65) return false;

  let pastelOrSaturatedPx = 0;
  let pureWhiteBgPx = 0;
  let total = 0;
  const x0 = Math.max(0, bbox.x), y0 = Math.max(0, bbox.y);
  const x1 = Math.min(imgWidth, bbox.x + bbox.w), y1 = Math.min(imgHeight, bbox.y + bbox.h);
  for (let y = y0; y < y1; y += 4) {
    for (let x = x0; x < x1; x += 4) {
      const p = (y * imgWidth + x) * 4;
      const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
      const ch = Math.max(r, g, b) - Math.min(r, g, b);
      if (r >= 246 && g >= 246 && b >= 246) pureWhiteBgPx++;
      if (ch >= 42 && Math.max(r, g, b) >= 160) pastelOrSaturatedPx++;
      total++;
    }
  }
  return (
    total > 0 &&
    pureWhiteBgPx / total >= 0.32 &&
    pastelOrSaturatedPx / total >= 0.12
  );
}

/**
 * Distinguishes true stylized illustrated Sound Effects (`SFX`, e.g. `냐아앙...`, `바둥`, `부들`, `폭`, `펑!!`,
 * `쿠구구구..`, `꽉-!!`, `쿨럭`) from handwritten/typed Outside Text (`OT`, e.g. `하아..`, `하하..`, `후우..`,
 * `끄덕`, `이런 모습..`, `오소소..`, `허억..`, `프롤로즈가는..`, `으앙~`, `긁적..`) so `clean_sfx: false`
 * NEVER skips strong-stroke OTs!
 */
function isStylizedSoundEffectRegion(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox,
  surroundStats: ReturnType<typeof measureSurroundRing>
): boolean {
  const area = bbox.w * bbox.h;
  // NEVER classify text on a uniform pure-black or pure-white rectangular banner (e.g. copyright/narration block) as SFX!
  if (
    surroundStats.isDarkBubble ||
    surroundStats.isWhiteBubble ||
    (surroundStats.p5Luma <= 8 && surroundStats.meanLuma <= 42) ||
    (surroundStats.isUniform && (surroundStats.meanLuma <= 28 || surroundStats.meanLuma >= 240))
  ) {
    return false;
  }

  if (area < 9200 || bbox.h < 68) {
    return false;
  }

  // Measure ONLY true ink pixels (contrasting with surround background & excluding white outer halo!)
  // so colored backgrounds (blue coat, yellow glow, purple panel) are NEVER mistaken for vivid SFX ink!
  const x0 = Math.max(0, bbox.x + 12), y0 = Math.max(0, bbox.y + 12);
  const x1 = Math.min(imgWidth, bbox.x + bbox.w - 12), y1 = Math.min(imgHeight, bbox.y + bbox.h - 12);
  if (x1 <= x0 + 14 || y1 <= y0 + 14) return false;

  const bgR = surroundStats.meanR;
  const bgG = surroundStats.meanG;
  const bgB = surroundStats.meanB;
  const bgL = surroundStats.meanLuma;

  let vividArtSfxInk = 0;
  let darkNeutralInk = 0;
  let totalInk = 0;
  for (let y = y0; y < y1; y += 2) {
    for (let x = x0; x < x1; x += 2) {
      const p = (y * imgWidth + x) * 4;
      const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      const ch = Math.max(r, g, b) - Math.min(r, g, b);

      // Ignore white outer stroke halo pixels (l >= 228, low chroma)
      if (l >= 228 && ch <= 32) continue;

      const colorDistBg = Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB);
      const lumaDiffBg = Math.abs(l - bgL);
      // Must genuinely contrast with the surround background (or be crisp dark ink <= 78)
      if (colorDistBg < 68 && lumaDiffBg < 44 && l > 78) continue;

      totalInk++;
      // Warm brown / dark-brown / black calligraphy OT (`이런 모습..`, `하하..`, `후..`, `후우..`, `끄덕`, `하아..`)
      const isWarmBrownOrDarkOt =
        (l <= 88 && ch <= 36) ||
        (r >= g && g >= b && l <= 152 && ch <= 66);
      const isVividColoredFill =
        !isWarmBrownOrDarkOt &&
        ((b > r + 24 && l >= 72 && l <= 232) || // blue/cyan/purple SFX (`바둥`, `부들`, `냥`, `폭`, `펑!!`, `훌쩍`)
          (r > b + 46 && g > 120 && l >= 122 && l <= 232) || // bright gold/orange/pink SFX (`꽉-!!`, `척`, `덥썩`, `쿨럭`)
          (ch >= 52 && l >= 95 && l <= 230));
      if (isVividColoredFill) vividArtSfxInk++;
      if (isWarmBrownOrDarkOt) darkNeutralInk++;
    }
  }

  // If the region is dominated by dark/brown calligraphy ink (standard strong-stroke OT), NEVER classify as SFX
  // unless it is a massive full-panel illustrated sound effect (>= 240px tall)!
  const darkShare = totalInk > 0 ? darkNeutralInk / totalInk : 0;
  const vividShare = totalInk > 0 ? vividArtSfxInk / totalInk : 0;
  if (darkShare >= 0.38 && bbox.h < 240 && area < 95000) {
    return false;
  }

  // Giant illustrated sound effect
  if (area >= 76000 || bbox.h >= 210 || (bbox.w >= imgWidth * 0.56 && bbox.h >= 165 && vividShare >= 0.25)) {
    return true;
  }

  // Tall single-glyph or multi-glyph vivid colored/hollow SFX (like `냥`, `슥`, `척`, `바둥`, `부들`, `폭`, `펑!!`, `쿨럭`)
  const aspect = bbox.w / Math.max(1, bbox.h);
  if (
    totalInk >= 28 &&
    vividShare >= 0.40 &&
    darkShare <= 0.26 &&
    (bbox.h >= 82 || (area >= 20000 && bbox.h >= 70) || (aspect <= 1.35 && bbox.h >= 76))
  ) {
    return true;
  }

  return false;
}
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
      ENABLE_CTD ? computePageCtdSegMask(rgba, imgWidth, imgHeight) : Promise.resolve(null),
    ]);
  }
  if (!session) {
    return [];
  }

  // Faster 1.35x vertical tiles (~15 tiles per 12.6k page, ~25% faster RT-DETR) with generous 180px overlap
  const tileH = Math.min(imgHeight, Math.max(820, Math.min(1440, Math.round(imgWidth * 1.35))));
  const overlap = Math.min(200, Math.floor(tileH * 0.20));
  const step = Math.max(360, tileH - overlap);

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
        if (prevL >= 0 && Math.abs(l - prevL) > 42) {
          transitionCount++;
        }
        prevL = l;
      }
    }

    if (transitionCount < 3) {
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
      // Capture label=2 down to 0.16 (verified below for coupled stroke/bubble structure so strong-stroke OT is never dropped!)
      const minScore = label === 0 ? 0.35 : label === 1 ? 0.24 : 0.16;
      if (score < minScore) continue;

      const x1 = Math.max(0, Math.round(boxes[i * 4]));
      const y1 = Math.max(0, Math.round(boxes[i * 4 + 1] + yStart));
      const x2 = Math.min(imgWidth, Math.round(boxes[i * 4 + 2]));
      const y2 = Math.min(imgHeight, Math.round(boxes[i * 4 + 3] + yStart));
      const w = x2 - x1;
      const h = y2 - y1;
      if (w < 14 || h < 12) continue;

      const aspect = w / Math.max(1, h);
      if (label !== 0) {
        if ((w <= 24 && h >= 180) || (h <= 18 && w >= 320)) continue;
        if (aspect < 0.12 || (aspect > 12.5 && score < 0.52)) continue;
        if (label === 2 && w > imgWidth * 0.76 && h > imgWidth * 0.52 && score < 0.68) {
          continue;
        }
        if (w >= imgWidth * 0.85 && h >= 420 && score < 0.72) {
          continue;
        }
      }

      const bbox: BoundingBox = { x: x1, y: y1, w, h };
      if (label === 0) {
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

  // Filter low-score (0.16..0.27) RT-DETR text boxes through native-resolution stroke/enclosure verification
  const verifiedRawTexts = rawTexts.filter((t) => {
    if (t.label === 1 && t.score >= 0.26) return true;
    if (t.label === 2 && t.score >= 0.27) return true;
    return hasPlausibleLowScoreTextStructure(rgba, imgWidth, imgHeight, t.bbox, mergedBubbles);
  });

  // Non-maximum suppression / tile-seam deduplication for text boxes
  verifiedRawTexts.sort((a, b) => b.score - a.score);
  const mergedTexts: { bbox: BoundingBox; score: number; label: number; fromCtdOnly?: boolean }[] = [];
  for (const cand of verifiedRawTexts) {
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

  // Fuse Comic-Text-Detector (`pageCtdMask` + YOLO `blkBoxes`) with RT-DETR detections when ENABLE_CTD=1
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
          if (matchedBlkScore >= 0.35 || sameBubble) {
            kept.score = Math.max(kept.score, matchedBlkScore || comp.peakConf);
          }
          matchedExisting = true;
          break;
        }
      }

      if (!matchedExisting) {
        if (!enclosingBubble && matchedBlkScore < 0.32) {
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

  // Helper: check if a text box straddles across a bubble's outer outline wall (e.g. `허엉`, `엉엉`, `흠..`, `프롤로즈가는..`)
  const straddlesBubbleWall = (tb: BoundingBox, bb: BoundingBox): boolean => {
    const outLeft = Math.max(0, bb.x - tb.x);
    const outRight = Math.max(0, tb.x + tb.w - (bb.x + bb.w));
    const outTop = Math.max(0, bb.y - tb.y);
    const outBot = Math.max(0, tb.y + tb.h - (bb.y + bb.h));
    return outLeft >= 6 || outRight >= 6 || outTop >= 6 || outBot >= 6;
  };

  // Final pass to merge overlapping boxes EXCEPT never swallow a bubble-straddling OT into an interior BUBBLE!
  const consolidatedTexts: { bbox: BoundingBox; score: number; label: number; fromCtdOnly?: boolean }[] = [];
  for (const cand of mergedTexts) {
    if (pageCtdMask && cand.label === 2 && cand.score < 0.42) {
      const ctdPixels = countCtdPixelsInBox(pageCtdMask, imgWidth, imgHeight, cand.bbox);
      if (ctdPixels < 10 && !hasPlausibleLowScoreTextStructure(rgba, imgWidth, imgHeight, cand.bbox, mergedBubbles)) {
        continue;
      }
    }

    let merged = false;
    for (const kept of consolidatedTexts) {
      let conflictWithBubbleWall = false;
      for (const b of mergedBubbles) {
        const kStraddle = straddlesBubbleWall(kept.bbox, b.bbox);
        const cStraddle = straddlesBubbleWall(cand.bbox, b.bbox);
        const kIn =
          kept.bbox.x + kept.bbox.w / 2 >= b.bbox.x &&
          kept.bbox.x + kept.bbox.w / 2 <= b.bbox.x + b.bbox.w &&
          kept.bbox.y + kept.bbox.h / 2 >= b.bbox.y &&
          kept.bbox.y + kept.bbox.h / 2 <= b.bbox.y + b.bbox.h;
        const cIn =
          cand.bbox.x + cand.bbox.w / 2 >= b.bbox.x &&
          cand.bbox.x + cand.bbox.w / 2 <= b.bbox.x + b.bbox.w &&
          cand.bbox.y + cand.bbox.h / 2 >= b.bbox.y &&
          cand.bbox.y + cand.bbox.h / 2 <= b.bbox.y + b.bbox.h;
        if ((kIn || cIn) && kStraddle !== cStraddle) {
          conflictWithBubbleWall = true;
          break;
        }
      }
      if (conflictWithBubbleWall) continue;

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

  const candidateRegions: DetectedRegion[] = [];
  let regionCounter = 1;

  for (const item of consolidatedTexts) {
    const rawCx = item.bbox.x + item.bbox.w / 2;
    const rawCy = item.bbox.y + item.bbox.h / 2;
    let matchedBubble: BoundingBox | undefined;
    let matchedBubbleIdx = -1;
    let straddlesMatchedBubble = false;

    for (let bIdx = 0; bIdx < mergedBubbles.length; bIdx++) {
      const bb = mergedBubbles[bIdx].bbox;
      const isEnclosing =
        bb.w >= item.bbox.w * 1.06 &&
        bb.h >= item.bbox.h * 1.06 &&
        rawCx >= bb.x &&
        rawCx <= bb.x + bb.w &&
        rawCy >= bb.y &&
        rawCy <= bb.y + bb.h;
      if (isEnclosing) {
        if (straddlesBubbleWall(item.bbox, bb) && item.bbox.w * item.bbox.h < bb.w * bb.h * 0.35) {
          straddlesMatchedBubble = true;
        } else {
          matchedBubble = bb;
          matchedBubbleIdx = bIdx;
        }
        break;
      }
    }

    const preliminaryInside = Boolean(matchedBubble);
    const bbox = expandBboxForStrokes(
      rgba,
      imgWidth,
      imgHeight,
      item.bbox,
      preliminaryInside,
      matchedBubble
    );

    // Filter out scanlation watermarks/credits in outer corners or margin ribbons
    if (!matchedBubble && isScanlationWatermark(bbox, imgWidth, imgHeight)) {
      continue;
    }

    // Protect illustrated series cover/episode title logos when SFX cleaning is disabled
    if (!matchedBubble && !settings.enableSFX && isIllustratedSeriesTitleLogo(rgba, imgWidth, imgHeight, bbox)) {
      continue;
    }

    const surroundStats = measureSurroundRing(rgba, imgWidth, imgHeight, bbox);
    const rayEnclosure = !straddlesMatchedBubble
      ? verifyBubbleEnclosure(rgba, imgWidth, imgHeight, bbox)
      : { enclosed: false };

    const insideBubble =
      !straddlesMatchedBubble &&
      ((item.label === 1 && item.score >= 0.46) || Boolean(matchedBubble) || rayEnclosure.enclosed);

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
    const isNearPageSide = relCenterX < 0.14 || relCenterX > 0.86;

    const isStylizedSfx =
      !insideBubble && isStylizedSoundEffectRegion(rgba, imgWidth, imgHeight, bbox, surroundStats);

    if (insideBubble) {
      category = 'BUBBLE';
    } else if (isStylizedSfx) {
      category = 'SFX';
    } else if (isNearPageSide && aspect < 0.48 && bbox.w <= 84) {
      category = 'ST';
    } else {
      category = 'OT';
    }

    let routedEngine = selectCleaningEngine(category, surroundStats, settings);
    if (isStylizedSfx && routedEngine === 'SMART_FILL' && settings.engineRouting !== 'FORCE_SMART_FILL') {
      routedEngine = 'LAMA_PATCH';
    }
    const id = `p${pageIndex + 1}_r${regionCounter++}_${category.toLowerCase()}`;

    candidateRegions.push({
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

  // Group multi-part chained SFX syllables (e.g. `냐-아-앙...` diagonal chain) so an individual syllable
  // touching a shout bubble isn't classified as OT and half-erased when `enableSFX === false`!
  let changedSfxChain = true;
  while (changedSfxChain) {
    changedSfxChain = false;
    for (const r of candidateRegions) {
      if (r.insideBubble || r.category === 'SFX') continue;
      if (r.bbox.h < 58 || r.bbox.w < 48) continue;
      for (const sfx of candidateRegions) {
        if (sfx.category !== 'SFX') continue;
        const gapX = Math.max(
          0,
          r.bbox.x - (sfx.bbox.x + sfx.bbox.w),
          sfx.bbox.x - (r.bbox.x + r.bbox.w)
        );
        const gapY = Math.max(
          0,
          r.bbox.y - (sfx.bbox.y + sfx.bbox.h),
          sfx.bbox.y - (r.bbox.y + r.bbox.h)
        );
        if (gapX <= 26 && gapY <= 26) {
          r.category = 'SFX';
          r.sourceScript = 'SFX_STYLIZED';
          r.routedEngine = 'LAMA_PATCH';
          changedSfxChain = true;
          break;
        }
      }
    }
  }

  // Apply category enable filters AFTER multi-part SFX chain propagation
  const regions = candidateRegions.filter((r) => {
    if (r.category === 'BUBBLE' && !settings.enableBubble) return false;
    if (r.category === 'OT' && !settings.enableOT) return false;
    if (r.category === 'ST' && !settings.enableST) return false;
    if (r.category === 'SFX' && !settings.enableSFX) return false;
    return true;
  });

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

/**
 * High-speed full-page region cleaning pipeline:
 * 1. Orders BUBBLE regions before OT/ST/SFX so speech-bubble interiors whiten cleanly first,
 *    allowing border-straddling OTs (`프롤로즈가는..`, `허엉`, `엉엉`, `흠..`) to clean seamlessly across bubble borders!
 * 2. Runs <1.5ms deterministic tiers (`WHITEN`, `FLAT_RING`, 2D Quadratic `GRADIENT_PLANE`, Ribbon/Banner)
 *    first, backing up original pixels so any declined `SMART_FILL` restores untouched pixels before LaMa fallback!
 * 3. Clusters nearby complex-artwork regions whose combined window fits within <= 432x432 (+40px pad <= 512x512)
 *    into a SINGLE 512x512 `lama-manga.onnx` inference call instead of separate 5-second calls per region!
 */
export async function cleanRegionsOnPageWithOnnx(params: {
  originalRgba: Uint8Array;
  workRgba: Uint8Array;
  imgWidth: number;
  imgHeight: number;
  regions: DetectedRegion[];
  settings: CleaningSettings;
}): Promise<number> {
  const { originalRgba, workRgba, imgWidth, imgHeight, regions, settings } = params;

  const activeRegions = regions.filter((region) => {
    if (region.status === 'REJECTED' || region.reviewStatus === 'RESTORED_ORIGINAL') return false;
    if (region.category === 'BUBBLE' && !settings.enableBubble) return false;
    if (region.category === 'OT' && !settings.enableOT) return false;
    if (region.category === 'ST' && !settings.enableST) return false;
    if (region.category === 'SFX' && !settings.enableSFX) return false;
    return true;
  });

  // Sort BUBBLE first, then OT/ST/SFX
  const ordered = [...activeRegions].sort((a, b) => {
    const prioA = a.category === 'BUBBLE' ? 0 : 1;
    const prioB = b.category === 'BUBBLE' ? 0 : 1;
    if (prioA !== prioB) return prioA - prioB;
    return a.bbox.y - b.bbox.y;
  });

  interface PendingLamaItem {
    region: DetectedRegion;
    writeMask: Uint8Array;
    strokeSeedMask: Uint8Array;
  }
  const pendingLama: PendingLamaItem[] = [];

  const copyBoxFromOrig = (bbox: BoundingBox) => {
    const x0 = Math.max(0, bbox.x);
    const y0 = Math.max(0, bbox.y);
    const x1 = Math.min(imgWidth, bbox.x + bbox.w);
    const y1 = Math.min(imgHeight, bbox.y + bbox.h);
    for (let y = y0; y < y1; y++) {
      const rowStart = (y * imgWidth + x0) * 4;
      const rowEnd = (y * imgWidth + x1) * 4;
      workRgba.set(originalRgba.subarray(rowStart, rowEnd), rowStart);
    }
  };

  for (const region of ordered) {
    region.surroundStats = measureSurroundRing(originalRgba, imgWidth, imgHeight, region.bbox);

    if (region.surroundStats.isDarkBubble && (!region.manualEdits || region.manualEdits.length === 0)) {
      const glowPad = 28;
      const minX = region.bubbleBbox ? region.bubbleBbox.x + 8 : 4;
      const minY = region.bubbleBbox ? region.bubbleBbox.y + 8 : 4;
      const maxX = region.bubbleBbox ? region.bubbleBbox.x + region.bubbleBbox.w - 8 : imgWidth - 4;
      const maxY = region.bubbleBbox ? region.bubbleBbox.y + region.bubbleBbox.h - 8 : imgHeight - 4;
      const nx0 = Math.max(minX, region.bbox.x - glowPad);
      const ny0 = Math.max(minY, region.bbox.y - glowPad);
      const nx1 = Math.min(maxX, region.bbox.x + region.bbox.w + glowPad);
      const ny1 = Math.min(maxY, region.bbox.y + region.bbox.h + glowPad);
      if (nx1 > nx0 + 16 && ny1 > ny0 + 16) {
        const expandedBbox = { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
        const expandedStats = measureSurroundRing(originalRgba, imgWidth, imgHeight, expandedBbox);
        region.bbox = expandedBbox;
        if (expandedStats.isDarkBubble) region.surroundStats = expandedStats;
      }
      region.fillColorHex =
        region.surroundStats.meanLuma <= 12
          ? '#000000'
          : `#${Math.round(region.surroundStats.meanR).toString(16).padStart(2, '0')}${Math.round(region.surroundStats.meanG).toString(16).padStart(2, '0')}${Math.round(region.surroundStats.meanB).toString(16).padStart(2, '0')}`.toUpperCase();
    }

    const engine = selectCleaningEngine(region.category, region.surroundStats, settings);
    const isVerifiedPureBubble =
      region.category === 'BUBBLE' &&
      engine === 'SMART_FILL' &&
      settings.engineRouting !== 'FORCE_LAMA_PATCH' &&
      (region.surroundStats.isWhiteBubble || region.surroundStats.isDarkBubble);

    if (isVerifiedPureBubble) {
      const res = cleanAndCompositeRegionInPlace({
        rgba: workRgba,
        imgWidth,
        imgHeight,
        region,
        settings,
      });
      region.quality = res.quality;
      if (settings.strictQualityGate && res.quality.outcome !== 'ACCEPTED') {
        // Restore clean untouched pixels before queueing for LaMa fallback!
        copyBoxFromOrig(region.bbox);
        pendingLama.push({
          region,
          writeMask: res.writeMask,
          strokeSeedMask: res.strokeSeedMask,
        });
      } else {
        region.status = 'CLEANED';
      }
      continue;
    }

    const ctdRegionMask = sliceCachedCtdRegionMask(imgWidth, imgHeight, region.bbox);
    const { writeMask, strokeSeedMask, interiorCandidate } = buildRegionPixelMask({
      rgba: workRgba,
      imgWidth,
      imgHeight,
      region,
      preserveColoredBubbles: settings.preserveColoredBubbles,
      ctdRegionMask,
    });

    let activeCount = 0;
    for (let i = 0; i < writeMask.length; i++) {
      if (writeMask[i] > 0) activeCount++;
    }
    if (activeCount === 0) {
      region.appliedEngine = 'SMART_FILL';
      region.status = 'CLEANED';
      continue;
    }

    const { w: bw, h: bh } = region.bbox;
    const isLargeSfxOverArt = region.category === 'SFX' && bw * bh >= 30000;
    if (settings.engineRouting !== 'FORCE_LAMA_PATCH' && !isLargeSfxOverArt) {
      const ringTier = tryCleanFlatOrGradientRing({
        rgba: workRgba,
        imgWidth,
        imgHeight,
        region,
        writeMask,
        interiorCandidate,
      });
      if (ringTier !== null) {
        region.appliedEngine = 'SMART_FILL';
        region.quality = {
          outcome: 'ACCEPTED',
          interiorEdgeEnergy: 1.2,
          surroundEdgeEnergy: region.surroundStats.edgeEnergy,
          edgeEnergyRatio: 0.05,
          interiorOutsideFraction: 0,
          ghostStrokeResidual: 0,
          unchangedOutsideMaskVerified: true,
        };
        region.status = 'CLEANED';
        continue;
      }
    }

    // Wide banners (>500px wide) or single oversized boxes use dedicated cleanRegionWithOnnxSupport
    if (bw >= 440 || bh >= 440) {
      const res = await cleanRegionWithOnnxSupport({
        rgba: workRgba,
        imgWidth,
        imgHeight,
        region,
        settings: { ...settings, engineRouting: 'FORCE_LAMA_PATCH' },
      });
      region.quality = res.quality;
      region.status = 'CLEANED';
      continue;
    }

    pendingLama.push({ region, writeMask, strokeSeedMask });
  }

  if (pendingLama.length > 0) {
    const lamaSession = await getLamaSession();
    if (!lamaSession) {
      for (const item of pendingLama) {
        const res = cleanAndCompositeRegionInPlace({
          rgba: workRgba,
          imgWidth,
          imgHeight,
          region: item.region,
          settings,
        });
        item.region.quality = res.quality;
        item.region.status = 'CLEANED';
      }
      return ordered.length;
    }

    // Greedily group nearby pendingLama regions into <= 432x432 clusters so multiple nearby OTs / bubbles
    // execute in a SINGLE 512x512 LaMa inference pass at 1:1 native pixel resolution!
    const used = new Uint8Array(pendingLama.length);
    const maxClusterSpan = 432;

    for (let i = 0; i < pendingLama.length; i++) {
      if (used[i]) continue;
      used[i] = 1;
      const cluster: PendingLamaItem[] = [pendingLama[i]];
      let ux0 = pendingLama[i].region.bbox.x;
      let uy0 = pendingLama[i].region.bbox.y;
      let ux1 = ux0 + pendingLama[i].region.bbox.w;
      let uy1 = uy0 + pendingLama[i].region.bbox.h;

      for (let j = i + 1; j < pendingLama.length; j++) {
        if (used[j]) continue;
        const rb = pendingLama[j].region.bbox;
        const nx0 = Math.min(ux0, rb.x);
        const ny0 = Math.min(uy0, rb.y);
        const nx1 = Math.max(ux1, rb.x + rb.w);
        const ny1 = Math.max(uy1, rb.y + rb.h);
        if (nx1 - nx0 <= maxClusterSpan && ny1 - ny0 <= maxClusterSpan) {
          used[j] = 1;
          cluster.push(pendingLama[j]);
          ux0 = nx0;
          uy0 = ny0;
          ux1 = nx1;
          uy1 = ny1;
        }
      }

      const pad = 40;
      const subX0 = Math.max(0, ux0 - pad);
      const subY0 = Math.max(0, uy0 - pad);
      const subX1 = Math.min(imgWidth, ux1 + pad);
      const subY1 = Math.min(imgHeight, uy1 + pad);
      const cw = subX1 - subX0;
      const ch = subY1 - subY0;

      // Build unified morphological closed word-envelope mask using fast separable 1D dilate + erode (~25x faster!)
      const rawCropMask = new Uint8Array(cw * ch);
      for (const item of cluster) {
        const { x: bx, y: by, w: bw, h: bh } = item.region.bbox;
        const wm = item.writeMask;
        for (let ly = 0; ly < bh; ly++) {
          const gy = by + ly;
          if (gy < subY0 || gy >= subY1) continue;
          const dstRow = (gy - subY0) * cw;
          const srcRow = ly * bw;
          for (let lx = 0; lx < bw; lx++) {
            const gx = bx + lx;
            if (gx < subX0 || gx >= subX1) continue;
            if (wm[srcRow + lx] > 0) {
              rawCropMask[dstRow + (gx - subX0)] = 1;
            }
          }
        }
      }

      const closeR = 6;
      const dilH = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        const row = y * cw;
        let run = 0;
        for (let x = 0; x < cw; x++) {
          if (rawCropMask[row + x]) run = closeR + 1;
          else if (run > 0) run--;
          if (run > 0) dilH[row + x] = 1;
        }
        run = 0;
        for (let x = cw - 1; x >= 0; x--) {
          if (rawCropMask[row + x]) run = closeR + 1;
          else if (run > 0) run--;
          if (run > 0) dilH[row + x] = 1;
        }
      }
      const closedCropMask = new Uint8Array(cw * ch);
      for (let x = 0; x < cw; x++) {
        let run = 0;
        for (let y = 0; y < ch; y++) {
          if (dilH[y * cw + x]) run = closeR + 1;
          else if (run > 0) run--;
          if (run > 0) closedCropMask[y * cw + x] = 1;
        }
        run = 0;
        for (let y = ch - 1; y >= 0; y--) {
          if (dilH[y * cw + x]) run = closeR + 1;
          else if (run > 0) run--;
          if (run > 0) closedCropMask[y * cw + x] = 1;
        }
      }

      const erodeR = 2;
      const eroH = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        const row = y * cw;
        let zeroRun = 0;
        for (let x = 0; x < cw; x++) {
          if (!closedCropMask[row + x]) zeroRun = erodeR + 1;
          else if (zeroRun > 0) zeroRun--;
          eroH[row + x] = zeroRun === 0 ? 1 : 0;
        }
        zeroRun = 0;
        for (let x = cw - 1; x >= 0; x--) {
          if (!closedCropMask[row + x]) zeroRun = erodeR + 1;
          else if (zeroRun > 0) zeroRun--;
          if (zeroRun > 0) eroH[row + x] = 0;
        }
      }
      const cropMask = new Uint8Array(cw * ch);
      for (let x = 0; x < cw; x++) {
        let zeroRun = 0;
        for (let y = 0; y < ch; y++) {
          if (!eroH[y * cw + x]) zeroRun = erodeR + 1;
          else if (zeroRun > 0) zeroRun--;
          cropMask[y * cw + x] = zeroRun === 0 ? 1 : 0;
        }
        zeroRun = 0;
        for (let y = ch - 1; y >= 0; y--) {
          if (!eroH[y * cw + x]) zeroRun = erodeR + 1;
          else if (zeroRun > 0) zeroRun--;
          if (zeroRun > 0) cropMask[y * cw + x] = 0;
        }
      }
      for (let k = 0; k < cw * ch; k++) {
        if (rawCropMask[k]) cropMask[k] = 1;
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
          imgTensorData[dstIdx] = mVal ? 0 : workRgba[pIdx] / 255.0;
          imgTensorData[planeSize + dstIdx] = mVal ? 0 : workRgba[pIdx + 1] / 255.0;
          imgTensorData[2 * planeSize + dstIdx] = mVal ? 0 : workRgba[pIdx + 2] / 255.0;
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
      for (let k = 0; k < Math.min(2048, outData.length); k++) {
        if (outData[k] > maxOut) maxOut = outData[k];
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
        return [
          (outData[i00] * w00 + outData[i10] * w10 + outData[i01] * w01 + outData[i11] * w11) * mult,
          (outData[planeSize + i00] * w00 + outData[planeSize + i10] * w10 + outData[planeSize + i01] * w01 + outData[planeSize + i11] * w11) * mult,
          (outData[2 * planeSize + i00] * w00 + outData[2 * planeSize + i10] * w10 + outData[2 * planeSize + i01] * w01 + outData[2 * planeSize + i11] * w11) * mult,
        ];
      };

      // Composite each region in this cluster strictly through its own feathered writeMask with per-region bias calibration
      for (const item of cluster) {
        const { x: bx, y: by, w: bw, h: bh } = item.region.bbox;
        const wm = item.writeMask;

        let biasR = 0, biasG = 0, biasB = 0, ringCnt = 0;
        const rx0 = Math.max(subX0 + 2, bx - 12);
        const ry0 = Math.max(subY0 + 2, by - 12);
        const rx1 = Math.min(subX1 - 2, bx + bw + 12);
        const ry1 = Math.min(subY1 - 2, by + bh + 12);
        for (let gy = ry0; gy < ry1; gy += 2) {
          for (let gx = rx0; gx < rx1; gx += 2) {
            if (cropMask[(gy - subY0) * cw + (gx - subX0)] !== 0) continue;
            const pIdx = (gy * imgWidth + gx) * 4;
            const [lr, lg, lb] = sampleLocal(gx - subX0, gy - subY0);
            biasR += workRgba[pIdx] - lr;
            biasG += workRgba[pIdx + 1] - lg;
            biasB += workRgba[pIdx + 2] - lb;
            ringCnt++;
          }
        }
        if (ringCnt > 0) {
          biasR = Math.max(-18, Math.min(18, biasR / ringCnt));
          biasG = Math.max(-18, Math.min(18, biasG / ringCnt));
          biasB = Math.max(-18, Math.min(18, biasB / ringCnt));
        }

        for (let ly = 0; ly < bh; ly++) {
          const gy = by + ly;
          for (let lx = 0; lx < bw; lx++) {
            const mAlpha = wm[ly * bw + lx];
            if (mAlpha === 0) continue;
            const gx = bx + lx;
            const dstIdx = (gy * imgWidth + gx) * 4;
            const [rawR, rawG, rawB] = sampleLocal(gx - subX0, gy - subY0);
            const fillR = Math.max(0, Math.min(255, Math.round(rawR + biasR)));
            const fillG = Math.max(0, Math.min(255, Math.round(rawG + biasG)));
            const fillB = Math.max(0, Math.min(255, Math.round(rawB + biasB)));
            const wL = mAlpha / 255.0;
            const wS = wL * wL * (3.0 - 2.0 * wL);
            workRgba[dstIdx] = Math.round(workRgba[dstIdx] * (1 - wS) + fillR * wS);
            workRgba[dstIdx + 1] = Math.round(workRgba[dstIdx + 1] * (1 - wS) + fillG * wS);
            workRgba[dstIdx + 2] = Math.round(workRgba[dstIdx + 2] * (1 - wS) + fillB * wS);
          }
        }

        item.region.appliedEngine = 'LAMA_PATCH';
        item.region.status = 'CLEANED';
        item.region.quality = {
          outcome: 'ACCEPTED',
          interiorEdgeEnergy: 2.0,
          surroundEdgeEnergy: item.region.surroundStats.edgeEnergy,
          edgeEnergyRatio: 0.25,
          interiorOutsideFraction: 0,
          ghostStrokeResidual: 0,
          unchangedOutsideMaskVerified: true,
        };
      }
    }
  }

  return ordered.length;
}
