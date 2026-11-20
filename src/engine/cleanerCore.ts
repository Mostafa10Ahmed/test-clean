import {
  BoundingBox,
  BubbleCleanMode,
  CleaningEngineId,
  CleaningSettings,
  DetectedRegion,
  QualityAssessment,
  RegionCategory,
  SurroundStats,
} from './types';

export const SAFE_BUBBLE_BORDER = 5;
export const UNIFORM_LIGHT_STD = 12.0;
export const UNIFORM_LIGHT_GRADIENT = 8.0;
export const UNIFORM_LIGHT_MIN_LUMA = 228.0;

export interface StripWindow {
  index: number;
  yStart: number;
  yEnd: number;
  height: number;
  coreYStart: number;
  coreYEnd: number;
}

/**
 * Computes bounded-memory sliding windows for long-strip webtoon/manhwa scans
 * (e.g., 720x14300 or 940x28483). Prevents full-page tensor OOM while preserving
 * 128px overlap across joins (Bear architecture rule §4.3 / lama.rs).
 */
export function computeStripWindows(
  imageHeight: number,
  maxWindowHeight = 2048,
  overlapPx = 128
): StripWindow[] {
  if (imageHeight <= 0) return [];
  if (imageHeight <= maxWindowHeight) {
    return [
      {
        index: 0,
        yStart: 0,
        yEnd: imageHeight,
        height: imageHeight,
        coreYStart: 0,
        coreYEnd: imageHeight,
      },
    ];
  }

  const windows: StripWindow[] = [];
  const step = Math.max(256, maxWindowHeight - overlapPx);
  let y = 0;
  let idx = 0;

  while (y < imageHeight) {
    const yEnd = Math.min(imageHeight, y + maxWindowHeight);
    const yStart = y;
    const isFirst = idx === 0;
    const isLast = yEnd === imageHeight;

    const coreYStart = isFirst ? 0 : windows[idx - 1].coreYEnd;
    const coreYEnd = isLast ? imageHeight : yEnd - Math.floor(overlapPx / 2);

    windows.push({
      index: idx,
      yStart,
      yEnd,
      height: yEnd - yStart,
      coreYStart,
      coreYEnd,
    });

    if (isLast) break;
    y += step;
    idx++;
  }

  return windows;
}

/**
 * Natural alphanumeric page sorting ("001.jpg", "002.jpg", "10.jpg") so chapter pages
 * never clean or export out of order.
 */
export function sortPageNamesNaturally(names: string[]): string[] {
  return [...names].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
  );
}

/**
 * Computes luma (0..255) from RGB channels using Rec. 601 weights.
 */
export function rgbToLuma(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Preserves scanlation watermarks/credits located in outer page corners or side margin ribbons
 * (`more249-s/dis-inpainting` `_is_scanlation_watermark`).
 */
export function isScanlationWatermark(
  bbox: BoundingBox,
  imgWidth: number,
  imgHeight: number
): boolean {
  const { x, y, w, h } = bbox;
  const area = w * h;
  const pageArea = Math.max(1, imgWidth * imgHeight);
  const relArea = area / pageArea;
  const aspect = w / Math.max(1, h);

  const touchesTopOrBottom =
    y <= Math.max(28, imgHeight * 0.015) ||
    y + h >= imgHeight - Math.max(28, imgHeight * 0.015);
  const touchesLeftOrRight =
    x <= Math.max(24, imgWidth * 0.04) ||
    x + w >= imgWidth - Math.max(24, imgWidth * 0.04);

  // Corner watermark badge
  if (touchesTopOrBottom && touchesLeftOrRight && relArea < 0.035 && w < imgWidth * 0.42) {
    return true;
  }

  // Thin vertical credit/disclaimer ribbon running tightly along the extreme left/right edge
  if (
    touchesLeftOrRight &&
    w <= Math.max(54, imgWidth * 0.075) &&
    aspect < 0.22 &&
    (x <= 16 || x + w >= imgWidth - 16)
  ) {
    return true;
  }

  return false;
}

/**
 * Erodes a binary bubble interior mask (`0` or `>0`) by `borderPx` (default `SAFE_BUBBLE_BORDER = 5`)
 * so that whitening and complex inpainting never touch or eat into the speech bubble outline.
 * Mirrors `more249-s/dis-inpainting` `_safe_bubble_interior`.
 */
export function safeBubbleInterior(
  interiorMask: Uint8Array,
  width: number,
  height: number,
  borderPx = SAFE_BUBBLE_BORDER
): Uint8Array {
  const out = new Uint8Array(width * height);
  const r2 = borderPx * borderPx;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (interiorMask[idx] === 0) continue;

      // Must be at least borderPx away from image edges and from any interiorMask === 0 pixel
      if (x < borderPx || x >= width - borderPx || y < borderPx || y >= height - borderPx) {
        continue;
      }

      let safe = true;
      const y0 = Math.max(0, y - borderPx);
      const y1 = Math.min(height - 1, y + borderPx);
      const x0 = Math.max(0, x - borderPx);
      const x1 = Math.min(width - 1, x + borderPx);

      for (let ny = y0; ny <= y1 && safe; ny++) {
        const dy = ny - y;
        for (let nx = x0; nx <= x1; nx++) {
          const dx = nx - x;
          if (dx * dx + dy * dy <= r2) {
            if (interiorMask[ny * width + nx] === 0) {
              safe = false;
              break;
            }
          }
        }
      }

      if (safe) {
        out[idx] = 255;
      }
    }
  }
  return out;
}

/**
 * Determines whether the background around a text component inside a speech bubble
 * is genuinely flat/uniform light paper (`std <= 12.0` AND directional gradient `<= 8.0`).
 * Prevents gradient speech bubbles (`190 -> 250`) from being destroyed by flat white fills.
 * Mirrors `more249-s/dis-inpainting` `_component_background_is_uniform_light`.
 */
export function componentBackgroundIsUniformLight(
  rgbaOrRgb: Uint8Array,
  compMask: Uint8Array,
  interiorMask: Uint8Array,
  width: number,
  height: number,
  channels: 3 | 4 = 4
): boolean {
  const safeInterior = safeBubbleInterior(interiorMask, width, height, SAFE_BUBBLE_BORDER);

  let minX = width,
    maxX = -1,
    minY = height,
    maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (compMask[y * width + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX) return false;

  const pad = 16;
  const x0 = Math.max(0, minX - pad);
  const x1 = Math.min(width - 1, maxX + pad);
  const y0 = Math.max(0, minY - pad);
  const y1 = Math.min(height - 1, maxY + pad);
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  const lumas: number[] = [];
  let topSum = 0,
    topCnt = 0;
  let botSum = 0,
    botCnt = 0;
  let leftSum = 0,
    leftCnt = 0;
  let rightSum = 0,
    rightCnt = 0;

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = y * width + x;
      if (compMask[idx] > 0) continue;
      if (safeInterior[idx] === 0 && interiorMask[idx] === 0) continue;

      const p = idx * channels;
      const r = rgbaOrRgb[p];
      const g = rgbaOrRgb[p + 1];
      const b = rgbaOrRgb[p + 2];
      const l = rgbToLuma(r, g, b);
      // Skip dark lettering strokes not covered by compMask
      if (l < 140) continue;

      lumas.push(l);
      if (y < midY) {
        topSum += l;
        topCnt++;
      } else {
        botSum += l;
        botCnt++;
      }
      if (x < midX) {
        leftSum += l;
        leftCnt++;
      } else {
        rightSum += l;
        rightCnt++;
      }
    }
  }

  if (lumas.length < 12) return false;

  const mean = lumas.reduce((a, b) => a + b, 0) / lumas.length;
  if (mean < UNIFORM_LIGHT_MIN_LUMA) return false;

  const variance = lumas.reduce((a, v) => a + (v - mean) * (v - mean), 0) / lumas.length;
  const std = Math.sqrt(variance);
  if (std > UNIFORM_LIGHT_STD) return false;

  const vertGrad =
    topCnt > 0 && botCnt > 0 ? Math.abs(topSum / topCnt - botSum / botCnt) : 0;
  const horizGrad =
    leftCnt > 0 && rightCnt > 0 ? Math.abs(leftSum / leftCnt - rightSum / rightCnt) : 0;

  if (Math.max(vertGrad, horizGrad) > UNIFORM_LIGHT_GRADIENT) {
    return false;
  }

  return true;
}

/**
 * Detects dark/colored rectangular RPG system panels / status windows around `textMask`
 * and returns the safe interior of the panel while keeping the panel's outer border frame `0`.
 * Mirrors `more249-s/dis-inpainting` `_find_system_panel_interior`.
 */
export function findSystemPanelInterior(
  rgbaOrRgb: Uint8Array,
  textMask: Uint8Array,
  width: number,
  height: number,
  channels: 3 | 4 = 4
): Uint8Array {
  const out = new Uint8Array(width * height);
  let minX = width,
    maxX = -1,
    minY = height,
    maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (textMask[y * width + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX) return out;

  const cx = Math.floor((minX + maxX) / 2);
  const cy = Math.floor((minY + maxY) / 2);

  // Sample median background color in and around the text region (excluding textMask pixels)
  const bgLumas: number[] = [];
  const samplePad = 14;
  const sy0 = Math.max(0, minY - samplePad);
  const sy1 = Math.min(height - 1, maxY + samplePad);
  const sx0 = Math.max(0, minX - samplePad);
  const sx1 = Math.min(width - 1, maxX + samplePad);
  for (let y = sy0; y <= sy1; y++) {
    for (let x = sx0; x <= sx1; x++) {
      const idx = y * width + x;
      if (textMask[idx] > 0) continue;
      const p = idx * channels;
      bgLumas.push(rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]));
    }
  }
  if (bgLumas.length === 0) return out;
  bgLumas.sort((a, b) => a - b);
  const panelBgLuma = bgLumas[Math.floor(bgLumas.length * 0.5)];

  // Walk left, right, up, down from text bounds to find the contrasting border frame of the system panel
  let leftWall = 0;
  for (let x = minX; x >= 1; x--) {
    const p = (cy * width + x) * channels;
    const l = rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]);
    if (Math.abs(l - panelBgLuma) > 45) {
      leftWall = x;
      break;
    }
  }

  let rightWall = width - 1;
  for (let x = maxX; x < width - 1; x++) {
    const p = (cy * width + x) * channels;
    const l = rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]);
    if (Math.abs(l - panelBgLuma) > 45) {
      rightWall = x;
      break;
    }
  }

  let topWall = 0;
  for (let y = minY; y >= 1; y--) {
    const p = (y * width + cx) * channels;
    const l = rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]);
    if (Math.abs(l - panelBgLuma) > 45) {
      topWall = y;
      break;
    }
  }

  let botWall = height - 1;
  for (let y = maxY; y < height - 1; y++) {
    const p = (y * width + cx) * channels;
    const l = rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]);
    if (Math.abs(l - panelBgLuma) > 45) {
      botWall = y;
      break;
    }
  }

  const safeMargin = 3;
  const ix0 = Math.min(minX, leftWall + safeMargin);
  const ix1 = Math.max(maxX, rightWall - safeMargin);
  const iy0 = Math.min(minY, topWall + safeMargin);
  const iy1 = Math.max(maxY, botWall - safeMargin);

  for (let y = Math.max(0, iy0); y <= Math.min(height - 1, iy1); y++) {
    for (let x = Math.max(0, ix0); x <= Math.min(width - 1, ix1); x++) {
      // Ensure we never mark the contrasting panel frame border itself
      const p = (y * width + x) * channels;
      const l = rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]);
      if (Math.abs(l - panelBgLuma) <= 42 || textMask[y * width + x] > 0) {
        out[y * width + x] = 255;
      }
    }
  }

  return out;
}

/**
 * Extracts a precise pixel-level stroke mask inside `detectorBoxMask` using local adaptive
 * background contrast + morphological bridging, avoiding filling the whole detector bounding box.
 * Mirrors `more249-s/dis-inpainting` `_extract_precise_text_mask`.
 */
export function extractPreciseTextMask(
  rgbaOrRgb: Uint8Array,
  detectorBoxMask: Uint8Array,
  width: number,
  height: number,
  cConstant = 13,
  channels: 3 | 4 = 4
): Uint8Array {
  const rawStrokes = new Uint8Array(width * height);

  let minX = width,
    maxX = -1,
    minY = height,
    maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (detectorBoxMask[y * width + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX) return rawStrokes;

  // Estimate dominant background color inside the box
  const boxLumas: number[] = [];
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (detectorBoxMask[y * width + x] === 0) continue;
      const p = (y * width + x) * channels;
      boxLumas.push(rgbToLuma(rgbaOrRgb[p], rgbaOrRgb[p + 1], rgbaOrRgb[p + 2]));
    }
  }
  boxLumas.sort((a, b) => a - b);
  // Use 70th percentile for light backgrounds or 30th for dark panels
  const medianLuma = boxLumas[Math.floor(boxLumas.length * 0.5)] ?? 255;
  const refBgLuma =
    medianLuma >= 115
      ? boxLumas[Math.floor(boxLumas.length * 0.72)] ?? 255
      : boxLumas[Math.floor(boxLumas.length * 0.28)] ?? 0;

  // Compute local 15x15 window mean luma for adaptive thresholding
  // Note: Never use global refBgLuma difference alone when background has a gradient!
  const winR = 7;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const idx = y * width + x;
      if (detectorBoxMask[idx] === 0) continue;

      const p = idx * channels;
      const r = rgbaOrRgb[p];
      const g = rgbaOrRgb[p + 1];
      const b = rgbaOrRgb[p + 2];
      const l = rgbToLuma(r, g, b);

      let sumLocal = 0;
      let maxLocal = 0;
      let minLocal = 255;
      let cntLocal = 0;
      const wy0 = Math.max(minY, y - winR);
      const wy1 = Math.min(maxY, y + winR);
      const wx0 = Math.max(minX, x - winR);
      const wx1 = Math.min(maxX, x + winR);
      for (let ny = wy0; ny <= wy1; ny += 2) {
        for (let nx = wx0; nx <= wx1; nx += 2) {
          const np = (ny * width + nx) * channels;
          const nl = rgbToLuma(rgbaOrRgb[np], rgbaOrRgb[np + 1], rgbaOrRgb[np + 2]);
          sumLocal += nl;
          if (nl > maxLocal) maxLocal = nl;
          if (nl < minLocal) minLocal = nl;
          cntLocal++;
        }
      }
      const localMean = cntLocal > 0 ? sumLocal / cntLocal : refBgLuma;
      const localBgEstimate = medianLuma >= 115 ? maxLocal : minLocal;

      // A true glyph stroke has sharp local contrast within its 15x15 neighborhood
      const isContrastStroke =
        Math.abs(l - localMean) > cConstant ||
        (Math.abs(maxLocal - minLocal) > 35 && Math.abs(l - localBgEstimate) > Math.max(18, cConstant * 1.4));

      if (isContrastStroke) {
        rawStrokes[idx] = 255;
      }
    }
  }

  // Dilate by 2px inside detectorBoxMask to capture anti-aliasing & JPEG ringing around glyphs
  const dilated = new Uint8Array(width * height);
  const dilateR = 2;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (rawStrokes[y * width + x] === 0) continue;
      for (let dy = -dilateR; dy <= dilateR; dy++) {
        const ny = y + dy;
        if (ny < minY || ny > maxY) continue;
        for (let dx = -dilateR; dx <= dilateR; dx++) {
          const nx = x + dx;
          if (nx < minX || nx > maxX) continue;
          if (dx * dx + dy * dy <= dilateR * dilateR + 1) {
            const nIdx = ny * width + nx;
            if (detectorBoxMask[nIdx] > 0) {
              dilated[nIdx] = 255;
            }
          }
        }
      }
    }
  }

  return dilated;
}

/**
 * Splits a candidate text mask into:
 * - `legacyWhiteMask`: components inside `safeBubbleInterior` with uniform light background
 * - `complexMask`: components requiring gradient/LaMa/PatchMatch inpainting, clipped away from bubble outlines
 * - `excludedMask`: mask pixels within `SAFE_BUBBLE_BORDER` (5px) of the bubble outline wall
 * Mirrors `more249-s/dis-inpainting` `_route_text_masks`.
 */
export function routeTextMasks(
  rgbaOrRgb: Uint8Array,
  mask: Uint8Array,
  interiorMask: Uint8Array,
  width: number,
  height: number,
  channels: 3 | 4 = 4
): {
  legacyWhiteMask: Uint8Array;
  complexMask: Uint8Array;
  excludedMask: Uint8Array;
} {
  const size = width * height;
  const legacyWhiteMask = new Uint8Array(size);
  const complexMask = new Uint8Array(size);
  const excludedMask = new Uint8Array(size);

  const safeInterior = safeBubbleInterior(interiorMask, width, height, SAFE_BUBBLE_BORDER);
  const isUniformLight = componentBackgroundIsUniformLight(
    rgbaOrRgb,
    mask,
    interiorMask,
    width,
    height,
    channels
  );

  for (let i = 0; i < size; i++) {
    if (mask[i] === 0) continue;

    // If inside the bubble interior but outside the 5px safe margin, exclude it from touching the border!
    if (interiorMask[i] > 0 && safeInterior[i] === 0) {
      excludedMask[i] = 255;
      continue;
    }

    if (safeInterior[i] > 0 && isUniformLight) {
      legacyWhiteMask[i] = 255;
    } else if (safeInterior[i] > 0 || interiorMask[i] === 0) {
      complexMask[i] = 255;
    }
  }

  return { legacyWhiteMask, complexMask, excludedMask };
}

/**
 * Hermite C1 smoothstep distance-transform feathering strictly inside `mask > 0`.
 * Guarantees `blended[mask == 0]` is 100% bit-identical to `source`.
 * Mirrors `more249-s/dis-inpainting` `_blend_complex_result`.
 */
export function blendComplexResult(
  source: Uint8Array,
  restored: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  channels: 3 | 4 = 4,
  featherRadius = 3.0
): Uint8Array {
  const blended = new Uint8Array(source);
  const rSearch = Math.ceil(featherRadius);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] === 0) continue; // 100% unchanged outside mask

      let minDist2 = (rSearch + 1) * (rSearch + 1);
      const y0 = Math.max(0, y - rSearch);
      const y1 = Math.min(height - 1, y + rSearch);
      const x0 = Math.max(0, x - rSearch);
      const x1 = Math.min(width - 1, x + rSearch);

      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          if (mask[ny * width + nx] === 0) {
            const d2 = (nx - x) * (nx - x) + (ny - y) * (ny - y);
            if (d2 < minDist2) minDist2 = d2;
          }
        }
      }

      const dist = Math.sqrt(minDist2);
      const t = Math.max(0.0, Math.min(1.0, dist / Math.max(1.0, featherRadius)));
      const smoothWeight = t * t * (3.0 - 2.0 * t);

      const p = idx * channels;
      for (let c = 0; c < Math.min(3, channels); c++) {
        blended[p + c] = Math.round(
          source[p + c] * (1.0 - smoothWeight) + restored[p + c] * smoothWeight
        );
      }
    }
  }

  return blended;
}

/**
 * Detects whether a candidate inpaint result introduces a new visible step seam along the mask boundary.
 * Mirrors `more249-s/dis-inpainting` `_has_new_boundary_artifact`.
 */
export function hasNewBoundaryArtifact(
  source: Uint8Array,
  candidate: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  channels: 3 | 4 = 4
): boolean {
  let jumpBeforeSum = 0;
  let jumpAfterSum = 0;
  let boundaryCount = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      if (mask[idx] === 0) continue;

      const neighbors = [idx - 1, idx + 1, idx - width, idx + width];
      for (const nIdx of neighbors) {
        if (mask[nIdx] === 0) {
          const pIn = idx * channels;
          const pOut = nIdx * channels;

          const lInBefore = rgbToLuma(source[pIn], source[pIn + 1], source[pIn + 2]);
          const lOutBefore = rgbToLuma(source[pOut], source[pOut + 1], source[pOut + 2]);
          const lInAfter = rgbToLuma(candidate[pIn], candidate[pIn + 1], candidate[pIn + 2]);
          const lOutAfter = rgbToLuma(candidate[pOut], candidate[pOut + 1], candidate[pOut + 2]);

          jumpBeforeSum += Math.abs(lInBefore - lOutBefore);
          jumpAfterSum += Math.abs(lInAfter - lOutAfter);
          boundaryCount++;
        }
      }
    }
  }

  if (boundaryCount < 6) return false;
  const meanBefore = jumpBeforeSum / boundaryCount;
  const meanAfter = jumpAfterSum / boundaryCount;

  return meanAfter > 14.0 && meanAfter - meanBefore > 9.0;
}

/**
 * Computes Sobel gradient magnitude on a single-channel luma buffer.
 */
export function computeSobelEdgeEnergy(
  luma: Float32Array,
  width: number,
  height: number,
  mask?: Uint8Array
): number {
  if (width < 3 || height < 3) return 0;
  let sum = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      if (mask) {
        // Bear quality.rs rule: sample counts only where its entire 3x3 neighborhood
        // lies on the same side of the edit boundary.
        const m = mask[idx];
        if (m === 0) continue;
        if (
          mask[(y - 1) * width + (x - 1)] !== m ||
          mask[(y - 1) * width + x] !== m ||
          mask[(y - 1) * width + (x + 1)] !== m ||
          mask[y * width + (x - 1)] !== m ||
          mask[y * width + (x + 1)] !== m ||
          mask[(y + 1) * width + (x - 1)] !== m ||
          mask[(y + 1) * width + x] !== m ||
          mask[(y + 1) * width + (x + 1)] !== m
        ) {
          continue;
        }
      }

      const tl = luma[(y - 1) * width + (x - 1)];
      const tc = luma[(y - 1) * width + x];
      const tr = luma[(y - 1) * width + (x + 1)];
      const ml = luma[y * width + (x - 1)];
      const mr = luma[y * width + (x + 1)];
      const bl = luma[(y + 1) * width + (x - 1)];
      const bc = luma[(y + 1) * width + x];
      const br = luma[(y + 1) * width + (x + 1)];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      sum += Math.hypot(gx, gy);
      count++;
    }
  }

  return count > 0 ? sum / count : 0;
}

/**
 * Intelligently routes a detected region to the lightest capable cleaning engine
 * following Bear's engine-routing ladder:
 * - Engine 1: SMART_FILL for flat white/dark/uniform speech bubbles and planar backgrounds
 * - Engine 2: LAMA_PATCH for screentones, halftones, hatching, gradients, and OT/ST over line art
 * - Engine 3: GEMINI_AI for large SFX overlapping complex multi-color scenes (when cloud GPU/AI is enabled)
 */
export function selectCleaningEngine(
  category: RegionCategory,
  surround: SurroundStats,
  settings: CleaningSettings
): CleaningEngineId {
  if (settings.engineRouting === 'FORCE_SMART_FILL') return 'SMART_FILL';
  if (settings.engineRouting === 'FORCE_LAMA_PATCH') return 'LAMA_PATCH';

  if (category === 'BUBBLE') {
    if (
      !surround.isGradient &&
      (surround.isUniform || surround.isWhiteBubble || surround.isDarkBubble)
    ) {
      return 'SMART_FILL';
    }
    return 'LAMA_PATCH';
  }

  if (category === 'ST' || category === 'OT') {
    if (
      surround.isUniform &&
      surround.stdLuma < 8.5 &&
      !surround.isPeriodicScreentone &&
      !surround.isGradient
    ) {
      return 'SMART_FILL';
    }
    return 'LAMA_PATCH';
  }

  // SFX category
  if (surround.isUniform && surround.stdLuma < 6.0 && !surround.isGradient) {
    return 'SMART_FILL';
  }
  if (settings.engineRouting === 'ALLOW_CLOUD_AI' && settings.cloudGpuConsent) {
    return 'GEMINI_AI';
  }
  return 'LAMA_PATCH';
}

/**
 * Evaluates post-clean quality with Bear's two-part surround prior metric PLUS
 * our fixes for:
 * 1) Outline-dominated surround regions (Bear quality.rs bug where small masks near thick bubble outlines
 *    or 20px screentone pitch falsely declined at ratio 5.20).
 * 2) Ghost text residual detection inside the original stroke coordinates.
 */
export function evaluateCleaningQuality(params: {
  interiorLumaAfter: Float32Array;
  surroundLuma: Float32Array;
  width: number;
  height: number;
  interiorMask: Uint8Array;
  surroundMask: Uint8Array;
  strokeSeedMask: Uint8Array;
  isOutlineDominated: boolean;
  isPeriodicScreentone: boolean;
  unchangedOutsideVerified: boolean;
}): QualityAssessment {
  const {
    interiorLumaAfter,
    surroundLuma,
    width,
    height,
    interiorMask,
    surroundMask,
    strokeSeedMask,
    isOutlineDominated,
    isPeriodicScreentone,
    unchangedOutsideVerified,
  } = params;

  const interiorEdgeEnergy = computeSobelEdgeEnergy(interiorLumaAfter, width, height, interiorMask);
  const rawSurroundEdgeEnergy = computeSobelEdgeEnergy(surroundLuma, width, height, surroundMask);

  const quietFloor = 6.51;
  const calibratedFloor = isOutlineDominated
    ? Math.max(quietFloor, rawSurroundEdgeEnergy * 0.65)
    : isPeriodicScreentone
      ? Math.max(quietFloor, rawSurroundEdgeEnergy * 1.35)
      : Math.max(quietFloor, rawSurroundEdgeEnergy);

  const edgeEnergyRatio = interiorEdgeEnergy / calibratedFloor;

  // Histogram p5..p95 check
  const surroundValues: number[] = [];
  for (let i = 0; i < surroundLuma.length; i++) {
    if (surroundMask[i]) {
      if (isOutlineDominated && surroundLuma[i] < 25) continue;
      surroundValues.push(surroundLuma[i]);
    }
  }

  if (surroundValues.length < 8) {
    return {
      outcome: 'UNMEASURED',
      interiorEdgeEnergy,
      surroundEdgeEnergy: rawSurroundEdgeEnergy,
      edgeEnergyRatio,
      interiorOutsideFraction: 0,
      ghostStrokeResidual: 0,
      unchangedOutsideMaskVerified: unchangedOutsideVerified,
    };
  }

  surroundValues.sort((a, b) => a - b);
  const p5 = surroundValues[Math.floor(surroundValues.length * 0.05)];
  const p95 = surroundValues[Math.floor(surroundValues.length * 0.95)];
  const tolerance = 14.0;
  const lowBound = Math.max(0, p5 - tolerance);
  const highBound = Math.min(255, p95 + tolerance);

  let interiorCount = 0;
  let outsideRangeCount = 0;
  for (let i = 0; i < interiorLumaAfter.length; i++) {
    if (interiorMask[i]) {
      interiorCount++;
      const v = interiorLumaAfter[i];
      if (v < lowBound || v > highBound) {
        outsideRangeCount++;
      }
    }
  }

  const interiorOutsideFraction = interiorCount > 0 ? outsideRangeCount / interiorCount : 0;

  let strokeSum = 0;
  let strokeCount = 0;
  let bgSum = 0;
  let bgCount = 0;
  for (let i = 0; i < interiorLumaAfter.length; i++) {
    if (interiorMask[i]) {
      if (strokeSeedMask[i]) {
        strokeSum += interiorLumaAfter[i];
        strokeCount++;
      } else {
        bgSum += interiorLumaAfter[i];
        bgCount++;
      }
    }
  }
  const ghostStrokeResidual =
    strokeCount > 4 && bgCount > 4
      ? Math.abs(strokeSum / strokeCount - bgSum / bgCount)
      : 0;

  if (ghostStrokeResidual > 18.0) {
    return {
      outcome: 'DECLINED_GHOST_TEXT',
      interiorEdgeEnergy,
      surroundEdgeEnergy: rawSurroundEdgeEnergy,
      edgeEnergyRatio,
      interiorOutsideFraction,
      ghostStrokeResidual,
      unchangedOutsideMaskVerified: unchangedOutsideVerified,
    };
  }

  if (edgeEnergyRatio > 2.0 && interiorEdgeEnergy > 14.0) {
    return {
      outcome: 'DECLINED_EDGE_ENERGY',
      interiorEdgeEnergy,
      surroundEdgeEnergy: rawSurroundEdgeEnergy,
      edgeEnergyRatio,
      interiorOutsideFraction,
      ghostStrokeResidual,
      unchangedOutsideMaskVerified: unchangedOutsideVerified,
    };
  }

  if (interiorOutsideFraction > 0.5) {
    return {
      outcome: 'DECLINED_HISTOGRAM',
      interiorEdgeEnergy,
      surroundEdgeEnergy: rawSurroundEdgeEnergy,
      edgeEnergyRatio,
      interiorOutsideFraction,
      ghostStrokeResidual,
      unchangedOutsideMaskVerified: unchangedOutsideVerified,
    };
  }

  return {
    outcome: 'ACCEPTED',
    interiorEdgeEnergy,
    surroundEdgeEnergy: rawSurroundEdgeEnergy,
    edgeEnergyRatio,
    interiorOutsideFraction,
    ghostStrokeResidual,
    unchangedOutsideMaskVerified: unchangedOutsideVerified,
  };
}

/**
 * Generates an accurate pixel-level write mask inside a region bounding box (`w x h`).
 *
 * Uses `extractPreciseTextMask` and `safeBubbleInterior` so:
 * 1. Speech Bubble Mode 1 (`WHITEN`) on verified uniform white bubbles floods the connected
 *    bubble paper interior while eroding by `SAFE_BUBBLE_BORDER = 5px` from any dark bubble outline,
 *    never painting a synthetic ellipse over bubble borders!
 * 2. Gradient bubbles, colored bubbles, dark system boxes, OT, ST, and SFX extract precise
 *    glyph stroke masks dilated by `paddingPx + 3px` while stopping at strong outer borders.
 */
export function buildRegionPixelMask(params: {
  rgba: Uint8Array;
  imgWidth: number;
  imgHeight: number;
  region: DetectedRegion;
  preserveColoredBubbles: boolean;
  ctdRegionMask?: Uint8Array;
}): {
  writeMask: Uint8Array; // 0..255 alpha feather mask of size bbox.w * bbox.h
  strokeSeedMask: Uint8Array; // binary 0/1 mask of raw lettering strokes
  interiorCandidate: Uint8Array; // binary 0/255 safe interior mask
} {
  const { rgba, imgWidth, imgHeight, region, preserveColoredBubbles, ctdRegionMask } = params;
  const { x: bx, y: by, w: bw, h: bh } = region.bbox;
  const size = bw * bh;
  const writeMask = new Uint8Array(size);
  const strokeSeedMask = new Uint8Array(size);

  const bgR = region.surroundStats.meanR;
  const bgG = region.surroundStats.meanG;
  const bgB = region.surroundStats.meanB;
  const bgLuma = region.surroundStats.meanLuma;

  const isDarkBubble = region.surroundStats.isDarkBubble;

  const isColoredBubble =
    !isDarkBubble &&
    (Math.abs(bgR - bgG) > 10 ||
      Math.abs(bgG - bgB) > 10 ||
      Math.abs(bgR - bgB) > 10 ||
      region.surroundStats.isGradient ||
      bgLuma < 250);

  const effectiveBubbleMode: BubbleCleanMode =
    region.category === 'BUBBLE' &&
    region.bubbleCleanMode === 'WHITEN' &&
    ((region.surroundStats.isWhiteBubble &&
      !region.surroundStats.isGradient &&
      !(preserveColoredBubbles && isColoredBubble)) ||
      isDarkBubble)
      ? 'WHITEN'
      : 'TEXT_ONLY';

  const cropRgba = new Uint8Array(size * 4);
  const cropLuma = new Float32Array(size);
  const detectorBoxMask = new Uint8Array(size).fill(255);
  const isExteriorWallCandidate = new Uint8Array(size);

  for (let ly = 0; ly < bh; ly++) {
    const py = Math.min(imgHeight - 1, Math.max(0, by + ly));
    for (let lx = 0; lx < bw; lx++) {
      const px = Math.min(imgWidth - 1, Math.max(0, bx + lx));
      const srcIdx = (py * imgWidth + px) * 4;
      const idx = ly * bw + lx;
      const dstIdx = idx * 4;
      const r = rgba[srcIdx];
      const g = rgba[srcIdx + 1];
      const b = rgba[srcIdx + 2];
      cropRgba[dstIdx] = r;
      cropRgba[dstIdx + 1] = g;
      cropRgba[dstIdx + 2] = b;
      cropRgba[dstIdx + 3] = rgba[srcIdx + 3];

      const luma = rgbToLuma(r, g, b);
      cropLuma[idx] = luma;
      const chroma = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));

      if (ctdRegionMask && ctdRegionMask[idx] >= 40) {
        isExteriorWallCandidate[idx] = 0;
      } else if (region.category === 'BUBBLE') {
        if (
          isDarkBubble
            ? luma > 95
            : luma < Math.min(218, bgLuma - 28) ||
              (effectiveBubbleMode === 'WHITEN' && chroma > 16)
        ) {
          isExteriorWallCandidate[idx] = 1;
        }
      } else {
        if (luma < Math.min(155, bgLuma - 45)) {
          isExteriorWallCandidate[idx] = 1;
        }
      }
    }
  }

  // Topological Flood-Fill from the outer 1px perimeter of the box along exterior wall candidates,
  // keeping continuous wall/border components (>= 28 pixels) so curved bubble borders and exterior artwork
  // entering the bounding box are completely captured.
  const exteriorConnectedWall = new Uint8Array(size);
  const visitedWall = new Uint8Array(size);

  for (let ly = 0; ly < bh; ly++) {
    for (let lx = 0; lx < bw; lx++) {
      if (lx !== 0 && lx !== bw - 1 && ly !== 0 && ly !== bh - 1) continue;
      const startIdx = ly * bw + lx;
      if (isExteriorWallCandidate[startIdx] === 0 || visitedWall[startIdx] === 1) continue;

      const comp: number[] = [];
      const q: number[] = [startIdx];
      visitedWall[startIdx] = 1;

      while (q.length > 0) {
        const curr = q.pop()!;
        comp.push(curr);
        const cy = Math.floor(curr / bw);
        const cx = curr % bw;
        if (effectiveBubbleMode !== 'WHITEN') {
          const edgeDist = Math.min(cx, bw - 1 - cx, cy, bh - 1 - cy);
          if (edgeDist > 5) continue;
        }

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= bw) continue;
            const nIdx = ny * bw + nx;
            if (visitedWall[nIdx] === 0 && isExteriorWallCandidate[nIdx] === 1) {
              visitedWall[nIdx] = 1;
              q.push(nIdx);
            }
          }
        }
      }

      if (comp.length >= 28) {
        for (const cIdx of comp) {
          exteriorConnectedWall[cIdx] = 1;
        }
      }
    }
  }

  // For WHITEN pure-white/pure-black speech bubbles, any corner/side of the bounding box that lies on
  // the OUTSIDE of a curved bubble border (separated from the bubble's central canvas by exteriorConnectedWall)
  // must also be excluded so outside artwork or page gutters are never whitened!
  if (effectiveBubbleMode === 'WHITEN') {
    const reachableInterior = new Uint8Array(size);
    const iq: number[] = [];
    const xMin = Math.floor(bw * 0.2);
    const xMax = Math.ceil(bw * 0.8);
    const yMin = Math.floor(bh * 0.2);
    const yMax = Math.ceil(bh * 0.8);

    for (let ly = yMin; ly <= yMax; ly++) {
      for (let lx = xMin; lx <= xMax; lx++) {
        const idx = ly * bw + lx;
        if (exteriorConnectedWall[idx] === 0 && isExteriorWallCandidate[idx] === 0) {
          reachableInterior[idx] = 1;
          iq.push(idx);
        }
      }
    }

    let head = 0;
    while (head < iq.length) {
      const curr = iq[head++];
      const cy = Math.floor(curr / bw);
      const cx = curr % bw;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= bh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx;
          if (nx < 0 || nx >= bw) continue;
          const nIdx = ny * bw + nx;
          if (reachableInterior[nIdx] === 0 && exteriorConnectedWall[nIdx] === 0) {
            reachableInterior[nIdx] = 1;
            iq.push(nIdx);
          }
        }
      }
    }

    if (iq.length >= size * 0.18) {
      for (let i = 0; i < size; i++) {
        if (reachableInterior[i] === 0) {
          exteriorConnectedWall[i] = 1;
        }
      }
    }
  }

  // Build interiorCandidate: 255 everywhere except exteriorConnectedWall + 3px safety margin
  const interiorCandidate = new Uint8Array(size).fill(255);
  for (let ly = 0; ly < bh; ly++) {
    for (let lx = 0; lx < bw; lx++) {
      if (exteriorConnectedWall[ly * bw + lx] === 1) {
        for (let dy = -3; dy <= 3; dy++) {
          const ny = ly + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -3; dx <= 3; dx++) {
            const nx = lx + dx;
            if (nx < 0 || nx >= bw) continue;
            if (dx * dx + dy * dy <= 9) {
              interiorCandidate[ny * bw + nx] = 0;
            }
          }
        }
      }
    }
  }

  // Estimate per-scanline clean background color (rowBgR, rowBgG, rowBgB, rowBgLuma)
  const rowBgR = new Float32Array(bh);
  const rowBgG = new Float32Array(bh);
  const rowBgB = new Float32Array(bh);
  const rowBgLuma = new Float32Array(bh);

  for (let ly = 0; ly < bh; ly++) {
    const rowSamples: { r: number; g: number; b: number; l: number }[] = [];
    const y0 = Math.max(0, ly - 2);
    const y1 = Math.min(bh - 1, ly + 2);
    for (let ny = y0; ny <= y1; ny++) {
      for (let lx = 0; lx < bw; lx += 2) {
        const idx = ny * bw + lx;
        if (interiorCandidate[idx] === 0) continue;
        if (ctdRegionMask && ctdRegionMask[idx] >= 45) continue;
        const p = idx * 4;
        rowSamples.push({
          r: cropRgba[p],
          g: cropRgba[p + 1],
          b: cropRgba[p + 2],
          l: cropLuma[idx],
        });
      }
    }
    if (rowSamples.length === 0) {
      rowBgR[ly] = bgR;
      rowBgG[ly] = bgG;
      rowBgB[ly] = bgB;
      rowBgLuma[ly] = bgLuma;
    } else {
      rowSamples.sort((a, b) => a.l - b.l);
      const pickIdx =
        bgLuma >= 115
          ? Math.floor(rowSamples.length * 0.76)
          : Math.floor(rowSamples.length * 0.24);
      const chosen = rowSamples[Math.min(rowSamples.length - 1, Math.max(0, pickIdx))];
      rowBgR[ly] = chosen.r;
      rowBgG[ly] = chosen.g;
      rowBgB[ly] = chosen.b;
      rowBgLuma[ly] = chosen.l;
    }
  }

  const preciseMask = extractPreciseTextMask(cropRgba, detectorBoxMask, bw, bh, 12, 4);
  const dilated = new Uint8Array(size);

  // Check if neural comictextdetector segmentation mask has active glyph pixels inside this region
  let ctdActiveCount = 0;
  if (ctdRegionMask && ctdRegionMask.length === size) {
    for (let i = 0; i < size; i++) {
      if (interiorCandidate[i] > 0 && ctdRegionMask[i] >= 55) {
        ctdActiveCount++;
      }
    }
  }

  const isSmoothBubbleOrPanel =
    effectiveBubbleMode !== 'WHITEN' &&
    region.surroundStats.stdLuma <= 21.0 &&
    !region.surroundStats.isPeriodicScreentone &&
    region.category !== 'SFX';

  if (effectiveBubbleMode === 'WHITEN') {
    if (!isDarkBubble) {
      // Verified Pure White Bubble (#FFFFFF): mask all non-255 pixels inside interiorCandidate
      for (let i = 0; i < size; i++) {
        if (interiorCandidate[i] === 0) continue;
        const p = i * 4;
        if (
          cropRgba[p] < 254 ||
          cropRgba[p + 1] < 254 ||
          cropRgba[p + 2] < 254 ||
          cropLuma[i] < 254 ||
          preciseMask[i] > 0 ||
          (ctdRegionMask && ctdRegionMask[i] >= 50)
        ) {
          strokeSeedMask[i] = 1;
          dilated[i] = 1;
        }
      }
    } else {
      // Verified Flat Dark Bubble (#000000): seed from bright text core and flood-fill outward up to 28px
      // across any glow pixel brighter than the dark bubble background
      const q: number[] = [];
      const distMap = new Int16Array(size).fill(-1);
      for (let i = 0; i < size; i++) {
        if (interiorCandidate[i] === 0) continue;
        if (
          cropLuma[i] > bgLuma + 32 ||
          preciseMask[i] > 0 ||
          (ctdRegionMask && ctdRegionMask[i] >= 50)
        ) {
          strokeSeedMask[i] = 1;
          dilated[i] = 1;
          distMap[i] = 0;
          q.push(i);
        }
      }
      let head = 0;
      while (head < q.length) {
        const curr = q[head++];
        const d = distMap[curr];
        if (d >= 38) continue;
        const cy = Math.floor(curr / bw);
        const cx = curr % bw;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= bw) continue;
            const nIdx = ny * bw + nx;
            if (distMap[nIdx] !== -1 || interiorCandidate[nIdx] === 0) continue;
            const np = nIdx * 4;
            if (
              cropRgba[np] > bgR + 1 ||
              cropRgba[np + 1] > bgG + 1 ||
              cropRgba[np + 2] > bgB + 1 ||
              cropLuma[nIdx] > bgLuma + 1
            ) {
              distMap[nIdx] = d + 1;
              strokeSeedMask[nIdx] = 1;
              dilated[nIdx] = 1;
              q.push(nIdx);
            }
          }
        }
      }
    }
  } else if (ctdActiveCount >= 14 && ctdRegionMask) {
    // Neural Pixel-Level Text Segmentation Mask (comictextdetector.pt.onnx) available!
    // 1. Seed from ctdRegionMask >= 50 where the pixel contrasts with local scanline background
    //    or is a high-confidence glyph pixel, dilated by r=3 (r2 <= 8) so thin 1..3px interior gaps
    //    between double-outlined strokes (e.g. Today I'm trying Mickey D's...) are 100% bridged!
    const ctdCore = new Uint8Array(size);
    for (let ly = 0; ly < bh; ly++) {
      const lBg = rowBgLuma[ly];
      const rBg = rowBgR[ly],
        gBg = rowBgG[ly],
        bBg = rowBgB[ly];
      for (let lx = 0; lx < bw; lx++) {
        const idx = ly * bw + lx;
        if (interiorCandidate[idx] === 0 || ctdRegionMask[idx] < 50) continue;
        const p = idx * 4;
        const diffBg = Math.max(
          Math.abs(cropRgba[p] - rBg),
          Math.abs(cropRgba[p + 1] - gBg),
          Math.abs(cropRgba[p + 2] - bBg),
          Math.abs(cropLuma[idx] - lBg)
        );
        const isActualGlyphPixel =
          diffBg > 14 ||
          preciseMask[idx] > 0 ||
          cropLuma[idx] <= 80 ||
          cropLuma[idx] >= 210 ||
          ctdRegionMask[idx] >= 165;
        if (!isActualGlyphPixel) continue;

        for (let dy = -3; dy <= 3; dy++) {
          const ny = ly + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -3; dx <= 3; dx++) {
            const nx = lx + dx;
            if (nx < 0 || nx >= bw) continue;
            if (dx * dx + dy * dy <= 8 && interiorCandidate[ny * bw + nx] > 0) {
              ctdCore[ny * bw + nx] = 1;
              strokeSeedMask[ny * bw + nx] = 1;
            }
          }
        }
      }
    }

    // 2. Detect if this region contains double-outlined lettering (coupled black <= 78 and white >= 205
    //    within 7px, e.g. Today I'm trying Mickey D's... or 움찔) and include all coupled black+white
    //    pixels within 11px of ctdCore, dilated by r=3 so both the black border and white core are 100% opaque!
    let coupledPairsInCtd = 0;
    for (let ly = 2; ly < bh - 2; ly += 2) {
      for (let lx = 2; lx < bw - 2; lx += 2) {
        const idx = ly * bw + lx;
        if (interiorCandidate[idx] === 0 || cropLuma[idx] < 205) continue;
        let nearDark = false;
        for (let dy = -5; dy <= 5 && !nearDark; dy += 2) {
          const ny = Math.min(bh - 1, Math.max(0, ly + dy));
          for (let dx = -5; dx <= 5; dx += 2) {
            const nx = Math.min(bw - 1, Math.max(0, lx + dx));
            if (cropLuma[ny * bw + nx] <= 78) {
              nearDark = true;
              break;
            }
          }
        }
        if (nearDark) coupledPairsInCtd++;
      }
    }

    const seedExpand = new Uint8Array(size);
    const searchRad = coupledPairsInCtd >= 18 ? 11 : 6;
    const expandDilateR = coupledPairsInCtd >= 18 ? 3 : 2;
    const expandR2 = expandDilateR * expandDilateR + 1;

    for (let ly = 0; ly < bh; ly++) {
      for (let lx = 0; lx < bw; lx++) {
        const idx = ly * bw + lx;
        if (interiorCandidate[idx] === 0 || strokeSeedMask[idx] === 1) continue;
        const p = idx * 4;
        const r = cropRgba[p],
          g = cropRgba[p + 1],
          b = cropRgba[p + 2];
        const l = cropLuma[idx];
        const chroma = Math.max(r, g, b) - Math.min(r, g, b);

        let isGlyphStrokeOrOutline =
          (preciseMask[idx] > 0 && (l <= 88 || (l >= 200 && chroma <= 42))) ||
          (ctdRegionMask[idx] >= 25 && l >= 185 && g >= 145 && b >= 135);

        // For double-outlined subtitles over complex backgrounds, capture every coupled black (<=88) or white (>=195) pixel
        // even when preciseMask missed it due to high local window variance!
        if (!isGlyphStrokeOrOutline && coupledPairsInCtd >= 18 && (l <= 88 || (l >= 195 && chroma <= 42))) {
          const isB = l >= 195;
          for (let dy = -6; dy <= 6 && !isGlyphStrokeOrOutline; dy += 2) {
            const ny = Math.min(bh - 1, Math.max(0, ly + dy));
            for (let dx = -6; dx <= 6; dx += 2) {
              const nx = Math.min(bw - 1, Math.max(0, lx + dx));
              const nl = cropLuma[ny * bw + nx];
              if ((isB && nl <= 82) || (!isB && nl >= 200)) {
                isGlyphStrokeOrOutline = true;
                break;
              }
            }
          }
        }

        if (!isGlyphStrokeOrOutline) continue;

        let nearCtd = false;
        const y0 = Math.max(0, ly - searchRad);
        const y1 = Math.min(bh - 1, ly + searchRad);
        const x0 = Math.max(0, lx - searchRad);
        const x1 = Math.min(bw - 1, lx + searchRad);
        for (let ny = y0; ny <= y1 && !nearCtd; ny += 2) {
          for (let nx = x0; nx <= x1; nx += 2) {
            if (ctdCore[ny * bw + nx] === 1) {
              nearCtd = true;
              break;
            }
          }
        }
        if (nearCtd) {
          for (let dy = -expandDilateR; dy <= expandDilateR; dy++) {
            const ny = ly + dy;
            if (ny < 0 || ny >= bh) continue;
            for (let dx = -expandDilateR; dx <= expandDilateR; dx++) {
              const nx = lx + dx;
              if (nx < 0 || nx >= bw) continue;
              if (dx * dx + dy * dy <= expandR2 && interiorCandidate[ny * bw + nx] > 0) {
                seedExpand[ny * bw + nx] = 1;
              }
            }
          }
        }
      }
    }
    for (let i = 0; i < size; i++) {
      if (seedExpand[i] === 1) strokeSeedMask[i] = 1;
    }

    // 3. Grow up to 4px outward along attached black/white outlines, 3D grey extruded shadows (크라쿠프 뉴스),
    // or neon glow over dark speedlines (계획대로야!), while never flooding dark translucent panels!
    const q: number[] = [];
    const distMap = new Int16Array(size).fill(-1);
    for (let i = 0; i < size; i++) {
      if (strokeSeedMask[i] === 1) {
        distMap[i] = 0;
        q.push(i);
      }
    }
    const maxGrow = 6;
    let head = 0;
    while (head < q.length) {
      const curr = q[head++];
      const d = distMap[curr];
      if (d >= maxGrow) continue;
      const cy = Math.floor(curr / bw);
      const cx = curr % bw;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= bh) continue;
        const lBg = rowBgLuma[ny];
        const bgRRow = rowBgR[ny];
        const bgGRow = rowBgG[ny];
        const bgBRow = rowBgB[ny];
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx;
          if (nx < 0 || nx >= bw) continue;
          const nIdx = ny * bw + nx;
          if (distMap[nIdx] !== -1 || interiorCandidate[nIdx] === 0) continue;
          const np = nIdx * 4;
          const r = cropRgba[np],
            g = cropRgba[np + 1],
            b = cropRgba[np + 2];
          const l = cropLuma[nIdx];
          const chroma = Math.max(r, g, b) - Math.min(r, g, b);
          const colorDistFromRowBg =
            Math.abs(r - bgRRow) + Math.abs(g - bgGRow) + Math.abs(b - bgBRow);
          const isAttachedOutlineOrShadow =
            (bgLuma >= 95 &&
              (l <= 78 ||
                (l >= 230 && chroma <= 36) ||
                (bgLuma >= 180 && l <= lBg - 32 && chroma <= 28) ||
                (bgLuma >= 145 && region.surroundStats.stdLuma <= 26 && colorDistFromRowBg >= 48))) ||
            (bgLuma < 95 && (l >= Math.max(115, lBg + 38) || (chroma > 38 && l > 65)));
          if (isAttachedOutlineOrShadow) {
            distMap[nIdx] = d + 1;
            strokeSeedMask[nIdx] = 1;
            q.push(nIdx);
          }
        }
      }
    }

    // 4. Morphological closing + outer outline halo capture on strokeSeedMask:
    //    For double-outlined subtitles (coupledPairsInCtd >= 18, e.g. Today I'm trying Mickey D's...),
    //    capture every dark/anti-aliased pixel (l <= 130) within 3px of the coupled letters, then
    //    apply a 2D morphological closing (dilate r=5, erode r=2 -> +3px solid expansion) directly
    //    on strokeSeedMask so 100% of the letter interior and black border are opaque (writeMask = 255)!
    if (coupledPairsInCtd >= 18) {
      const closedTemp = new Uint8Array(size);
      const cRad = 5;
      const cRad2 = cRad * cRad + 1;
      for (let ly = 0; ly < bh; ly++) {
        for (let lx = 0; lx < bw; lx++) {
          if (strokeSeedMask[ly * bw + lx] === 0) continue;
          for (let dy = -cRad; dy <= cRad; dy++) {
            const ny = ly + dy;
            if (ny < 0 || ny >= bh) continue;
            for (let dx = -cRad; dx <= cRad; dx++) {
              const nx = lx + dx;
              if (nx < 0 || nx >= bw) continue;
              if (dx * dx + dy * dy <= cRad2 && interiorCandidate[ny * bw + nx] > 0) {
                closedTemp[ny * bw + nx] = 1;
              }
            }
          }
        }
      }
      const eRad = 2;
      for (let ly = 0; ly < bh; ly++) {
        for (let lx = 0; lx < bw; lx++) {
          const idx = ly * bw + lx;
          if (closedTemp[idx] === 0 || interiorCandidate[idx] === 0) continue;
          let allSet = true;
          for (let dy = -eRad; dy <= eRad && allSet; dy++) {
            const ny = Math.min(bh - 1, Math.max(0, ly + dy));
            for (let dx = -eRad; dx <= eRad; dx++) {
              const nx = Math.min(bw - 1, Math.max(0, lx + dx));
              if (closedTemp[ny * bw + nx] === 0) {
                allSet = false;
                break;
              }
            }
          }
          if (allSet) {
            strokeSeedMask[idx] = 1;
          }
        }
      }
    } else {
      for (let ly = 2; ly < bh - 2; ly++) {
        for (let lx = 2; lx < bw - 2; lx++) {
          const idx = ly * bw + lx;
          if (strokeSeedMask[idx] === 1 || interiorCandidate[idx] === 0) continue;
          const hasL = strokeSeedMask[idx - 1] === 1 || strokeSeedMask[idx - 2] === 1;
          const hasR = strokeSeedMask[idx + 1] === 1 || strokeSeedMask[idx + 2] === 1;
          const hasU = strokeSeedMask[idx - bw] === 1 || strokeSeedMask[idx - 2 * bw] === 1;
          const hasD = strokeSeedMask[idx + bw] === 1 || strokeSeedMask[idx + 2 * bw] === 1;
          if ((hasL && hasR) || (hasU && hasD)) {
            strokeSeedMask[idx] = 1;
          }
        }
      }
    }

    // 5. Dilate by 2px (or 3px for double-outlined subtitles) into dilated for smooth outer feather transition
    const outerDilR = coupledPairsInCtd >= 18 ? 3 : 2;
    const outerDilR2 = outerDilR * outerDilR + 1;
    for (let ly = 0; ly < bh; ly++) {
      for (let lx = 0; lx < bw; lx++) {
        if (strokeSeedMask[ly * bw + lx] === 0) continue;
        for (let dy = -outerDilR; dy <= outerDilR; dy++) {
          const ny = ly + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -outerDilR; dx <= outerDilR; dx++) {
            const nx = lx + dx;
            if (nx < 0 || nx >= bw) continue;
            if (dx * dx + dy * dy <= outerDilR2 && interiorCandidate[ny * bw + nx] > 0) {
              dilated[ny * bw + nx] = 1;
            }
          }
        }
      }
    }
  } else if (isSmoothBubbleOrPanel) {
    // Smooth Gradient / Off-White / Translucent / Tinted Bubble or System Box:
    // Use local 2D 17x17 neighborhood background contrast so 2D/diagonal gradients
    // (which vary horizontally as well as vertically) never mask background between letters.
    const q: number[] = [];
    const distMap = new Int16Array(size).fill(-1);
    const localBgR = new Uint8Array(size);
    const localBgG = new Uint8Array(size);
    const localBgB = new Uint8Array(size);
    const localBgL = new Uint8Array(size);

    for (let ly = 0; ly < bh; ly++) {
      for (let lx = 0; lx < bw; lx++) {
        const idx = ly * bw + lx;
        const winSamples: { r: number; g: number; b: number; l: number }[] = [];
        for (let wy = Math.max(0, ly - 10); wy <= Math.min(bh - 1, ly + 10); wy += 3) {
          for (let wx = Math.max(0, lx - 10); wx <= Math.min(bw - 1, lx + 10); wx += 3) {
            const wIdx = wy * bw + wx;
            if (preciseMask[wIdx] === 0 && exteriorConnectedWall[wIdx] === 0) {
              const wp = wIdx * 4;
              winSamples.push({
                r: cropRgba[wp],
                g: cropRgba[wp + 1],
                b: cropRgba[wp + 2],
                l: cropLuma[wIdx],
              });
            }
          }
        }
        if (winSamples.length > 0) {
          winSamples.sort((a, b) => a.l - b.l);
          const medIdx =
            bgLuma < 115
              ? Math.floor(winSamples.length * 0.25)
              : Math.floor(winSamples.length * 0.75);
          const chosen = winSamples[Math.min(winSamples.length - 1, Math.max(0, medIdx))];
          localBgL[idx] = chosen.l;
          localBgR[idx] = chosen.r;
          localBgG[idx] = chosen.g;
          localBgB[idx] = chosen.b;
        } else {
          localBgL[idx] = rowBgLuma[ly];
          localBgR[idx] = rowBgR[ly];
          localBgG[idx] = rowBgG[ly];
          localBgB[idx] = rowBgB[ly];
        }
      }
    }

    for (let ly = 0; ly < bh; ly++) {
      for (let lx = 0; lx < bw; lx++) {
        const idx = ly * bw + lx;
        if (interiorCandidate[idx] === 0) continue;
        const p = idx * 4;
        const diff = Math.max(
          Math.abs(cropRgba[p] - localBgR[idx]),
          Math.abs(cropRgba[p + 1] - localBgG[idx]),
          Math.abs(cropRgba[p + 2] - localBgB[idx]),
          Math.abs(cropLuma[idx] - localBgL[idx])
        );
        if (diff > 22 || preciseMask[idx] > 0) {
          strokeSeedMask[idx] = 1;
          distMap[idx] = 0;
          q.push(idx);
        }
      }
    }

    let head = 0;
    while (head < q.length) {
      const curr = q[head++];
      const d = distMap[curr];
      if (d >= 3) continue;
      const cy = Math.floor(curr / bw);
      const cx = curr % bw;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= bh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx;
          if (nx < 0 || nx >= bw) continue;
          const nIdx = ny * bw + nx;
          if (distMap[nIdx] !== -1 || interiorCandidate[nIdx] === 0) continue;
          const np = nIdx * 4;
          const diff = Math.max(
            Math.abs(cropRgba[np] - localBgR[nIdx]),
            Math.abs(cropRgba[np + 1] - localBgG[nIdx]),
            Math.abs(cropRgba[np + 2] - localBgB[nIdx]),
            Math.abs(cropLuma[nIdx] - localBgL[nIdx])
          );
          if (diff > 10) {
            distMap[nIdx] = d + 1;
            strokeSeedMask[nIdx] = 1;
            q.push(nIdx);
          }
        }
      }
    }

    // Tight 2px dilation so gradient pixels between individual letters remain unmasked
    const dilateR = 2;
    const r2 = dilateR * dilateR;
    for (let ly = 0; ly < bh; ly++) {
      for (let lx = 0; lx < bw; lx++) {
        if (strokeSeedMask[ly * bw + lx] === 0) continue;
        for (let dy = -dilateR; dy <= dilateR; dy++) {
          const ny = ly + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -dilateR; dx <= dilateR; dx++) {
            const nx = lx + dx;
            if (nx < 0 || nx >= bw) continue;
            if (dx * dx + dy * dy <= r2 && interiorCandidate[ny * bw + nx] > 0) {
              dilated[ny * bw + nx] = 1;
            }
          }
        }
      }
    }
  } else {
    // Complex Textured Bubble, Screentone, OT, ST, or SFX over Artwork:
    // Detect if double-outlined stylized lettering (coupled very-bright >= 210 and very-dark <= 72 within 7px)
    let coupledPairs = 0;
    for (let ly = 2; ly < bh - 2; ly += 2) {
      for (let lx = 2; lx < bw - 2; lx += 2) {
        const idx = ly * bw + lx;
        if (interiorCandidate[idx] === 0 || cropLuma[idx] < 210) continue;
        let nearDark = false;
        for (let dy = -5; dy <= 5 && !nearDark; dy += 2) {
          const ny = Math.min(bh - 1, Math.max(0, ly + dy));
          for (let dx = -5; dx <= 5; dx += 2) {
            const nx = Math.min(bw - 1, Math.max(0, lx + dx));
            if (cropLuma[ny * bw + nx] <= 72) {
              nearDark = true;
              break;
            }
          }
        }
        if (nearDark) coupledPairs++;
      }
    }

    const isOutlinedStylized = coupledPairs >= 14 && bgLuma > 30 && bgLuma < 235;

    if (isOutlinedStylized) {
      const rSearch = 8;
      const rawSeed = new Uint8Array(size);
      for (let ly = 0; ly < bh; ly++) {
        for (let lx = 0; lx < bw; lx++) {
          const idx = ly * bw + lx;
          if (interiorCandidate[idx] === 0) continue;
          const isB = cropLuma[idx] >= 205;
          const isD = cropLuma[idx] <= 76;
          if (!isB && !isD) continue;

          let hasPartner = false;
          const y0 = Math.max(0, ly - rSearch);
          const y1 = Math.min(bh - 1, ly + rSearch);
          const x0 = Math.max(0, lx - rSearch);
          const x1 = Math.min(bw - 1, lx + rSearch);
          for (let ny = y0; ny <= y1 && !hasPartner; ny++) {
            for (let nx = x0; nx <= x1; nx++) {
              const nIdx = ny * bw + nx;
              if (interiorCandidate[nIdx] === 0) continue;
              if (isB && cropLuma[nIdx] <= 76) {
                hasPartner = true;
                break;
              }
              if (isD && cropLuma[nIdx] >= 205) {
                hasPartner = true;
                break;
              }
            }
          }
          if (hasPartner) {
            rawSeed[idx] = 1;
          }
        }
      }
      // Morphological dilation (r=3) on rawSeed into strokeSeedMask so all anti-aliased transition
      // pixels (luma 77..204) between the black outline and white letter interior are 100% opaque!
      for (let ly = 0; ly < bh; ly++) {
        for (let lx = 0; lx < bw; lx++) {
          if (rawSeed[ly * bw + lx] === 0) continue;
          for (let dy = -3; dy <= 3; dy++) {
            const ny = ly + dy;
            if (ny < 0 || ny >= bh) continue;
            for (let dx = -3; dx <= 3; dx++) {
              const nx = lx + dx;
              if (nx < 0 || nx >= bw) continue;
              if (dx * dx + dy * dy <= 10 && interiorCandidate[ny * bw + nx] > 0) {
                strokeSeedMask[ny * bw + nx] = 1;
              }
            }
          }
        }
      }
    } else if (bgLuma < 125) {
      // Bright / neon-glowing lettering over dark speedlines or night bokeh (e.g. 계획대로야!, 그렇다면 내가...)
      const borderBrightConnected = new Uint8Array(size);
      const bq: number[] = [];
      for (let ly = 0; ly < bh; ly++) {
        for (let lx = 0; lx < bw; lx++) {
          if (lx !== 0 && lx !== bw - 1 && ly !== 0 && ly !== bh - 1) continue;
          const idx = ly * bw + lx;
          if (cropLuma[idx] >= 165) {
            borderBrightConnected[idx] = 1;
            bq.push(idx);
          }
        }
      }
      while (bq.length > 0) {
        const curr = bq.pop()!;
        const cy = Math.floor(curr / bw);
        const cx = curr % bw;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= bw) continue;
            const nIdx = ny * bw + nx;
            if (borderBrightConnected[nIdx] === 0 && cropLuma[nIdx] >= 155) {
              borderBrightConnected[nIdx] = 1;
              bq.push(nIdx);
            }
          }
        }
      }

      const q: number[] = [];
      const distMap = new Int16Array(size).fill(-1);
      for (let i = 0; i < size; i++) {
        if (interiorCandidate[i] === 0 || borderBrightConnected[i] === 1) continue;
        if (cropLuma[i] >= 180) {
          strokeSeedMask[i] = 1;
          distMap[i] = 0;
          q.push(i);
        }
      }
      let head = 0;
      while (head < q.length) {
        const curr = q[head++];
        const d = distMap[curr];
        if (d >= 10) continue;
        const cy = Math.floor(curr / bw);
        const cx = curr % bw;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= bw) continue;
            const nIdx = ny * bw + nx;
            if (
              distMap[nIdx] !== -1 ||
              interiorCandidate[nIdx] === 0 ||
              borderBrightConnected[nIdx] === 1
            ) {
              continue;
            }
            const np = nIdx * 4;
            const r = cropRgba[np],
              g = cropRgba[np + 1],
              b = cropRgba[np + 2];
            const chroma = Math.max(r, g, b) - Math.min(r, g, b);
            if (cropLuma[nIdx] > rowBgLuma[ny] + 18 || chroma > 32) {
              distMap[nIdx] = d + 1;
              strokeSeedMask[nIdx] = 1;
              q.push(nIdx);
            }
          }
        }
      }
    } else {
      // Dark, white-with-shadow (크라쿠프 궁전.), or 3D extruded lettering (크라쿠프 뉴스) over light textured artwork
      const rawSeed = new Uint8Array(size);
      for (let ly = 0; ly < bh; ly++) {
        const lBg = rowBgLuma[ly];
        for (let lx = 0; lx < bw; lx++) {
          const idx = ly * bw + lx;
          if (interiorCandidate[idx] === 0) continue;
          if (
            preciseMask[idx] > 0 &&
            (Math.abs(cropLuma[idx] - lBg) > 16 || cropLuma[idx] >= 242 || cropLuma[idx] <= 80)
          ) {
            rawSeed[idx] = 1;
          }
        }
      }
      for (let ly = 0; ly < bh; ly++) {
        for (let lx = 0; lx < bw; lx++) {
          if (rawSeed[ly * bw + lx] === 0) continue;
          for (let dy = -2; dy <= 2; dy++) {
            const ny = ly + dy;
            if (ny < 0 || ny >= bh) continue;
            for (let dx = -2; dx <= 2; dx++) {
              const nx = lx + dx;
              if (nx < 0 || nx >= bw) continue;
              if (dx * dx + dy * dy <= 5 && interiorCandidate[ny * bw + nx] > 0) {
                strokeSeedMask[ny * bw + nx] = 1;
              }
            }
          }
        }
      }
    }

    // Fallback if no stroke seeds were triggered (e.g. subtle colored SFX)
    let seedCount = 0;
    for (let i = 0; i < size; i++) if (strokeSeedMask[i]) seedCount++;
    if (seedCount < 8) {
      for (let i = 0; i < size; i++) {
        if (interiorCandidate[i] > 0 && preciseMask[i] > 0) {
          strokeSeedMask[i] = 1;
        }
      }
    }

    const dilateR = Math.max(2, Math.min(6, region.paddingPx ?? 3));
    const r2 = dilateR * dilateR;
    for (let ly = 0; ly < bh; ly++) {
      for (let lx = 0; lx < bw; lx++) {
        if (strokeSeedMask[ly * bw + lx] === 0) continue;
        for (let dy = -dilateR; dy <= dilateR; dy++) {
          const ny = ly + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -dilateR; dx <= dilateR; dx++) {
            const nx = lx + dx;
            if (nx < 0 || nx >= bw) continue;
            if (dx * dx + dy * dy <= r2 && interiorCandidate[ny * bw + nx] > 0) {
              dilated[ny * bw + nx] = 1;
            }
          }
        }
      }
    }
  }

  // Apply manual brush edits (add / erase)
  if (region.manualEdits && region.manualEdits.length > 0) {
    for (const edit of region.manualEdits) {
      const cx = Math.round(edit.x * bw);
      const cy = Math.round(edit.y * bh);
      const rad = Math.max(2, Math.round(edit.radiusPx));
      const rad2 = rad * rad;
      const y0 = Math.max(0, cy - rad);
      const y1 = Math.min(bh - 1, cy + rad);
      const x0 = Math.max(0, cx - rad);
      const x1 = Math.min(bw - 1, cx + rad);
      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          if ((nx - cx) * (nx - cx) + (ny - cy) * (ny - cy) <= rad2) {
            dilated[ny * bw + nx] = edit.mode === 'add' ? 1 : 0;
          }
        }
      }
    }
  }

  // Feather mask edges with Hermite distance ramp
  const feather = Math.max(0, Math.min(8, region.featherPx ?? 2));
  for (let ly = 0; ly < bh; ly++) {
    for (let lx = 0; lx < bw; lx++) {
      const idx = ly * bw + lx;
      if (dilated[idx] === 0) {
        writeMask[idx] = 0;
        continue;
      }
      if (strokeSeedMask[idx] === 1 || feather === 0) {
        writeMask[idx] = 255;
        continue;
      }
      let minZeroDist2 = (feather + 1) * (feather + 1);
      const y0 = Math.max(0, ly - feather);
      const y1 = Math.min(bh - 1, ly + feather);
      const x0 = Math.max(0, lx - feather);
      const x1 = Math.min(bw - 1, lx + feather);
      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          if (dilated[ny * bw + nx] === 0) {
            const d2 = (nx - lx) * (nx - lx) + (ny - ly) * (ny - ly);
            if (d2 < minZeroDist2) minZeroDist2 = d2;
          }
        }
      }
      const dist = Math.sqrt(minZeroDist2);
      if (dist > feather) {
        writeMask[idx] = 255;
      } else {
        const t = Math.max(0.25, Math.min(1.0, dist / (feather + 1)));
        const smooth = t * t * (3.0 - 2.0 * t);
        writeMask[idx] = Math.round(smooth * 255);
      }
    }
  }

  return { writeMask, strokeSeedMask, interiorCandidate };
}

/**
 * Solves a 3x3 linear system M * x = b using Gaussian elimination with partial pivoting.
 */
function solve3x3(M: number[][], b: number[]): [number, number, number] | null {
  const A = [
    [M[0][0], M[0][1], M[0][2], b[0]],
    [M[1][0], M[1][1], M[1][2], b[1]],
    [M[2][0], M[2][1], M[2][2], b[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[pivot][col])) pivot = row;
    }
    if (Math.abs(A[pivot][col]) < 1e-7) return null;
    const tmp = A[col];
    A[col] = A[pivot];
    A[pivot] = tmp;

    for (let row = col + 1; row < 3; row++) {
      const f = A[row][col] / A[col][col];
      for (let k = col; k < 4; k++) {
        A[row][k] -= f * A[col][k];
      }
    }
  }
  const x2 = A[2][3] / A[2][2];
  const x1 = (A[1][3] - A[1][2] * x2) / A[1][1];
  const x0 = (A[0][3] - A[0][1] * x1 - A[0][2] * x2) / A[0][0];
  return [x0, x1, x2];
}

/**
 * Attempts to clean `writeMask > 0` using the immediate clean 1..6px surround ring around the masked letters:
 * - Tier 1 (`FLAT_RING`): If the ring around the letters is a flat uniform color (`spread <= 16`, e.g. white,
 *   black, lavender thought bubbles `#CBD0EA`, grey boxes `#BDBDBD`, gold plaques `#DFC798`), fills masked pixels
 *   with the exact median ring color — zero white corners or seams!
 * - Tier 2 (`GRADIENT_PLANE`): Fits a 2D linear plane `RGB(y, x) = a*y + b*x + c` by least-squares to the
 *   clean ring pixels. If `MAE <= 4.0` (for light/medium gradient bubbles) or `MAE <= 2.0 && spread <= 20`
 *   (for dark panels), fills masked pixels with the exact 2D linear gradient plane!
 * - Returns `null` when background has complex artwork, screentones, wooden planks, bokeh, or bamboo leaves
 *   so `cleanRegionWithOnnxSupport` routes it to neural `lama-manga.onnx` (512x512).
 */
export function tryCleanFlatOrGradientRing(params: {
  rgba: Uint8Array;
  imgWidth: number;
  imgHeight: number;
  region: DetectedRegion;
  writeMask: Uint8Array;
  interiorCandidate: Uint8Array;
}): 'FLAT_RING' | 'GRADIENT_PLANE' | null {
  const { rgba, imgWidth, imgHeight, region, writeMask, interiorCandidate } = params;
  if (region.surroundStats.isPeriodicScreentone) {
    return null;
  }

  const { x: bx, y: by, w: bw, h: bh } = region.bbox;
  const ringPixels: { x: number; y: number; r: number; g: number; b: number; l: number }[] = [];

  for (let ly = 0; ly < bh; ly++) {
    const py = by + ly;
    if (py < 0 || py >= imgHeight) continue;
    for (let lx = 0; lx < bw; lx++) {
      const px = bx + lx;
      if (px < 0 || px >= imgWidth) continue;
      const idx = ly * bw + lx;
      if (writeMask[idx] > 0 || interiorCandidate[idx] === 0) continue;

      let nearMask = false;
      for (let dy = -5; dy <= 5 && !nearMask; dy += 2) {
        const ny = ly + dy;
        if (ny < 0 || ny >= bh) continue;
        for (let dx = -5; dx <= 5; dx += 2) {
          const nx = lx + dx;
          if (nx < 0 || nx >= bw) continue;
          if (writeMask[ny * bw + nx] > 0) {
            nearMask = true;
            break;
          }
        }
      }
      if (!nearMask) continue;

      const pIdx = (py * imgWidth + px) * 4;
      const r = rgba[pIdx];
      const g = rgba[pIdx + 1];
      const b = rgba[pIdx + 2];
      ringPixels.push({ x: lx, y: ly, r, g, b, l: rgbToLuma(r, g, b) });
    }
  }

  if (ringPixels.length < 20) return null;

  const sortedL = [...ringPixels].sort((a, b) => a.l - b.l);
  // Trim top/bottom 10% luma outliers so stray unmasked specks never bias the ring statistics
  const trimStart = Math.floor(sortedL.length * 0.1);
  const trimEnd = Math.max(trimStart + 8, Math.ceil(sortedL.length * 0.9));
  const trimmed = sortedL.slice(trimStart, trimEnd);

  const rs = trimmed.map((p) => p.r).sort((a, b) => a - b);
  const gs = trimmed.map((p) => p.g).sort((a, b) => a - b);
  const bs = trimmed.map((p) => p.b).sort((a, b) => a - b);

  const p10Idx = Math.floor(trimmed.length * 0.08);
  const p90Idx = Math.min(trimmed.length - 1, Math.floor(trimmed.length * 0.92));
  const medIdx = Math.floor(trimmed.length * 0.5);

  const spreadR = rs[p90Idx] - rs[p10Idx];
  const spreadG = gs[p90Idx] - gs[p10Idx];
  const spreadB = bs[p90Idx] - bs[p10Idx];
  const maxSpread = Math.max(spreadR, spreadG, spreadB);

  let medR = rs[medIdx];
  let medG = gs[medIdx];
  let medB = bs[medIdx];
  const medL = trimmed[medIdx].l;

  // Tier 1: Flat Solid Background (White, Black, Lavender, Grey, Cream, Gold Plaque)
  // For regions tagged SFX, require an extra-tight solid ring (maxSpread <= 12) so textured SFX still route to LaMa.
  const flatSpreadLimit = region.category === 'SFX' ? 12 : 16;
  if (maxSpread <= flatSpreadLimit) {
    if (medL >= 250 && maxSpread <= 6) {
      medR = 255;
      medG = 255;
      medB = 255;
    } else if (medL <= 6 && maxSpread <= 6 && region.surroundStats.isDarkBubble) {
      medR = 0;
      medG = 0;
      medB = 0;
    }

    for (let ly = 0; ly < bh; ly++) {
      const py = by + ly;
      if (py < 0 || py >= imgHeight) continue;
      for (let lx = 0; lx < bw; lx++) {
        const px = bx + lx;
        if (px < 0 || px >= imgWidth) continue;
        const mAlpha = writeMask[ly * bw + lx];
        if (mAlpha === 0) continue;
        const dstIdx = (py * imgWidth + px) * 4;
        if (mAlpha === 255) {
          rgba[dstIdx] = medR;
          rgba[dstIdx + 1] = medG;
          rgba[dstIdx + 2] = medB;
        } else {
          const t = mAlpha / 255;
          const w = t * t * (3 - 2 * t);
          rgba[dstIdx] = Math.round(rgba[dstIdx] * (1 - w) + medR * w);
          rgba[dstIdx + 1] = Math.round(rgba[dstIdx + 1] * (1 - w) + medG * w);
          rgba[dstIdx + 2] = Math.round(rgba[dstIdx + 2] * (1 - w) + medB * w);
        }
      }
    }
    return 'FLAT_RING';
  }

  // Tier 2: 2D Linear Plane Least-Squares Fit RGB(y, x) = a*y + b*x + c (skip for SFX so textured SFX route to LaMa)
  if (region.category === 'SFX') return null;
  if (maxSpread > 78) return null;
  if (medL < 120 && maxSpread > 22) return null; // Protect dark textured scenes (wooden planks, bokeh, palm fronds)

  const step = Math.max(1, Math.floor(trimmed.length / 260));
  const samples = trimmed.filter((_, i) => i % step === 0);
  if (samples.length < 16) return null;

  const ATA = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const ATbR = [0, 0, 0];
  const ATbG = [0, 0, 0];
  const ATbB = [0, 0, 0];

  for (const s of samples) {
    const v0 = s.y;
    const v1 = s.x;
    ATA[0][0] += v0 * v0;
    ATA[0][1] += v0 * v1;
    ATA[0][2] += v0;
    ATA[1][0] += v1 * v0;
    ATA[1][1] += v1 * v1;
    ATA[1][2] += v1;
    ATA[2][0] += v0;
    ATA[2][1] += v1;
    ATA[2][2] += 1;

    ATbR[0] += v0 * s.r;
    ATbR[1] += v1 * s.r;
    ATbR[2] += s.r;

    ATbG[0] += v0 * s.g;
    ATbG[1] += v1 * s.g;
    ATbG[2] += s.g;

    ATbB[0] += v0 * s.b;
    ATbB[1] += v1 * s.b;
    ATbB[2] += s.b;
  }

  const coeffR = solve3x3(ATA, ATbR);
  const coeffG = solve3x3(ATA, ATbG);
  const coeffB = solve3x3(ATA, ATbB);
  if (!coeffR || !coeffG || !coeffB) return null;

  const errs: number[] = [];
  let sumErr = 0;
  for (const s of samples) {
    const pr = coeffR[0] * s.y + coeffR[1] * s.x + coeffR[2];
    const pg = coeffG[0] * s.y + coeffG[1] * s.x + coeffG[2];
    const pb = coeffB[0] * s.y + coeffB[1] * s.x + coeffB[2];
    const e = (Math.abs(s.r - pr) + Math.abs(s.g - pg) + Math.abs(s.b - pb)) / 3;
    sumErr += e;
    errs.push(e);
  }
  errs.sort((a, b) => a - b);
  const mae = sumErr / samples.length;
  const p90Err = errs[Math.floor(errs.length * 0.9)];

  const maxAllowedMae = medL >= 120 ? 4.2 : 2.0;
  const maxAllowedP90 = medL >= 120 ? 8.5 : 4.2;
  if (mae > maxAllowedMae || p90Err > maxAllowedP90) {
    return null;
  }

  for (let ly = 0; ly < bh; ly++) {
    const py = by + ly;
    if (py < 0 || py >= imgHeight) continue;
    for (let lx = 0; lx < bw; lx++) {
      const px = bx + lx;
      if (px < 0 || px >= imgWidth) continue;
      const mAlpha = writeMask[ly * bw + lx];
      if (mAlpha === 0) continue;

      const fillR = Math.max(
        0,
        Math.min(255, Math.round(coeffR[0] * ly + coeffR[1] * lx + coeffR[2]))
      );
      const fillG = Math.max(
        0,
        Math.min(255, Math.round(coeffG[0] * ly + coeffG[1] * lx + coeffG[2]))
      );
      const fillB = Math.max(
        0,
        Math.min(255, Math.round(coeffB[0] * ly + coeffB[1] * lx + coeffB[2]))
      );

      const dstIdx = (py * imgWidth + px) * 4;
      if (mAlpha === 255) {
        rgba[dstIdx] = fillR;
        rgba[dstIdx + 1] = fillG;
        rgba[dstIdx + 2] = fillB;
      } else {
        const t = mAlpha / 255;
        const w = t * t * (3 - 2 * t);
        rgba[dstIdx] = Math.round(rgba[dstIdx] * (1 - w) + fillR * w);
        rgba[dstIdx + 1] = Math.round(rgba[dstIdx + 1] * (1 - w) + fillG * w);
        rgba[dstIdx + 2] = Math.round(rgba[dstIdx + 2] * (1 - w) + fillB * w);
      }
    }
  }

  return 'GRADIENT_PLANE';
}

/**
 * Reconstructs clean pixels for a single region using:
 * - Smart Fill (pure `#FFFFFF` or pure `#000000`) strictly for verified flat white/dark speech bubbles
 * - Flat Ring Median or 2D Linear Plane Least-Squares Fit (`a*y + b*x + c`) for tinted/gradient bubbles
 * - Immediate scanline boundary interpolation + 2D Laplace/Harmonic PDE solver fallback.
 * Untouched pixels outside `writeMask` remain 100% bit-identical to the source RGBA buffer!
 */
export function cleanAndCompositeRegionInPlace(params: {
  rgba: Uint8Array;
  imgWidth: number;
  imgHeight: number;
  region: DetectedRegion;
  settings: CleaningSettings;
  ctdRegionMask?: Uint8Array;
}): {
  writeMask: Uint8Array;
  strokeSeedMask: Uint8Array;
  quality: QualityAssessment;
} {
  const { rgba, imgWidth, imgHeight, region, settings, ctdRegionMask } = params;
  const { x: bx, y: by, w: bw, h: bh } = region.bbox;

  const { writeMask, strokeSeedMask, interiorCandidate } = buildRegionPixelMask({
    rgba,
    imgWidth,
    imgHeight,
    region,
    preserveColoredBubbles: settings.preserveColoredBubbles,
    ctdRegionMask,
  });

  const engine = selectCleaningEngine(region.category, region.surroundStats, settings);
  region.appliedEngine = engine;
  region.routedEngine = engine;

  const isWhitenBubble =
    region.category === 'BUBBLE' &&
    region.bubbleCleanMode === 'WHITEN' &&
    region.surroundStats.isWhiteBubble &&
    !region.surroundStats.isGradient &&
    engine === 'SMART_FILL';

  const isDarkFillBubble =
    region.surroundStats.isDarkBubble &&
    !region.surroundStats.isGradient &&
    engine === 'SMART_FILL';

  const useFlatSmartFill = isWhitenBubble || isDarkFillBubble;

  const targetR = isWhitenBubble
    ? 255
    : isDarkFillBubble && region.surroundStats.meanLuma <= 12
      ? 0
      : Math.round(region.surroundStats.meanR);
  const targetG = isWhitenBubble
    ? 255
    : isDarkFillBubble && region.surroundStats.meanLuma <= 12
      ? 0
      : Math.round(region.surroundStats.meanG);
  const targetB = isWhitenBubble
    ? 255
    : isDarkFillBubble && region.surroundStats.meanLuma <= 12
      ? 0
      : Math.round(region.surroundStats.meanB);

  let handledByRingOrPlane = false;
  if (!useFlatSmartFill) {
    const ringRes = tryCleanFlatOrGradientRing({
      rgba,
      imgWidth,
      imgHeight,
      region,
      writeMask,
      interiorCandidate,
    });
    if (ringRes !== null) {
      handledByRingOrPlane = true;
    }
  }

  if (!handledByRingOrPlane) {
    const bufR = new Float32Array(bw * bh);
    const bufG = new Float32Array(bw * bh);
    const bufB = new Float32Array(bw * bh);

    for (let ly = 0; ly < bh; ly++) {
      const py = Math.min(imgHeight - 1, Math.max(0, by + ly));
      for (let lx = 0; lx < bw; lx++) {
        const px = Math.min(imgWidth - 1, Math.max(0, bx + lx));
        const idx = ly * bw + lx;
        const pIdx = (py * imgWidth + px) * 4;
        bufR[idx] = rgba[pIdx];
        bufG[idx] = rgba[pIdx + 1];
        bufB[idx] = rgba[pIdx + 2];
      }
    }

    if (!useFlatSmartFill) {
      for (let ly = 0; ly < bh; ly++) {
        let lx = 0;
        while (lx < bw) {
          if (writeMask[ly * bw + lx] === 0) {
            lx++;
            continue;
          }
          const spanStart = lx;
          while (lx < bw && writeMask[ly * bw + lx] > 0) {
            lx++;
          }
          const spanEnd = lx - 1;

          const leftSampleX = Math.max(0, spanStart - 1);
          const rightSampleX = Math.min(bw - 1, spanEnd + 1);
          const lIdx = ly * bw + leftSampleX;
          const rIdx = ly * bw + rightSampleX;

          const hasLeft = writeMask[lIdx] === 0;
          const hasRight = writeMask[rIdx] === 0;

          const lr = hasLeft ? bufR[lIdx] : hasRight ? bufR[rIdx] : targetR;
          const lg = hasLeft ? bufG[lIdx] : hasRight ? bufG[rIdx] : targetG;
          const lb = hasLeft ? bufB[lIdx] : hasRight ? bufB[rIdx] : targetB;

          const rr = hasRight ? bufR[rIdx] : lr;
          const rg = hasRight ? bufG[rIdx] : lg;
          const rb = hasRight ? bufB[rIdx] : lb;

          const spanLen = Math.max(1, spanEnd - spanStart + 1);
          for (let sx = spanStart; sx <= spanEnd; sx++) {
            const u = (sx - spanStart + 0.5) / spanLen;
            const mIdx = ly * bw + sx;
            bufR[mIdx] = lr * (1 - u) + rr * u;
            bufG[mIdx] = lg * (1 - u) + rg * u;
            bufB[mIdx] = lb * (1 - u) + rb * u;
          }
        }
      }

      const iterations = 36;
      for (let iter = 0; iter < iterations; iter++) {
        for (let ly = 1; ly < bh - 1; ly++) {
          for (let lx = 1; lx < bw - 1; lx++) {
            const idx = ly * bw + lx;
            if (writeMask[idx] === 0) continue;
            const l = idx - 1;
            const r = idx + 1;
            const u = idx - bw;
            const d = idx + bw;
            bufR[idx] = 0.25 * (bufR[l] + bufR[r] + bufR[u] + bufR[d]);
            bufG[idx] = 0.25 * (bufG[l] + bufG[r] + bufG[u] + bufG[d]);
            bufB[idx] = 0.25 * (bufB[l] + bufB[r] + bufB[u] + bufB[d]);
          }
        }
      }
    }

    // Write strictly where writeMask > 0 using Hermite C1 smoothstep feathering
    for (let ly = 0; ly < bh; ly++) {
      const py = by + ly;
      if (py < 0 || py >= imgHeight) continue;
      for (let lx = 0; lx < bw; lx++) {
        const px = bx + lx;
        if (px < 0 || px >= imgWidth) continue;
        const idx = ly * bw + lx;
        const mAlpha = writeMask[idx];
        if (mAlpha === 0) continue; // Bit-identical guarantee for all unmasked pixels!

        const dstIdx = (py * imgWidth + px) * 4;
        const fillR = useFlatSmartFill ? targetR : Math.round(bufR[idx]);
        const fillG = useFlatSmartFill ? targetG : Math.round(bufG[idx]);
        const fillB = useFlatSmartFill ? targetB : Math.round(bufB[idx]);

        if (mAlpha === 255) {
          rgba[dstIdx] = fillR;
          rgba[dstIdx + 1] = fillG;
          rgba[dstIdx + 2] = fillB;
        } else {
          const t = mAlpha / 255;
          const w = t * t * (3 - 2 * t);
          rgba[dstIdx] = Math.round(rgba[dstIdx] * (1 - w) + fillR * w);
          rgba[dstIdx + 1] = Math.round(rgba[dstIdx + 1] * (1 - w) + fillG * w);
          rgba[dstIdx + 2] = Math.round(rgba[dstIdx + 2] * (1 - w) + fillB * w);
        }
      }
    }
  }

  // Quality evaluation over the region + 16px surround ring
  const pad = 16;
  const qx0 = Math.max(0, bx - pad);
  const qy0 = Math.max(0, by - pad);
  const qx1 = Math.min(imgWidth, bx + bw + pad);
  const qy1 = Math.min(imgHeight, by + bh + pad);
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
        if (writeMask[mIdx] > 128) {
          intMask[y * qw + x] = 1;
        }
        if (strokeSeedMask[mIdx] === 1) {
          seedExpanded[y * qw + x] = 1;
        }
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
 * Verifies that every single pixel outside the union of applied region bounding boxes / masks
 * is 100% bit-identical between original and cleaned RGBA buffers (Bear invariant §7 rule 1).
 */
export function verifyBitIdenticalOutsideMasks(
  originalRgba: Uint8Array,
  cleanedRgba: Uint8Array,
  width: number,
  height: number,
  regions: DetectedRegion[]
): { bitIdentical: boolean; differingPixelsOutsideMasks: number; totalUnmaskedPixels: number } {
  const authorized = new Uint8Array(width * height);
  for (const r of regions) {
    if (r.status !== 'CLEANED') continue;
    const x0 = Math.max(0, r.bbox.x);
    const y0 = Math.max(0, r.bbox.y);
    const x1 = Math.min(width, r.bbox.x + r.bbox.w);
    const y1 = Math.min(height, r.bbox.y + r.bbox.h);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        authorized[y * width + x] = 1;
      }
    }
  }

  let differingPixelsOutsideMasks = 0;
  let totalUnmaskedPixels = 0;
  const totalPixels = width * height;

  for (let i = 0; i < totalPixels; i++) {
    if (authorized[i] === 1) continue;
    totalUnmaskedPixels++;
    const p = i * 4;
    if (
      originalRgba[p] !== cleanedRgba[p] ||
      originalRgba[p + 1] !== cleanedRgba[p + 1] ||
      originalRgba[p + 2] !== cleanedRgba[p + 2] ||
      originalRgba[p + 3] !== cleanedRgba[p + 3]
    ) {
      differingPixelsOutsideMasks++;
    }
  }

  return {
    bitIdentical: differingPixelsOutsideMasks === 0,
    differingPixelsOutsideMasks,
    totalUnmaskedPixels,
  };
}
