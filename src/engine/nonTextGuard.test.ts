import test from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluateNonTextGuard,
  detectXOrCheckmarkOrCircleSymbol,
  detectCharacterFaceRegion,
  isAnimeSkinColor,
} from './nonTextGuard';
import { SurroundStats } from './types';

function createCanvas(w: number, h: number, r: number, g: number, b: number): Uint8Array {
  const buf = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    buf[i * 4] = r;
    buf[i * 4 + 1] = g;
    buf[i * 4 + 2] = b;
    buf[i * 4 + 3] = 255;
  }
  return buf;
}

function drawThickLine(
  buf: Uint8Array,
  w: number,
  h: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  thickness: number,
  r: number,
  g: number,
  b: number
) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) * 2 + 1;
  const rad = Math.max(1, Math.floor(thickness / 2));
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const cx = Math.round(x0 + (x1 - x0) * t);
    const cy = Math.round(y0 + (y1 - y0) * t);
    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        if (dx * dx + dy * dy > rad * rad + 1) continue;
        const px = cx + dx;
        const py = cy + dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        const idx = (py * w + px) * 4;
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
      }
    }
  }
}

function drawFilledCircle(
  buf: Uint8Array,
  w: number,
  h: number,
  cx: number,
  cy: number,
  radius: number,
  r: number,
  g: number,
  b: number
) {
  for (let y = Math.max(0, Math.floor(cy - radius)); y <= Math.min(h - 1, Math.ceil(cy + radius)); y++) {
    for (let x = Math.max(0, Math.floor(cx - radius)); x <= Math.min(w - 1, Math.ceil(cx + radius)); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= radius * radius) {
        const idx = (y * w + x) * 4;
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
      }
    }
  }
}

function drawRing(
  buf: Uint8Array,
  w: number,
  h: number,
  cx: number,
  cy: number,
  outerR: number,
  innerR: number,
  r: number,
  g: number,
  b: number
) {
  for (let y = Math.max(0, Math.floor(cy - outerR)); y <= Math.min(h - 1, Math.ceil(cy + outerR)); y++) {
    for (let x = Math.max(0, Math.floor(cx - outerR)); x <= Math.min(w - 1, Math.ceil(cx + outerR)); x++) {
      const d2 = (x - cx) ** 2 + (y - cy) ** 2;
      if (d2 <= outerR * outerR && d2 >= innerR * innerR) {
        const idx = (y * w + x) * 4;
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
      }
    }
  }
}

function makeSurround(overrides: Partial<SurroundStats>): SurroundStats {
  return {
    meanR: 254,
    meanG: 254,
    meanB: 254,
    meanLuma: 254,
    stdLuma: 2,
    p5Luma: 250,
    p95Luma: 255,
    edgeEnergy: 4,
    isUniform: true,
    isWhiteBubble: true,
    isDarkBubble: false,
    isPeriodicScreentone: false,
    isGradient: false,
    isOutlineDominated: false,
    gradSlopeX: 0,
    gradSlopeY: 0,
    ...overrides,
  };
}

const WHITE_BUBBLE_SURROUND: SurroundStats = makeSurround({});

const COMPLEX_ART_SURROUND: SurroundStats = makeSurround({
  meanR: 130,
  meanG: 115,
  meanB: 100,
  meanLuma: 118,
  stdLuma: 48,
  p5Luma: 40,
  p95Luma: 210,
  edgeEnergy: 120,
  isUniform: false,
  isWhiteBubble: false,
});

test('rejects standalone red X mark inside white speech bubble', () => {
  const w = 100, h = 100;
  const img = createCanvas(w, h, 40, 45, 55);
  // White speech bubble with black border
  drawFilledCircle(img, w, h, 50, 50, 42, 20, 20, 20);
  drawFilledCircle(img, w, h, 50, 50, 38, 255, 255, 255);
  // Red X inside bubble
  drawThickLine(img, w, h, 32, 32, 68, 68, 10, 200, 35, 35);
  drawThickLine(img, w, h, 68, 32, 32, 68, 10, 200, 35, 35);

  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 8, y: 8, w: 84, h: 84 },
    confidence: 0.42,
    insideBubble: true,
    surroundStats: COMPLEX_ART_SURROUND,
    ctdMetrics: { ctd65: 900, ctd120: 400, ctdCov65: 0.13, ctdMax: 240 },
  });

  assert.equal(res.reject, true);
  assert.equal(res.reason, 'NON_TRANSLATABLE_X_MARK');
});

test('rejects monochrome tilted black X mark on white background', () => {
  const w = 90, h = 90;
  const img = createCanvas(w, h, 255, 255, 255);
  // Slightly tilted black X
  drawThickLine(img, w, h, 26, 22, 64, 68, 9, 18, 18, 18);
  drawThickLine(img, w, h, 67, 26, 23, 64, 9, 18, 18, 18);

  const sym = detectXOrCheckmarkOrCircleSymbol(img, w, h, { x: 10, y: 10, w: 70, h: 70 }, 0.55);
  assert.equal(sym.matched, true);
  assert.equal(sym.reason, 'NON_TRANSLATABLE_X_MARK');
});

test('rejects standalone green checkmark (✓) and monochrome checkmark', () => {
  const w = 100, h = 100;
  // Green checkmark on white badge
  const greenImg = createCanvas(w, h, 255, 255, 255);
  drawThickLine(greenImg, w, h, 22, 54, 42, 74, 8, 30, 175, 65);
  drawThickLine(greenImg, w, h, 42, 74, 80, 24, 8, 30, 175, 65);

  const greenRes = detectXOrCheckmarkOrCircleSymbol(
    greenImg,
    w,
    h,
    { x: 10, y: 10, w: 80, h: 80 },
    0.58
  );
  assert.equal(greenRes.matched, true);
  assert.equal(greenRes.reason, 'NON_TRANSLATABLE_CHECKMARK');

  // Black checkmark on white background
  const blackImg = createCanvas(w, h, 255, 255, 255);
  drawThickLine(blackImg, w, h, 20, 52, 40, 72, 7, 15, 15, 15);
  drawThickLine(blackImg, w, h, 40, 72, 82, 22, 7, 15, 15, 15);

  const blackRes = detectXOrCheckmarkOrCircleSymbol(
    blackImg,
    w,
    h,
    { x: 10, y: 10, w: 80, h: 80 },
    0.58
  );
  assert.equal(blackRes.matched, true);
  assert.equal(blackRes.reason, 'NON_TRANSLATABLE_CHECKMARK');
});

test('rejects standalone red hollow circle mark (⭕)', () => {
  const w = 100, h = 100;
  const img = createCanvas(w, h, 255, 255, 255);
  drawRing(img, w, h, 50, 50, 32, 22, 210, 35, 35);

  const res = detectXOrCheckmarkOrCircleSymbol(img, w, h, { x: 10, y: 10, w: 80, h: 80 }, 0.55);
  assert.equal(res.matched, true);
  assert.equal(res.reason, 'NON_TRANSLATABLE_CIRCLE_MARK');
});

test('protects realistic skin-tone character face with paired eyes and mouth', () => {
  const w = 180, h = 180;
  const img = createCanvas(w, h, 238, 198, 172); // warm anime peach skin
  assert.equal(isAnimeSkinColor(238, 198, 172), true);
  // Subtle blush/shading variance so it is not a flat uniform box
  drawFilledCircle(img, w, h, 42, 105, 16, 244, 182, 162);
  drawFilledCircle(img, w, h, 138, 105, 16, 244, 182, 162);
  // Left & right dark anime eyes + small mouth
  drawFilledCircle(img, w, h, 62, 74, 11, 25, 20, 28);
  drawFilledCircle(img, w, h, 118, 74, 11, 25, 20, 28);
  drawThickLine(img, w, h, 80, 126, 100, 126, 4, 90, 45, 45);

  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 20, y: 20, w: 140, h: 140 },
    confidence: 0.48,
    insideBubble: false,
    surroundStats: COMPLEX_ART_SURROUND,
    ctdMetrics: { ctd65: 260, ctd120: 45, ctdCov65: 0.013, ctdMax: 140 },
  });

  assert.equal(res.reject, true);
  assert.equal(res.reason, 'CHARACTER_FACE');
});

test('protects white chibi shocked character face with paired oval eyes', () => {
  const w = 180, h = 180;
  const img = createCanvas(w, h, 12, 18, 35); // dark background
  drawFilledCircle(img, w, h, 90, 90, 68, 25, 55, 120); // outline
  drawFilledCircle(img, w, h, 90, 90, 62, 252, 252, 255); // white chibi head
  // Two symmetric eyes + open mouth
  drawFilledCircle(img, w, h, 62, 78, 13, 30, 40, 75);
  drawFilledCircle(img, w, h, 118, 78, 13, 30, 40, 75);
  drawThickLine(img, w, h, 82, 118, 98, 118, 8, 30, 40, 75);

  const faceRes = detectCharacterFaceRegion(
    img,
    w,
    h,
    { x: 16, y: 16, w: 148, h: 148 },
    COMPLEX_ART_SURROUND,
    0.45,
    { ctd65: 250, ctd120: 35, ctdCov65: 0.011, ctdMax: 130 }
  );
  assert.equal(faceRes.matched, true);
});

test('rejects low-CTD artwork false positives (sword hilt / cloud / lace) outside white bubbles', () => {
  const w = 160, h = 140;
  const img = createCanvas(w, h, 60, 20, 20);
  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 10, y: 10, w: 140, h: 120 },
    confidence: 0.52,
    insideBubble: false,
    surroundStats: COMPLEX_ART_SURROUND,
    ctdMetrics: { ctd65: 12, ctd120: 0, ctdCov65: 0.0007, ctdMax: 90 },
  });

  assert.equal(res.reject, true);
  assert.equal(res.reason, 'NO_TEXT_CONSENSUS');
});

test('rejects diagonal striped UI progress bar (////)', () => {
  const w = 320, h = 80;
  const img = createCanvas(w, h, 240, 248, 255);
  for (let x = 20; x < 290; x += 28) {
    drawThickLine(img, w, h, x, 60, x + 18, 18, 6, 80, 165, 240);
  }
  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 10, y: 10, w: 300, h: 60 },
    confidence: 0.39,
    insideBubble: true,
    surroundStats: makeSurround({
      meanR: 225,
      meanG: 238,
      meanB: 252,
      meanLuma: 235,
      stdLuma: 13,
      edgeEnergy: 31,
      isUniform: false,
      isWhiteBubble: false,
    }),
    ctdMetrics: { ctd65: 900, ctd120: 350, ctdCov65: 0.05, ctdMax: 230 },
  });

  assert.equal(res.reject, true);
  assert.equal(res.reason, 'NON_TEXT_UI_PATTERN');
});

test('PRESERVES real text printed over a character face when strong CTD text consensus exists', () => {
  const w = 220, h = 120;
  const img = createCanvas(w, h, 236, 196, 170);
  drawFilledCircle(img, w, h, 70, 50, 10, 25, 20, 25);
  drawFilledCircle(img, w, h, 150, 50, 10, 25, 20, 25);

  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 10, y: 10, w: 200, h: 100 },
    confidence: 0.68,
    insideBubble: false,
    surroundStats: COMPLEX_ART_SURROUND,
    ctdMetrics: { ctd65: 4600, ctd120: 3800, ctdCov65: 0.23, ctdMax: 255 },
  });

  assert.equal(res.reject, false);
});

test('PRESERVES real single-character and multi-character dialogue inside white speech bubbles', () => {
  const w = 140, h = 90;
  const img = createCanvas(w, h, 255, 255, 255);
  // Draw two letters "HI" (vertical bars + crossbar)
  drawThickLine(img, w, h, 26, 24, 26, 66, 6, 15, 15, 15);
  drawThickLine(img, w, h, 52, 24, 52, 66, 6, 15, 15, 15);
  drawThickLine(img, w, h, 26, 45, 52, 45, 6, 15, 15, 15);
  drawThickLine(img, w, h, 78, 24, 78, 66, 6, 15, 15, 15);

  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 10, y: 10, w: 120, h: 70 },
    confidence: 0.82,
    insideBubble: true,
    surroundStats: WHITE_BUBBLE_SURROUND,
    ctdMetrics: { ctd65: 1800, ctd120: 1400, ctdCov65: 0.21, ctdMax: 255 },
  });

  assert.equal(res.reject, false);
});

test('PRESERVES cream/peach narration boxes (uniform low-variance surround)', () => {
  const w = 260, h = 120;
  const img = createCanvas(w, h, 248, 226, 198); // uniform peach narration box
  // Multi-stroke lettering inside box
  for (let x = 30; x <= 210; x += 28) {
    drawThickLine(img, w, h, x, 35, x + 14, 35, 4, 35, 28, 22);
    drawThickLine(img, w, h, x + 7, 35, x + 7, 75, 4, 35, 28, 22);
  }
  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 10, y: 10, w: 240, h: 100 },
    confidence: 0.92,
    insideBubble: true,
    surroundStats: makeSurround({
      meanR: 248,
      meanG: 226,
      meanB: 198,
      meanLuma: 229,
      stdLuma: 4,
      edgeEnergy: 6,
      isUniform: true,
      isWhiteBubble: false,
    }),
    ctdMetrics: { ctd65: 4200, ctd120: 3600, ctdCov65: 0.17, ctdMax: 255 },
  });

  assert.equal(res.reject, false);
});

test('rejects border-straddling vertical comic stitch mark on speech bubble edge', () => {
  const w = 100, h = 200;
  const img = createCanvas(w, h, 255, 255, 255);
  // Dark background above bubble (y = 0..45)
  for (let y = 0; y < 45; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      img[i] = 52;
      img[i + 1] = 60;
      img[i + 2] = 78;
    }
  }
  // Single continuous vertical stitch stem + 3 short cross-ticks
  drawThickLine(img, w, h, 48, 22, 52, 168, 4, 15, 15, 15);
  drawThickLine(img, w, h, 36, 52, 62, 56, 3, 15, 15, 15);
  drawThickLine(img, w, h, 35, 92, 63, 95, 3, 15, 15, 15);
  drawThickLine(img, w, h, 38, 134, 62, 128, 3, 15, 15, 15);

  const res = evaluateNonTextGuard({
    rgba: img,
    imgWidth: w,
    imgHeight: h,
    bbox: { x: 10, y: 10, w: 76, h: 170 },
    confidence: 0.95,
    insideBubble: true,
    surroundStats: WHITE_BUBBLE_SURROUND,
    ctdMetrics: { ctd65: 1400, ctd120: 900, ctdCov65: 0.11, ctdMax: 240 },
  });

  assert.equal(res.reject, true);
  assert.equal(res.reason, 'NON_TEXT_UI_PATTERN');
});

