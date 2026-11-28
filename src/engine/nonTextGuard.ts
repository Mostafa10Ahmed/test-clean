import { BoundingBox, SurroundStats } from './types';
import { rgbToLuma } from './cleanerCore';

export type NonTextGuardReason =
  | 'NO_TEXT_CONSENSUS'
  | 'NON_TRANSLATABLE_X_MARK'
  | 'NON_TRANSLATABLE_CHECKMARK'
  | 'NON_TRANSLATABLE_CIRCLE_MARK'
  | 'CHARACTER_FACE'
  | 'NON_TEXT_UI_PATTERN';

export interface CtdBoxMetrics {
  ctd65: number;
  ctd120: number;
  ctdCov65: number;
  ctdMax: number;
}

export interface NonTextGuardInput {
  rgba: Uint8Array;
  imgWidth: number;
  imgHeight: number;
  bbox: BoundingBox;
  confidence: number;
  insideBubble: boolean;
  surroundStats: SurroundStats;
  ctdMetrics?: CtdBoxMetrics | null;
  fromCtdOnly?: boolean;
}

export interface NonTextGuardDecision {
  reject: boolean;
  reason?: NonTextGuardReason;
  detail?: string;
}

/**
 * Computes pixel-level Comic-Text-Detector (CTD) confidence statistics inside `bbox`.
 */
export function computeCtdBoxMetrics(
  ctdSegMask: Uint8Array | null,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): CtdBoxMetrics | null {
  if (!ctdSegMask) return null;
  const x0 = Math.max(0, Math.floor(bbox.x));
  const y0 = Math.max(0, Math.floor(bbox.y));
  const x1 = Math.min(imgWidth, Math.ceil(bbox.x + bbox.w));
  const y1 = Math.min(imgHeight, Math.ceil(bbox.y + bbox.h));
  const total = Math.max(1, (x1 - x0) * (y1 - y0));

  let ctd65 = 0;
  let ctd120 = 0;
  let ctdMax = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * imgWidth;
    for (let x = x0; x < x1; x++) {
      const v = ctdSegMask[row + x];
      if (v > ctdMax) ctdMax = v;
      if (v >= 65) ctd65++;
      if (v >= 120) ctd120++;
    }
  }

  return {
    ctd65,
    ctd120,
    ctdCov65: ctd65 / total,
    ctdMax,
  };
}

/**
 * Detects warm anime/manhwa skin chromaticity (ivory, peach, warm tan).
 */
export function isAnimeSkinColor(r: number, g: number, b: number): boolean {
  return (
    r >= 165 &&
    g >= 115 &&
    b >= 85 &&
    r > g &&
    g > b &&
    r - b >= 14 &&
    r - b <= 95 &&
    r - g <= 58
  );
}

interface ConnectedComponent {
  area: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  touchesBorderEdges: number;
  mask: Uint8Array; // tight (gw x gh) binary mask
  gw: number;
  gh: number;
}

function extractBinaryComponents(
  binary: Uint8Array,
  w: number,
  h: number,
  minArea = 20
): ConnectedComponent[] {
  const vis = new Uint8Array(w * h);
  const q = new Int32Array(w * h);
  const comps: ConnectedComponent[] = [];

  for (let i = 0; i < w * h; i++) {
    if (!binary[i] || vis[i]) continue;
    let hd = 0;
    let tl = 0;
    q[tl++] = i;
    vis[i] = 1;

    let minX = w;
    let minY = h;
    let maxX = -1;
    let maxY = -1;
    let touchTop = false;
    let touchBot = false;
    let touchLeft = false;
    let touchRight = false;

    while (hd < tl) {
      const c = q[hd++];
      const cy = Math.floor(c / w);
      const cx = c - cy * w;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
      if (cy === 0) touchTop = true;
      if (cy === h - 1) touchBot = true;
      if (cx === 0) touchLeft = true;
      if (cx === w - 1) touchRight = true;

      if (cy > 0 && binary[c - w] && !vis[c - w]) {
        vis[c - w] = 1;
        q[tl++] = c - w;
      }
      if (cy + 1 < h && binary[c + w] && !vis[c + w]) {
        vis[c + w] = 1;
        q[tl++] = c + w;
      }
      if (cx > 0 && binary[c - 1] && !vis[c - 1]) {
        vis[c - 1] = 1;
        q[tl++] = c - 1;
      }
      if (cx + 1 < w && binary[c + 1] && !vis[c + 1]) {
        vis[c + 1] = 1;
        q[tl++] = c + 1;
      }
    }

    if (tl < minArea) continue;
    const gw = maxX - minX + 1;
    const gh = maxY - minY + 1;
    const subMask = new Uint8Array(gw * gh);
    for (let k = 0; k < tl; k++) {
      const idx = q[k];
      const py = Math.floor(idx / w) - minY;
      const px = idx - Math.floor(idx / w) * w - minX;
      subMask[py * gw + px] = 1;
    }

    comps.push({
      area: tl,
      x0: minX,
      y0: minY,
      x1: maxX,
      y1: maxY,
      touchesBorderEdges:
        (touchTop ? 1 : 0) + (touchBot ? 1 : 0) + (touchLeft ? 1 : 0) + (touchRight ? 1 : 0),
      mask: subMask,
      gw,
      gh,
    });
  }

  comps.sort((a, b) => b.area - a.area);
  return comps;
}

function countEnclosedHolePixels(mask: Uint8Array, gw: number, gh: number): number {
  const pw = gw + 2;
  const ph = gh + 2;
  const visited = new Uint8Array(pw * ph);
  const q = new Int32Array(pw * ph);
  let hd = 0;
  let tl = 0;
  q[tl++] = 0;
  visited[0] = 1;

  while (hd < tl) {
    const c = q[hd++];
    const cy = Math.floor(c / pw);
    const cx = c - cy * pw;

    const tryPush = (nx: number, ny: number) => {
      if (nx < 0 || nx >= pw || ny < 0 || ny >= ph) return;
      const ni = ny * pw + nx;
      if (visited[ni]) return;
      const mx = nx - 1;
      const my = ny - 1;
      if (mx >= 0 && mx < gw && my >= 0 && my < gh && mask[my * gw + mx] > 0) return;
      visited[ni] = 1;
      q[tl++] = ni;
    };

    tryPush(cx - 1, cy);
    tryPush(cx + 1, cy);
    tryPush(cx, cy - 1);
    tryPush(cx, cy + 1);
  }

  let holePixels = 0;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      if (mask[y * gw + x] === 0 && visited[(y + 1) * pw + (x + 1)] === 0) {
        holePixels++;
      }
    }
  }
  return holePixels;
}

function computeEdgeOrientationStats(
  rgba: Uint8Array,
  imgWidth: number,
  x: number,
  y: number,
  w: number,
  h: number
): {
  total: number;
  orthoFrac: number;
  diagPosFrac: number;
  diagNegFrac: number;
  diagAsym: number;
} {
  let ortho = 0;
  let diagPos = 0;
  let diagNeg = 0;
  let total = 0;

  for (let ly = 1; ly < h - 1; ly++) {
    for (let lx = 1; lx < w - 1; lx++) {
      const iL = ((y + ly) * imgWidth + (x + lx - 1)) * 4;
      const iR = ((y + ly) * imgWidth + (x + lx + 1)) * 4;
      const iT = ((y + ly - 1) * imgWidth + (x + lx)) * 4;
      const iB = ((y + ly + 1) * imgWidth + (x + lx)) * 4;

      const gx =
        rgbToLuma(rgba[iR], rgba[iR + 1], rgba[iR + 2]) -
        rgbToLuma(rgba[iL], rgba[iL + 1], rgba[iL + 2]);
      const gy =
        rgbToLuma(rgba[iB], rgba[iB + 1], rgba[iB + 2]) -
        rgbToLuma(rgba[iT], rgba[iT + 1], rgba[iT + 2]);

      const mag = Math.hypot(gx, gy);
      if (mag < 28) continue;
      total++;
      const ax = Math.abs(gx);
      const ay = Math.abs(gy);
      if (ax > ay * 2.5 || ay > ax * 2.5) ortho++;
      else if (gx * gy > 0) diagPos++;
      else diagNeg++;
    }
  }

  return {
    total,
    orthoFrac: ortho / Math.max(1, total),
    diagPosFrac: diagPos / Math.max(1, total),
    diagNegFrac: diagNeg / Math.max(1, total),
    diagAsym: Math.abs(diagPos - diagNeg) / Math.max(1, diagPos + diagNeg),
  };
}

function computeHorizontalPeriodicity(
  rgba: Uint8Array,
  imgWidth: number,
  x: number,
  y: number,
  w: number,
  h: number
): number {
  if (w < 40 || h < 12) return 0;
  const y0 = Math.floor(h * 0.2);
  const y1 = Math.max(y0 + 1, Math.floor(h * 0.8));
  const prof = new Float64Array(w);
  for (let lx = 0; lx < w; lx++) {
    let s = 0;
    for (let ly = y0; ly < y1; ly++) {
      const idx = ((y + ly) * imgWidth + (x + lx)) * 4;
      s += rgbToLuma(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
    }
    prof[lx] = s / (y1 - y0);
  }

  let mean = 0;
  for (let i = 0; i < w; i++) mean += prof[i];
  mean /= w;
  let variance = 0;
  for (let i = 0; i < w; i++) {
    prof[i] -= mean;
    variance += prof[i] * prof[i];
  }
  if (variance < w * 15) return 0;

  const minLag = 8;
  const maxLag = Math.min(Math.floor(w / 3), 90);
  let maxCorr = 0;
  let dipped = false;

  for (let lag = 3; lag <= maxLag; lag++) {
    let num = 0;
    let d1 = 0;
    let d2 = 0;
    const n = w - lag;
    for (let i = 0; i < n; i++) {
      num += prof[i] * prof[i + lag];
      d1 += prof[i] * prof[i];
      d2 += prof[i + lag] * prof[i + lag];
    }
    const corr = num / Math.max(1e-6, Math.sqrt(d1 * d2));
    if (corr < 0.35) dipped = true;
    if (dipped && lag >= minLag && corr > maxCorr) {
      maxCorr = corr;
    }
  }
  return maxCorr;
}

function isXCrossGlyph(
  comp: ConnectedComponent,
  isColored: boolean,
  edgeStats: { orthoFrac: number; diagPosFrac: number; diagNegFrac: number }
): boolean {
  const { gw, gh, mask, area } = comp;
  if (gw < 14 || gh < 14 || gw > 145 || gh > 145) return false;
  const aspect = gw / Math.max(1, gh);
  if (aspect < 0.58 || aspect > 1.72) return false;

  const fillFrac = area / Math.max(1, gw * gh);
  if (fillFrac < 0.14 || fillFrac > 0.68) return false;

  // 3x3 grid densities
  const grid = new Float64Array(9);
  const gridTot = new Float64Array(9);
  let diagHits = 0;

  // Outer 22% mid-edge notch bands (top, bottom, left, right)
  let topNotch = 0, topNotchTot = 0;
  let botNotch = 0, botNotchTot = 0;
  let leftNotch = 0, leftNotchTot = 0;
  let rightNotch = 0, rightNotchTot = 0;

  for (let y = 0; y < gh; y++) {
    const ny = gh > 1 ? y / (gh - 1) : 0.5;
    const gy = Math.min(2, Math.floor((y / gh) * 3));
    for (let x = 0; x < gw; x++) {
      const nx = gw > 1 ? x / (gw - 1) : 0.5;
      const gx = Math.min(2, Math.floor((x / gw) * 3));
      const v = mask[y * gw + x];
      const gi = gy * 3 + gx;
      gridTot[gi]++;
      if (v) {
        grid[gi]++;
        if (Math.abs(nx - ny) <= 0.36 || Math.abs(nx + ny - 1) <= 0.36) {
          diagHits++;
        }
      }

      if (nx >= 0.36 && nx <= 0.64 && ny <= 0.22) {
        topNotchTot++;
        if (v) topNotch++;
      }
      if (nx >= 0.36 && nx <= 0.64 && ny >= 0.78) {
        botNotchTot++;
        if (v) botNotch++;
      }
      if (ny >= 0.36 && ny <= 0.64 && nx <= 0.22) {
        leftNotchTot++;
        if (v) leftNotch++;
      }
      if (ny >= 0.36 && ny <= 0.64 && nx >= 0.78) {
        rightNotchTot++;
        if (v) rightNotch++;
      }
    }
  }

  const d = Array.from({ length: 9 }, (_, i) => grid[i] / Math.max(1, gridTot[i]));
  const center = d[4];
  const corners = [d[0], d[2], d[6], d[8]];
  const minCorner = Math.min(...corners);
  const avgCorner = corners.reduce((a, b) => a + b, 0) / 4;

  const notches = [
    topNotch / Math.max(1, topNotchTot),
    botNotch / Math.max(1, botNotchTot),
    leftNotch / Math.max(1, leftNotchTot),
    rightNotch / Math.max(1, rightNotchTot),
  ];
  const emptyNotches = notches.filter((n) => n <= 0.18).length;
  const avgNotch = notches.reduce((a, b) => a + b, 0) / 4;
  const diagFrac = diagHits / Math.max(1, area);

  // Core X-cross invariants:
  // 1) Solid crossing center
  // 2) All 4 diagonal corners populated
  // 3) Outer cardinal mid-edges notched/empty
  // 4) Ink concentrated along the two diagonals
  if (center < 0.45) return false;
  if (minCorner < 0.18 || avgCorner < 0.26) return false;
  if (emptyNotches < 3 || avgNotch > 0.18) return false;
  if (avgCorner < avgNotch * 1.6 + 0.08) return false;
  if (diagFrac < 0.72) return false;

  // Require balanced diagonal edge signature for monochrome (or colored confirmation)
  if (!isColored) {
    const diagEdgeSum = edgeStats.diagPosFrac + edgeStats.diagNegFrac;
    if (diagEdgeSum < 0.45 || edgeStats.orthoFrac > 0.52) return false;
  }

  return true;
}

function isCheckmarkGlyph(comp: ConnectedComponent): boolean {
  const { gw, gh, mask, area } = comp;
  if (gw < 14 || gh < 14 || gw > 155 || gh > 155) return false;
  const aspect = gw / Math.max(1, gh);
  if (aspect < 0.52 || aspect > 2.15) return false;

  const fillFrac = area / Math.max(1, gw * gh);
  if (fillFrac < 0.06 || fillFrac > 0.46) return false;

  // Checkmark is an open V-tick with zero enclosed holes
  const holePx = countEnclosedHolePixels(mask, gw, gh);
  if (holePx > Math.max(6, area * 0.03)) return false;

  // Inspect spatial quadrants & column trajectory
  let botXSum = 0;
  let botCnt = 0;
  let topLeftHits = 0, topLeftTot = 0;
  let botRightHits = 0, botRightTot = 0;
  let leftYSum = 0, leftCnt = 0;
  let rightMinY = gh, rightCnt = 0;

  for (let y = 0; y < gh; y++) {
    const ny = gh > 1 ? y / (gh - 1) : 0.5;
    for (let x = 0; x < gw; x++) {
      const nx = gw > 1 ? x / (gw - 1) : 0.5;
      const v = mask[y * gw + x];

      if (nx <= 0.35 && ny <= 0.25) {
        topLeftTot++;
        if (v) topLeftHits++;
      }
      if (nx >= 0.66 && ny >= 0.62) {
        botRightTot++;
        if (v) botRightHits++;
      }
      if (!v) continue;

      if (ny >= 0.84) {
        botXSum += nx;
        botCnt++;
      }
      if (nx <= 0.22) {
        leftYSum += ny;
        leftCnt++;
      }
      if (nx >= 0.80) {
        if (ny < rightMinY) rightMinY = ny;
        rightCnt++;
      }
    }
  }

  if (botCnt === 0 || leftCnt === 0 || rightCnt === 0) return false;

  const xAtBottom = botXSum / botCnt;
  const leftAvgY = leftYSum / leftCnt;
  const topLeftFill = topLeftHits / Math.max(1, topLeftTot);
  const botRightFill = botRightHits / Math.max(1, botRightTot);

  // V-trough lies left-of-center (16%..58%), top-left and bottom-right are empty,
  // left arm starts lower than the top-right tip
  if (xAtBottom < 0.16 || xAtBottom > 0.58) return false;
  if (topLeftFill > 0.08 || botRightFill > 0.08) return false;
  if (rightMinY > 0.36) return false;
  if (leftAvgY < rightMinY + 0.18 || leftAvgY > 0.88) return false;

  // Verify clean ascending single-stroke right arm from xAtBottom to ~90% width
  const startCol = Math.min(gw - 2, Math.max(1, Math.round(xAtBottom * (gw - 1))));
  const endCol = Math.min(gw - 1, Math.round(0.92 * (gw - 1)));
  if (endCol - startCol < 5) return false;

  let prevY = 1.0;
  let monoViolations = 0;
  let colSamples = 0;
  let spanSum = 0;
  let firstColY = -1;
  let lastColY = -1;

  for (let x = startCol; x <= endCol; x++) {
    let minY = gh;
    let maxY = -1;
    let sumY = 0;
    let cnt = 0;
    for (let y = 0; y < gh; y++) {
      if (mask[y * gw + x]) {
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumY += y;
        cnt++;
      }
    }
    if (cnt === 0) continue;
    const avgY = sumY / cnt / Math.max(1, gh - 1);
    const span = (maxY - minY + 1) / Math.max(1, gh);
    spanSum += span;
    colSamples++;
    if (firstColY < 0) firstColY = avgY;
    lastColY = avgY;
    // Ascending stroke means avgY should decrease (allowing tiny anti-alias jitter)
    if (avgY > prevY + 0.08) monoViolations++;
    prevY = Math.min(prevY, avgY);
  }

  if (colSamples < 4) return false;
  const avgColSpan = spanSum / colSamples;
  const rise = firstColY - lastColY;

  if (rise < 0.45) return false;
  if (avgColSpan > 0.38) return false;
  if (monoViolations > Math.max(1, Math.floor(colSamples * 0.15))) return false;

  return true;
}

function isHollowCircleMarkGlyph(
  comp: ConnectedComponent,
  isColored: boolean,
  confidence: number
): boolean {
  if (!isColored && confidence >= 0.65) return false;
  const { gw, gh, mask, area } = comp;
  if (gw < 18 || gh < 18 || gw > 140 || gh > 140) return false;
  const aspect = gw / Math.max(1, gh);
  if (aspect < 0.72 || aspect > 1.38) return false;

  const holePx = countEnclosedHolePixels(mask, gw, gh);
  if (holePx < gw * gh * 0.16) return false;

  // 3x3 center empty, 4 cardinal sides populated
  const grid = new Float64Array(9);
  const gridTot = new Float64Array(9);
  for (let y = 0; y < gh; y++) {
    const gy = Math.min(2, Math.floor((y / gh) * 3));
    for (let x = 0; x < gw; x++) {
      const gx = Math.min(2, Math.floor((x / gw) * 3));
      const gi = gy * 3 + gx;
      gridTot[gi]++;
      if (mask[y * gw + x]) grid[gi]++;
    }
  }
  const d = Array.from({ length: 9 }, (_, i) => grid[i] / Math.max(1, gridTot[i]));
  if (d[4] > 0.04) return false;
  if (Math.min(d[1], d[3], d[5], d[7]) < 0.22) return false;

  // High H & V symmetry
  let interH = 0, unionH = 0, interV = 0, unionV = 0;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const a = mask[y * gw + x];
      const ah = mask[y * gw + (gw - 1 - x)];
      const av = mask[(gh - 1 - y) * gw + x];
      if (a || ah) unionH++;
      if (a && ah) interH++;
      if (a || av) unionV++;
      if (a && av) interV++;
    }
  }
  const symH = interH / Math.max(1, unionH);
  const symV = interV / Math.max(1, unionV);
  return symH >= 0.72 && symV >= 0.72 && area / (gw * gh) <= 0.62;
}

/**
 * Detects standalone non-translatable graphic marks (`X` / `✕` / `❌`, `✓` / `✔`, `⭕` / `○`)
 * whether colored (red X, green checkmark) or monochrome, inside a speech bubble or UI badge.
 */
export function detectXOrCheckmarkOrCircleSymbol(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox,
  confidence: number
): { matched: boolean; reason?: NonTextGuardReason; detail?: string } {
  const x = Math.max(0, Math.floor(bbox.x));
  const y = Math.max(0, Math.floor(bbox.y));
  const w = Math.min(imgWidth - x, Math.ceil(bbox.w));
  const h = Math.min(imgHeight - y, Math.ceil(bbox.h));
  if (w < 16 || h < 16 || w > 220 || h > 220) return { matched: false };

  const totalPx = w * h;
  const redMask = new Uint8Array(totalPx);
  const greenMask = new Uint8Array(totalPx);
  let redCount = 0;
  let greenCount = 0;
  let brightCount = 0;
  const lumas: number[] = [];

  for (let ly = 0; ly < h; ly++) {
    for (let lx = 0; lx < w; lx++) {
      const idx = ((y + ly) * imgWidth + (x + lx)) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];
      const l = rgbToLuma(r, g, b);
      lumas.push(l);
      if (l >= 215) brightCount++;
      if (r >= 100 && r - g >= 35 && r - b >= 35) {
        redMask[ly * w + lx] = 1;
        redCount++;
      }
      if (g >= 95 && g - r >= 28 && g - b >= 18) {
        greenMask[ly * w + lx] = 1;
        greenCount++;
      }
    }
  }

  const edgeStats = computeEdgeOrientationStats(rgba, imgWidth, x, y, w, h);

  const testMaskCandidate = (
    binary: Uint8Array,
    isColored: boolean,
    label: string
  ): { matched: boolean; reason?: NonTextGuardReason; detail?: string } | null => {
    const rawComps = extractBinaryComponents(binary, w, h, 24);
    if (rawComps.length === 0) return null;

    // Filter out speech-bubble outer outlines that wrap around >= 2 bbox edges and span > 80% of box
    const interiorComps = rawComps.filter(
      (c) => !(c.touchesBorderEdges >= 2 && (c.gw >= w * 0.82 || c.gh >= h * 0.82))
    );
    const comps = interiorComps.length > 0 ? interiorComps : rawComps;
    if (comps.length === 0) return null;

    const top = comps[0];
    const totalInteriorArea = comps.reduce((s, c) => s + c.area, 0);
    // Must be a single standalone symbol (not multi-letter text)
    if (top.area / Math.max(1, totalInteriorArea) < 0.78) return null;
    if (comps.length >= 2 && comps[1].area >= top.area * 0.18) return null;

    if (isXCrossGlyph(top, isColored, edgeStats)) {
      return {
        matched: true,
        reason: 'NON_TRANSLATABLE_X_MARK',
        detail: `${label} X-mark (${top.gw}x${top.gh})`,
      };
    }
    if (isCheckmarkGlyph(top)) {
      return {
        matched: true,
        reason: 'NON_TRANSLATABLE_CHECKMARK',
        detail: `${label} checkmark (${top.gw}x${top.gh})`,
      };
    }
    if (isHollowCircleMarkGlyph(top, isColored, confidence)) {
      return {
        matched: true,
        reason: 'NON_TRANSLATABLE_CIRCLE_MARK',
        detail: `${label} circle symbol (${top.gw}x${top.gh})`,
      };
    }
    return null;
  };

  // Pass 1: Red/crimson symbol (e.g. ❌ inside white bubble or UI button)
  if (redCount >= 60 && redCount <= totalPx * 0.55) {
    const hit = testMaskCandidate(redMask, true, 'red');
    if (hit) return hit;
  }

  // Pass 2: Green/teal symbol (e.g. ✅ / ✓ checkmark)
  if (greenCount >= 40 && greenCount <= totalPx * 0.55) {
    const hit = testMaskCandidate(greenMask, true, 'green');
    if (hit) return hit;
  }

  // Pass 3: Monochrome ink symbol against interior background (e.g. black X / ✓ on white, or white X / ✓ on dark)
  lumas.sort((a, b) => a - b);
  const bgLuma =
    brightCount >= totalPx * 0.35 ? 245 : lumas[Math.floor(lumas.length * 0.5)];
  const monoMask = new Uint8Array(totalPx);
  for (let ly = 0; ly < h; ly++) {
    for (let lx = 0; lx < w; lx++) {
      const idx = ((y + ly) * imgWidth + (x + lx)) * 4;
      const l = rgbToLuma(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
      if (Math.abs(l - bgLuma) >= 65) {
        monoMask[ly * w + lx] = 1;
      }
    }
  }
  const monoHit = testMaskCandidate(monoMask, false, 'monochrome');
  if (monoHit) return monoHit;

  return { matched: false };
}

/**
 * Protects both realistic skin-tone character faces (eyes/brows/mouth over skin)
 * and white/stylized chibi faces (round head with two symmetric eyes) from accidental erasure.
 */
export function detectCharacterFaceRegion(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox,
  surroundStats: SurroundStats,
  confidence: number,
  ctdMetrics?: CtdBoxMetrics | null
): { matched: boolean; detail?: string } {
  // Never confuse uniform flat cream/white speech/narration bubbles with character faces
  if (surroundStats.isUniform && surroundStats.stdLuma <= 11 && surroundStats.edgeEnergy <= 18) {
    return { matched: false };
  }

  const x = Math.max(0, Math.floor(bbox.x));
  const y = Math.max(0, Math.floor(bbox.y));
  const w = Math.min(imgWidth - x, Math.ceil(bbox.w));
  const h = Math.min(imgHeight - y, Math.ceil(bbox.h));
  if (w < 24 || h < 24) return { matched: false };

  // Strong neural text consensus (e.g. large Korean dialogue printed directly across a character's face like `아니다.`)
  // should NOT be rejected as a bare face.
  const hasStrongTextConsensus =
    ctdMetrics != null &&
    ctdMetrics.ctdCov65 >= 0.16 &&
    ctdMetrics.ctd120 >= 300 &&
    confidence >= 0.62;
  if (hasStrongTextConsensus) {
    return { matched: false };
  }

  const pad = 12;
  const sx0 = Math.max(0, x - pad);
  const sy0 = Math.max(0, y - pad);
  const sx1 = Math.min(imgWidth, x + w + pad);
  const sy1 = Math.min(imgHeight, y + h + pad);

  let skinIn = 0;
  let totIn = 0;
  let skinRing = 0;
  let totRing = 0;
  let whiteIn = 0;
  const darkMask = new Uint8Array(w * h);

  for (let py = sy0; py < sy1; py++) {
    for (let px = sx0; px < sx1; px++) {
      const idx = (py * imgWidth + px) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];
      const inside = px >= x && px < x + w && py >= y && py < y + h;
      const sk = isAnimeSkinColor(r, g, b);
      if (inside) {
        totIn++;
        if (sk) skinIn++;
        const l = rgbToLuma(r, g, b);
        if (l >= 225) whiteIn++;
        if (l < 95) {
          darkMask[(py - y) * w + (px - x)] = 1;
        }
      } else {
        totRing++;
        if (sk) skinRing++;
      }
    }
  }

  const skinInFrac = skinIn / Math.max(1, totIn);
  const skinRingFrac = skinRing / Math.max(1, totRing);
  const whiteInFrac = whiteIn / Math.max(1, totIn);

  // Extract interior dark blobs (potential eyes/mouth/brows) ignoring outer border-touching hair/frame
  const darkComps = extractBinaryComponents(darkMask, w, h, Math.max(12, Math.floor(w * h * 0.0015)))
    .filter((c) => !(c.touchesBorderEdges >= 2 && (c.gw >= w * 0.78 || c.gh >= h * 0.78)));

  // Check for horizontally paired symmetric eye blobs in the upper/middle 70% of the region
  let hasPairedEyes = false;
  for (let i = 0; i < Math.min(darkComps.length, 8); i++) {
    const a = darkComps[i];
    const acx = (a.x0 + a.x1) / 2;
    const acy = (a.y0 + a.y1) / 2;
    if (acy > h * 0.72 || acy < h * 0.12) continue;
    if (a.gw > w * 0.42 || a.gh > h * 0.48) continue;

    for (let j = i + 1; j < Math.min(darkComps.length, 8); j++) {
      const b = darkComps[j];
      const bcx = (b.x0 + b.x1) / 2;
      const bcy = (b.y0 + b.y1) / 2;
      if (bcy > h * 0.72 || bcy < h * 0.12) continue;
      if (b.gw > w * 0.42 || b.gh > h * 0.48) continue;

      const dy = Math.abs(acy - bcy);
      const dx = Math.abs(acx - bcx);
      const sizeRatio = Math.max(a.area, b.area) / Math.max(1, Math.min(a.area, b.area));

      if (
        dy <= h * 0.14 &&
        dx >= w * 0.18 &&
        dx <= w * 0.68 &&
        sizeRatio <= 2.3 &&
        Math.min(acx, bcx) <= w * 0.48 &&
        Math.max(acx, bcx) >= w * 0.52
      ) {
        hasPairedEyes = true;
        break;
      }
    }
    if (hasPairedEyes) break;
  }

  // Case 1: Realistic skin-tone face with paired dark eyes and weak/missing text confirmation
  if (skinInFrac >= 0.28 && hasPairedEyes && darkComps.length <= 7) {
    const weakCtd = !ctdMetrics || ctdMetrics.ctdCov65 < 0.14 || ctdMetrics.ctd120 < 220;
    if (weakCtd || confidence < 0.62) {
      return {
        matched: true,
        detail: `skin portrait with paired eyes (skinIn=${skinInFrac.toFixed(2)}, comps=${darkComps.length})`,
      };
    }
  }

  // Case 2: Heavy skin-tone region with low CTD text confirmation (nose/cheek/chin/neck/face crop)
  if (skinInFrac >= 0.34 && skinRingFrac >= 0.24 && darkComps.length <= 5) {
    const weakCtd = !ctdMetrics || ctdMetrics.ctdCov65 < 0.08 || ctdMetrics.ctd120 < 120;
    if (weakCtd && confidence < 0.62) {
      return {
        matched: true,
        detail: `low-text skin face/portrait crop (skinIn=${skinInFrac.toFixed(2)}, skinRing=${skinRingFrac.toFixed(2)})`,
      };
    }
  }

  // Case 3: White/stylized round chibi face (e.g. 😱 round white head with 2 symmetric eyes + mouth on dark/colored background)
  const aspect = w / Math.max(1, h);
  if (
    aspect >= 0.65 &&
    aspect <= 1.55 &&
    whiteInFrac >= 0.25 &&
    !surroundStats.isWhiteBubble &&
    hasPairedEyes &&
    darkComps.length <= 6
  ) {
    const weakCtd = !ctdMetrics || ctdMetrics.ctdCov65 < 0.08 || ctdMetrics.ctd120 < 100;
    if (weakCtd || confidence < 0.52) {
      return {
        matched: true,
        detail: `stylized/chibi face with paired eyes (whiteIn=${whiteInFrac.toFixed(2)}, comps=${darkComps.length})`,
      };
    }
  }

  return { matched: false };
}

/**
 * Rejects purely non-text geometric UI/architectural patterns:
 * - Diagonal striped UI progress bars (`////`)
 * - Purely rectilinear architectural window grilles/bars (`orthoFrac >= 0.86`)
 * - Single tall vertical stitch/zipper/seam lines with minimal CTD text signal
 */
export function detectNonTextUiOrArchitecturalPattern(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox,
  surroundStats: SurroundStats,
  confidence: number,
  ctdMetrics?: CtdBoxMetrics | null
): { matched: boolean; detail?: string } {
  const x = Math.max(0, Math.floor(bbox.x));
  const y = Math.max(0, Math.floor(bbox.y));
  const w = Math.min(imgWidth - x, Math.ceil(bbox.w));
  const h = Math.min(imgHeight - y, Math.ceil(bbox.h));
  if (w < 24 || h < 20) return { matched: false };

  const edgeStats = computeEdgeOrientationStats(rgba, imgWidth, x, y, w, h);

  // 1) One-way diagonal striped UI progress bar (`////` with diagAsym >= 0.78 and periodic repeats)
  if (
    w >= h * 2.2 &&
    confidence < 0.62 &&
    edgeStats.diagAsym >= 0.78 &&
    Math.max(edgeStats.diagPosFrac, edgeStats.diagNegFrac) >= 0.20
  ) {
    const per = computeHorizontalPeriodicity(rgba, imgWidth, x, y, w, h);
    if (per >= 0.45) {
      return {
        matched: true,
        detail: `diagonal UI progress bar (diagAsym=${edgeStats.diagAsym.toFixed(2)}, per=${per.toFixed(2)})`,
      };
    }
  }

  // 2) Purely rectilinear architectural window grille / glowing vertical bar strip (`orthoFrac >= 0.86`)
  if (
    !surroundStats.isWhiteBubble &&
    confidence < 0.65 &&
    edgeStats.total >= 250 &&
    edgeStats.orthoFrac >= 0.86 &&
    (!ctdMetrics || ctdMetrics.ctdCov65 < 0.14)
  ) {
    return {
      matched: true,
      detail: `rectilinear architectural grille/frame (orthoFrac=${edgeStats.orthoFrac.toFixed(2)})`,
    };
  }

  // 3) Repeating traditional wooden/shoji backlit lattice window:
  //    Warm brown/tan wood frame containing >= 28 small BRIGHTER backlit grid panes (`luma > bgLuma + 35`)
  //    with balanced orthogonal symmetry (`diagAsym <= 0.08`). Unlike cream narration boxes (dark ink on
  //    bright paper), backlit window lattices have bright holes inside a darker warm wood frame.
  if (!surroundStats.isWhiteBubble && w >= 120 && h >= 50 && confidence < 0.80) {
    let warmWoodPx = 0;
    let sampled = 0;
    for (let ly = 0; ly < h; ly += 2) {
      for (let lx = 0; lx < w; lx += 2) {
        const idx = ((y + ly) * imgWidth + (x + lx)) * 4;
        const r = rgba[idx];
        const g = rgba[idx + 1];
        const b = rgba[idx + 2];
        if (isAnimeSkinColor(r, g, b) || (r > g && g >= b && r - b >= 18 && r >= 90)) {
          warmWoodPx++;
        }
        sampled++;
      }
    }
    const warmFrac = warmWoodPx / Math.max(1, sampled);
    if (warmFrac >= 0.65 && edgeStats.diagAsym <= 0.08 && edgeStats.orthoFrac >= 0.56) {
      const bgLuma = surroundStats.meanLuma;
      const brightHoleMask = new Uint8Array(w * h);
      for (let ly = 0; ly < h; ly++) {
        for (let lx = 0; lx < w; lx++) {
          const idx = ((y + ly) * imgWidth + (x + lx)) * 4;
          const lum = rgbToLuma(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
          if (lum > bgLuma + 30 && lum >= 175) {
            brightHoleMask[ly * w + lx] = 1;
          }
        }
      }
      const brightComps = extractBinaryComponents(brightHoleMask, w, h, 10);
      if (brightComps.length >= 24) {
        return {
          matched: true,
          detail: `backlit warm architectural lattice (brightPanes=${brightComps.length}, warmFrac=${warmFrac.toFixed(2)})`,
        };
      }
    }
  }

  // 4) Single continuous tall vertical seam/stitch line (e.g. comic stitch mark straddling a speech bubble edge)
  if (h >= w * 1.55 && h >= 90) {
    const darkMask = new Uint8Array(w * h);
    let topLumaSum = 0, topCnt = 0;
    let botLumaSum = 0, botCnt = 0;
    const topBand = Math.max(4, Math.floor(h * 0.14));
    const botStart = Math.floor(h * 0.72);
    for (let ly = 0; ly < h; ly++) {
      for (let lx = 0; lx < w; lx++) {
        const idx = ((y + ly) * imgWidth + (x + lx)) * 4;
        const lum = rgbToLuma(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
        if (lum < 42) {
          darkMask[ly * w + lx] = 1;
        }
        if (ly < topBand) {
          topLumaSum += lum;
          topCnt++;
        } else if (ly >= botStart) {
          botLumaSum += lum;
          botCnt++;
        }
      }
    }
    const topAvg = topLumaSum / Math.max(1, topCnt);
    const botAvg = botLumaSum / Math.max(1, botCnt);
    const straddlesBubbleBorder =
      (topAvg < 155 && botAvg >= 225) || (botAvg < 155 && topAvg >= 225);
    const lowConfOrLowCtd =
      confidence < 0.65 &&
      (!ctdMetrics || (ctdMetrics.ctd120 < 240 && ctdMetrics.ctdCov65 < 0.08));

    if (straddlesBubbleBorder || lowConfOrLowCtd) {
      const comps = extractBinaryComponents(darkMask, w, h, 30);
      if (comps.length >= 1 && comps.length <= 3) {
        const tallest = comps.reduce((best, c) => (c.gh > best.gh ? c : best), comps[0]);
        const inkFill = tallest.area / Math.max(1, tallest.gw * tallest.gh);
        if (tallest.gh >= h * 0.65 && tallest.gw <= w * 0.62 && inkFill <= 0.42) {
          return {
            matched: true,
            detail: `vertical seam/stitch artwork line (tallest=${tallest.gw}x${tallest.gh}, borderStraddle=${straddlesBubbleBorder})`,
          };
        }
      }
    }
  }

  return { matched: false };
}

/**
 * Native-resolution check for genuine multi-glyph coupled stroke+halo text (e.g. heavy-stroke OT
 * `하하..`, `후우..`, `끄덕`, `이런 모습..`, `오소소..`) so face/artwork guards never falsely reject real OT,
 * while standalone artwork false positives (cat eye, uniform embroidery, empty dashed boxes, mouth,
 * breath puffs, sweat drops) are cleanly rejected even when full-page CTD is disabled!
 */
export function analyzeNativeGlyphConsensus(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): {
  hasStrokedGlyphs: boolean;
  interiorGlyphComps: number;
  inkFraction: number;
  contrastSpan: number;
  minLuma: number;
  maxLuma: number;
} {
  const x = Math.max(0, Math.floor(bbox.x));
  const y = Math.max(0, Math.floor(bbox.y));
  const w = Math.min(imgWidth - x, Math.ceil(bbox.w));
  const h = Math.min(imgHeight - y, Math.ceil(bbox.h));
  if (w < 12 || h < 10) {
    return {
      hasStrokedGlyphs: false,
      interiorGlyphComps: 0,
      inkFraction: 0,
      contrastSpan: 0,
      minLuma: 255,
      maxLuma: 0,
    };
  }

  let minLuma = 255;
  let maxLuma = 0;
  const lumas = new Float32Array(w * h);
  const chromas = new Uint8Array(w * h);
  const rimLumas: number[] = [];

  for (let ly = 0; ly < h; ly++) {
    for (let lx = 0; lx < w; lx++) {
      const p = ((y + ly) * imgWidth + (x + lx)) * 4;
      const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
      const l = rgbToLuma(r, g, b);
      const idx = ly * w + lx;
      lumas[idx] = l;
      chromas[idx] = Math.max(r, g, b) - Math.min(r, g, b);
      if (l < minLuma) minLuma = l;
      if (l > maxLuma) maxLuma = l;
      if (lx <= 2 || lx >= w - 3 || ly <= 2 || ly >= h - 3) {
        if ((lx + ly) % 2 === 0) rimLumas.push(l);
      }
    }
  }

  rimLumas.sort((a, b) => a - b);
  const bgLuma = rimLumas.length > 0 ? rimLumas[Math.floor(rimLumas.length * 0.5)] : 240;
  const contrastSpan = maxLuma - minLuma;

  const inkMask = new Uint8Array(w * h);
  let inkPx = 0;
  let coupledStrokePairs = 0;

  for (let ly = 2; ly < h - 2; ly++) {
    for (let lx = 2; lx < w - 2; lx++) {
      const idx = ly * w + lx;
      const l = lumas[idx];
      if (Math.abs(l - bgLuma) >= 42 || (bgLuma >= 140 && l <= 95) || (bgLuma < 100 && l >= 205)) {
        inkMask[idx] = 1;
        inkPx++;
      }
      if ((lx & 1) === 0 && (ly & 1) === 0 && (l <= 115 || (chromas[idx] >= 45 && l <= 175))) {
        let nearWhite = false;
        for (let dy = -5; dy <= 5 && !nearWhite; dy += 2) {
          const ny = Math.min(h - 1, Math.max(0, ly + dy));
          for (let dx = -5; dx <= 5; dx += 2) {
            const nx = Math.min(w - 1, Math.max(0, lx + dx));
            const nIdx = ny * w + nx;
            if (lumas[nIdx] >= 238 && chromas[nIdx] <= 28) {
              nearWhite = true;
              break;
            }
          }
        }
        if (nearWhite) coupledStrokePairs++;
      }
    }
  }

  const comps = extractBinaryComponents(inkMask, w, h, 8).filter(
    (c) => !(c.touchesBorderEdges >= 2 && (c.gw >= w * 0.84 || c.gh >= h * 0.84))
  );

  const hasStrokedGlyphs =
    coupledStrokePairs >= 10 && comps.length >= 2 && contrastSpan >= 115;

  return {
    hasStrokedGlyphs,
    interiorGlyphComps: comps.length,
    inkFraction: inkPx / Math.max(1, w * h),
    contrastSpan,
    minLuma,
    maxLuma,
  };
}

/**
 * Master False-Positive Protection Gate for Manhwa/Manga Cleaning.
 * Prevents erasing character faces, non-translatable symbols (X, ✓, ⭕),
 * decorative ornaments/weapons, UI progress bars, and architectural lattices
 * while preserving genuine dialogue bubbles, OT, ST, and SFX.
 */
export function evaluateNonTextGuard(input: NonTextGuardInput): NonTextGuardDecision {
  const {
    rgba,
    imgWidth,
    imgHeight,
    bbox,
    confidence,
    insideBubble,
    surroundStats,
    ctdMetrics,
  } = input;

  // 1. Neural Text Consensus Gate (when CTD metrics are present):
  if (ctdMetrics && !surroundStats.isWhiteBubble) {
    const isNearlyZeroCtd = ctdMetrics.ctd120 < 25 && ctdMetrics.ctdCov65 < 0.01;
    const isWeakLowConfCtd =
      confidence < 0.48 && ctdMetrics.ctd120 < 150 && ctdMetrics.ctdCov65 < 0.025;
    if ((isNearlyZeroCtd && confidence < 0.72) || isWeakLowConfCtd) {
      return {
        reject: true,
        reason: 'NO_TEXT_CONSENSUS',
        detail: `non-white region has weak CTD text signal (ctd120=${ctdMetrics.ctd120}, cov65=${ctdMetrics.ctdCov65.toFixed(4)}, conf=${confidence.toFixed(2)})`,
      };
    }
  }

  const nativeGlyph = analyzeNativeGlyphConsensus(rgba, imgWidth, imgHeight, bbox);

  // 1b. Native-Resolution Non-Text Consensus Gate (protects empty dashed panels, uniform silver
  // embroidery leaves, cat eyes, mouths, breath puffs, and bokeh dots even without CTD!):
  if (!insideBubble && !surroundStats.isWhiteBubble) {
    // Giant empty / faint dashed panel box (e.g. 720x800 white panel with dashed curve)
    if (bbox.w * bbox.h >= 36000 && nativeGlyph.inkFraction < 0.018 && confidence < 0.78) {
      return {
        reject: true,
        reason: 'NO_TEXT_CONSENSUS',
        detail: `large nearly-empty panel false positive (inkFrac=${nativeGlyph.inkFraction.toFixed(4)})`,
      };
    }
    // Low-contrast grey/pastel embroidery or lace with zero crisp dark or bright text strokes
    if (
      confidence < 0.56 &&
      nativeGlyph.contrastSpan < 105 &&
      nativeGlyph.minLuma > 108 &&
      nativeGlyph.maxLuma < 248 &&
      (!ctdMetrics || ctdMetrics.ctd120 < 180)
    ) {
      return {
        reject: true,
        reason: 'NO_TEXT_CONSENSUS',
        detail: `low-contrast decorative/embroidery pattern (span=${nativeGlyph.contrastSpan.toFixed(0)}, minL=${nativeGlyph.minLuma.toFixed(0)})`,
      };
    }
    // Tiny isolated single-blob artwork mark or single animal/character eye (e.g. cat pupil+iris on 002.jpg,
    // breath puff 'ㅡ', bokeh dot, mouth)
    if (
      confidence < 0.62 &&
      !nativeGlyph.hasStrokedGlyphs &&
      ((nativeGlyph.interiorGlyphComps <= 1 &&
        (bbox.w * bbox.h <= 4200 || nativeGlyph.inkFraction < 0.035)) ||
        (bbox.w <= 72 &&
          bbox.h <= 72 &&
          bbox.w / Math.max(1, bbox.h) >= 0.55 &&
          bbox.w / Math.max(1, bbox.h) <= 1.8 &&
          nativeGlyph.interiorGlyphComps <= 2 &&
          confidence < 0.52)) &&
      (!ctdMetrics || ctdMetrics.ctd120 < 110)
    ) {
      return {
        reject: true,
        reason: 'NO_TEXT_CONSENSUS',
        detail: `isolated single-blob artwork/eye mark (comps=${nativeGlyph.interiorGlyphComps}, area=${bbox.w * bbox.h})`,
      };
    }
  } else if (bbox.w * bbox.h >= 85000 && nativeGlyph.inkFraction < 0.012) {
    return {
      reject: true,
      reason: 'NO_TEXT_CONSENSUS',
      detail: `giant empty bubble/panel box (inkFrac=${nativeGlyph.inkFraction.toFixed(4)})`,
    };
  }

  // 2. Standalone Non-Translatable Symbol Guard (X / ✕ / ❌, ✓ / ✔, ⭕ / ○):
  const symbolHit = detectXOrCheckmarkOrCircleSymbol(rgba, imgWidth, imgHeight, bbox, confidence);
  if (symbolHit.matched) {
    return {
      reject: true,
      reason: symbolHit.reason,
      detail: symbolHit.detail,
    };
  }

  // 3. Character Face & Portrait Guard (skip when strong multi-glyph stroked OT is clearly present):
  if (!nativeGlyph.hasStrokedGlyphs || nativeGlyph.interiorGlyphComps < 3) {
    const faceHit = detectCharacterFaceRegion(
      rgba,
      imgWidth,
      imgHeight,
      bbox,
      surroundStats,
      confidence,
      ctdMetrics
    );
    if (faceHit.matched) {
      return {
        reject: true,
        reason: 'CHARACTER_FACE',
        detail: faceHit.detail,
      };
    }
  }

  // 4. Non-Text UI / Architectural Pattern Guard (progress bars, window grilles, lattices, stitch seams):
  const patternHit = detectNonTextUiOrArchitecturalPattern(
    rgba,
    imgWidth,
    imgHeight,
    bbox,
    surroundStats,
    confidence,
    ctdMetrics
  );
  if (patternHit.matched) {
    return {
      reject: true,
      reason: 'NON_TEXT_UI_PATTERN',
      detail: patternHit.detail,
    };
  }

  return { reject: false };
}
