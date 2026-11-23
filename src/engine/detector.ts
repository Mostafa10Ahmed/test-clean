import {
  computeStripWindows,
  rgbToLuma,
  selectCleaningEngine,
} from './cleanerCore';
import {
  BoundingBox,
  CleaningSettings,
  DetectedRegion,
  RegionCategory,
  SurroundStats,
} from './types';

/**
 * Analyzes the 20px surround ring around a candidate bounding box to measure:
 * - Mean RGB and Luma
 * - Luma spread (std, p5, p95)
 * - Enclosure by a speech bubble outline (`isWhiteBubble`, `isDarkBubble`, `isUniform`)
 * - Periodic screentone vs gradient vs complex art (`isPeriodicScreentone`, `isOutlineDominated`)
 */
export function measureSurroundRing(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): SurroundStats {
  const ringWidth = 28;
  const x0 = Math.max(0, bbox.x - ringWidth);
  const y0 = Math.max(0, bbox.y - ringWidth);
  const x1 = Math.min(imgWidth, bbox.x + bbox.w + ringWidth);
  const y1 = Math.min(imgHeight, bbox.y + bbox.h + ringWidth);

  const allSamples: { r: number; g: number; b: number; l: number; x: number; y: number }[] = [];
  const outerRingSamples: { r: number; g: number; b: number; l: number; x: number; y: number }[] = [];
  const innerRimSamples: { r: number; g: number; b: number; l: number; x: number; y: number }[] = [];
  let outlineDarkPixels = 0;
  let veryDarkOuterPixels = 0;
  let brightOuterSparklePixels = 0;
  let totalRingPixels = 0;
  let transitions = 0;
  let prevLuma = -1;

  for (let y = y0; y < y1; y += 2) {
    for (let x = x0; x < x1; x += 2) {
      const idx = (y * imgWidth + x) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];
      const l = rgbToLuma(r, g, b);

      const inBox =
        x >= bbox.x && x < bbox.x + bbox.w && y >= bbox.y && y < bbox.y + bbox.h;
      const inInnerRim =
        inBox &&
        (x < bbox.x + 8 ||
          x >= bbox.x + bbox.w - 8 ||
          y < bbox.y + 8 ||
          y >= bbox.y + bbox.h - 8);
      const inOuterRing =
        x < bbox.x - 4 ||
        x >= bbox.x + bbox.w + 4 ||
        y < bbox.y - 4 ||
        y >= bbox.y + bbox.h + 4;

      if (inInnerRim) {
        innerRimSamples.push({ r, g, b, l, x, y });
      }

      if (inOuterRing) {
        totalRingPixels++;
        outerRingSamples.push({ r, g, b, l, x, y });
        if (l < 65) outlineDarkPixels++;
        if (r <= 6 && g <= 6 && b <= 6) veryDarkOuterPixels++;
        if (l >= 90 || r >= 75) brightOuterSparklePixels++;
        if (prevLuma >= 0 && Math.abs(l - prevLuma) > 22) {
          transitions++;
        }
        prevLuma = l;
      }

      allSamples.push({ r, g, b, l, x, y });
    }
  }

  if (allSamples.length === 0) {
    return {
      meanR: 255,
      meanG: 255,
      meanB: 255,
      meanLuma: 255,
      stdLuma: 0,
      p5Luma: 255,
      p95Luma: 255,
      edgeEnergy: 0,
      isUniform: true,
      isWhiteBubble: true,
      isDarkBubble: false,
      isPeriodicScreentone: false,
      isGradient: false,
      isOutlineDominated: false,
      gradSlopeX: 0,
      gradSlopeY: 0,
    };
  }

  const sortedByLuma = [...allSamples].sort((a, b) => a.l - b.l);
  const p5Luma = sortedByLuma[Math.floor(sortedByLuma.length * 0.05)].l;
  const p95Luma = sortedByLuma[Math.floor(sortedByLuma.length * 0.95)].l;
  const medianLuma = sortedByLuma[Math.floor(sortedByLuma.length * 0.65)].l;

  // Check if the outer surround ring is predominantly a flat pure-black speech bubble interior
  const darkOuterRatio = veryDarkOuterPixels / Math.max(1, totalRingPixels);
  const sparkleOuterRatio = brightOuterSparklePixels / Math.max(1, totalRingPixels);
  const isLikelyDarkBg = (darkOuterRatio >= 0.48 && sparkleOuterRatio <= 0.18) || medianLuma < 36;

  // Detect translucent / semi-transparent bubbles over artwork using ALL innerRimSamples (unfiltered!)
  let innerHasArtworkShowthrough = false;
  if (innerRimSamples.length >= 16) {
    const innerSortedL = innerRimSamples.map((s) => s.l).sort((a, b) => a - b);
    const innerP10 = innerSortedL[Math.floor(innerSortedL.length * 0.10)];
    const innerP90 = innerSortedL[Math.floor(innerSortedL.length * 0.90)];
    const innerSpread = innerP90 - innerP10;
    const midDarkCount = innerRimSamples.filter((s) => s.l >= 35 && s.l < 228).length;
    const midDarkRatio = midDarkCount / innerRimSamples.length;
    const coloredCount = innerRimSamples.filter(
      (s) => Math.max(Math.abs(s.r - s.g), Math.abs(s.g - s.b), Math.abs(s.r - s.b)) >= 14
    ).length;
    const coloredRatio = coloredCount / innerRimSamples.length;
    // Translucent bubble over artwork shows mid-tone/colored background bleed or >18 luma variation inside the rim
    if (innerSpread > 18 || midDarkRatio > 0.22 || coloredRatio > 0.16) {
      innerHasArtworkShowthrough = true;
    }
  }

  // Prefer innerRimSamples for light bubbles when enough non-stroke samples exist so we measure
  // the actual paper inside the box (preventing outer gutter from masking a gray gradient box!)
  const lightInnerRim = innerRimSamples.filter((s) => s.l >= 165);
  const bgPool =
    !isLikelyDarkBg && lightInnerRim.length >= 24
      ? [...lightInnerRim].sort((a, b) => a.l - b.l)
      : sortedByLuma;

  const sortedOuter = [...outerRingSamples].sort((a, b) => a.l - b.l);
  const bgSlice = isLikelyDarkBg
    ? sortedOuter.length >= 16
      ? sortedOuter.slice(0, Math.max(1, Math.floor(sortedOuter.length * 0.82)))
      : sortedByLuma.slice(0, Math.max(1, Math.floor(sortedByLuma.length * 0.65)))
    : bgPool.slice(
        Math.floor(bgPool.length * 0.18),
        Math.max(1, Math.floor(bgPool.length * 0.94))
      );

  let sumR = 0,
    sumG = 0,
    sumB = 0,
    sumL = 0;
  let leftSum = 0,
    leftCnt = 0;
  let rightSum = 0,
    rightCnt = 0;
  let topSum = 0,
    topCnt = 0;
  let botSum = 0,
    botCnt = 0;

  const midX = bbox.x + bbox.w / 2;
  const midY = bbox.y + bbox.h / 2;

  for (const s of bgSlice) {
    sumR += s.r;
    sumG += s.g;
    sumB += s.b;
    sumL += s.l;
    if (s.x < midX) {
      leftSum += s.l;
      leftCnt++;
    } else {
      rightSum += s.l;
      rightCnt++;
    }
    if (s.y < midY) {
      topSum += s.l;
      topCnt++;
    } else {
      botSum += s.l;
      botCnt++;
    }
  }

  const n = Math.max(1, bgSlice.length);
  const rawMeanR = sumR / n;
  const rawMeanG = sumG / n;
  const rawMeanB = sumB / n;
  const rawMeanLuma = sumL / n;
  const p20BgLuma = bgSlice[Math.floor(bgSlice.length * 0.15)]?.l ?? rawMeanLuma;

  const variance =
    bgSlice.reduce((acc, s) => acc + (s.l - rawMeanLuma) * (s.l - rawMeanLuma), 0) / n;
  const stdLuma = Math.sqrt(variance);

  const horizDiff = leftCnt > 0 && rightCnt > 0 ? rightSum / rightCnt - leftSum / leftCnt : 0;
  const vertDiff = topCnt > 0 && botCnt > 0 ? botSum / botCnt - topSum / topCnt : 0;
  const gradSlopeX = horizDiff / Math.max(1, bbox.w);
  const gradSlopeY = vertDiff / Math.max(1, bbox.h);
  const hasDirectionalGradient = Math.max(Math.abs(horizDiff), Math.abs(vertDiff)) > 4.5;
  const maxChroma = Math.max(
    Math.abs(rawMeanR - rawMeanG),
    Math.abs(rawMeanG - rawMeanB),
    Math.abs(rawMeanR - rawMeanB)
  );

  const outlineFraction = outlineDarkPixels / Math.max(1, totalRingPixels);
  const transitionRate = transitions / Math.max(1, totalRingPixels);

  // Strict pure-white check: off-white (236..249), tinted/translucent (maxChroma > 6), or subtle
  // gradients (hasDirectionalGradient) must NOT be painted with a flat #FFFFFF rectangle!
  const isWhiteBubble =
    !innerHasArtworkShowthrough &&
    rawMeanLuma >= 250.5 &&
    p20BgLuma >= 246 &&
    stdLuma <= 4.0 &&
    maxChroma <= 6.0 &&
    !hasDirectionalGradient;

  // Strict pure-black speech bubble check: requires >= 76% of outer ring to be pure black (<=6)
  // and almost zero bright features (ropes, bamboo leaves, bokeh, palm fronds) crossing the ring.
  const isDarkBubble =
    darkOuterRatio >= 0.76 &&
    sparkleOuterRatio <= 0.04 &&
    rawMeanLuma <= 8.0 &&
    stdLuma <= 4.5 &&
    !hasDirectionalGradient;

  const isUniform =
    (!innerHasArtworkShowthrough && stdLuma <= 3.5 && !hasDirectionalGradient && maxChroma <= 7.0) ||
    isWhiteBubble ||
    isDarkBubble;
  const isPeriodicScreentone = !isUniform && transitionRate > 0.22 && stdLuma < 48;
  const isOutlineDominated = outlineFraction >= 0.12 && outlineFraction <= 0.65 && !isDarkBubble;
  const isGradient =
    !isUniform &&
    !isPeriodicScreentone &&
    (hasDirectionalGradient ||
      Math.abs(gradSlopeX) > 0.05 ||
      Math.abs(gradSlopeY) > 0.05 ||
      (rawMeanLuma >= 175 && stdLuma <= 19.5));

  const snapPureBlack = isDarkBubble && rawMeanLuma <= 12;

  return {
    meanR: isWhiteBubble ? 255 : snapPureBlack ? 0 : rawMeanR,
    meanG: isWhiteBubble ? 255 : snapPureBlack ? 0 : rawMeanG,
    meanB: isWhiteBubble ? 255 : snapPureBlack ? 0 : rawMeanB,
    meanLuma: isWhiteBubble ? 255 : snapPureBlack ? 0 : rawMeanLuma,
    stdLuma,
    p5Luma,
    p95Luma,
    edgeEnergy: stdLuma * 2.4 + transitionRate * 35,
    isUniform,
    isWhiteBubble,
    isDarkBubble,
    isPeriodicScreentone,
    isGradient,
    isOutlineDominated,
    gradSlopeX,
    gradSlopeY,
  };
}

/**
 * Verifies whether a candidate text region sits inside an enclosed speech bubble
 * by walking rays in 8 directions from the text bounding box and checking for
 * a uniform paper band (>= 12px) bounded by a dark bubble border or wide clean margin.
 * Fixes Bear's false-positive on white-outlined SFX and character eyes/collars (`balloon.rs`).
 */
export function verifyBubbleEnclosure(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): { enclosed: boolean; bubbleBbox?: BoundingBox } {
  const cx = Math.floor(bbox.x + bbox.w / 2);
  const cy = Math.floor(bbox.y + bbox.h / 2);
  const maxRay = Math.min(220, Math.floor(Math.max(imgWidth, imgHeight) * 0.28));

  const dirs = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [-1, 1],
    [1, -1],
    [-1, -1],
  ];

  let validRays = 0;
  let minBubbleX = bbox.x;
  let maxBubbleX = bbox.x + bbox.w;
  let minBubbleY = bbox.y;
  let maxBubbleY = bbox.y + bbox.h;

  for (const [dx, dy] of dirs) {
    let step = Math.floor(Math.min(bbox.w, bbox.h) * 0.35);
    let whiteRun = 0;
    let hitOutline = false;
    let endX = cx;
    let endY = cy;

    for (; step < maxRay; step += 2) {
      const px = cx + dx * step;
      const py = cy + dy * step;
      if (px < 2 || px >= imgWidth - 2 || py < 2 || py >= imgHeight - 2) break;

      const idx = (py * imgWidth + px) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];
      const l = rgbToLuma(r, g, b);
      const chromaSpread = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));

      if (l >= 238 && chromaSpread <= 14) {
        whiteRun += 2;
        endX = px;
        endY = py;
      } else if (l < 115 && whiteRun >= 12) {
        hitOutline = true;
        break;
      } else if (l < 210 && whiteRun < 12) {
        break;
      }
    }

    if (hitOutline || whiteRun >= 26) {
      validRays++;
      minBubbleX = Math.min(minBubbleX, endX);
      maxBubbleX = Math.max(maxBubbleX, endX);
      minBubbleY = Math.min(minBubbleY, endX === cx ? endY : endY);
      maxBubbleY = Math.max(maxBubbleY, endY);
    }
  }

  if (validRays >= 6) {
    return {
      enclosed: true,
      bubbleBbox: {
        x: Math.max(0, minBubbleX - 4),
        y: Math.max(0, minBubbleY - 4),
        w: Math.min(imgWidth - Math.max(0, minBubbleX - 4), maxBubbleX - minBubbleX + 8),
        h: Math.min(imgHeight - Math.max(0, minBubbleY - 4), maxBubbleY - minBubbleY + 8),
      },
    };
  }

  return { enclosed: false };
}

/**
 * Checks whether a candidate bounding box has true multi-glyph text rhythm
 * (horizontal and vertical stroke-to-paper alternations) rather than a single
 * continuous character hair/clothing line or panel border.
 */
function hasGlyphStrokePeriodicity(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  bbox: BoundingBox
): { isGlyphCluster: boolean; whitePaperFraction: number; darkInkFraction: number } {
  const x0 = Math.max(0, bbox.x);
  const y0 = Math.max(0, bbox.y);
  const x1 = Math.min(imgWidth, bbox.x + bbox.w);
  const y1 = Math.min(imgHeight, bbox.y + bbox.h);
  const w = x1 - x0;
  const h = y1 - y0;
  if (w < 24 || h < 20) {
    return { isGlyphCluster: false, whitePaperFraction: 0, darkInkFraction: 0 };
  }

  let whitePixels = 0;
  let darkPixels = 0;
  let totalSamples = 0;
  let activeScanlinesWithMultiStrokes = 0;
  let totalScanlinesChecked = 0;

  const stepY = Math.max(2, Math.floor(h / 12));
  for (let y = y0 + 2; y < y1 - 2; y += stepY) {
    totalScanlinesChecked++;
    let inStroke = false;
    let strokeRunsOnLine = 0;
    for (let x = x0 + 2; x < x1 - 2; x += 2) {
      const idx = (y * imgWidth + x) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];
      const l = rgbToLuma(r, g, b);
      const chroma = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));
      totalSamples++;
      if (l >= 238 && chroma <= 16) whitePixels++;
      if (l <= 95) {
        darkPixels++;
        if (!inStroke) {
          inStroke = true;
          strokeRunsOnLine++;
        }
      } else if (l >= 180) {
        inStroke = false;
      }
    }
    // A line of text has at least 2 distinct stroke crossings (usually 3..14 across glyphs)
    if (strokeRunsOnLine >= 2 && strokeRunsOnLine <= 22) {
      activeScanlinesWithMultiStrokes++;
    }
  }

  const whitePaperFraction = whitePixels / Math.max(1, totalSamples);
  const darkInkFraction = darkPixels / Math.max(1, totalSamples);
  const multiStrokeLineRatio =
    activeScanlinesWithMultiStrokes / Math.max(1, totalScanlinesChecked);

  const isGlyphCluster =
    multiStrokeLineRatio >= 0.45 &&
    darkInkFraction >= 0.05 &&
    darkInkFraction <= 0.52 &&
    whitePaperFraction >= 0.35;

  return { isGlyphCluster, whitePaperFraction, darkInkFraction };
}

/**
 * Runs native-resolution connected-component text & speech-bubble detection across an entire
 * page (including ultra-tall 14,000px–28,483px Manhwa long-strips via sliding windows).
 *
 * Categorizes every verified region into one of the 4 independent categories:
 * - BUBBLE (Speech Bubbles, thought bubbles, narration boxes)
 * - OT (Outside Text over panel backgrounds / screentones)
 * - ST (Side Text / vertical edge annotations / narrow margin notes)
 * - SFX (Stylized Sound Effects with high gradient / outlined strokes over artwork)
 */
export function detectRegionsInRgbaBuffer(
  rgba: Uint8Array,
  imgWidth: number,
  imgHeight: number,
  pageIndex: number,
  settings: CleaningSettings
): DetectedRegion[] {
  const cellSize = 16;
  const gridW = Math.floor(imgWidth / cellSize);
  const gridH = Math.floor(imgHeight / cellSize);
  if (gridW < 4 || gridH < 4) return [];

  computeStripWindows(imgHeight, settings.maxStripWindowHeight || 2048, 128);

  const candidateGrid = new Uint8Array(gridW * gridH);
  const cellStrokeRatio = new Float32Array(gridW * gridH);
  const cellBrightRatio = new Float32Array(gridW * gridH);

  // Step 1: Evaluate 16x16 native-resolution cells for glyph stroke bi-modality
  for (let gy = 1; gy < gridH - 1; gy++) {
    const yStart = gy * cellSize;
    for (let gx = 1; gx < gridW - 1; gx++) {
      const xStart = gx * cellSize;
      let darkCount = 0;
      let brightCount = 0;
      let midPureWhite = 0;
      let transitions = 0;
      let minL = 255;
      let maxL = 0;

      for (let dy = 0; dy < cellSize; dy += 2) {
        const py = yStart + dy;
        let prevL = -1;
        for (let dx = 0; dx < cellSize; dx += 2) {
          const px = xStart + dx;
          const idx = (py * imgWidth + px) * 4;
          const r = rgba[idx];
          const g = rgba[idx + 1];
          const b = rgba[idx + 2];
          const l = rgbToLuma(r, g, b);
          const chroma = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));
          if (l < minL) minL = l;
          if (l > maxL) maxL = l;
          if (l < 90) darkCount++;
          if (l > 220) brightCount++;
          if (l > 244 && chroma <= 12) midPureWhite++;
          if (prevL >= 0 && Math.abs(l - prevL) > 55) transitions++;
          prevL = l;
        }
      }

      const samples = 64;
      const dRatio = darkCount / samples;
      const bRatio = brightCount / samples;
      const wRatio = midPureWhite / samples;
      const contrast = maxL - minL;

      const cIdx = gy * gridW + gx;
      cellStrokeRatio[cIdx] = dRatio;
      cellBrightRatio[cIdx] = bRatio;

      if (
        contrast > 115 &&
        dRatio >= 0.08 &&
        dRatio <= 0.55 &&
        bRatio >= 0.35 &&
        transitions >= 6
      ) {
        candidateGrid[cIdx] = wRatio >= 0.45 ? 2 : 1;
      }
    }
  }

  // Step 2: Morphological bridge across adjacent glyph cells
  const bridged = new Uint8Array(gridW * gridH);
  for (let gy = 1; gy < gridH - 1; gy++) {
    for (let gx = 1; gx < gridW - 1; gx++) {
      const idx = gy * gridW + gx;
      if (candidateGrid[idx] > 0) {
        bridged[idx] = candidateGrid[idx];
        continue;
      }
      const left = candidateGrid[idx - 1];
      const right = candidateGrid[idx + 1];
      const up = candidateGrid[idx - gridW];
      const down = candidateGrid[idx + gridW];
      if (((left > 0 && right > 0) || (up > 0 && down > 0)) && cellBrightRatio[idx] >= 0.55) {
        bridged[idx] = Math.max(left, right, up, down);
      }
    }
  }

  // Step 3: Connected-component labeling with glyph periodicity verification
  const visited = new Uint8Array(gridW * gridH);
  const rawRegions: DetectedRegion[] = [];
  let regionCounter = 1;

  for (let gy = 1; gy < gridH - 1; gy++) {
    for (let gx = 1; gx < gridW - 1; gx++) {
      const startIdx = gy * gridW + gx;
      if (bridged[startIdx] === 0 || visited[startIdx]) continue;

      const queue: number[] = [startIdx];
      visited[startIdx] = 1;

      let minGx = gx,
        maxGx = gx;
      let minGy = gy,
        maxGy = gy;
      let cellCount = 0;
      let bubbleVoteCount = 0;
      let strokeSum = 0;

      while (queue.length > 0) {
        const curr = queue.pop()!;
        const cy = Math.floor(curr / gridW);
        const cx = curr % gridW;
        cellCount++;
        if (bridged[curr] === 2) bubbleVoteCount++;
        strokeSum += cellStrokeRatio[curr];

        if (cx < minGx) minGx = cx;
        if (cx > maxGx) maxGx = cx;
        if (cy < minGy) minGy = cy;
        if (cy > maxGy) maxGy = cy;

        if (maxGy - minGy > 26 || maxGx - minGx > 22) continue;

        const neighbors = [curr - 1, curr + 1, curr - gridW, curr + gridW];
        for (const n of neighbors) {
          if (n >= 0 && n < gridW * gridH && !visited[n] && bridged[n] > 0) {
            visited[n] = 1;
            queue.push(n);
          }
        }
      }

      const bw = (maxGx - minGx + 1) * cellSize;
      const bh = (maxGy - minGy + 1) * cellSize;
      const aspect = bw / Math.max(1, bh);

      // Require at least 3 glyph cells and realistic text block dimensions
      if (cellCount < 3) continue;
      if (bw < 32 || bh < 28) continue;
      if (bw > imgWidth * 0.72 || bh > 520) continue;
      if (aspect > 6.5 || aspect < 0.16) continue;

      const boxAreaCells = (maxGx - minGx + 1) * (maxGy - minGy + 1);
      const fillDensity = cellCount / Math.max(1, boxAreaCells);
      if (fillDensity < 0.42 && cellCount > 5) continue;

      const bbox: BoundingBox = {
        x: Math.max(4, minGx * cellSize - 4),
        y: Math.max(4, minGy * cellSize - 4),
        w: Math.min(imgWidth - 8, bw + 8),
        h: Math.min(imgHeight - 8, bh + 8),
      };

      // Verify multi-stroke glyph periodicity to reject character hair/clothes line art
      const glyphCheck = hasGlyphStrokePeriodicity(rgba, imgWidth, imgHeight, bbox);
      if (!glyphCheck.isGlyphCluster) continue;

      const surroundStats = measureSurroundRing(rgba, imgWidth, imgHeight, bbox);
      const enclosure = verifyBubbleEnclosure(rgba, imgWidth, imgHeight, bbox);

      const insideBubble =
        enclosure.enclosed ||
        (surroundStats.isWhiteBubble && glyphCheck.whitePaperFraction >= 0.52) ||
        (bubbleVoteCount / cellCount >= 0.75 && surroundStats.stdLuma <= 11.5);

      let category: RegionCategory = 'BUBBLE';
      const relCenterX = (bbox.x + bbox.w / 2) / imgWidth;
      const isNearPageSide = relCenterX < 0.22 || relCenterX > 0.78;

      if (insideBubble) {
        category = 'BUBBLE';
      } else if (
        !surroundStats.isUniform &&
        surroundStats.edgeEnergy > 45 &&
        glyphCheck.whitePaperFraction >= 0.42 &&
        cellCount >= 5
      ) {
        category = 'SFX';
      } else if (isNearPageSide || (aspect < 0.5 && bw <= 76)) {
        category = 'ST';
      } else {
        category = 'OT';
      }

      if (category === 'BUBBLE' && !settings.enableBubble) continue;
      if (category === 'OT' && !settings.enableOT) continue;
      if (category === 'ST' && !settings.enableST) continue;
      if (category === 'SFX' && !settings.enableSFX) continue;

      const confidence = Math.min(
        0.99,
        Math.max(
          0.68,
          insideBubble
            ? 0.93 + (surroundStats.isWhiteBubble ? 0.05 : 0)
            : category === 'SFX'
              ? 0.81
              : 0.87
        )
      );

      const routedEngine = selectCleaningEngine(category, surroundStats, settings);
      const id = `p${pageIndex + 1}_r${regionCounter++}_${category.toLowerCase()}`;

      rawRegions.push({
        id,
        pageIndex,
        category,
        bbox,
        bubbleBbox: enclosure.bubbleBbox,
        bubbleId: insideBubble ? `bub_${pageIndex + 1}_${regionCounter}` : undefined,
        insideBubble,
        confidence: Number(confidence.toFixed(2)),
        sourceScript: category === 'SFX' ? 'SFX_STYLIZED' : 'KO',
        routedEngine,
        bubbleCleanMode: settings.bubbleCleanMode,
        paddingPx: settings.defaultPaddingPx,
        featherPx: settings.defaultFeatherPx,
        fillColorHex: surroundStats.isWhiteBubble
          ? '#FFFFFF'
          : `#${Math.round(surroundStats.meanR).toString(16).padStart(2, '0')}${Math.round(surroundStats.meanG).toString(16).padStart(2, '0')}${Math.round(surroundStats.meanB).toString(16).padStart(2, '0')}`.toUpperCase(),
        inkCoverageRatio: Number((strokeSum / cellCount).toFixed(3)),
        surroundStats,
        status: 'DETECTED',
        reviewStatus:
          category === 'SFX' && confidence < 0.84 ? 'NEEDS_REVIEW' : 'AUTO_APPROVED',
        manualEdits: [],
      });
    }
  }

  return rawRegions;
}
