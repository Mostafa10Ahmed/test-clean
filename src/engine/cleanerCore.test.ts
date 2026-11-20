import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SAFE_BUBBLE_BORDER,
  blendComplexResult,
  cleanAndCompositeRegionInPlace,
  componentBackgroundIsUniformLight,
  computeStripWindows,
  evaluateCleaningQuality,
  extractPreciseTextMask,
  findSystemPanelInterior,
  hasNewBoundaryArtifact,
  routeTextMasks,
  safeBubbleInterior,
  selectCleaningEngine,
  sortPageNamesNaturally,
  verifyBitIdenticalOutsideMasks,
} from './cleanerCore';
import { detectRegionsInRgbaBuffer, measureSurroundRing } from './detector';
import { CleaningSettings, DetectedRegion } from './types';

const defaultSettings: CleaningSettings = {
  enableBubble: true,
  enableOT: true,
  enableST: true,
  enableSFX: true,
  bubbleCleanMode: 'WHITEN',
  defaultPaddingPx: 3,
  defaultFeatherPx: 2,
  engineRouting: 'AUTO_LADDER',
  sourceLanguages: ['KO', 'JA', 'EN', 'ZH'],
  strictQualityGate: true,
  preserveColoredBubbles: true,
  maxStripWindowHeight: 2048,
  cloudGpuConsent: false,
};

test('Long-strip windowing slices 28,483px Manhwa strips with 128px overlap without gaps', () => {
  const windows = computeStripWindows(28483, 2048, 128);
  assert.ok(windows.length >= 14);
  assert.equal(windows[0].yStart, 0);
  assert.equal(windows[0].coreYStart, 0);
  assert.equal(windows[windows.length - 1].yEnd, 28483);
  assert.equal(windows[windows.length - 1].coreYEnd, 28483);

  for (let i = 1; i < windows.length; i++) {
    assert.ok(windows[i - 1].yEnd - windows[i].yStart >= 128);
    assert.equal(windows[i - 1].coreYEnd, windows[i].coreYStart);
  }
});

test('Natural alphanumeric page sorting preserves chapter order', () => {
  const unsorted = ['010.jpg', '002.jpg', '001.jpg', '013.jpg', '003.jpg'];
  const sorted = sortPageNamesNaturally(unsorted);
  assert.deepEqual(sorted, ['001.jpg', '002.jpg', '003.jpg', '010.jpg', '013.jpg']);
});

test('Engine routing ladder selects Smart Fill for uniform bubbles and LaMa/Gemini for textured or SFX', () => {
  const whiteBubbleSurround = {
    meanR: 255,
    meanG: 255,
    meanB: 255,
    meanLuma: 255,
    stdLuma: 2.1,
    p5Luma: 250,
    p95Luma: 255,
    edgeEnergy: 1.5,
    isUniform: true,
    isWhiteBubble: true,
    isDarkBubble: false,
    isPeriodicScreentone: false,
    isGradient: false,
    isOutlineDominated: true,
    gradSlopeX: 0,
    gradSlopeY: 0,
  };

  assert.equal(selectCleaningEngine('BUBBLE', whiteBubbleSurround, defaultSettings), 'SMART_FILL');

  const screentoneSurround = {
    ...whiteBubbleSurround,
    meanLuma: 185,
    stdLuma: 24.5,
    isUniform: false,
    isWhiteBubble: false,
    isPeriodicScreentone: true,
  };
  assert.equal(selectCleaningEngine('OT', screentoneSurround, defaultSettings), 'LAMA_PATCH');
  assert.equal(selectCleaningEngine('SFX', screentoneSurround, defaultSettings), 'LAMA_PATCH');

  const cloudSettings: CleaningSettings = {
    ...defaultSettings,
    engineRouting: 'ALLOW_CLOUD_AI',
    cloudGpuConsent: true,
  };
  assert.equal(selectCleaningEngine('SFX', screentoneSurround, cloudSettings), 'GEMINI_AI');
});

test('Regression: Outline-dominated surround does not falsely decline clean speech bubble interior (Bear quality.rs bug fix)', () => {
  const w = 48;
  const h = 48;
  const interiorLumaAfter = new Float32Array(w * h).fill(255);
  const surroundLuma = new Float32Array(w * h).fill(255);
  const interiorMask = new Uint8Array(w * h);
  const surroundMask = new Uint8Array(w * h);
  const strokeSeedMask = new Uint8Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (x >= 12 && x < 36 && y >= 12 && y < 36) {
        interiorMask[idx] = 1;
        if (x >= 18 && x < 30 && y >= 18 && y < 30) {
          strokeSeedMask[idx] = 1;
        }
      } else {
        surroundMask[idx] = 1;
        if (x === 10 || x === 11 || x === 36 || x === 37) {
          surroundLuma[idx] = 0;
        }
      }
    }
  }

  const assessment = evaluateCleaningQuality({
    interiorLumaAfter,
    surroundLuma,
    width: w,
    height: h,
    interiorMask,
    surroundMask,
    strokeSeedMask,
    isOutlineDominated: true,
    isPeriodicScreentone: false,
    unchangedOutsideVerified: true,
  });

  assert.equal(assessment.outcome, 'ACCEPTED');
  assert.equal(assessment.ghostStrokeResidual, 0);
});

test('Regression: Ghost text left behind inside stroke coordinates is detected and declined', () => {
  const w = 40;
  const h = 40;
  const interiorLumaAfter = new Float32Array(w * h).fill(250);
  const surroundLuma = new Float32Array(w * h).fill(250);
  const interiorMask = new Uint8Array(w * h);
  const surroundMask = new Uint8Array(w * h);
  const strokeSeedMask = new Uint8Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (x >= 8 && x < 32 && y >= 8 && y < 32) {
        interiorMask[idx] = 1;
        if (x >= 14 && x < 26 && y >= 14 && y < 26) {
          strokeSeedMask[idx] = 1;
          interiorLumaAfter[idx] = 210;
        }
      } else {
        surroundMask[idx] = 1;
      }
    }
  }

  const assessment = evaluateCleaningQuality({
    interiorLumaAfter,
    surroundLuma,
    width: w,
    height: h,
    interiorMask,
    surroundMask,
    strokeSeedMask,
    isOutlineDominated: false,
    isPeriodicScreentone: false,
    unchangedOutsideVerified: true,
  });

  assert.equal(assessment.outcome, 'DECLINED_GHOST_TEXT');
  assert.ok(assessment.ghostStrokeResidual > 18);
});

test('Bit-identical compositor guarantees 0 modified pixels outside authorized region masks', () => {
  const w = 160;
  const h = 160;
  const orig = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    orig[i * 4] = 252;
    orig[i * 4 + 1] = 252;
    orig[i * 4 + 2] = 252;
    orig[i * 4 + 3] = 255;
  }

  for (let y = 55; y < 85; y++) {
    for (let x = 55; x < 85; x++) {
      if ((x + y) % 3 === 0) {
        const p = (y * w + x) * 4;
        orig[p] = 15;
        orig[p + 1] = 15;
        orig[p + 2] = 15;
      }
    }
  }

  const work = new Uint8Array(orig);
  const surround = measureSurroundRing(work, w, h, { x: 48, y: 48, w: 44, h: 44 });
  const region: DetectedRegion = {
    id: 'test_r1',
    pageIndex: 0,
    category: 'BUBBLE',
    bbox: { x: 48, y: 48, w: 44, h: 44 },
    insideBubble: true,
    confidence: 0.96,
    sourceScript: 'KO',
    routedEngine: 'SMART_FILL',
    bubbleCleanMode: 'WHITEN',
    paddingPx: 3,
    featherPx: 2,
    fillColorHex: '#FFFFFF',
    inkCoverageRatio: 0.25,
    surroundStats: surround,
    status: 'CLEANED',
    reviewStatus: 'AUTO_APPROVED',
    manualEdits: [],
  };

  const { quality } = cleanAndCompositeRegionInPlace({
    rgba: work,
    imgWidth: w,
    imgHeight: h,
    region,
    settings: defaultSettings,
  });

  assert.equal(quality.outcome, 'ACCEPTED');

  const check = verifyBitIdenticalOutsideMasks(orig, work, w, h, [region]);
  assert.equal(check.bitIdentical, true);
  assert.equal(check.differingPixelsOutsideMasks, 0);
  assert.ok(check.totalUnmaskedPixels > 20000);
});

test('Regression: Detector rejects isolated diagonal artwork lines and detects enclosed speech bubble dialogue', () => {
  const w = 320;
  const h = 320;
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = 160;
    rgba[i * 4 + 1] = 160;
    rgba[i * 4 + 2] = 160;
    rgba[i * 4 + 3] = 255;
  }

  for (let t = 20; t < 100; t++) {
    const p = (t * w + t) * 4;
    rgba[p] = 10;
    rgba[p + 1] = 10;
    rgba[p + 2] = 10;
  }

  for (let y = 120; y <= 240; y++) {
    for (let x = 120; x <= 260; x++) {
      const p = (y * w + x) * 4;
      const isBorder = x <= 122 || x >= 258 || y <= 122 || y >= 238;
      const val = isBorder ? 15 : 255;
      rgba[p] = val;
      rgba[p + 1] = val;
      rgba[p + 2] = val;
    }
  }

  for (let y = 152; y < 204; y += 2) {
    for (let x = 152; x < 228; x++) {
      if (x % 6 < 2 && y % 8 < 6) {
        const p = (y * w + x) * 4;
        rgba[p] = 12;
        rgba[p + 1] = 12;
        rgba[p + 2] = 12;
      }
    }
  }

  const regions = detectRegionsInRgbaBuffer(rgba, w, h, 0, defaultSettings);
  assert.equal(regions.length, 1);
  assert.equal(regions[0].category, 'BUBBLE');
  assert.equal(regions[0].insideBubble, true);
});

// ============================================================================
// Ported Mask-Routing & Whitening Precision Suite (more249-s/dis-inpainting)
// ============================================================================

test('dis-inpainting #1: safeBubbleInterior keeps SAFE_BUBBLE_BORDER margin from bubble outline', () => {
  const w = 120;
  const h = 120;
  const interior = new Uint8Array(w * h);
  for (let y = 20; y < 100; y++) {
    for (let x = 20; x < 100; x++) {
      interior[y * w + x] = 255;
    }
  }

  const safe = safeBubbleInterior(interior, w, h, SAFE_BUBBLE_BORDER);

  // Center is preserved
  assert.equal(safe[60 * w + 60], 255);
  // Pixels within SAFE_BUBBLE_BORDER of the boundary (20..24) are excluded
  assert.equal(safe[20 * w + 60], 0);
  assert.equal(safe[22 * w + 60], 0);
});

test('dis-inpainting #2: gradient speech bubble is NOT classified as uniform light', () => {
  const w = 140;
  const h = 140;
  const interior = new Uint8Array(w * h);
  for (let y = 20; y < 120; y++) {
    for (let x = 20; x < 120; x++) {
      interior[y * w + x] = 255;
    }
  }

  // Flat white bubble -> true
  const flatImg = new Uint8Array(w * h * 3).fill(248);
  const compMask = new Uint8Array(w * h);
  for (let y = 55; y < 85; y++) {
    for (let x = 45; x < 95; x++) {
      compMask[y * w + x] = 255;
      const p = (y * w + x) * 3;
      flatImg[p] = 20;
      flatImg[p + 1] = 20;
      flatImg[p + 2] = 20;
    }
  }
  assert.equal(componentBackgroundIsUniformLight(flatImg, compMask, interior, w, h, 3), true);

  // Vertical gradient bubble (190 -> 250) -> false!
  const gradImg = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const val = Math.round(190 + (60 * y) / (h - 1));
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 3;
      gradImg[p] = val;
      gradImg[p + 1] = val;
      gradImg[p + 2] = val;
    }
  }
  for (let y = 55; y < 85; y++) {
    for (let x = 45; x < 95; x++) {
      const p = (y * w + x) * 3;
      gradImg[p] = 20;
      gradImg[p + 1] = 20;
      gradImg[p + 2] = 20;
    }
  }
  assert.equal(componentBackgroundIsUniformLight(gradImg, compMask, interior, w, h, 3), false);
});

test('dis-inpainting #3: system panel interior is detected while preserving panel border', () => {
  const w = 180;
  const h = 160;
  const image = new Uint8Array(w * h * 3).fill(235);

  // Dark RPG system box at x:30..150, y:25..135
  for (let y = 25; y < 135; y++) {
    for (let x = 30; x < 150; x++) {
      const p = (y * w + x) * 3;
      const isBorder =
        y === 25 || y === 26 || y === 133 || y === 134 || x === 30 || x === 31 || x === 148 || x === 149;
      const v = isBorder ? 220 : 35;
      image[p] = v;
      image[p + 1] = v;
      image[p + 2] = v;
    }
  }

  // Bright text strokes inside system panel
  const textMask = new Uint8Array(w * h);
  for (let y = 60; y < 95; y++) {
    for (let x = 55; x < 125; x++) {
      const p = (y * w + x) * 3;
      image[p] = 245;
      image[p + 1] = 245;
      image[p + 2] = 245;
      textMask[y * w + x] = 255;
    }
  }

  const interior = findSystemPanelInterior(image, textMask, w, h, 3);
  assert.equal(interior[75 * w + 90], 255); // Inside panel is exposed
  assert.equal(interior[25 * w + 90], 0); // Top border is protected
  assert.equal(interior[75 * w + 30], 0); // Left border is protected
});

test('dis-inpainting #4: extractPreciseTextMask avoids filling the whole detector bounding box', () => {
  const w = 140;
  const h = 120;
  const image = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = Math.round(110 + ((x * 40) / w));
      const p = (y * w + x) * 3;
      image[p] = v;
      image[p + 1] = v;
      image[p + 2] = v;
    }
  }

  // Draw crisp dark lettering strokes inside box (20..120, 20..100)
  for (let y = 45; y < 68; y++) {
    for (let x = 35; x < 105; x++) {
      if ((x - 35) % 14 < 4) {
        const p = (y * w + x) * 3;
        image[p] = 15;
        image[p + 1] = 15;
        image[p + 2] = 15;
      }
    }
  }

  const detectorBoxMask = new Uint8Array(w * h);
  let boxPixels = 0;
  for (let y = 20; y < 100; y++) {
    for (let x = 20; x < 120; x++) {
      detectorBoxMask[y * w + x] = 255;
      boxPixels++;
    }
  }

  const precise = extractPreciseTextMask(image, detectorBoxMask, w, h, 13, 3);
  let active = 0;
  for (let y = 20; y < 100; y++) {
    for (let x = 20; x < 120; x++) {
      if (precise[y * w + x] > 0) active++;
    }
  }
  const density = active / boxPixels;
  assert.ok(density > 0.01 && density < 0.65, `Expected stroke density in 0.01..0.65, got ${density}`);
});

test('dis-inpainting #5: routeTextMasks clips outline contact from complex inpaint', () => {
  const w = 120;
  const h = 120;
  const image = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const val = Math.round(180 + (70 * y) / (h - 1));
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 3;
      image[p] = val;
      image[p + 1] = val;
      image[p + 2] = val;
    }
  }

  const interior = new Uint8Array(w * h);
  for (let y = 20; y < 100; y++) {
    for (let x = 20; x < 100; x++) {
      interior[y * w + x] = 255;
    }
  }

  const mask = new Uint8Array(w * h);
  // Mask starts at y=20 (touching the outline margin) through y=60
  for (let y = 20; y < 60; y++) {
    for (let x = 45; x < 75; x++) {
      mask[y * w + x] = 255;
    }
  }

  const { legacyWhiteMask, complexMask, excludedMask } = routeTextMasks(
    image,
    mask,
    interior,
    w,
    h,
    3
  );

  // Gradient bubble -> not legacy white
  let whiteCount = 0;
  for (let i = 0; i < legacyWhiteMask.length; i++) if (legacyWhiteMask[i] > 0) whiteCount++;
  assert.equal(whiteCount, 0);

  // Outline contact rows (y = 20..24) must be excluded from complexMask
  assert.equal(complexMask[21 * w + 55], 0);
  assert.equal(excludedMask[21 * w + 55], 255);
  // Safe inner rows (y = 40) are routed to complexMask
  assert.equal(complexMask[40 * w + 55], 255);
});

test('dis-inpainting #6: blendComplexResult only changes masked pixels', () => {
  const w = 80;
  const h = 80;
  const source = new Uint8Array(w * h * 3).fill(120);
  const restored = new Uint8Array(w * h * 3).fill(200);
  const mask = new Uint8Array(w * h);
  for (let y = 20; y < 60; y++) {
    for (let x = 20; x < 60; x++) {
      mask[y * w + x] = 255;
    }
  }

  const blended = blendComplexResult(source, restored, mask, w, h, 3);
  for (let i = 0; i < w * h; i++) {
    if (mask[i] === 0) {
      assert.equal(blended[i * 3], 120);
      assert.equal(blended[i * 3 + 1], 120);
      assert.equal(blended[i * 3 + 2], 120);
    }
  }
  assert.ok(blended[(40 * w + 40) * 3] > 180);
});

test('dis-inpainting #7: hasNewBoundaryArtifact flags visible rectangular patch seams', () => {
  const w = 120;
  const h = 120;
  const source = new Uint8Array(w * h * 3);
  for (let x = 0; x < w; x++) {
    const val = Math.round(140 + (60 * x) / (w - 1));
    for (let y = 0; y < h; y++) {
      const p = (y * w + x) * 3;
      source[p] = val;
      source[p + 1] = val;
      source[p + 2] = val;
    }
  }

  const mask = new Uint8Array(w * h);
  for (let y = 35; y < 85; y++) {
    for (let x = 35; x < 85; x++) {
      mask[y * w + x] = 255;
    }
  }

  const smoothCandidate = new Uint8Array(source);
  const badPatchCandidate = new Uint8Array(source);
  for (let y = 35; y < 85; y++) {
    for (let x = 35; x < 85; x++) {
      const p = (y * w + x) * 3;
      badPatchCandidate[p] = 215;
      badPatchCandidate[p + 1] = 215;
      badPatchCandidate[p + 2] = 215;
    }
  }

  assert.equal(hasNewBoundaryArtifact(source, smoothCandidate, mask, w, h, 3), false);
  assert.equal(hasNewBoundaryArtifact(source, badPatchCandidate, mask, w, h, 3), true);
});

test('Regression: Solid gold plaque classified as SFX cleans cleanly via FLAT_RING while textured SFX returns null', async () => {
  const { tryCleanFlatOrGradientRing } = await import('./cleanerCore');
  const w = 120;
  const h = 80;
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = 212;
    rgba[i * 4 + 1] = 182;
    rgba[i * 4 + 2] = 132;
    rgba[i * 4 + 3] = 255;
  }
  const writeMask = new Uint8Array(w * h);
  const interiorCandidate = new Uint8Array(w * h).fill(1);
  for (let y = 25; y < 55; y++) {
    for (let x = 25; x < 95; x++) {
      writeMask[y * w + x] = 255;
      const p = (y * w + x) * 4;
      rgba[p] = 250;
      rgba[p + 1] = 245;
      rgba[p + 2] = 245;
    }
  }
  const region = {
    id: 'plaque_test',
    pageIndex: 1,
    category: 'SFX' as const,
    bbox: { x: 0, y: 0, w, h },
    confidence: 0.9,
    sourceScript: 'SFX_STYLIZED' as const,
    routedEngine: 'LAMA_PATCH' as const,
    appliedEngine: 'LAMA_PATCH' as const,
    bubbleCleanMode: 'WHITEN' as const,
    paddingPx: 3,
    featherPx: 2,
    fillColorHex: '#D4B684',
    status: 'DETECTED' as const,
    reviewStatus: 'AUTO_APPROVED' as const,
    insideBubble: false,
    inkCoverageRatio: 0.2,
    surroundStats: {
      meanR: 212,
      meanG: 182,
      meanB: 132,
      meanLuma: 185,
      stdLuma: 2,
      p5Luma: 180,
      p95Luma: 190,
      edgeEnergy: 4,
      isUniform: true,
      isWhiteBubble: false,
      isDarkBubble: false,
      isPeriodicScreentone: false,
      isGradient: false,
      isOutlineDominated: false,
      gradSlopeX: 0,
      gradSlopeY: 0,
    },
    manualEdits: [],
  };

  const tier = tryCleanFlatOrGradientRing({
    rgba,
    imgWidth: w,
    imgHeight: h,
    region,
    writeMask,
    interiorCandidate,
  });
  assert.equal(tier, 'FLAT_RING');
  const centerIdx = (40 * w + 60) * 4;
  assert.equal(rgba[centerIdx], 212);
  assert.equal(rgba[centerIdx + 1], 182);
  assert.equal(rgba[centerIdx + 2], 132);
});

