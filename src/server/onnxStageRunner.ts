import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import {
  computePageCtdSegMask,
  detectRegionsWithOnnx,
  cleanRegionWithOnnxSupport,
  getCachedPageCtd,
  setCachedPageCtd,
  clearPageCtdCache,
  releaseAllOnnxSessions,
} from '../engine/onnxPipeline';
import { detectRegionsInRgbaBuffer, measureSurroundRing } from '../engine/detector';
import { CleaningSettings, DetectedRegion } from '../engine/types';
import { createTwoLayerPsd } from '../engine/psdWriter';

sharp.cache(false);
sharp.concurrency(1);

interface BatchManifest {
  pages: {
    index: number;
    name: string;
    inPath: string;
    outPath: string;
    scratchDir: string;
  }[];
  settings: CleaningSettings;
  outputFormat?: 'jpg' | 'png' | 'psd';
}

async function runCtdBatch(manifest: BatchManifest) {
  for (const p of manifest.pages) {
    fs.mkdirSync(p.scratchDir, { recursive: true });
    const image = sharp(p.inPath, { limitInputPixels: false });
    const meta = await image.metadata();
    const width = meta.width || 800;
    const height = meta.height || 1200;
    const rawBuffer = await image.ensureAlpha().raw().toBuffer();
    const rgba = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);

    const mask = await computePageCtdSegMask(rgba, width, height);
    const cached = getCachedPageCtd();
    if (mask && cached) {
      fs.writeFileSync(
        path.join(p.scratchDir, 'ctd_mask.bin'),
        Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength)
      );
      fs.writeFileSync(
        path.join(p.scratchDir, 'ctd_meta.json'),
        JSON.stringify({
          width: cached.width,
          height: cached.height,
          fingerprint: cached.fingerprint,
          blkBoxes: cached.blkBoxes,
        }),
        'utf-8'
      );
    }
    clearPageCtdCache();
    console.log(
      `PROGRESS:DETECTING_CTD:${p.index + 1}:${manifest.pages.length}`
    );
    console.log(
      `[Subprocess:CTD] Page ${p.index + 1}/${manifest.pages.length} (${p.name}) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
    );
  }
  await releaseAllOnnxSessions();
}

async function runRtdetrBatch(manifest: BatchManifest) {
  for (const p of manifest.pages) {
    fs.mkdirSync(p.scratchDir, { recursive: true });
    const image = sharp(p.inPath, { limitInputPixels: false });
    const meta = await image.metadata();
    const width = meta.width || 800;
    const height = meta.height || 1200;
    const rawBuffer = await image.ensureAlpha().raw().toBuffer();
    const rgba = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);

    const maskPath = path.join(p.scratchDir, 'ctd_mask.bin');
    const ctdMetaPath = path.join(p.scratchDir, 'ctd_meta.json');
    if (fs.existsSync(maskPath) && fs.existsSync(ctdMetaPath)) {
      const ctdMeta = JSON.parse(fs.readFileSync(ctdMetaPath, 'utf-8')) as {
        width: number;
        height: number;
        fingerprint: number;
        blkBoxes: { bbox: { x: number; y: number; w: number; h: number }; score: number }[];
      };
      const maskBuf = fs.readFileSync(maskPath);
      setCachedPageCtd({
        width: ctdMeta.width,
        height: ctdMeta.height,
        fingerprint: ctdMeta.fingerprint,
        segMask: new Uint8Array(maskBuf.buffer, maskBuf.byteOffset, maskBuf.byteLength),
        blkBoxes: ctdMeta.blkBoxes || [],
      });
    } else {
      clearPageCtdCache();
    }

    const onnxRegions = await detectRegionsWithOnnx(rgba, width, height, p.index, manifest.settings);
    const regions: DetectedRegion[] =
      onnxRegions.length > 0
        ? onnxRegions
        : detectRegionsInRgbaBuffer(rgba, width, height, p.index, manifest.settings);

    fs.writeFileSync(path.join(p.scratchDir, 'regions.json'), JSON.stringify(regions), 'utf-8');
    clearPageCtdCache();
    // Keep ctd_mask.bin & ctd_meta.json for Stage 3 (CLEAN/LaMa) so LaMa gets pixel-exact text masks!

    console.log(
      `PROGRESS:DETECTING_RTDETR:${p.index + 1}:${manifest.pages.length}`
    );
    console.log(
      `[Subprocess:RT-DETR] Page ${p.index + 1}/${manifest.pages.length} (${regions.length} regions) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
    );
  }
  await releaseAllOnnxSessions();
}

async function runCleanBatch(manifest: BatchManifest) {
  let totalCleaned = 0;
  for (const p of manifest.pages) {
    const t0 = performance.now();
    const image = sharp(p.inPath, { limitInputPixels: false });
    const meta = await image.metadata();
    const width = meta.width || 800;
    const height = meta.height || 1200;
    const rawBuffer = await image.ensureAlpha().raw().toBuffer();
    const rgba = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);

    // Restore pixel-exact CTD segmentation mask for Stage 3 (eliminates halos & muddy LaMa patches!)
    const maskPath = path.join(p.scratchDir, 'ctd_mask.bin');
    const ctdMetaPath = path.join(p.scratchDir, 'ctd_meta.json');
    if (fs.existsSync(maskPath) && fs.existsSync(ctdMetaPath)) {
      const ctdMeta = JSON.parse(fs.readFileSync(ctdMetaPath, 'utf-8')) as {
        width: number;
        height: number;
        fingerprint: number;
        blkBoxes: { bbox: { x: number; y: number; w: number; h: number }; score: number }[];
      };
      const maskBuf = fs.readFileSync(maskPath);
      setCachedPageCtd({
        width: ctdMeta.width,
        height: ctdMeta.height,
        fingerprint: ctdMeta.fingerprint,
        segMask: new Uint8Array(maskBuf.buffer, maskBuf.byteOffset, maskBuf.byteLength),
        blkBoxes: ctdMeta.blkBoxes || [],
      });
    } else {
      clearPageCtdCache();
    }

    const regionsPath = path.join(p.scratchDir, 'regions.json');
    const regions = fs.existsSync(regionsPath)
      ? (JSON.parse(fs.readFileSync(regionsPath, 'utf-8')) as DetectedRegion[])
      : [];
    const workRgba = new Uint8Array(rgba);
    const settings = manifest.settings;
    let cleanedCount = 0;

    for (const region of regions) {
      if (region.status === 'REJECTED' || region.reviewStatus === 'RESTORED_ORIGINAL') continue;
      if (region.category === 'BUBBLE' && !settings.enableBubble) continue;
      if (region.category === 'OT' && !settings.enableOT) continue;
      if (region.category === 'ST' && !settings.enableST) continue;
      if (region.category === 'SFX' && !settings.enableSFX) continue;

      region.surroundStats = measureSurroundRing(rgba, width, height, region.bbox);

      if (region.surroundStats.isDarkBubble && (!region.manualEdits || region.manualEdits.length === 0)) {
        const glowPad = 28;
        const minX = region.bubbleBbox ? region.bubbleBbox.x + 8 : 4;
        const minY = region.bubbleBbox ? region.bubbleBbox.y + 8 : 4;
        const maxX = region.bubbleBbox ? region.bubbleBbox.x + region.bubbleBbox.w - 8 : width - 4;
        const maxY = region.bubbleBbox ? region.bubbleBbox.y + region.bubbleBbox.h - 8 : height - 4;
        const nx0 = Math.max(minX, region.bbox.x - glowPad);
        const ny0 = Math.max(minY, region.bbox.y - glowPad);
        const nx1 = Math.min(maxX, region.bbox.x + region.bbox.w + glowPad);
        const ny1 = Math.min(maxY, region.bbox.y + region.bbox.h + glowPad);
        if (nx1 > nx0 + 16 && ny1 > ny0 + 16) {
          const expandedBbox = { x: nx0, y: ny0, w: nx1 - nx0, h: ny1 - ny0 };
          const expandedStats = measureSurroundRing(rgba, width, height, expandedBbox);
          region.bbox = expandedBbox;
          if (expandedStats.isDarkBubble) region.surroundStats = expandedStats;
        }
        region.fillColorHex =
          region.surroundStats.meanLuma <= 12
            ? '#000000'
            : `#${Math.round(region.surroundStats.meanR).toString(16).padStart(2, '0')}${Math.round(region.surroundStats.meanG).toString(16).padStart(2, '0')}${Math.round(region.surroundStats.meanB).toString(16).padStart(2, '0')}`.toUpperCase();
      }

      const { quality } = await cleanRegionWithOnnxSupport({
        rgba: workRgba,
        imgWidth: width,
        imgHeight: height,
        region,
        settings,
      });

      if (settings.strictQualityGate && quality.outcome !== 'ACCEPTED' && region.appliedEngine === 'SMART_FILL') {
        await cleanRegionWithOnnxSupport({
          rgba: workRgba,
          imgWidth: width,
          imgHeight: height,
          region,
          settings: { ...settings, engineRouting: 'FORCE_LAMA_PATCH' },
        });
      }
      cleanedCount++;
    }

    const fmt = manifest.outputFormat || 'jpg';
    if (fmt === 'psd') {
      const psdBuf = createTwoLayerPsd({
        width,
        height,
        originalRgba: rgba,
        cleanedRgba: workRgba,
      });
      await fs.promises.writeFile(p.outPath, psdBuf);
    } else if (fmt === 'jpg') {
      await sharp(Buffer.from(workRgba.buffer, workRgba.byteOffset, workRgba.byteLength), {
        raw: { width, height, channels: 4 },
        limitInputPixels: false,
      })
        .flatten({ background: '#ffffff' })
        .jpeg({ quality: 91, mozjpeg: false })
        .toFile(p.outPath);
    } else {
      await sharp(Buffer.from(workRgba.buffer, workRgba.byteOffset, workRgba.byteLength), {
        raw: { width, height, channels: 4 },
        limitInputPixels: false,
      })
        .png({ compressionLevel: 3 })
        .toFile(p.outPath);
    }

    totalCleaned += cleanedCount;
    clearPageCtdCache();
    fs.promises.rm(p.scratchDir, { recursive: true, force: true }).catch(() => {});
    fs.promises.unlink(p.inPath).catch(() => {});

    const elapsedMs = Math.round(performance.now() - t0);
    console.log(
      `PROGRESS:CLEANING_LAMA:${p.index + 1}:${manifest.pages.length}`
    );
    console.log(
      `[Subprocess:CLEAN/LaMa] Page ${p.index + 1}/${manifest.pages.length} (${cleanedCount} cleaned, ${elapsedMs}ms) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
    );
  }

  fs.writeFileSync(
    path.join(path.dirname(manifest.pages[0].outPath), 'batch_stats.json'),
    JSON.stringify({ cleanedRegions: totalCleaned }),
    'utf-8'
  );
  await releaseAllOnnxSessions();
}

async function main() {
  const [stage, manifestPath] = process.argv.slice(2);
  if (!stage || !manifestPath) {
    console.error('Usage: node onnxStageRunner.mjs <ctd|rtdetr|clean> <manifest.json>');
    process.exit(2);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as BatchManifest;
  if (stage === 'ctd') {
    await runCtdBatch(manifest);
  } else if (stage === 'rtdetr') {
    await runRtdetrBatch(manifest);
  } else if (stage === 'clean') {
    await runCleanBatch(manifest);
  } else {
    throw new Error(`Unknown batch stage: ${stage}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('[Subprocess Error]:', err);
    process.exit(1);
  });
