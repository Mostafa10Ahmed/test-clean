import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import {
  computePageCtdSegMask,
  detectRegionsWithOnnx,
  cleanRegionWithOnnxSupport,
  getCachedPageCtd,
  setCachedPageCtd,
  releaseAllOnnxSessions,
} from '../engine/onnxPipeline';
import { detectRegionsInRgbaBuffer, measureSurroundRing } from '../engine/detector';
import { CleaningSettings, DetectedRegion } from '../engine/types';

sharp.cache(false);
sharp.concurrency(1);

async function main() {
  const [stage, imagePath, workDir, settingsJsonPath, outPngPath] = process.argv.slice(2);
  if (!stage || !imagePath || !workDir) {
    console.error('Usage: tsx onnxStageRunner.ts <ctd|rtdetr|clean> <imagePath> <workDir> [settingsJson] [outPngPath]');
    process.exit(2);
  }

  fs.mkdirSync(workDir, { recursive: true });
  const image = sharp(imagePath, { limitInputPixels: false });
  const meta = await image.metadata();
  const width = meta.width || 800;
  const height = meta.height || 1200;
  const rawBuffer = await image.ensureAlpha().raw().toBuffer();
  const rgba = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);

  const maskPath = path.join(workDir, 'ctd_mask.bin');
  const ctdMetaPath = path.join(workDir, 'ctd_meta.json');
  const regionsPath = path.join(workDir, 'regions.json');
  const statsPath = path.join(workDir, 'clean_stats.json');

  if (stage === 'ctd') {
    const mask = await computePageCtdSegMask(rgba, width, height);
    const cached = getCachedPageCtd();
    if (mask && cached) {
      fs.writeFileSync(maskPath, Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength));
      fs.writeFileSync(
        ctdMetaPath,
        JSON.stringify({
          width: cached.width,
          height: cached.height,
          fingerprint: cached.fingerprint,
          blkBoxes: cached.blkBoxes,
        }),
        'utf-8'
      );
    }
    await releaseAllOnnxSessions();
    console.log(`[Subprocess:CTD] Done ${path.basename(imagePath)} | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`);
    return;
  }

  const settings = JSON.parse(fs.readFileSync(settingsJsonPath, 'utf-8')) as CleaningSettings;

  if (stage === 'rtdetr') {
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
    }

    const onnxRegions = await detectRegionsWithOnnx(rgba, width, height, 0, settings);
    const regions: DetectedRegion[] =
      onnxRegions.length > 0
        ? onnxRegions
        : detectRegionsInRgbaBuffer(rgba, width, height, 0, settings);

    fs.writeFileSync(regionsPath, JSON.stringify(regions), 'utf-8');
    await releaseAllOnnxSessions();
    console.log(`[Subprocess:RT-DETR] Done ${path.basename(imagePath)} (${regions.length} regions) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`);
    return;
  }

  if (stage === 'clean') {
    const t0 = performance.now();
    const regions = JSON.parse(fs.readFileSync(regionsPath, 'utf-8')) as DetectedRegion[];
    const workRgba = new Uint8Array(rgba);
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

    await sharp(Buffer.from(workRgba.buffer, workRgba.byteOffset, workRgba.byteLength), {
      raw: { width, height, channels: 4 },
      limitInputPixels: false,
    })
      .png({ compressionLevel: 6 })
      .toFile(outPngPath);

    await releaseAllOnnxSessions();
    const elapsedMs = Math.round(performance.now() - t0);
    fs.writeFileSync(statsPath, JSON.stringify({ cleanedRegions: cleanedCount, elapsedMs }), 'utf-8');
    console.log(`[Subprocess:CLEAN/LaMa] Done ${path.basename(imagePath)} (${cleanedCount} cleaned, ${elapsedMs}ms) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`);
    return;
  }

  throw new Error(`Unknown stage: ${stage}`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('[Subprocess Error]:', err);
    process.exit(1);
  });
