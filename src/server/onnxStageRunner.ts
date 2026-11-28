import fs from 'fs';
import os from 'os';
import path from 'path';
import sharp from 'sharp';
import {
  computePageCtdSegMask,
  detectRegionsWithOnnx,
  cleanRegionsOnPageWithOnnx,
  getCachedPageCtd,
  setCachedPageCtd,
  clearPageCtdCache,
  releaseAllOnnxSessions,
} from '../engine/onnxPipeline';
import { detectRegionsInRgbaBuffer } from '../engine/detector';
import { CleaningSettings, DetectedRegion } from '../engine/types';
import { createTwoLayerPsd } from '../engine/psdWriter';

sharp.cache(false);
sharp.concurrency(Math.min(4, Math.max(1, os.cpus().length)));

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
    const cleanedCount = await cleanRegionsOnPageWithOnnx({
      originalRgba: rgba,
      workRgba,
      imgWidth: width,
      imgHeight: height,
      regions,
      settings,
    });

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
