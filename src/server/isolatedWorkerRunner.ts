import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import JSZip from 'jszip';
import { BubbleCleanMode, CleaningSettings } from '../engine/types';

export interface IsolatedJobProgress {
  stage: string;
  currentPage: number;
  totalPages: number;
}

export interface IsolatedJobResult {
  outputPath: string;
  outputFileName: string;
  totalPages: number;
  cleanedRegions: number;
  totalProcessingTimeMs: number;
  bubbleCleanMode: BubbleCleanMode;
}

function naturalSort(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

function getTsxBin(): string {
  const localBin = path.resolve(process.cwd(), 'node_modules', '.bin', 'tsx');
  return fs.existsSync(localBin) ? localBin : 'npx';
}

function runStageSubprocess(args: string[], stageLabel: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const bin = getTsxBin();
    const runnerPath = path.resolve(process.cwd(), 'src/server/onnxStageRunner.ts');
    const fullArgs = bin === 'npx' ? ['tsx', runnerPath, ...args] : [runnerPath, ...args];

    const child = spawn(bin, fullArgs, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        NODE_OPTIONS: process.env.SUBPROCESS_NODE_OPTIONS || '--max-old-space-size=320',
      },
    });

    let stderrTail = '';
    child.stdout?.on('data', (d) => process.stdout.write(d));
    child.stderr?.on('data', (d) => {
      const s = d.toString();
      stderrTail = (stderrTail + s).slice(-800);
      process.stderr.write(d);
    });

    child.on('error', reject);
    child.on('close', (code, signal) => {
      if (code === 0) {
        resolve();
      } else {
        reject(
          new Error(
            `Stage ${stageLabel} exited with ${signal ? `signal ${signal}` : `code ${code}`}${stderrTail ? `: ${stderrTail.trim().slice(-200)}` : ''}`
          )
        );
      }
    });
  });
}

export async function runIsolatedWorkerJob(params: {
  jobId: string;
  inputDiskPath: string;
  originalName: string;
  cleanSfx: boolean;
  bubbleCleanMode?: BubbleCleanMode;
  baseSettings: CleaningSettings;
  onProgress?: (p: IsolatedJobProgress) => void;
}): Promise<IsolatedJobResult> {
  const t0 = Date.now();
  const jobRoot = path.join(os.tmpdir(), 'bearbot-isolated-jobs', params.jobId);
  const inDir = path.join(jobRoot, 'in');
  const outDir = path.join(jobRoot, 'out');
  const scratchDir = path.join(jobRoot, 'scratch');
  fs.mkdirSync(inDir, { recursive: true });
  fs.mkdirSync(outDir, { recursive: true });
  fs.mkdirSync(scratchDir, { recursive: true });

  const settings: CleaningSettings = {
    ...params.baseSettings,
    enableSFX: params.cleanSfx,
    bubbleCleanMode: params.bubbleCleanMode || params.baseSettings.bubbleCleanMode || 'WHITEN',
  };
  const settingsPath = path.join(jobRoot, 'settings.json');
  fs.writeFileSync(settingsPath, JSON.stringify(settings), 'utf-8');

  params.onProgress?.({ stage: 'EXTRACTING', currentPage: 0, totalPages: 0 });

  const pageNames: string[] = [];
  if (params.originalName.toLowerCase().endsWith('.zip')) {
    const zipBuf = await fs.promises.readFile(params.inputDiskPath);
    const zip = await JSZip.loadAsync(zipBuf);
    const entries = Object.keys(zip.files)
      .filter((k) => !zip.files[k].dir && /\.(png|jpe?g|webp)$/i.test(k) && !k.includes('__MACOSX'))
      .sort(naturalSort);
    for (const entryName of entries) {
      const base = path.basename(entryName);
      const buf = await zip.files[entryName].async('nodebuffer');
      await fs.promises.writeFile(path.join(inDir, base), buf);
      pageNames.push(base);
    }
  } else if (/\.(png|jpe?g|webp)$/i.test(params.originalName)) {
    const base = path.basename(params.originalName);
    await fs.promises.copyFile(params.inputDiskPath, path.join(inDir, base));
    pageNames.push(base);
  }

  // Delete original upload immediately after extracting
  await fs.promises.unlink(params.inputDiskPath).catch(() => {});

  if (pageNames.length === 0) {
    throw new Error('No valid PNG/JPG/WebP images found in upload');
  }

  let totalCleanedRegions = 0;
  const cleanedFiles: { name: string; fullPath: string }[] = [];

  for (let i = 0; i < pageNames.length; i++) {
    const name = pageNames[i];
    const inPath = path.join(inDir, name);
    const outName = `${name.replace(/\.[^.]+$/, '')}_cleaned.png`;
    const outPath = path.join(outDir, outName);
    const pageScratch = path.join(scratchDir, `p_${i}`);
    fs.mkdirSync(pageScratch, { recursive: true });

    // Stage 1: CTD in its own isolated subprocess (100% RAM reclaimed on exit)
    params.onProgress?.({ stage: 'DETECTING_CTD', currentPage: i + 1, totalPages: pageNames.length });
    await runStageSubprocess(['ctd', inPath, pageScratch], `CTD p${i + 1}`);

    // Stage 2: RT-DETR in its own isolated subprocess (100% RAM reclaimed on exit)
    params.onProgress?.({ stage: 'DETECTING_RTDETR', currentPage: i + 1, totalPages: pageNames.length });
    await runStageSubprocess(['rtdetr', inPath, pageScratch, settingsPath], `RT-DETR p${i + 1}`);

    // Stage 3: Clean + LaMa inpainting in its own isolated subprocess (100% RAM reclaimed on exit)
    params.onProgress?.({ stage: 'CLEANING_LAMA', currentPage: i + 1, totalPages: pageNames.length });
    await runStageSubprocess(['clean', inPath, pageScratch, settingsPath, outPath], `CLEAN/LaMa p${i + 1}`);

    const statsPath = path.join(pageScratch, 'clean_stats.json');
    if (fs.existsSync(statsPath)) {
      try {
        const st = JSON.parse(fs.readFileSync(statsPath, 'utf-8')) as { cleanedRegions?: number };
        totalCleanedRegions += st.cleanedRegions || 0;
      } catch {
        // ignore
      }
    }

    cleanedFiles.push({ name: outName, fullPath: outPath });
    // Free disk scratch & original page immediately
    await fs.promises.rm(pageScratch, { recursive: true, force: true }).catch(() => {});
    await fs.promises.unlink(inPath).catch(() => {});
  }

  params.onProgress?.({ stage: 'PACKAGING', currentPage: pageNames.length, totalPages: pageNames.length });

  let outputPath: string;
  let outputFileName: string;

  if (cleanedFiles.length === 1) {
    outputPath = cleanedFiles[0].fullPath;
    outputFileName = cleanedFiles[0].name;
  } else {
    const outZip = new JSZip();
    for (const cf of cleanedFiles) {
      const data = await fs.promises.readFile(cf.fullPath);
      outZip.file(cf.name, data);
    }
    const zipBuf = await outZip.generateAsync({ type: 'nodebuffer', compression: 'STORE' });
    outputFileName = `${params.jobId}_cleaned.zip`;
    outputPath = path.join(jobRoot, outputFileName);
    await fs.promises.writeFile(outputPath, zipBuf);
    await fs.promises.rm(outDir, { recursive: true, force: true }).catch(() => {});
  }

  return {
    outputPath,
    outputFileName,
    totalPages: pageNames.length,
    cleanedRegions: totalCleanedRegions,
    totalProcessingTimeMs: Date.now() - t0,
    bubbleCleanMode: settings.bubbleCleanMode,
  };
}
