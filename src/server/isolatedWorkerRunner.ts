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

function runBatchSubprocess(
  stage: 'ctd' | 'rtdetr' | 'clean',
  manifestPath: string,
  extraEnv: Record<string, string>,
  onProgress?: (p: IsolatedJobProgress) => void
): Promise<{ ok: boolean; code: number | null; signal: string | null; errorText: string }> {
  return new Promise((resolve) => {
    const compiledRunner = path.resolve(process.cwd(), 'dist-server', 'onnxStageRunner.mjs');
    const useCompiled = fs.existsSync(compiledRunner);
    const localTsx = path.resolve(process.cwd(), 'node_modules', '.bin', 'tsx');

    const cmd = useCompiled ? process.execPath : fs.existsSync(localTsx) ? localTsx : 'npx';
    const args = useCompiled
      ? [compiledRunner, stage, manifestPath]
      : fs.existsSync(localTsx)
        ? [path.resolve(process.cwd(), 'src/server/onnxStageRunner.ts'), stage, manifestPath]
        : ['tsx', path.resolve(process.cwd(), 'src/server/onnxStageRunner.ts'), stage, manifestPath];

    const child = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        ...extraEnv,
        NODE_OPTIONS: process.env.SUBPROCESS_NODE_OPTIONS || '--max-old-space-size=256',
      },
    });

    let stderrTail = '';
    let stdoutBuf = '';

    child.stdout?.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      stdoutBuf += text;
      const lines = stdoutBuf.split('\n');
      stdoutBuf = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('PROGRESS:')) {
          const [, st, cur, tot] = line.trim().split(':');
          onProgress?.({
            stage: st,
            currentPage: Number(cur) || 1,
            totalPages: Number(tot) || 1,
          });
        } else if (line.trim()) {
          console.log(line);
        }
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      const s = chunk.toString();
      stderrTail = (stderrTail + s).slice(-800);
      process.stderr.write(chunk);
    });

    child.on('error', (err) => {
      resolve({ ok: false, code: -1, signal: null, errorText: err.message });
    });

    child.on('close', (code, signal) => {
      resolve({
        ok: code === 0,
        code,
        signal: signal ? String(signal) : null,
        errorText: stderrTail.trim().slice(-240),
      });
    });
  });
}

export async function runIsolatedWorkerJob(params: {
  jobId: string;
  inputDiskPath: string;
  originalName: string;
  cleanSfx: boolean;
  bubbleCleanMode?: BubbleCleanMode;
  outputFormat?: 'jpg' | 'png' | 'psd';
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

  await fs.promises.unlink(params.inputDiskPath).catch(() => {});

  if (pageNames.length === 0) {
    throw new Error('No valid PNG/JPG/WebP images found in upload');
  }

  const outputFormat: 'jpg' | 'png' | 'psd' = params.outputFormat || 'jpg';
  const ext = outputFormat === 'psd' ? 'psd' : outputFormat === 'png' ? 'png' : 'jpg';
  const pages = pageNames.map((name, index) => ({
    index,
    name,
    inPath: path.join(inDir, name),
    outPath: path.join(outDir, `${name.replace(/\.[^.]+$/, '')}_cleaned.${ext}`),
    scratchDir: path.join(scratchDir, `p_${index}`),
  }));

  const manifestPath = path.join(jobRoot, 'manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify({ pages, settings, outputFormat }), 'utf-8');

  // Stage 1 (BATCH, optional when ENABLE_CTD=1): CTD loads ONCE for entire chapter, then exits
  if (process.env.ENABLE_CTD === '1' || process.env.ENABLE_CTD === 'true') {
    params.onProgress?.({ stage: 'DETECTING_CTD', currentPage: 1, totalPages: pages.length });
    const ctdRes = await runBatchSubprocess('ctd', manifestPath, {}, params.onProgress);
    if (!ctdRes.ok) {
      // If CTD hit OOM (137) on an ultra-tall stitched chapter, continue with RT-DETR + LaMa instead of failing!
      console.warn(
        `[Worker] CTD batch exited (${ctdRes.signal || ctdRes.code}) — continuing cleanly with RT-DETR + LaMa.`
      );
    }
  }

  // Stage 2 (BATCH): RT-DETR INT8 loads ONCE for entire chapter, fuses CTD masks if present, then exits!
  params.onProgress?.({ stage: 'DETECTING_RTDETR', currentPage: 1, totalPages: pages.length });
  const detRes = await runBatchSubprocess(
    'rtdetr',
    manifestPath,
    { ENABLE_CTD: '0' }, // never reload CTD inside RT-DETR stage
    params.onProgress
  );
  if (!detRes.ok) {
    throw new Error(`Stage RT-DETR failed (${detRes.signal || detRes.code}): ${detRes.errorText}`);
  }

  // Stage 3 (BATCH): Cleaning + LaMa inpainting loads ONCE for entire chapter, then exits!
  params.onProgress?.({ stage: 'CLEANING_LAMA', currentPage: 1, totalPages: pages.length });
  const cleanRes = await runBatchSubprocess('clean', manifestPath, {}, params.onProgress);
  if (!cleanRes.ok) {
    // If LaMa hit OOM (137) on a huge stitched chapter, retry clean stage with built-in JS inpainting fallback!
    if (cleanRes.code === 137 || cleanRes.signal === 'SIGKILL') {
      console.warn('[Worker] LaMa clean batch hit OOM (137) — retrying with low-RAM inpainting fallback.');
      const fallbackRes = await runBatchSubprocess(
        'clean',
        manifestPath,
        { ENABLE_LAMA: '0' },
        params.onProgress
      );
      if (!fallbackRes.ok) {
        throw new Error(
          `Stage CLEAN fallback failed (${fallbackRes.signal || fallbackRes.code}): ${fallbackRes.errorText}`
        );
      }
    } else {
      throw new Error(`Stage CLEAN failed (${cleanRes.signal || cleanRes.code}): ${cleanRes.errorText}`);
    }
  }

  let totalCleanedRegions = 0;
  const statsPath = path.join(outDir, 'batch_stats.json');
  if (fs.existsSync(statsPath)) {
    try {
      const st = JSON.parse(fs.readFileSync(statsPath, 'utf-8')) as { cleanedRegions?: number };
      totalCleanedRegions = st.cleanedRegions || 0;
    } catch {
      // ignore
    }
  }

  params.onProgress?.({ stage: 'PACKAGING', currentPage: pages.length, totalPages: pages.length });

  let outputPath: string;
  let outputFileName: string;

  if (pages.length === 1) {
    outputPath = pages[0].outPath;
    outputFileName = path.basename(pages[0].outPath);
  } else {
    const outZip = new JSZip();
    for (const p of pages) {
      if (fs.existsSync(p.outPath)) {
        outZip.file(path.basename(p.outPath), await fs.promises.readFile(p.outPath));
      }
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
    totalPages: pages.length,
    cleanedRegions: totalCleanedRegions,
    totalProcessingTimeMs: Date.now() - t0,
    bubbleCleanMode: settings.bubbleCleanMode,
  };
}
