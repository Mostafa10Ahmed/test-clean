import dotenv from 'dotenv';
import fs from 'fs';
import os from 'os';
import path from 'path';
import express from 'express';
import multer from 'multer';
import { BubbleCleanMode, CleaningSettings } from '../engine/types';
import { runIsolatedWorkerJob } from './isolatedWorkerRunner';

dotenv.config({ path: '.env', override: false });
if (fs.existsSync(path.resolve(process.cwd(), 'env'))) {
  dotenv.config({ path: 'env', override: false });
}

const DEFAULT_WORKER_SETTINGS: CleaningSettings = {
  enableBubble: true,
  enableOT: true,
  enableST: true,
  enableSFX: false,
  bubbleCleanMode: 'WHITEN',
  engineRouting: 'AUTO',
  strictQualityGate: true,
  maxStripWindowHeight: 2048,
  autoReviewFlagSfx: true,
  defaultPaddingPx: 6,
  defaultFeatherPx: 3,
  preserveColoredBubbles: true,
};

interface WorkerJobRecord {
  jobId: string;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  stage: string;
  currentPage: number;
  totalPages: number;
  processedPages: number;
  cleanedRegions: number;
  totalProcessingTimeMs: number;
  bubbleCleanMode: BubbleCleanMode;
  outputPath?: string;
  outputFileName?: string;
  error?: string;
  createdAt: number;
}

const uploadTmpDir = path.join(os.tmpdir(), 'bearbot-uploads');
fs.mkdirSync(uploadTmpDir, { recursive: true });

const workerDiskUpload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadTmpDir),
    filename: (_req, file, cb) => cb(null, `${Date.now()}_${path.basename(file.originalname)}`),
  }),
  limits: { fileSize: 150 * 1024 * 1024 },
});

const app = express();
const PORT = Number(process.env.PORT || process.env.DEFAULT_APP_PORT) || 3000;

const workerJobs = new Map<string, WorkerJobRecord>();
const pendingQueueIds: string[] = [];
let activeJobId: string | null = null;
let workerQueueChain: Promise<void> = Promise.resolve();

function checkWorkerAuth(req: express.Request, res: express.Response): boolean {
  const expected = (process.env.WORKER_TOKEN || '').trim();
  if (!expected) return true;
  const auth = (req.headers.authorization || '').replace(/^Bearer\s+/i, '').trim();
  const headerTok = String(req.headers['x-worker-token'] || '').trim();
  if (auth === expected || headerTok === expected) return true;
  res.status(401).json({ error: 'Unauthorized worker token' });
  return false;
}

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    status: 'ready',
    mode: 'worker-lite',
    rssMb: Number((process.memoryUsage().rss / 1048576).toFixed(1)),
  });
});

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, status: 'ready', mode: 'worker-lite' });
});

app.post('/worker/jobs', workerDiskUpload.single('file'), (req, res) => {
  if (!checkWorkerAuth(req, res)) return;
  const uploaded = req.file;
  if (!uploaded) {
    res.status(400).json({ error: 'Missing file upload' });
    return;
  }

  const cleanSfx = String(req.body?.clean_sfx || 'false').toLowerCase() === 'true';
  const rawMode = String(req.body?.bubble_mode || '').toUpperCase();
  const bubbleCleanMode: BubbleCleanMode | undefined =
    rawMode === 'TEXT_ONLY' ? 'TEXT_ONLY' : rawMode === 'WHITEN' ? 'WHITEN' : undefined;
  const rawFmt = String(req.body?.output_format || 'jpg').toLowerCase();
  const outputFormat: 'jpg' | 'png' | 'psd' =
    rawFmt === 'psd' ? 'psd' : rawFmt === 'png' ? 'png' : 'jpg';

  const jobId = `wjob_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const record: WorkerJobRecord = {
    jobId,
    status: 'QUEUED',
    stage: 'QUEUED',
    currentPage: 0,
    totalPages: 0,
    processedPages: 0,
    cleanedRegions: 0,
    totalProcessingTimeMs: 0,
    bubbleCleanMode: bubbleCleanMode || 'WHITEN',
    createdAt: Date.now(),
  };
  workerJobs.set(jobId, record);
  pendingQueueIds.push(jobId);

  const diskPath = uploaded.path;
  const origName = uploaded.originalname || 'chapter.zip';
  const queuePosNow = pendingQueueIds.length + (activeJobId ? 1 : 0);

  console.log(
    `[Worker] Queued ${jobId} (#${queuePosNow}, ${origName}, ${(uploaded.size / 1048576).toFixed(2)} MB) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
  );

  workerQueueChain = workerQueueChain.then(async () => {
    const idx = pendingQueueIds.indexOf(jobId);
    if (idx >= 0) pendingQueueIds.splice(idx, 1);
    activeJobId = jobId;
    record.status = 'PROCESSING';
    record.stage = 'EXTRACTING';
    console.log(
      `[Worker] Starting ${jobId} (${origName}, clean_sfx=${cleanSfx}) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
    );
    try {
      const result = await runIsolatedWorkerJob({
        jobId,
        inputDiskPath: diskPath,
        originalName: origName,
        cleanSfx,
        bubbleCleanMode,
        outputFormat,
        baseSettings: DEFAULT_WORKER_SETTINGS,
        onProgress: (p) => {
          record.stage = p.stage;
          record.currentPage = p.currentPage;
          record.totalPages = p.totalPages;
          record.processedPages = Math.max(0, p.currentPage - 1);
        },
      });
      record.totalPages = result.totalPages;
      record.processedPages = result.totalPages;
      record.currentPage = result.totalPages;
      record.cleanedRegions = result.cleanedRegions;
      record.totalProcessingTimeMs = result.totalProcessingTimeMs;
      record.bubbleCleanMode = result.bubbleCleanMode;
      record.outputPath = result.outputPath;
      record.outputFileName = result.outputFileName;
      record.stage = 'COMPLETED';
      record.status = 'COMPLETED';
      console.log(
        `[Worker] Completed ${jobId} (${record.totalPages} pages, ${record.cleanedRegions} regions, ${(record.totalProcessingTimeMs / 1000).toFixed(1)}s) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
      );
    } catch (err) {
      console.error(`[Worker] Failed ${jobId}:`, err);
      await fs.promises.unlink(diskPath).catch(() => {});
      record.stage = 'FAILED';
      record.status = 'FAILED';
      record.error = (err as Error).message;
    } finally {
      activeJobId = null;
    }
  });

  // Clean up completed/failed job metadata and temp dirs older than 1 hour
  const cutoff = Date.now() - 60 * 60 * 1000;
  for (const [k, v] of workerJobs.entries()) {
    if (v.createdAt < cutoff && (v.status === 'COMPLETED' || v.status === 'FAILED')) {
      workerJobs.delete(k);
      fs.promises
        .rm(path.join(os.tmpdir(), 'bearbot-isolated-jobs', k), { recursive: true, force: true })
        .catch(() => {});
    }
  }

  res.status(202).json({ jobId, status: record.status });
});

app.get('/worker/jobs/:id', (req, res) => {
  if (!checkWorkerAuth(req, res)) return;
  const rec = workerJobs.get(req.params.id);
  if (!rec) {
    res.status(404).json({ error: 'Job not found' });
    return;
  }
  const qIdx = pendingQueueIds.indexOf(rec.jobId);
  const queuePosition = rec.status === 'QUEUED' ? (qIdx >= 0 ? qIdx + 1 : 1) : 0;
  const queueLength = pendingQueueIds.length + (activeJobId ? 1 : 0);
  res.json({
    jobId: rec.jobId,
    status: rec.status,
    stage: rec.stage,
    queuePosition,
    queueLength,
    currentPage: rec.currentPage,
    totalPages: rec.totalPages,
    processedPages: rec.processedPages,
    cleanedRegions: rec.cleanedRegions,
    totalProcessingTimeMs: rec.totalProcessingTimeMs,
    bubbleCleanMode: rec.bubbleCleanMode,
    outputFileName: rec.outputFileName,
    error: rec.error,
  });
});

app.get('/worker/jobs/:id/result', (req, res) => {
  if (!checkWorkerAuth(req, res)) return;
  const rec = workerJobs.get(req.params.id);
  if (!rec || rec.status !== 'COMPLETED' || !rec.outputPath || !fs.existsSync(rec.outputPath)) {
    res.status(404).json({ error: 'Job result not ready' });
    return;
  }
  res.download(rec.outputPath, rec.outputFileName || 'cleaned_output');
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(
    `BearBot LITE WORKER listening on http://0.0.0.0:${PORT} | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`
  );
});
