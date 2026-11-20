import dotenv from 'dotenv';
import fs from 'fs';
import os from 'os';
import express from 'express';
import multer from 'multer';
import path from 'path';
import { BubbleCleanMode } from './src/engine/types';

// Load .env and fallback to legacy 'env' file without overriding Railway environment variables
dotenv.config({ path: '.env', override: false });
if (fs.existsSync(path.resolve(process.cwd(), 'env'))) {
  dotenv.config({ path: 'env', override: false });
}
import { createServer as createViteServer } from 'vite';
import { ChapterCleaningManager } from './src/server/chapterProcessor';
import { BearDiscordBot } from './src/server/discordBot';
import { handleOAuthCallback } from './src/server/driveOAuth';
import { runIsolatedWorkerJob } from './src/server/isolatedWorkerRunner';
import { buildTestCleanRepoZip } from './src/server/repoExporter';

const uploadTmpDir = path.join(os.tmpdir(), 'bearbot-uploads');
fs.mkdirSync(uploadTmpDir, { recursive: true });

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 150 * 1024 * 1024 }, // 150MB max chapter ZIP upload
});

const workerDiskUpload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadTmpDir),
    filename: (_req, file, cb) => cb(null, `${Date.now()}_${path.basename(file.originalname)}`),
  }),
  limits: { fileSize: 150 * 1024 * 1024 },
});

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

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT || process.env.DEFAULT_APP_PORT) || 3000;
  const isWorkerMode = process.env.WORKER_MODE === '1' || process.env.WORKER_MODE === 'true';

  app.use(express.json({ limit: '25mb' }));

  const manager = new ChapterCleaningManager();
  const discordBot = isWorkerMode ? null : new BearDiscordBot(manager);
  if (discordBot) {
    discordBot.startIfConfigured().catch(console.error);
  }

  // --- WORKER ASYNC FIFO QUEUE (1-at-a-time + subprocess isolation for full quality on 1GB RAM) ---
  const workerJobs = new Map<string, WorkerJobRecord>();
  const pendingQueueIds: string[] = [];
  let activeJobId: string | null = null;
  let workerQueueChain: Promise<void> = Promise.resolve();

  const checkWorkerAuth = (req: express.Request, res: express.Response): boolean => {
    const expected = (process.env.WORKER_TOKEN || '').trim();
    if (!expected) return true;
    const auth = (req.headers.authorization || '').replace(/^Bearer\s+/i, '').trim();
    const headerTok = String(req.headers['x-worker-token'] || '').trim();
    if (auth === expected || headerTok === expected) return true;
    res.status(401).json({ error: 'Unauthorized worker token' });
    return false;
  };

  app.get('/health', (_req, res) => {
    res.json({ ok: true, status: 'ready', mode: isWorkerMode ? 'worker' : 'bot' });
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

    console.log(`[Worker] Queued ${jobId} (#${queuePosNow}, ${origName}, ${(uploaded.size / 1048576).toFixed(2)} MB) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`);
    workerQueueChain = workerQueueChain.then(async () => {
      const idx = pendingQueueIds.indexOf(jobId);
      if (idx >= 0) pendingQueueIds.splice(idx, 1);
      activeJobId = jobId;
      record.status = 'PROCESSING';
      record.stage = 'EXTRACTING';
      console.log(`[Worker] Starting ${jobId} (${origName}, clean_sfx=${cleanSfx}) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`);
      try {
        const result = await runIsolatedWorkerJob({
          jobId,
          inputDiskPath: diskPath,
          originalName: origName,
          cleanSfx,
          bubbleCleanMode,
          baseSettings: manager.getSettings(),
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
        console.log(`[Worker] Completed ${jobId} (${record.totalPages} pages, ${record.cleanedRegions} regions, ${(record.totalProcessingTimeMs / 1000).toFixed(1)}s) | RSS=${(process.memoryUsage().rss / 1048576).toFixed(1)}MB`);
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

    // Prune completed/failed job metadata older than 1 hour
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

  if (isWorkerMode) {
    app.get('/api/health', (_req, res) => {
      res.json({ ok: true, status: 'ready', mode: 'worker' });
    });
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`BearBot WORKER Server listening on http://0.0.0.0:${PORT} (WORKER_MODE=1)`);
    });
    return;
  }

  // Serve static files from public (test chapters, cleaned PNGs, ZIP exports)
  app.use(
    express.static(path.resolve(process.cwd(), 'public'), {
      setHeaders: (res, filePath) => {
        if (filePath.endsWith('.zip')) {
          res.setHeader('Content-Type', 'application/zip');
          res.setHeader('Content-Disposition', `attachment; filename="${path.basename(filePath)}"`);
          res.setHeader('Cache-Control', 'no-store');
        }
      },
    })
  );

  // --- API ROUTES ---

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, status: 'ready', mode: 'bot' });
  });

  // Google Drive OAuth callback (for /profile Google Drive linking)
  app.get('/oauth/callback', async (req, res) => {
    const code = String(req.query.code || '');
    const state = String(req.query.state || '');
    if (!code || !state) {
      res.status(400).send('<h3>Missing OAuth code or state. Please retry from /profile.</h3>');
      return;
    }
    const result = await handleOAuthCallback(code, state);
    const color = result.ok ? '#16a34a' : '#dc2626';
    const title = result.ok ? '✅ تم ربط Google Drive بنجاح' : '❌ فشل ربط Google Drive';
    res.status(result.ok ? 200 : 400).send(
      `<!doctype html><html dir="rtl" lang="ar"><head><meta charset="utf-8"><title>${title}</title></head>` +
        `<body style="font-family:system-ui,sans-serif;background:#0f172a;color:#f8fafc;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;">` +
        `<div style="max-width:480px;padding:32px;border-radius:16px;background:#1e293b;border:2px solid ${color};text-align:center;">` +
        `<h2 style="margin-top:0;color:${color};">${title}</h2>` +
        `<p style="line-height:1.7;">${result.message}</p>` +
        `</div></body></html>`
    );
  });

  // Get full application state (chapters, bot messages, settings, Discord gateway status)
  app.get('/api/state', (_req, res) => {
    res.json({
      jobs: manager.getJobs(),
      messages: manager.getMessages(),
      settings: manager.getSettings(),
      discordStatus: discordBot ? discordBot.getStatus() : { configured: false, connected: false, registeredCommands: [] },
    });
  });

  // Execute a Bot Slash Command (/clean, /detect, /review, /status, /cancel, /settings, /export, /help)
  app.post('/api/bot/command', async (req, res) => {
    try {
      const { command } = req.body as { command: string };
      if (!command || typeof command !== 'string') {
        res.status(400).json({ error: 'Command string is required' });
        return;
      }
      const result = await manager.executeBotCommand(command);
      res.json({
        ...result,
        jobs: manager.getJobs(),
        messages: manager.getMessages(),
        settings: manager.getSettings(),
      });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Update global & chapter cleaning settings
  app.post('/api/settings', (req, res) => {
    try {
      const updated = manager.updateSettings(req.body || {});
      res.json({ settings: updated, jobs: manager.getJobs() });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Run Detect or Detect+Clean on a specific chapter
  app.post('/api/chapters/:id/process', async (req, res) => {
    try {
      const { id } = req.params;
      const mode = req.body?.mode === 'DETECT_ONLY' ? 'DETECT_ONLY' : 'DETECT_AND_CLEAN';
      const enableSFX =
        typeof req.body?.enableSFX === 'boolean' ? req.body.enableSFX : undefined;
      const bubbleCleanMode =
        req.body?.bubbleCleanMode === 'TEXT_ONLY' || req.body?.bubbleCleanMode === 'WHITEN'
          ? req.body.bubbleCleanMode
          : undefined;
      const job = await manager.processFullChapter(id, mode, { enableSFX, bubbleCleanMode });
      res.json({ job, jobs: manager.getJobs() });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Run Detect or Clean on a single page
  app.post('/api/chapters/:id/pages/:pageIndex/process', async (req, res) => {
    try {
      const { id, pageIndex } = req.params;
      const job = manager.getJob(id);
      if (!job) {
        res.status(404).json({ error: `Chapter ${id} not found` });
        return;
      }
      const mode = req.body?.mode || 'DETECT_AND_CLEAN';
      const page = await manager.processSinglePage(job, Number(pageIndex), mode);
      res.json({ page, job });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Modify a detected region (brush stroke add/erase, padding, bubble whitening vs text-only, engine override, reject, manual box)
  app.post('/api/chapters/:id/pages/:pageIndex/region-edit', async (req, res) => {
    try {
      const { id, pageIndex } = req.params;
      const page = await manager.modifyPageRegion({
        chapterId: id,
        pageIndex: Number(pageIndex),
        ...req.body,
      });
      const job = manager.getJob(id);
      res.json({ page, job });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Undo last region/mask edit on a chapter
  app.post('/api/chapters/:id/undo', async (req, res) => {
    try {
      const result = await manager.undo(req.params.id);
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // Redo last region/mask edit on a chapter
  app.post('/api/chapters/:id/redo', async (req, res) => {
    try {
      const result = await manager.redo(req.params.id);
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // Export chapter ZIP archive
  app.post('/api/chapters/:id/export', async (req, res) => {
    try {
      const job = manager.getJob(req.params.id);
      if (!job) {
        res.status(404).json({ error: 'Chapter not found' });
        return;
      }
      const zipUrl = await manager.buildChapterZipArchive(job);
      res.json({ zipUrl, job });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Upload single/multiple images or a full chapter ZIP archive
  app.post('/api/upload', upload.array('files', 100), async (req, res) => {
    try {
      const uploaded = (req.files as Express.Multer.File[]) || [];
      if (uploaded.length === 0) {
        res.status(400).json({ error: 'Please select at least one image or ZIP archive' });
        return;
      }
      const title = typeof req.body?.title === 'string' ? req.body.title : undefined;
      const enableSFX =
        req.body?.enableSFX === 'false' || req.body?.enableSFX === false
          ? false
          : req.body?.enableSFX === 'true' || req.body?.enableSFX === true
            ? true
            : undefined;
      const job = await manager.ingestUploadedFiles(
        uploaded.map((f) => ({ originalname: f.originalname, buffer: f.buffer })),
        title,
        enableSFX !== undefined ? { enableSFX } : undefined
      );
      res.json({ job, jobs: manager.getJobs() });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Export all final bot source files as a ready-to-push ZIP for Mostafa10Ahmed/test-clean
  app.get('/api/export-bot-repo', async (req, res) => {
    try {
      const result = await buildTestCleanRepoZip();
      if (req.query.format === 'json') {
        res.setHeader('Cache-Control', 'no-store');
        res.json({
          filename: 'test-clean-bot.zip',
          fileCount: result.fileCount,
          sizeBytes: result.sizeBytes,
          files: result.files,
          base64: result.buffer.toString('base64'),
        });
        return;
      }
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', 'attachment; filename="test-clean-bot.zip"');
      res.setHeader('Content-Length', String(result.buffer.byteLength));
      res.setHeader('Cache-Control', 'no-store');
      res.status(200).end(result.buffer);
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Server-side Gemini Vision inspection for an individual region crop
  app.post('/api/gemini/inspect-region', async (req, res) => {
    try {
      const { chapterId, pageIndex, regionId } = req.body as {
        chapterId: string;
        pageIndex: number;
        regionId: string;
      };
      const analysis = await manager.inspectRegionWithGemini(
        chapterId,
        Number(pageIndex),
        regionId
      );
      res.json(analysis);
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Vite middleware in development, static dist in production
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`BearBot Full-Stack Server listening on http://localhost:${PORT}`);
  });
}

startServer().catch(console.error);
