import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { ChapterCleaningManager } from './src/server/chapterProcessor';
import { BearDiscordBot } from './src/server/discordBot';
import { buildTestCleanRepoZip } from './src/server/repoExporter';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 150 * 1024 * 1024 }, // 150MB max chapter ZIP upload
});

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT || process.env.DEFAULT_APP_PORT) || 3000;

  app.use(express.json({ limit: '25mb' }));

  const manager = new ChapterCleaningManager();
  const discordBot = new BearDiscordBot(manager);
  discordBot.startIfConfigured().catch(console.error);

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
    res.json({ ok: true, status: 'ready' });
  });

  // Get full application state (chapters, bot messages, settings, Discord gateway status)
  app.get('/api/state', (_req, res) => {
    res.json({
      jobs: manager.getJobs(),
      messages: manager.getMessages(),
      settings: manager.getSettings(),
      discordStatus: discordBot.getStatus(),
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
