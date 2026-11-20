import fs from 'fs';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import os from 'os';
import path from 'path';
import { BubbleCleanMode } from '../engine/types';

export interface RemoteJobProgress {
  status: string;
  stage?: string;
  queuePosition?: number;
  queueLength?: number;
  currentPage?: number;
  totalPages?: number;
}

export interface RemoteCleanRequest {
  fileName: string;
  buffer: Buffer;
  cleanSfx: boolean;
  bubbleCleanMode?: BubbleCleanMode;
  onProgress?: (progress: RemoteJobProgress) => void;
}

export interface RemoteCleanResult {
  jobId: string;
  outputPath: string;
  outputFileName: string;
  totalPages: number;
  cleanedRegions: number;
  totalProcessingTimeMs: number;
  bubbleCleanMode: BubbleCleanMode;
}

function getWorkerBaseUrl(): string {
  return (process.env.WORKER_URL || '').trim().replace(/\/+$/, '');
}

function getWorkerHeaders(): Record<string, string> {
  const token = (process.env.WORKER_TOKEN || '').trim();
  return token ? { Authorization: `Bearer ${token}`, 'X-Worker-Token': token } : {};
}

export function isRemoteWorkerConfigured(): boolean {
  return Boolean(getWorkerBaseUrl());
}

export async function cleanWithRemoteWorker(req: RemoteCleanRequest): Promise<RemoteCleanResult> {
  const baseUrl = getWorkerBaseUrl();
  if (!baseUrl) {
    throw new Error('WORKER_URL is not configured');
  }

  const headers = getWorkerHeaders();
  const form = new FormData();
  const blob = new Blob([new Uint8Array(req.buffer)]);
  form.append('file', blob, req.fileName);
  form.append('clean_sfx', req.cleanSfx ? 'true' : 'false');
  if (req.bubbleCleanMode) {
    form.append('bubble_mode', req.bubbleCleanMode);
  }

  const submitRes = await fetch(`${baseUrl}/worker/jobs`, {
    method: 'POST',
    headers,
    body: form,
  });
  if (!submitRes.ok) {
    const errText = await submitRes.text().catch(() => '');
    throw new Error(`Worker submit failed (${submitRes.status}): ${errText.slice(0, 200)}`);
  }

  const submitted = (await submitRes.json()) as { jobId: string };
  const jobId = submitted.jobId;
  if (!jobId) {
    throw new Error('Worker did not return a jobId');
  }

  const pollIntervalMs = Number(process.env.WORKER_POLL_MS) || 3500;
  // Generous 45-minute timeout so queued chapters never time out prematurely
  const maxWaitMs = Number(process.env.WORKER_TIMEOUT_MS) || 45 * 60 * 1000;
  const startedAt = Date.now();

  let statusPayload: {
    status: string;
    stage?: string;
    queuePosition?: number;
    queueLength?: number;
    currentPage?: number;
    totalPages?: number;
    cleanedRegions?: number;
    totalProcessingTimeMs?: number;
    bubbleCleanMode?: BubbleCleanMode;
    outputFileName?: string;
    error?: string;
  } | null = null;

  let consecutiveErrors = 0;

  while (Date.now() - startedAt < maxWaitMs) {
    await new Promise((r) => setTimeout(r, pollIntervalMs));
    try {
      const pollRes = await fetch(`${baseUrl}/worker/jobs/${encodeURIComponent(jobId)}`, {
        headers,
      });
      if (!pollRes.ok) {
        if (pollRes.status === 404) {
          throw new Error(`Worker job ${jobId} was lost (worker may have restarted)`);
        }
        consecutiveErrors++;
        if (consecutiveErrors >= 15) {
          throw new Error(`Worker unreachable (${pollRes.status}) after repeated retries`);
        }
        continue;
      }
      consecutiveErrors = 0;
      statusPayload = (await pollRes.json()) as typeof statusPayload;
      if (statusPayload) {
        req.onProgress?.({
          status: statusPayload.status,
          stage: statusPayload.stage,
          queuePosition: statusPayload.queuePosition,
          queueLength: statusPayload.queueLength,
          currentPage: statusPayload.currentPage,
          totalPages: statusPayload.totalPages,
        });
      }
      if (statusPayload?.status === 'COMPLETED') {
        break;
      }
      if (statusPayload?.status === 'FAILED') {
        throw new Error(statusPayload.error || 'Remote worker failed to clean chapter');
      }
    } catch (err) {
      if ((err as Error).message.includes('was lost') || (err as Error).message.includes('failed to clean')) {
        throw err;
      }
      consecutiveErrors++;
      if (consecutiveErrors >= 15) {
        throw err;
      }
    }
  }

  if (!statusPayload || statusPayload.status !== 'COMPLETED') {
    throw new Error(`Worker job ${jobId} timed out after ${Math.round(maxWaitMs / 1000)}s`);
  }

  const downloadRes = await fetch(`${baseUrl}/worker/jobs/${encodeURIComponent(jobId)}/result`, {
    headers,
  });
  if (!downloadRes.ok) {
    const errText = await downloadRes.text().catch(() => '');
    throw new Error(`Worker result download failed (${downloadRes.status}): ${errText.slice(0, 200)}`);
  }

  const tmpDir = path.join(os.tmpdir(), 'bearbot-remote-results');
  fs.mkdirSync(tmpDir, { recursive: true });
  const outputFileName = statusPayload.outputFileName || `${jobId}_cleaned.zip`;
  const outputPath = path.join(tmpDir, `${jobId}_${path.basename(outputFileName)}`);
  if (downloadRes.body) {
    await pipeline(
      Readable.fromWeb(downloadRes.body as unknown as import('stream/web').ReadableStream),
      fs.createWriteStream(outputPath)
    );
  } else {
    const outBuf = Buffer.from(await downloadRes.arrayBuffer());
    await fs.promises.writeFile(outputPath, outBuf);
  }

  return {
    jobId,
    outputPath,
    outputFileName,
    totalPages: statusPayload.totalPages || 1,
    cleanedRegions: statusPayload.cleanedRegions || 0,
    totalProcessingTimeMs: statusPayload.totalProcessingTimeMs || Date.now() - startedAt,
    bubbleCleanMode: statusPayload.bubbleCleanMode || req.bubbleCleanMode || 'WHITEN',
  };
}
