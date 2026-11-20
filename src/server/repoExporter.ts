import fs from 'fs';
import path from 'path';
import JSZip from 'jszip';

const ROOT_FILES = [
  'package.json',
  'tsconfig.json',
  'vite.config.ts',
  'index.html',
  'server.ts',
  'metadata.json',
  '.env.example',
  '.gitignore',
  'Dockerfile',
  '.dockerignore',
  'README.md',
  'LICENSE',
];

const INCLUDE_DIRS = ['src', 'scripts'];

function collectFiles(dirAbs: string, baseRel: string): { abs: string; rel: string }[] {
  const results: { abs: string; rel: string }[] = [];
  if (!fs.existsSync(dirAbs)) return results;
  for (const entry of fs.readdirSync(dirAbs, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || entry.name === 'node_modules' || entry.name === 'dist') {
      continue;
    }
    const full = path.join(dirAbs, entry.name);
    const rel = path.posix.join(baseRel, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectFiles(full, rel));
    } else if (entry.isFile()) {
      results.push({ abs: full, rel });
    }
  }
  return results;
}

export interface RepoExportResult {
  outputPath: string;
  publicUrl: string;
  fileCount: number;
  sizeBytes: number;
  files: string[];
  buffer: Buffer;
}

export async function buildTestCleanRepoZip(): Promise<RepoExportResult> {
  const root = process.cwd();
  const outDir = path.join(root, 'public', 'exports');
  const outZip = path.join(outDir, 'test-clean-bot.zip');
  fs.mkdirSync(outDir, { recursive: true });

  const zip = new JSZip();
  const included: string[] = [];

  for (const rel of ROOT_FILES) {
    const abs = path.join(root, rel);
    if (fs.existsSync(abs) && fs.statSync(abs).isFile()) {
      zip.file(rel, fs.readFileSync(abs));
      included.push(rel);
    }
  }

  for (const dirName of INCLUDE_DIRS) {
    const files = collectFiles(path.join(root, dirName), dirName);
    for (const f of files) {
      zip.file(f.rel, fs.readFileSync(f.abs));
      included.push(f.rel);
    }
  }

  const buf = await zip.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
    platform: 'DOS',
  });

  const tmpZip = `${outZip}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmpZip, buf);
  fs.renameSync(tmpZip, outZip);
  return {
    outputPath: outZip,
    publicUrl: '/exports/test-clean-bot.zip',
    fileCount: included.length,
    sizeBytes: buf.byteLength,
    files: included,
    buffer: buf,
  };
}
