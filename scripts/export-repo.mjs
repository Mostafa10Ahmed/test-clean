#!/usr/bin/env node
/**
 * Builds a clean, git-ready export archive of BearBot for pushing to
 * https://github.com/Mostafa10Ahmed/test-clean
 * Output: public/exports/test-clean-bot.zip
 */
import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';
import JSZip from 'jszip';

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, 'public', 'exports');
const OUT_ZIP = path.join(OUT_DIR, 'test-clean-bot.zip');

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

function collectFiles(dirAbs, baseRel) {
  const results = [];
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

export async function buildTestCleanRepoZip() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const zip = new JSZip();
  const included = [];

  for (const rel of ROOT_FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs) && fs.statSync(abs).isFile()) {
      zip.file(rel, fs.readFileSync(abs));
      included.push(rel);
    }
  }

  for (const dirName of INCLUDE_DIRS) {
    const files = collectFiles(path.join(ROOT, dirName), dirName);
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

  const tmpZip = `${OUT_ZIP}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmpZip, buf);
  fs.renameSync(tmpZip, OUT_ZIP);
  return {
    outputPath: OUT_ZIP,
    publicUrl: '/exports/test-clean-bot.zip',
    fileCount: included.length,
    sizeBytes: buf.byteLength,
    files: included,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  buildTestCleanRepoZip()
    .then((res) => {
      console.log(
        `Exported ${res.fileCount} files (${(res.sizeBytes / 1024).toFixed(1)} KB) -> ${res.outputPath}`
      );
    })
    .catch((err) => {
      console.error('Export failed:', err);
      process.exit(1);
    });
}
