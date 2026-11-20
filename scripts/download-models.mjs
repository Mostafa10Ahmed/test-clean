#!/usr/bin/env node
/**
 * Pre-downloads all 3 BearBot ONNX models into ./models/
 * Models:
 *   1. detector-v4-s_int8.onnx       (RT-DETR Comic Text & Bubble Detector)
 *   2. comictextdetector.pt.onnx     (CTD Text Segmentation Mask Model)
 *   3. lama-manga.onnx               (Manga-LaMa Inpainting Model)
 */
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

const MODELS_DIR = process.env.MODEL_DIR ? path.resolve(process.env.MODEL_DIR) : path.resolve(process.cwd(), 'models');
fs.mkdirSync(MODELS_DIR, { recursive: true });

const MODELS = [
  {
    name: 'detector-v4-s_int8.onnx',
    url: 'https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector-v4-s_int8.onnx',
  },
  {
    name: 'comictextdetector.pt.onnx',
    url: 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx',
  },
  {
    name: 'lama-manga.onnx',
    url: 'https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx',
  },
];

for (const model of MODELS) {
  const dest = path.join(MODELS_DIR, model.name);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1024 * 1024) {
    const mb = (fs.statSync(dest).size / 1024 / 1024).toFixed(1);
    console.log(`[OK] ${model.name} already downloaded (${mb} MB)`);
    continue;
  }
  console.log(`[DOWNLOADING] ${model.name} ...`);
  const tmp = `${dest}.part`;
  execFileSync('curl', ['-L', '--progress-bar', '-o', tmp, model.url], { stdio: 'inherit' });
  fs.renameSync(tmp, dest);
  const mb = (fs.statSync(dest).size / 1024 / 1024).toFixed(1);
  console.log(`[DONE] ${model.name} (${mb} MB)`);
}

console.log('\nAll BearBot ONNX models are ready in ./models/');
