import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyRunOverrides,
  buildCleanSlashCommand,
  parseCleanCommandArgs,
} from './botCommandParser';
import { CleaningSettings } from './types';

const SAMPLE_SETTINGS: CleaningSettings = {
  enableBubble: true,
  enableOT: true,
  enableST: true,
  enableSFX: true,
  bubbleCleanMode: 'WHITEN',
  defaultPaddingPx: 3,
  defaultFeatherPx: 2,
  engineRouting: 'AUTO_LADDER',
  sourceLanguages: ['KO', 'JA', 'ZH', 'EN'],
  strictQualityGate: true,
  preserveColoredBubbles: true,
  maxStripWindowHeight: 2048,
  cloudGpuConsent: false,
};

test('parseCleanCommandArgs handles sfx:false and --no-sfx flags', () => {
  const a = parseCleanCommandArgs(['chapter1', 'sfx:false']);
  assert.equal(a.chapterHint, 'chapter1');
  assert.equal(a.enableSFX, false);

  const b = parseCleanCommandArgs(['--no-sfx', 'chapter2', 'mode:text_only']);
  assert.equal(b.chapterHint, 'chapter2');
  assert.equal(b.enableSFX, false);
  assert.equal(b.bubbleCleanMode, 'TEXT_ONLY');
});

test('parseCleanCommandArgs handles sfx:true and --sfx flags', () => {
  const a = parseCleanCommandArgs(['chapter3', 'sfx:true', 'mode:whiten']);
  assert.equal(a.chapterHint, 'chapter3');
  assert.equal(a.enableSFX, true);
  assert.equal(a.bubbleCleanMode, 'WHITEN');

  const b = parseCleanCommandArgs(['clean_sfx:true', 'chapter1']);
  assert.equal(b.chapterHint, 'chapter1');
  assert.equal(b.enableSFX, true);
});

test('parseCleanCommandArgs leaves enableSFX undefined when omitted', () => {
  const res = parseCleanCommandArgs(['chapter2']);
  assert.equal(res.chapterHint, 'chapter2');
  assert.equal(res.enableSFX, undefined);
  assert.equal(res.bubbleCleanMode, undefined);
});

test('buildCleanSlashCommand emits explicit sfx flag', () => {
  assert.equal(
    buildCleanSlashCommand('chapter1', { enableSFX: false }),
    '/clean chapter1 sfx:false'
  );
  assert.equal(
    buildCleanSlashCommand('chapter2', { enableSFX: true, bubbleCleanMode: 'TEXT_ONLY' }),
    '/clean chapter2 sfx:true mode:text_only'
  );
});

test('applyRunOverrides overrides enableSFX without mutating base settings', () => {
  const next = applyRunOverrides(SAMPLE_SETTINGS, { enableSFX: false });
  assert.equal(next.enableSFX, false);
  assert.equal(SAMPLE_SETTINGS.enableSFX, true);
});
