import { BubbleCleanMode, CleaningSettings } from './types';

export interface ParsedCleanArgs {
  chapterHint?: string;
  enableSFX?: boolean;
  bubbleCleanMode?: BubbleCleanMode;
}

/**
 * Parses arguments passed to `/clean`, supporting explicit SFX & bubble mode flags:
 *   /clean chapter1 sfx:false
 *   /clean chapter2 sfx:true mode:text_only
 *   /clean chapter3 --no-sfx
 *   /clean --sfx chapter1
 */
export function parseCleanCommandArgs(tokens: string[]): ParsedCleanArgs {
  let chapterHint: string | undefined;
  let enableSFX: boolean | undefined;
  let bubbleCleanMode: BubbleCleanMode | undefined;

  for (const rawToken of tokens) {
    const token = rawToken.trim().toLowerCase();
    if (!token) continue;

    if (
      token === '--no-sfx' ||
      token === 'sfx:false' ||
      token === 'sfx:off' ||
      token === 'sfx:0' ||
      token === 'sfx:no' ||
      token === 'nosfx' ||
      token === 'clean_sfx:false'
    ) {
      enableSFX = false;
      continue;
    }

    if (
      token === '--sfx' ||
      token === 'sfx:true' ||
      token === 'sfx:on' ||
      token === 'sfx:1' ||
      token === 'sfx:yes' ||
      token === 'clean_sfx:true'
    ) {
      enableSFX = true;
      continue;
    }

    if (
      token === 'mode:text_only' ||
      token === 'mode:text-only' ||
      token === '--text-only' ||
      token === 'bubble_mode:text_only'
    ) {
      bubbleCleanMode = 'TEXT_ONLY';
      continue;
    }

    if (
      token === 'mode:whiten' ||
      token === '--whiten' ||
      token === 'bubble_mode:whiten'
    ) {
      bubbleCleanMode = 'WHITEN';
      continue;
    }

    if (!chapterHint && !token.startsWith('--') && !token.includes(':')) {
      chapterHint = token;
    }
  }

  return { chapterHint, enableSFX, bubbleCleanMode };
}

/**
 * Formats a canonical `/clean` command string with explicit SFX selection.
 */
export function buildCleanSlashCommand(
  chapterId: string,
  options: { enableSFX: boolean; bubbleCleanMode?: BubbleCleanMode }
): string {
  const parts = ['/clean', chapterId.trim() || 'chapter1', `sfx:${options.enableSFX ? 'true' : 'false'}`];
  if (options.bubbleCleanMode) {
    parts.push(`mode:${options.bubbleCleanMode === 'TEXT_ONLY' ? 'text_only' : 'whiten'}`);
  }
  return parts.join(' ');
}

/**
 * Merges optional per-command overrides onto base settings without mutating the base object.
 */
export function applyRunOverrides(
  base: CleaningSettings,
  overrides?: { enableSFX?: boolean; bubbleCleanMode?: BubbleCleanMode }
): CleaningSettings {
  if (!overrides) return { ...base };
  return {
    ...base,
    enableSFX: overrides.enableSFX !== undefined ? overrides.enableSFX : base.enableSFX,
    bubbleCleanMode:
      overrides.bubbleCleanMode !== undefined ? overrides.bubbleCleanMode : base.bubbleCleanMode,
  };
}
