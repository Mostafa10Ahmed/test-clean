import React, { useState } from 'react';
import {
  Send,
  Download,
  Sliders,
  Eye,
  Play,
  Terminal,
  Upload,
  CheckCircle2,
  FolderArchive,
  Package,
} from 'lucide-react';
import { BotMessage, ChapterJob, CleaningSettings } from '../engine/types';
import { buildCleanSlashCommand } from '../engine/botCommandParser';

interface BotConsoleViewProps {
  messages: BotMessage[];
  jobs: ChapterJob[];
  settings: CleaningSettings;
  discordStatus: {
    configured: boolean;
    connected: boolean;
    botTag?: string;
    registeredCommands: string[];
  };
  isBusy: boolean;
  onExecuteCommand: (cmd: string) => Promise<void>;
  onOpenReview: (chapterId: string, pageIndex?: number) => void;
  onUpdateSettings: (partial: Partial<CleaningSettings>) => Promise<void>;
  onOpenUploadModal: () => void;
}

const SLASH_PRESETS = [
  { cmd: '/clean chapter1', desc: 'Clean Test Chapter 1 (13 pages)' },
  { cmd: '/clean chapter2', desc: 'Clean Test Chapter 2 (17 pages)' },
  { cmd: '/clean chapter3', desc: 'Clean Test Chapter 3 (13 pages)' },
  { cmd: '/detect chapter1', desc: 'Detect regions only (no pixel edits)' },
  { cmd: '/review chapter1', desc: 'Open Chapter 1 in Review Studio' },
  { cmd: '/status', desc: 'Show queue & worker progress' },
  { cmd: '/settings', desc: 'Inspect / toggle cleaning engine settings' },
  { cmd: '/export chapter1', desc: 'Package lossless PNG ZIP archive' },
  { cmd: '/help', desc: 'Show full bot command manual' },
];

export const BotConsoleView: React.FC<BotConsoleViewProps> = ({
  messages,
  jobs,
  settings,
  discordStatus,
  isBusy,
  onExecuteCommand,
  onOpenReview,
  onUpdateSettings,
  onOpenUploadModal,
}) => {
  const [inputCmd, setInputCmd] = useState('');
  const [isDownloadingRepo, setIsDownloadingRepo] = useState(false);
  const [repoDownloadInfo, setRepoDownloadInfo] = useState<string | null>(null);
  const [repoDownloadError, setRepoDownloadError] = useState<string | null>(null);
  const cleanWithSfx = settings.enableSFX;

  const handleDownloadRepoZip = async () => {
    if (isDownloadingRepo) return;
    setIsDownloadingRepo(true);
    setRepoDownloadError(null);
    setRepoDownloadInfo(null);
    try {
      const res = await fetch(`/api/export-bot-repo?format=json&t=${Date.now()}`, {
        cache: 'no-store',
      });
      if (!res.ok) {
        throw new Error(`Export request failed (${res.status})`);
      }
      const data = (await res.json()) as {
        filename: string;
        fileCount: number;
        sizeBytes: number;
        base64: string;
      };
      if (!data.base64) {
        throw new Error('Server returned empty ZIP payload');
      }
      const binaryStr = window.atob(data.base64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      // Verify PK\x03\x04 ZIP header magic bytes
      if (
        bytes.length < 22 ||
        bytes[0] !== 0x50 ||
        bytes[1] !== 0x4b ||
        bytes[2] !== 0x03 ||
        bytes[3] !== 0x04
      ) {
        throw new Error('Invalid ZIP header signature');
      }
      const blob = new Blob([bytes], { type: 'application/zip' });
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = data.filename || 'test-clean-bot.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 15000);
      setRepoDownloadInfo(
        `Verified ZIP (${data.fileCount} files · ${(bytes.length / 1024).toFixed(1)} KB · PK signature OK)`
      );
    } catch (err) {
      setRepoDownloadError((err as Error).message);
    } finally {
      setIsDownloadingRepo(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputCmd.trim() || isBusy) return;
    const cmd = inputCmd.trim();
    setInputCmd('');
    await onExecuteCommand(cmd);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Left 8 Columns: Interactive Discord / Bot Slash Command Terminal */}
      <div className="lg:col-span-8 border border-slate-200 bg-white rounded-lg flex flex-col min-h-[680px]">
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <Terminal className="w-4 h-4 text-slate-700" />
            <h2 className="text-sm font-semibold text-slate-900">
              BearBot Scanlation Cleaning Terminal
            </h2>
            <span className="text-xs text-slate-500">
              · Slash Commands & Interactive Chapter Pipeline
            </span>
          </div>
          <div className="text-xs text-slate-500 font-mono tabular-nums">
            {discordStatus.connected
              ? `Discord Connected (${discordStatus.botTag})`
              : 'Web Bot Gateway Active · Discord Ready'}
          </div>
        </div>

        {/* Quick Slash Command Bar with Explicit /clean SFX Toggle */}
        <div className="px-5 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center gap-2 overflow-x-auto">
          <label className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-800 bg-white border border-slate-300 rounded cursor-pointer whitespace-nowrap shrink-0">
            <input
              type="checkbox"
              aria-label="Clean SFX in /clean command"
              checked={cleanWithSfx}
              onChange={(e) => {
                onUpdateSettings({ enableSFX: e.target.checked });
              }}
              className="rounded border-slate-300 text-slate-900 focus:ring-slate-900"
            />
            <span>/clean SFX: {cleanWithSfx ? 'ON' : 'OFF'}</span>
          </label>
          <span className="text-slate-300" aria-hidden="true">|</span>
          {SLASH_PRESETS.slice(0, 6).map((item) => {
            const resolvedCmd = item.cmd.startsWith('/clean ')
              ? buildCleanSlashCommand(item.cmd.replace('/clean ', ''), {
                  enableSFX: cleanWithSfx,
                })
              : item.cmd;
            return (
              <button
                key={item.cmd}
                type="button"
                disabled={isBusy}
                onClick={() => onExecuteCommand(resolvedCmd)}
                className="px-2.5 py-1 text-xs font-mono font-medium text-slate-700 bg-white border border-slate-200 rounded hover:bg-slate-100 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
              >
                {resolvedCmd}
              </button>
            );
          })}
        </div>

        {/* Transcript Feed */}
        <div className="flex-1 p-5 space-y-5 overflow-y-auto max-h-[500px]">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.sender === 'USER' ? 'items-end' : 'items-start'
              }`}
            >
              <div className="flex items-center gap-2 mb-1 text-xs text-slate-500">
                <span className="font-semibold text-slate-800">
                  {msg.sender === 'USER' ? 'Scanlation Operator' : 'BearBot Engine'}
                </span>
                <span aria-hidden="true">·</span>
                <span className="font-mono tabular-nums">
                  {new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })}
                </span>
              </div>

              <div
                className={`max-w-2xl rounded-lg p-4 ${
                  msg.sender === 'USER'
                    ? 'bg-slate-900 text-white font-mono text-xs'
                    : 'bg-slate-50 border border-slate-200 text-slate-800 text-sm w-full'
                }`}
              >
                <p className="leading-relaxed whitespace-pre-line">{msg.text}</p>

                {msg.embed && (
                  <div className="mt-3 pt-3 border-t border-slate-200 space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="text-sm font-semibold text-slate-900">
                        {msg.embed.title}
                      </h3>
                      <span className="text-xs font-mono text-emerald-700 tabular-nums">
                        {msg.embed.status}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      {msg.embed.fields.map((f, i) => (
                        <div key={i} className="bg-white p-2.5 rounded border border-slate-200">
                          <div className="text-xs text-slate-500">{f.label}</div>
                          <div className="text-xs font-mono font-medium text-slate-900 mt-0.5 tabular-nums">
                            {f.value}
                          </div>
                        </div>
                      ))}
                    </div>

                    {msg.embed.actions && msg.embed.actions.length > 0 && (
                      <div className="flex flex-wrap items-center gap-2 pt-1">
                        {msg.embed.actions.map((act, idx) => (
                          <button
                            key={idx}
                            type="button"
                            disabled={isBusy}
                            onClick={() => {
                              if (act.command.startsWith('/review')) {
                                const chId = act.command.split(/\s+/)[1] || 'chapter1';
                                onOpenReview(chId, 0);
                              } else {
                                onExecuteCommand(act.command);
                              }
                            }}
                            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                              act.variant === 'primary'
                                ? 'bg-slate-900 text-white hover:bg-slate-800'
                                : 'bg-white text-slate-700 border border-slate-300 hover:bg-slate-100'
                            }`}
                          >
                            {act.label}
                          </button>
                        ))}
                        {msg.embed.zipUrl && (
                          <a
                            href={msg.embed.zipUrl}
                            download
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-emerald-800 bg-emerald-50 border border-emerald-200 rounded hover:bg-emerald-100 transition-colors whitespace-nowrap shrink-0"
                          >
                            <Download className="w-3.5 h-3.5" />
                            Download Cleaned Chapter ZIP
                          </a>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Command Input Bar */}
        <form
          onSubmit={handleSubmit}
          className="p-4 border-t border-slate-200 bg-slate-50 flex items-center gap-3"
        >
          <div className="relative flex-1">
            <input
              type="text"
              aria-label="Slash command input"
              value={inputCmd}
              onChange={(e) => setInputCmd(e.target.value)}
              placeholder="e.g. /clean chapter1 sfx:false, /clean chapter2 sfx:true, /detect chapter1, /export, /help..."
              className="w-full px-3.5 py-2 text-sm font-mono bg-white border border-slate-300 rounded-md text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-900"
            />
          </div>
          <button
            type="submit"
            disabled={isBusy || !inputCmd.trim()}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-md hover:bg-slate-800 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
          >
            <Send className="w-3.5 h-3.5" />
            {isBusy ? 'Processing...' : 'Run Command'}
          </button>
          <button
            type="button"
            onClick={onOpenUploadModal}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload Images / ZIP
          </button>
        </form>
      </div>

      {/* Right 4 Columns: Mounted Chapters + Independent Category & Engine Controls */}
      <div className="lg:col-span-4 space-y-6">
        {/* Mounted Real-World Test Chapters */}
        <div className="border border-slate-200 bg-white rounded-lg p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-900">
              Mounted Chapters ({jobs.length})
            </h3>
            <span className="text-xs text-slate-500 font-mono tabular-nums">
              Native Resolution
            </span>
          </div>

          <div className="space-y-3">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="p-3.5 rounded-md border border-slate-200 bg-slate-50/60 space-y-2.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold text-slate-900">{job.title}</div>
                    <div className="text-xs text-slate-500 font-mono tabular-nums mt-0.5">
                      {job.totalPages} pages · {job.cleanedRegions}/{job.totalRegions} regions cleaned ·{' '}
                      {job.avgTimePerPageMs}ms/p
                    </div>
                  </div>
                  <span className="text-xs font-mono font-medium text-slate-700 whitespace-nowrap shrink-0">
                    {job.status}
                  </span>
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    disabled={isBusy}
                    onClick={() =>
                      onExecuteCommand(
                        buildCleanSlashCommand(job.id, { enableSFX: cleanWithSfx })
                      )
                    }
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-white bg-slate-900 rounded hover:bg-slate-800 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
                  >
                    <Play className="w-3 h-3" />
                    {cleanWithSfx ? 'Clean (SFX ON)' : 'Clean (No SFX)'}
                  </button>
                  <button
                    type="button"
                    onClick={() => onOpenReview(job.id, 0)}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0"
                  >
                    <Eye className="w-3 h-3" />
                    Inspect & Edit
                  </button>
                  {job.zipDownloadUrl && (
                    <a
                      href={job.zipDownloadUrl}
                      download
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-emerald-800 bg-emerald-50 border border-emerald-200 rounded hover:bg-emerald-100 transition-colors whitespace-nowrap shrink-0"
                    >
                      <FolderArchive className="w-3 h-3" />
                      ZIP
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 4 Independent Cleaning Categories & Bubble Mode Controls */}
        <div className="border border-slate-200 bg-white rounded-lg p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-slate-700" />
            <h3 className="text-sm font-semibold text-slate-900">
              Cleaning Categories & Bubble Modes
            </h3>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-medium text-slate-600">
              Independent Region Categories
            </div>
            {(
              [
                {
                  key: 'enableBubble',
                  label: '1. Speech Bubbles (Dialogue & Narration)',
                  val: settings.enableBubble,
                },
                {
                  key: 'enableOT',
                  label: '2. OT (Outside Text over Screentones)',
                  val: settings.enableOT,
                },
                {
                  key: 'enableST',
                  label: '3. ST (Side Text & Margin Notes)',
                  val: settings.enableST,
                },
                {
                  key: 'enableSFX',
                  label: '4. SFX (Stylized Sound Effects)',
                  val: settings.enableSFX,
                },
              ] as const
            ).map((item) => (
              <label
                key={item.key}
                className="flex items-center justify-between py-1.5 px-2.5 rounded border border-slate-200 cursor-pointer hover:bg-slate-50"
              >
                <span className="text-xs text-slate-800">{item.label}</span>
                <input
                  type="checkbox"
                  checked={item.val}
                  onChange={(e) => onUpdateSettings({ [item.key]: e.target.checked })}
                  className="rounded border-slate-300 text-slate-900 focus:ring-slate-900"
                />
              </label>
            ))}
          </div>

          {/* Speech Bubble Cleaning Mode Selector */}
          <div className="space-y-1.5 pt-2 border-t border-slate-200">
            <div className="text-xs font-medium text-slate-600">
              Speech Bubble Cleaning Mode
            </div>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-100 rounded-md">
              <button
                type="button"
                onClick={() => onUpdateSettings({ bubbleCleanMode: 'WHITEN' })}
                className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  settings.bubbleCleanMode === 'WHITEN'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Smart Whitening (#FFF)
              </button>
              <button
                type="button"
                onClick={() => onUpdateSettings({ bubbleCleanMode: 'TEXT_ONLY' })}
                className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  settings.bubbleCleanMode === 'TEXT_ONLY'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Text-Only Removal
              </button>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {settings.bubbleCleanMode === 'WHITEN'
                ? 'Cleans white bubble interiors to pure #FFFFFF while preserving bubble outlines and colored/textured bubbles.'
                : 'Removes only lettering strokes inside bubbles while preserving original paper texture and gradients.'}
            </p>
          </div>

          <div className="pt-2 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
            <span className="inline-flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Immutable Originals & Lossless PNG
            </span>
            <span className="font-mono tabular-nums">Pad: {settings.defaultPaddingPx}px</span>
          </div>
        </div>

        {/* Ready-to-Push Repository Export Package for Mostafa10Ahmed/test-clean */}
        <div className="border border-slate-200 bg-white rounded-lg p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Package className="w-4 h-4 text-slate-700" />
            <h3 className="text-sm font-semibold text-slate-900">
              Export Bot Repo (test-clean)
            </h3>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Download all final BearBot source files (Discord bot with explicit <code className="font-mono text-slate-900">clean_sfx</code> option, ONNX cleaning engine, Dockerfile, model downloader, and README) packaged ready to push to <code className="font-mono text-slate-900">Mostafa10Ahmed/test-clean</code>.
          </p>
          <button
            type="button"
            onClick={handleDownloadRepoZip}
            disabled={isDownloadingRepo}
            className="w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 rounded-md hover:bg-slate-800 disabled:opacity-50 transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            {isDownloadingRepo ? 'Preparing Verified ZIP...' : 'Download test-clean-bot.zip'}
          </button>
          {repoDownloadInfo && (
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-2.5 py-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0 text-emerald-600" />
              <span>{repoDownloadInfo}</span>
            </div>
          )}
          {repoDownloadError && (
            <div className="text-[11px] font-medium text-rose-700 bg-rose-50 border border-rose-200 rounded px-2.5 py-1.5">
              Download error: {repoDownloadError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
