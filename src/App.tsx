import React, { useEffect, useState } from 'react';
import { Upload, X } from 'lucide-react';
import {
  BotMessage,
  ChapterJob,
  CleaningEngineId,
  CleaningSettings,
  DetectedRegion,
  RegionCategory,
} from './engine/types';
import { BotConsoleView } from './components/BotConsoleView';
import { ReviewStudioView } from './components/ReviewStudioView';
import { ChapterQueueView } from './components/ChapterQueueView';
import { BenchmarkReportView } from './components/BenchmarkReportView';

type ActiveTab = 'BOT_CONSOLE' | 'REVIEW_STUDIO' | 'CHAPTER_QUEUE' | 'BENCHMARK_AUDIT';

const INITIAL_SETTINGS: CleaningSettings = {
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

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('BOT_CONSOLE');
  const [jobs, setJobs] = useState<ChapterJob[]>([]);
  const [messages, setMessages] = useState<BotMessage[]>([]);
  const [settings, setSettings] = useState<CleaningSettings>(INITIAL_SETTINGS);
  const [discordStatus, setDiscordStatus] = useState({
    configured: false,
    connected: false,
    registeredCommands: [
      '/clean',
      '/detect',
      '/review',
      '/status',
      '/cancel',
      '/settings',
      '/export',
      '/help',
    ],
  });

  const [selectedChapterId, setSelectedChapterId] = useState<string>('chapter1');
  const [selectedPageIndex, setSelectedPageIndex] = useState<number>(0);
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Upload Modal State
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);
  const [uploadTitle, setUploadTitle] = useState<string>('');
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);
  const [uploadCleanSfx, setUploadCleanSfx] = useState<boolean>(true);

  const fetchAppState = async () => {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error('Failed to load state from BearBot server');
      const data = await res.json();
      setJobs(data.jobs || []);
      setMessages(data.messages || []);
      if (data.settings) setSettings(data.settings);
      if (data.discordStatus) setDiscordStatus(data.discordStatus);
    } catch (err) {
      setErrorBanner((err as Error).message);
    }
  };

  useEffect(() => {
    fetchAppState();
  }, []);

  const handleExecuteCommand = async (command: string) => {
    setIsBusy(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/bot/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Command failed');
      setJobs(data.jobs || []);
      setMessages(data.messages || []);
      if (data.settings) setSettings(data.settings);
      if (data.activeChapterId) {
        setSelectedChapterId(data.activeChapterId);
      }
      if (command.trim().toLowerCase().startsWith('/review')) {
        setActiveTab('REVIEW_STUDIO');
      }
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleUpdateSettings = async (partial: Partial<CleaningSettings>) => {
    setErrorBanner(null);
    const prevSettings = settings;
    setSettings((curr) => ({ ...curr, ...partial }));
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(partial),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to update settings');
      setSettings(data.settings);
      setJobs(data.jobs || []);
    } catch (err) {
      setSettings(prevSettings);
      setErrorBanner((err as Error).message);
    }
  };

  const handleProcessChapter = async (
    chapterId: string,
    mode: 'DETECT_ONLY' | 'DETECT_AND_CLEAN'
  ) => {
    setIsBusy(true);
    setErrorBanner(null);
    try {
      const res = await fetch(`/api/chapters/${chapterId}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Chapter processing failed');
      setJobs(data.jobs || []);
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleProcessSinglePage = async (
    chapterId: string,
    pageIndex: number,
    mode: 'DETECT_ONLY' | 'DETECT_AND_CLEAN'
  ) => {
    setIsBusy(true);
    setErrorBanner(null);
    try {
      const res = await fetch(`/api/chapters/${chapterId}/pages/${pageIndex}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Page processing failed');
      setJobs((prev) => prev.map((j) => (j.id === chapterId ? data.job : j)));
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleModifyRegion = async (
    chapterId: string,
    pageIndex: number,
    payload: {
      action:
        | 'UPDATE_REGION'
        | 'ADD_STROKE'
        | 'REJECT_REGION'
        | 'RESTORE_REGION'
        | 'ADD_MANUAL_REGION';
      regionId?: string;
      updates?: Partial<DetectedRegion>;
      stroke?: { mode: 'add' | 'erase'; x: number; y: number; radiusPx: number };
      newRegionBbox?: { x: number; y: number; w: number; h: number; category: RegionCategory };
    }
  ) => {
    setIsBusy(true);
    setErrorBanner(null);
    try {
      const res = await fetch(`/api/chapters/${chapterId}/pages/${pageIndex}/region-edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Region update failed');
      setJobs((prev) => prev.map((j) => (j.id === chapterId ? data.job : j)));
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleUndo = async (chapterId: string) => {
    setIsBusy(true);
    try {
      const res = await fetch(`/api/chapters/${chapterId}/undo`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Undo failed');
      setJobs((prev) => prev.map((j) => (j.id === chapterId ? data.job : j)));
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleRedo = async (chapterId: string) => {
    setIsBusy(true);
    try {
      const res = await fetch(`/api/chapters/${chapterId}/redo`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Redo failed');
      setJobs((prev) => prev.map((j) => (j.id === chapterId ? data.job : j)));
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleExportChapter = async (chapterId: string) => {
    setIsBusy(true);
    setErrorBanner(null);
    try {
      const res = await fetch(`/api/chapters/${chapterId}/export`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Export failed');
      setJobs((prev) => prev.map((j) => (j.id === chapterId ? data.job : j)));
      if (data.zipUrl) {
        const zipRes = await fetch(data.zipUrl, { cache: 'no-store' });
        if (!zipRes.ok) {
          throw new Error(`Failed to fetch chapter ZIP (${zipRes.status})`);
        }
        const buf = await zipRes.arrayBuffer();
        const bytes = new Uint8Array(buf);
        if (
          bytes.length < 22 ||
          bytes[0] !== 0x50 ||
          bytes[1] !== 0x4b ||
          bytes[2] !== 0x03 ||
          bytes[3] !== 0x04
        ) {
          throw new Error('Downloaded archive did not contain a valid PK ZIP signature');
        }
        const blob = new Blob([bytes], { type: 'application/zip' });
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `${chapterId}_cleaned.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(blobUrl), 30000);
      }
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleInspectWithGemini = async (
    chapterId: string,
    pageIndex: number,
    regionId: string
  ): Promise<{
    detectedText: string;
    scriptLanguage: string;
    recommendedCategory: RegionCategory;
    recommendedEngine: CleaningEngineId;
    reconstructionNotes: string;
  }> => {
    const res = await fetch('/api/gemini/inspect-region', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chapterId, pageIndex, regionId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Gemini region analysis failed');
    return data;
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFiles || uploadFiles.length === 0) {
      setErrorBanner('Please select at least one image or chapter ZIP file to upload.');
      return;
    }
    setIsBusy(true);
    setErrorBanner(null);
    try {
      const formData = new FormData();
      if (uploadTitle.trim()) {
        formData.append('title', uploadTitle.trim());
      }
      formData.append('enableSFX', String(uploadCleanSfx));
      for (let i = 0; i < uploadFiles.length; i++) {
        formData.append('files', uploadFiles[i]);
      }
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      setJobs(data.jobs || []);
      if (data.job?.id) {
        setSelectedChapterId(data.job.id);
        setSelectedPageIndex(0);
      }
      setIsUploadOpen(false);
      setUploadTitle('');
      setUploadFiles(null);
      setActiveTab('REVIEW_STUDIO');
    } catch (err) {
      setErrorBanner((err as Error).message);
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="flex items-center justify-between gap-8 px-6 py-4 bg-white border-b border-slate-200">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('BOT_CONSOLE');
          }}
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
        >
          BearBot Studio
        </a>

        {/* Zone 2: 4 clean single-line text navigation links */}
        <nav className="flex items-center gap-6 text-sm font-medium text-slate-600 overflow-x-auto">
          <button
            type="button"
            onClick={() => setActiveTab('BOT_CONSOLE')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'BOT_CONSOLE'
                ? 'text-slate-900 border-b-2 border-slate-900 font-semibold'
                : ''
            }`}
          >
            Bot Console
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('REVIEW_STUDIO')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'REVIEW_STUDIO'
                ? 'text-slate-900 border-b-2 border-slate-900 font-semibold'
                : ''
            }`}
          >
            Review Studio
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('CHAPTER_QUEUE')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'CHAPTER_QUEUE'
                ? 'text-slate-900 border-b-2 border-slate-900 font-semibold'
                : ''
            }`}
          >
            Chapter Queue
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('BENCHMARK_AUDIT')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
              activeTab === 'BENCHMARK_AUDIT'
                ? 'text-slate-900 border-b-2 border-slate-900 font-semibold'
                : ''
            }`}
          >
            3-Chapter Benchmark
          </button>
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => setIsUploadOpen(true)}
            className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Upload Chapter ZIP
          </button>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1400px] w-full mx-auto px-6 py-6 space-y-5">
        {errorBanner && (
          <div
            role="alert"
            className="p-3.5 bg-rose-50 border border-rose-200 rounded-lg flex items-center justify-between text-xs text-rose-800"
          >
            <span>{errorBanner}</span>
            <button
              type="button"
              aria-label="Dismiss error"
              onClick={() => setErrorBanner(null)}
              className="text-rose-600 hover:text-rose-900 font-semibold"
            >
              Dismiss
            </button>
          </div>
        )}

        {activeTab === 'BOT_CONSOLE' && (
          <BotConsoleView
            messages={messages}
            jobs={jobs}
            settings={settings}
            discordStatus={discordStatus}
            isBusy={isBusy}
            onExecuteCommand={handleExecuteCommand}
            onOpenReview={(chapterId, pageIdx = 0) => {
              setSelectedChapterId(chapterId);
              setSelectedPageIndex(pageIdx);
              setActiveTab('REVIEW_STUDIO');
            }}
            onUpdateSettings={handleUpdateSettings}
            onOpenUploadModal={() => setIsUploadOpen(true)}
          />
        )}

        {activeTab === 'REVIEW_STUDIO' && (
          <ReviewStudioView
            jobs={jobs}
            selectedChapterId={selectedChapterId}
            selectedPageIndex={selectedPageIndex}
            isBusy={isBusy}
            onSelectChapter={(id) => {
              setSelectedChapterId(id);
              setSelectedPageIndex(0);
            }}
            onSelectPage={(idx) => setSelectedPageIndex(idx)}
            onModifyRegion={handleModifyRegion}
            onUndo={handleUndo}
            onRedo={handleRedo}
            onInspectWithGemini={handleInspectWithGemini}
            onExportChapter={handleExportChapter}
          />
        )}

        {activeTab === 'CHAPTER_QUEUE' && (
          <ChapterQueueView
            jobs={jobs}
            settings={settings}
            isBusy={isBusy}
            onProcessChapter={handleProcessChapter}
            onProcessSinglePage={handleProcessSinglePage}
            onOpenReview={(chapterId, pageIdx) => {
              setSelectedChapterId(chapterId);
              setSelectedPageIndex(pageIdx);
              setActiveTab('REVIEW_STUDIO');
            }}
            onExportChapter={handleExportChapter}
            onUpdateSettings={handleUpdateSettings}
            onOpenUploadModal={() => setIsUploadOpen(true)}
          />
        )}

        {activeTab === 'BENCHMARK_AUDIT' && (
          <BenchmarkReportView
            jobs={jobs}
            isBusy={isBusy}
            onRunChapter={(id) => handleProcessChapter(id, 'DETECT_AND_CLEAN')}
            onOpenReview={(chapterId, pageIdx) => {
              setSelectedChapterId(chapterId);
              setSelectedPageIndex(pageIdx);
              setActiveTab('REVIEW_STUDIO');
            }}
          />
        )}
      </main>

      {/* Upload Chapter ZIP / Images Dialog */}
      {isUploadOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="upload-modal-title"
          className="fixed inset-0 z-50 bg-slate-900/50 flex items-center justify-center p-4"
        >
          <div className="bg-white border border-slate-200 rounded-lg max-w-md w-full p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 id="upload-modal-title" className="text-sm font-semibold text-slate-900">
                Upload Manhwa / Manga Chapter (ZIP or Images)
              </h2>
              <button
                type="button"
                aria-label="Close upload modal"
                onClick={() => setIsUploadOpen(false)}
                className="p-1 text-slate-500 hover:text-slate-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleUploadSubmit} className="space-y-4">
              <label className="block space-y-1">
                <span className="text-xs font-medium text-slate-700">Chapter Title</span>
                <input
                  type="text"
                  value={uploadTitle}
                  onChange={(e) => setUploadTitle(e.target.value)}
                  placeholder="e.g. Solo Leveling Ch. 45"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded focus:outline-none focus:ring-2 focus:ring-slate-900"
                />
              </label>

              <label className="block space-y-1">
                <span className="text-xs font-medium text-slate-700">
                  Select Chapter ZIP Archive or Multiple Page Images (.png, .jpg, .webp, .zip)
                </span>
                <input
                  type="file"
                  multiple
                  accept=".zip,.png,.jpg,.jpeg,.webp"
                  onChange={(e) => setUploadFiles(e.target.files)}
                  className="w-full text-xs text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-slate-900 file:text-white hover:file:bg-slate-800"
                />
              </label>

              <label className="flex items-center justify-between p-2.5 rounded border border-slate-200 bg-slate-50 cursor-pointer">
                <div>
                  <div className="text-xs font-medium text-slate-900">
                    Clean Sound Effects (SFX)
                  </div>
                  <div className="text-xs text-slate-500">
                    Uncheck to clean only Bubbles, OT &amp; ST and leave SFX untouched.
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={uploadCleanSfx}
                  onChange={(e) => setUploadCleanSfx(e.target.checked)}
                  className="rounded border-slate-300 text-slate-900 focus:ring-slate-900"
                />
              </label>

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsUploadOpen(false)}
                  className="px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isBusy}
                  className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded hover:bg-slate-800 disabled:opacity-50"
                >
                  <Upload className="w-3.5 h-3.5" />
                  {isBusy ? 'Uploading & Cleaning...' : 'Start Chapter Cleaning'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
