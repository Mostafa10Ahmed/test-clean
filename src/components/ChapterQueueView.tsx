import React from 'react';
import {
  Play,
  Scan,
  Eye,
  Download,
  RotateCcw,
  CheckCircle2,
  Cloud,
  Upload,
} from 'lucide-react';
import { ChapterJob, CleaningSettings } from '../engine/types';

interface ChapterQueueViewProps {
  jobs: ChapterJob[];
  settings: CleaningSettings;
  isBusy: boolean;
  onProcessChapter: (chapterId: string, mode: 'DETECT_ONLY' | 'DETECT_AND_CLEAN') => Promise<void>;
  onProcessSinglePage: (
    chapterId: string,
    pageIndex: number,
    mode: 'DETECT_ONLY' | 'DETECT_AND_CLEAN'
  ) => Promise<void>;
  onOpenReview: (chapterId: string, pageIndex: number) => void;
  onExportChapter: (chapterId: string) => Promise<void>;
  onUpdateSettings: (partial: Partial<CleaningSettings>) => Promise<void>;
  onOpenUploadModal: () => void;
}

export const ChapterQueueView: React.FC<ChapterQueueViewProps> = ({
  jobs,
  settings,
  isBusy,
  onProcessChapter,
  onProcessSinglePage,
  onOpenReview,
  onExportChapter,
  onUpdateSettings,
  onOpenUploadModal,
}) => {
  return (
    <div className="space-y-6">
      {/* Top Batch Controls & Optional Cloud GPU Consent */}
      <div className="p-5 bg-white border border-slate-200 rounded-lg flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Chapter Batch Queue & Long-Strip Worker Scheduler
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Processes full webtoon/manhwa chapters in 2,048px overlapping strip windows without rescaling or overwriting source files.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded text-xs text-slate-700 cursor-pointer">
            <Cloud className="w-3.5 h-3.5 text-slate-600" />
            <span>Approve Cloud GPU / AI Crop Inpainting</span>
            <input
              type="checkbox"
              checked={settings.cloudGpuConsent}
              onChange={(e) =>
                onUpdateSettings({
                  cloudGpuConsent: e.target.checked,
                  engineRouting: e.target.checked ? 'ALLOW_CLOUD_AI' : 'AUTO_LADDER',
                })
              }
              className="rounded border-slate-300 text-slate-900"
            />
          </label>

          <button
            type="button"
            onClick={onOpenUploadModal}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 rounded hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload New Chapter (ZIP / Images)
          </button>
        </div>
      </div>

      {/* Chapter Job Cards & High-Density Page Tables */}
      <div className="space-y-6">
        {jobs.map((job) => (
          <div key={job.id} className="bg-white border border-slate-200 rounded-lg overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-4 bg-slate-50/70">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="text-sm font-semibold text-slate-900">{job.title}</h3>
                  <span className="text-xs font-mono text-slate-500 tabular-nums">
                    ID: {job.id} · {job.status}
                  </span>
                </div>
                <div className="text-xs text-slate-500 font-mono tabular-nums mt-1">
                  Pages: {job.processedPages}/{job.totalPages} · Regions Cleaned:{' '}
                  {job.cleanedRegions}/{job.totalRegions} · Avg Speed: {job.avgTimePerPageMs} ms/page · Peak Memory: {job.peakMemoryMb} MB
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => onProcessChapter(job.id, 'DETECT_ONLY')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-50 whitespace-nowrap shrink-0"
                >
                  <Scan className="w-3.5 h-3.5" />
                  Detect Only
                </button>
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => onProcessChapter(job.id, 'DETECT_AND_CLEAN')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-slate-900 rounded hover:bg-slate-800 disabled:opacity-50 whitespace-nowrap shrink-0"
                >
                  <Play className="w-3.5 h-3.5" />
                  Clean All {job.totalPages} Pages
                </button>
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => onExportChapter(job.id)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 whitespace-nowrap shrink-0"
                >
                  <Download className="w-3.5 h-3.5" />
                  Build ZIP
                </button>
                {job.zipDownloadUrl && (
                  <a
                    href={job.zipDownloadUrl}
                    download
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-emerald-800 bg-emerald-50 border border-emerald-200 rounded hover:bg-emerald-100 whitespace-nowrap shrink-0"
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Download ZIP
                  </a>
                )}
              </div>
            </div>

            {/* Page-by-Page High-Density Data Grid */}
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-xs text-slate-500 bg-slate-50/40">
                    <th className="py-2.5 px-4 font-medium">Page</th>
                    <th className="py-2.5 px-4 font-medium">Dimensions</th>
                    <th className="py-2.5 px-4 font-medium text-right">Strip Windows</th>
                    <th className="py-2.5 px-4 font-medium text-right">Bubbles</th>
                    <th className="py-2.5 px-4 font-medium text-right">OT / ST</th>
                    <th className="py-2.5 px-4 font-medium text-right">SFX</th>
                    <th className="py-2.5 px-4 font-medium text-right">Time</th>
                    <th className="py-2.5 px-4 font-medium">Pixel Integrity</th>
                    <th className="py-2.5 px-4 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs font-mono tabular-nums">
                  {job.pages.map((p) => {
                    const bubbles = p.regions.filter((r) => r.category === 'BUBBLE').length;
                    const otSt = p.regions.filter(
                      (r) => r.category === 'OT' || r.category === 'ST'
                    ).length;
                    const sfx = p.regions.filter((r) => r.category === 'SFX').length;

                    return (
                      <tr key={p.pageIndex} className="hover:bg-slate-50">
                        <td className="py-2.5 px-4 font-medium text-slate-900">
                          #{String(p.pageIndex + 1).padStart(2, '0')} · {p.fileName}
                        </td>
                        <td className="py-2.5 px-4 text-slate-600">
                          {p.width}×{p.height}px
                        </td>
                        <td className="py-2.5 px-4 text-right text-slate-600">
                          {p.stripWindowCount}
                        </td>
                        <td className="py-2.5 px-4 text-right text-slate-900">{bubbles}</td>
                        <td className="py-2.5 px-4 text-right text-slate-700">{otSt}</td>
                        <td className="py-2.5 px-4 text-right text-slate-700">{sfx}</td>
                        <td className="py-2.5 px-4 text-right text-slate-600">
                          {p.processingTimeMs > 0 ? `${p.processingTimeMs} ms` : '—'}
                        </td>
                        <td className="py-2.5 px-4 text-emerald-700">
                          {p.untouchedPixelsBitIdentical
                            ? '100% Bit-Identical Outside Masks'
                            : 'Modified'}
                        </td>
                        <td className="py-2.5 px-4 text-right">
                          <div className="inline-flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={() => onOpenReview(job.id, p.pageIndex)}
                              className="inline-flex items-center gap-1 px-2 py-1 text-xs font-sans font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 whitespace-nowrap shrink-0"
                            >
                              <Eye className="w-3 h-3" />
                              Review
                            </button>
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() =>
                                onProcessSinglePage(job.id, p.pageIndex, 'DETECT_AND_CLEAN')
                              }
                              className="inline-flex items-center gap-1 px-2 py-1 text-xs font-sans font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-40 whitespace-nowrap shrink-0"
                            >
                              <RotateCcw className="w-3 h-3" />
                              Re-run
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
