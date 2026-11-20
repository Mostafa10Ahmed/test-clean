import React, { useState, useRef } from 'react';
import {
  Undo2,
  Redo2,
  Paintbrush,
  Eraser,
  PlusSquare,
  Check,
  RotateCcw,
  Download,
  ChevronLeft,
  ChevronRight,
  Layers,
  ScanSearch,
} from 'lucide-react';
import {
  BubbleCleanMode,
  ChapterJob,
  CleaningEngineId,
  DetectedRegion,
  RegionCategory,
} from '../engine/types';

interface ReviewStudioViewProps {
  jobs: ChapterJob[];
  selectedChapterId: string;
  selectedPageIndex: number;
  isBusy: boolean;
  onSelectChapter: (chapterId: string) => void;
  onSelectPage: (pageIndex: number) => void;
  onModifyRegion: (
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
  ) => Promise<void>;
  onUndo: (chapterId: string) => Promise<void>;
  onRedo: (chapterId: string) => Promise<void>;
  onInspectWithGemini: (
    chapterId: string,
    pageIndex: number,
    regionId: string
  ) => Promise<{
    detectedText: string;
    scriptLanguage: string;
    recommendedCategory: RegionCategory;
    recommendedEngine: CleaningEngineId;
    reconstructionNotes: string;
  }>;
  onExportChapter: (chapterId: string) => Promise<void>;
}

type ViewMode = 'SPLIT_COMPARE' | 'CLEANED' | 'MASKS_OVERLAY' | 'ORIGINAL';
type ToolMode = 'SELECT' | 'BRUSH_ADD' | 'BRUSH_ERASE' | 'ADD_SFX_BOX';

export const ReviewStudioView: React.FC<ReviewStudioViewProps> = ({
  jobs,
  selectedChapterId,
  selectedPageIndex,
  isBusy,
  onSelectChapter,
  onSelectPage,
  onModifyRegion,
  onUndo,
  onRedo,
  onInspectWithGemini,
  onExportChapter,
}) => {
  const currentJob = jobs.find((j) => j.id === selectedChapterId) || jobs[0];
  const currentPage =
    currentJob?.pages[selectedPageIndex] || currentJob?.pages[0];

  const [viewMode, setViewMode] = useState<ViewMode>('CLEANED');
  const [toolMode, setToolMode] = useState<ToolMode>('SELECT');
  const [splitPercent, setSplitPercent] = useState<number>(50);
  const [brushSize, setBrushSize] = useState<number>(8);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<'ALL' | RegionCategory>('ALL');
  const [geminiInsight, setGeminiInsight] = useState<{
    regionId: string;
    detectedText: string;
    scriptLanguage: string;
    recommendedCategory: RegionCategory;
    recommendedEngine: CleaningEngineId;
    reconstructionNotes: string;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);

  if (!currentJob || !currentPage) {
    return (
      <div className="p-8 border border-slate-200 bg-white rounded-lg text-center text-sm text-slate-600">
        No chapter pages available to review.
      </div>
    );
  }

  const filteredRegions = currentPage.regions.filter((r) =>
    categoryFilter === 'ALL' ? true : r.category === categoryFilter
  );

  const activeRegion =
    currentPage.regions.find((r) => r.id === selectedRegionId) ||
    filteredRegions[0] ||
    null;

  const handleCanvasClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current || isBusy) return;
    const rect = containerRef.current.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;
    const nativeX = Math.round(relX * currentPage.width);
    const nativeY = Math.round(relY * currentPage.height);

    if (toolMode === 'ADD_SFX_BOX') {
      const boxW = Math.min(180, Math.round(currentPage.width * 0.22));
      const boxH = 90;
      await onModifyRegion(currentJob.id, currentPage.pageIndex, {
        action: 'ADD_MANUAL_REGION',
        newRegionBbox: {
          x: Math.max(4, nativeX - Math.floor(boxW / 2)),
          y: Math.max(4, nativeY - Math.floor(boxH / 2)),
          w: boxW,
          h: boxH,
          category: 'SFX',
        },
      });
      setToolMode('SELECT');
      return;
    }

    if ((toolMode === 'BRUSH_ADD' || toolMode === 'BRUSH_ERASE') && activeRegion) {
      const localX = (nativeX - activeRegion.bbox.x) / Math.max(1, activeRegion.bbox.w);
      const localY = (nativeY - activeRegion.bbox.y) / Math.max(1, activeRegion.bbox.h);
      if (localX >= 0 && localX <= 1 && localY >= 0 && localY <= 1) {
        await onModifyRegion(currentJob.id, currentPage.pageIndex, {
          action: 'ADD_STROKE',
          regionId: activeRegion.id,
          stroke: {
            mode: toolMode === 'BRUSH_ADD' ? 'add' : 'erase',
            x: localX,
            y: localY,
            radiusPx: brushSize,
          },
        });
      }
      return;
    }

    // Select region under cursor
    const hit = currentPage.regions.find(
      (r) =>
        nativeX >= r.bbox.x &&
        nativeX <= r.bbox.x + r.bbox.w &&
        nativeY >= r.bbox.y &&
        nativeY <= r.bbox.y + r.bbox.h
    );
    if (hit) {
      setSelectedRegionId(hit.id);
    }
  };

  return (
    <div className="space-y-4">
      {/* Studio Top Control Toolbar */}
      <div className="p-4 bg-white border border-slate-200 rounded-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          {/* Chapter Selector */}
          <label className="flex items-center gap-2 text-xs font-medium text-slate-700">
            <span>Chapter:</span>
            <select
              aria-label="Select Chapter"
              value={currentJob.id}
              onChange={(e) => onSelectChapter(e.target.value)}
              className="h-9 px-2.5 py-0 leading-normal bg-slate-50 border border-slate-300 rounded text-xs font-semibold text-slate-900"
            >
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.title}
                </option>
              ))}
            </select>
          </label>

          {/* Page Stepper */}
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="Previous Page"
              disabled={currentPage.pageIndex <= 0}
              onClick={() => onSelectPage(Math.max(0, currentPage.pageIndex - 1))}
              className="h-9 px-2 border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-40 inline-flex items-center justify-center"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <select
              aria-label="Select Page"
              value={currentPage.pageIndex}
              onChange={(e) => onSelectPage(Number(e.target.value))}
              className="h-9 px-2.5 py-0 leading-normal bg-slate-50 border border-slate-300 rounded text-xs font-mono tabular-nums text-slate-900"
            >
              {currentJob.pages.map((p) => (
                <option key={p.pageIndex} value={p.pageIndex}>
                  Page {p.pageIndex + 1}/{currentJob.totalPages} — {p.fileName} ({p.regions.length} regions)
                </option>
              ))}
            </select>
            <button
              type="button"
              aria-label="Next Page"
              disabled={currentPage.pageIndex >= currentJob.pages.length - 1}
              onClick={() =>
                onSelectPage(Math.min(currentJob.pages.length - 1, currentPage.pageIndex + 1))
              }
              className="h-9 px-2 border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-40 inline-flex items-center justify-center"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* View Mode Segmented Control */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
            {(
              [
                { id: 'MASKS_OVERLAY', label: 'Detected Masks' },
                { id: 'CLEANED', label: 'Cleaned Output' },
                { id: 'SPLIT_COMPARE', label: 'Before/After Split' },
                { id: 'ORIGINAL', label: 'Original Scan' },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setViewMode(m.id)}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  viewMode === m.id
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Undo / Redo & Export */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={isBusy || currentJob.undoStack.length === 0}
            onClick={() => onUndo(currentJob.id)}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-40 whitespace-nowrap shrink-0"
          >
            <Undo2 className="w-3.5 h-3.5" />
            Undo ({currentJob.undoStack.length})
          </button>
          <button
            type="button"
            disabled={isBusy || currentJob.redoStack.length === 0}
            onClick={() => onRedo(currentJob.id)}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-40 whitespace-nowrap shrink-0"
          >
            <Redo2 className="w-3.5 h-3.5" />
            Redo ({currentJob.redoStack.length})
          </button>
          <button
            type="button"
            onClick={() => onExportChapter(currentJob.id)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-slate-900 rounded hover:bg-slate-800 whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Export ZIP
          </button>
        </div>
      </div>

      {/* Main 12-Col Review Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 8 Columns: Long-Strip Viewport Canvas with Interactive Mask Overlays */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-lg overflow-hidden">
          {/* Interactive Brush & Split Slider Sub-bar */}
          <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setToolMode('SELECT')}
                className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap shrink-0 ${
                  toolMode === 'SELECT'
                    ? 'bg-slate-900 text-white'
                    : 'bg-white text-slate-700 border border-slate-300'
                }`}
              >
                Inspect Region
              </button>
              <button
                type="button"
                onClick={() => setToolMode('BRUSH_ADD')}
                className={`inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap shrink-0 ${
                  toolMode === 'BRUSH_ADD'
                    ? 'bg-slate-900 text-white'
                    : 'bg-white text-slate-700 border border-slate-300'
                }`}
              >
                <Paintbrush className="w-3 h-3" />
                Add to Mask
              </button>
              <button
                type="button"
                onClick={() => setToolMode('BRUSH_ERASE')}
                className={`inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap shrink-0 ${
                  toolMode === 'BRUSH_ERASE'
                    ? 'bg-slate-900 text-white'
                    : 'bg-white text-slate-700 border border-slate-300'
                }`}
              >
                <Eraser className="w-3 h-3" />
                Erase from Mask
              </button>
              <button
                type="button"
                onClick={() => setToolMode('ADD_SFX_BOX')}
                className={`inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap shrink-0 ${
                  toolMode === 'ADD_SFX_BOX'
                    ? 'bg-slate-900 text-white'
                    : 'bg-white text-slate-700 border border-slate-300'
                }`}
              >
                <PlusSquare className="w-3 h-3" />
                Manual SFX Box
              </button>
            </div>

            <div className="flex items-center gap-4 text-xs text-slate-600">
              <label className="flex items-center gap-1.5 font-mono tabular-nums">
                <span>Brush: {brushSize}px</span>
                <input
                  type="range"
                  min={2}
                  max={28}
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-20 accent-slate-900"
                />
              </label>
              {viewMode === 'SPLIT_COMPARE' && (
                <label className="flex items-center gap-1.5 font-mono tabular-nums">
                  <span>Split: {splitPercent}%</span>
                  <input
                    type="range"
                    min={5}
                    max={95}
                    value={splitPercent}
                    onChange={(e) => setSplitPercent(Number(e.target.value))}
                    className="w-24 accent-slate-900"
                  />
                </label>
              )}
            </div>
          </div>

          {/* Scrollable Native-Aspect Strip Viewport */}
          <div className="max-h-[680px] overflow-y-auto bg-slate-900 p-4 flex justify-center">
            <div
              ref={containerRef}
              onClick={handleCanvasClick}
              data-testid="review-strip-canvas"
              className="relative w-full max-w-[680px] select-none cursor-crosshair bg-slate-800 shadow-sm"
            >
              {/* Base Image (Cleaned or Original) */}
              <img
                src={
                  viewMode === 'ORIGINAL'
                    ? currentPage.originalUrl
                    : currentPage.cleanedUrl
                }
                alt={`Page ${currentPage.pageIndex + 1} (${viewMode})`}
                referrerPolicy="no-referrer"
                className="w-full h-auto block"
              />

              {/* Split Before/After Overlay */}
              {viewMode === 'SPLIT_COMPARE' && (
                <div
                  className="absolute inset-0 overflow-hidden pointer-events-none"
                  style={{ clipPath: `inset(0 ${100 - splitPercent}% 0 0)` }}
                >
                  <img
                    src={currentPage.originalUrl}
                    alt="Original scan before cleaning"
                    referrerPolicy="no-referrer"
                    className="w-full h-auto block"
                  />
                </div>
              )}

              {/* Detected Regions Overlay Boxes */}
              {viewMode === 'MASKS_OVERLAY' &&
                filteredRegions.map((reg) => {
                  const leftPct = (reg.bbox.x / currentPage.width) * 100;
                  const topPct = (reg.bbox.y / currentPage.height) * 100;
                  const widthPct = (reg.bbox.w / currentPage.width) * 100;
                  const heightPct = (reg.bbox.h / currentPage.height) * 100;
                  const isSelected = activeRegion?.id === reg.id;

                  const borderColor =
                    reg.category === 'BUBBLE'
                      ? 'border-emerald-500 bg-emerald-500/15'
                      : reg.category === 'SFX'
                        ? 'border-amber-500 bg-amber-500/20'
                        : reg.category === 'ST'
                          ? 'border-sky-500 bg-sky-500/15'
                          : 'border-indigo-500 bg-indigo-500/15';

                  return (
                    <button
                      key={reg.id}
                      type="button"
                      aria-label={`Select region ${reg.id} (${reg.category})`}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        setSelectedRegionId(reg.id);
                      }}
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className={`absolute border-2 transition-opacity ${borderColor} ${
                        isSelected ? 'ring-2 ring-white z-10' : 'opacity-85 hover:opacity-100'
                      }`}
                    >
                      <span className="sr-only">
                        {reg.category} region at {reg.bbox.x},{reg.bbox.y}
                      </span>
                    </button>
                  );
                })}
            </div>
          </div>
        </div>

        {/* Right 4 Columns: Active Region Inspector + Manual Correction Controls */}
        <div className="lg:col-span-4 space-y-5">
          {/* Page Metadata & Category Filter */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-slate-700" />
                <h3 className="text-sm font-semibold text-slate-900">
                  Page {currentPage.pageIndex + 1} Regions ({currentPage.regions.length})
                </h3>
              </div>
              <span className="text-xs font-mono text-slate-500 tabular-nums">
                {currentPage.width}×{currentPage.height}px · {currentPage.stripWindowCount} strips
              </span>
            </div>

            <div className="flex flex-wrap gap-1 p-1 bg-slate-100 rounded-md">
              {(['ALL', 'BUBBLE', 'OT', 'ST', 'SFX'] as const).map((cat) => (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setCategoryFilter(cat)}
                  className={`px-2 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    categoryFilter === cat
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Region List */}
            <div className="max-h-48 overflow-y-auto divide-y divide-slate-100 border border-slate-200 rounded">
              {filteredRegions.length === 0 ? (
                <div className="p-3 text-xs text-slate-500 text-center">
                  No regions in this category on Page {currentPage.pageIndex + 1}.
                </div>
              ) : (
                filteredRegions.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => setSelectedRegionId(r.id)}
                    className={`w-full px-3 py-2 text-left flex items-center justify-between text-xs transition-colors ${
                      activeRegion?.id === r.id
                        ? 'bg-slate-900 text-white'
                        : 'bg-white text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <span className="font-mono truncate">
                      {r.id} · {r.category}
                    </span>
                    <span className="font-mono tabular-nums opacity-80 whitespace-nowrap shrink-0">
                      {r.bbox.w}×{r.bbox.h} · {r.appliedEngine || r.routedEngine}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Selected Region Parameter & Quality Editor */}
          {activeRegion ? (
            <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
                <div>
                  <h4 className="text-sm font-semibold text-slate-900 font-mono">
                    {activeRegion.id}
                  </h4>
                  <p className="text-xs text-slate-500 font-mono tabular-nums mt-0.5">
                    Box: ({activeRegion.bbox.x}, {activeRegion.bbox.y}) {activeRegion.bbox.w}×
                    {activeRegion.bbox.h}px · Conf: {(activeRegion.confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <span className="text-xs font-mono font-medium text-emerald-700">
                  {activeRegion.status}
                </span>
              </div>

              {/* Category & Engine Selector */}
              <div className="grid grid-cols-2 gap-3">
                <label className="block space-y-1">
                  <span className="text-xs text-slate-600">Category</span>
                  <select
                    aria-label="Region Category"
                    value={activeRegion.category}
                    onChange={(e) =>
                      onModifyRegion(currentJob.id, currentPage.pageIndex, {
                        action: 'UPDATE_REGION',
                        regionId: activeRegion.id,
                        updates: { category: e.target.value as RegionCategory },
                      })
                    }
                    className="w-full h-9 px-2.5 py-0 leading-normal text-xs bg-slate-50 border border-slate-300 rounded font-medium text-slate-900"
                  >
                    <option value="BUBBLE">Speech Bubble</option>
                    <option value="OT">Outside Text (OT)</option>
                    <option value="ST">Side Text (ST)</option>
                    <option value="SFX">Sound Effect (SFX)</option>
                  </select>
                </label>

                <label className="block space-y-1">
                  <span className="text-xs text-slate-600">Cleaning Engine</span>
                  <select
                    aria-label="Region Cleaning Engine"
                    value={activeRegion.routedEngine}
                    onChange={(e) =>
                      onModifyRegion(currentJob.id, currentPage.pageIndex, {
                        action: 'UPDATE_REGION',
                        regionId: activeRegion.id,
                        updates: { routedEngine: e.target.value as CleaningEngineId },
                      })
                    }
                    className="w-full h-9 px-2.5 py-0 leading-normal text-xs bg-slate-50 border border-slate-300 rounded font-medium text-slate-900"
                  >
                    <option value="SMART_FILL">Engine 1: Smart Fill</option>
                    <option value="LAMA_PATCH">Engine 2: LaMa / Patch Inpaint</option>
                    <option value="GEMINI_AI">Engine 3: Gemini AI Inpaint</option>
                  </select>
                </label>
              </div>

              {/* Bubble Mode Switch (Mode 1: Smart Whitening vs Mode 2: Text-Only Removal) */}
              <div className="space-y-1.5">
                <span className="text-xs text-slate-600">Bubble Cleaning Mode</span>
                <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-100 rounded">
                  {(
                    [
                      { id: 'WHITEN', label: 'Smart Whitening (#FFF)' },
                      { id: 'TEXT_ONLY', label: 'Text-Only Removal' },
                    ] as { id: BubbleCleanMode; label: string }[]
                  ).map((bm) => (
                    <button
                      key={bm.id}
                      type="button"
                      onClick={() =>
                        onModifyRegion(currentJob.id, currentPage.pageIndex, {
                          action: 'UPDATE_REGION',
                          regionId: activeRegion.id,
                          updates: { bubbleCleanMode: bm.id },
                        })
                      }
                      className={`px-2 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                        activeRegion.bubbleCleanMode === bm.id
                          ? 'bg-white text-slate-900 shadow-xs'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {bm.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Mask Dilation Padding & Feather Sliders */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <label className="block space-y-1 text-xs text-slate-600 font-mono tabular-nums">
                  <span>Mask Expansion: {activeRegion.paddingPx}px</span>
                  <input
                    type="range"
                    min={0}
                    max={14}
                    value={activeRegion.paddingPx}
                    onChange={(e) =>
                      onModifyRegion(currentJob.id, currentPage.pageIndex, {
                        action: 'UPDATE_REGION',
                        regionId: activeRegion.id,
                        updates: { paddingPx: Number(e.target.value) },
                      })
                    }
                    className="w-full accent-slate-900"
                  />
                </label>
                <label className="block space-y-1 text-xs text-slate-600 font-mono tabular-nums">
                  <span>Edge Feather: {activeRegion.featherPx}px</span>
                  <input
                    type="range"
                    min={0}
                    max={8}
                    value={activeRegion.featherPx}
                    onChange={(e) =>
                      onModifyRegion(currentJob.id, currentPage.pageIndex, {
                        action: 'UPDATE_REGION',
                        regionId: activeRegion.id,
                        updates: { featherPx: Number(e.target.value) },
                      })
                    }
                    className="w-full accent-slate-900"
                  />
                </label>
              </div>

              {/* Quality Assessment Diagnostics */}
              {activeRegion.quality && (
                <div className="p-3 bg-slate-50 border border-slate-200 rounded text-xs space-y-1 font-mono tabular-nums">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Quality Gate:</span>
                    <span className="font-semibold text-slate-900">
                      {activeRegion.quality.outcome}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Edge Energy Ratio:</span>
                    <span>{activeRegion.quality.edgeEnergyRatio.toFixed(2)}x (limit ≤2.0x)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Ghost Stroke Residual:</span>
                    <span>{activeRegion.quality.ghostStrokeResidual.toFixed(2)} luma</span>
                  </div>
                </div>
              )}

              {/* Reject / Restore / Gemini Inspect Buttons */}
              <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-200">
                {activeRegion.status !== 'REJECTED' ? (
                  <button
                    type="button"
                    onClick={() =>
                      onModifyRegion(currentJob.id, currentPage.pageIndex, {
                        action: 'REJECT_REGION',
                        regionId: activeRegion.id,
                      })
                    }
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 border border-rose-200 rounded hover:bg-rose-100 whitespace-nowrap shrink-0"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Reject & Restore Original
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() =>
                      onModifyRegion(currentJob.id, currentPage.pageIndex, {
                        action: 'RESTORE_REGION',
                        regionId: activeRegion.id,
                      })
                    }
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 rounded hover:bg-emerald-100 whitespace-nowrap shrink-0"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Re-Apply Cleaning
                  </button>
                )}

                <button
                  type="button"
                  onClick={async () => {
                    const res = await onInspectWithGemini(
                      currentJob.id,
                      currentPage.pageIndex,
                      activeRegion.id
                    );
                    setGeminiInsight({ regionId: activeRegion.id, ...res });
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-100 whitespace-nowrap shrink-0"
                >
                  <ScanSearch className="w-3.5 h-3.5" />
                  Analyze Crop
                </button>
              </div>

              {geminiInsight && geminiInsight.regionId === activeRegion.id && (
                <div className="p-3 bg-slate-50 border border-slate-200 rounded text-xs space-y-1">
                  <div className="font-semibold text-slate-900">
                    Crop Analysis ({geminiInsight.scriptLanguage})
                  </div>
                  <div className="text-slate-700 font-mono">{geminiInsight.detectedText}</div>
                  <div className="text-slate-500">{geminiInsight.reconstructionNotes}</div>
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
