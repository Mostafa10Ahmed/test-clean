export type RegionCategory = 'BUBBLE' | 'OT' | 'ST' | 'SFX';

export type BubbleCleanMode = 'WHITEN' | 'TEXT_ONLY';

export type CleaningEngineId = 'SMART_FILL' | 'LAMA_PATCH' | 'GEMINI_AI';

export type QualityOutcome =
  | 'ACCEPTED'
  | 'DECLINED_EDGE_ENERGY'
  | 'DECLINED_HISTOGRAM'
  | 'DECLINED_GHOST_TEXT'
  | 'UNMEASURED';

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SurroundStats {
  meanR: number;
  meanG: number;
  meanB: number;
  meanLuma: number;
  stdLuma: number;
  p5Luma: number;
  p95Luma: number;
  edgeEnergy: number;
  isUniform: boolean;
  isWhiteBubble: boolean;
  isDarkBubble: boolean;
  isPeriodicScreentone: boolean;
  isGradient: boolean;
  isOutlineDominated: boolean;
  gradSlopeX: number;
  gradSlopeY: number;
}

export interface QualityAssessment {
  outcome: QualityOutcome;
  interiorEdgeEnergy: number;
  surroundEdgeEnergy: number;
  edgeEnergyRatio: number;
  interiorOutsideFraction: number;
  ghostStrokeResidual: number;
  unchangedOutsideMaskVerified: boolean;
  fallbackUsed?: CleaningEngineId;
}

export interface MaskStrokeEdit {
  mode: 'add' | 'erase';
  x: number; // normalized 0..1 within region bbox
  y: number;
  radiusPx: number;
}

export interface DetectedRegion {
  id: string;
  pageIndex: number;
  category: RegionCategory;
  bbox: BoundingBox;
  bubbleBbox?: BoundingBox;
  bubbleId?: string;
  insideBubble: boolean;
  confidence: number;
  sourceScript: 'JA' | 'KO' | 'ZH' | 'EN' | 'MIXED' | 'SFX_STYLIZED';
  routedEngine: CleaningEngineId;
  appliedEngine?: CleaningEngineId;
  bubbleCleanMode: BubbleCleanMode;
  paddingPx: number;
  featherPx: number;
  fillColorHex: string;
  inkCoverageRatio: number;
  surroundStats: SurroundStats;
  status: 'DETECTED' | 'CLEANED' | 'HELD_FOR_REVIEW' | 'REJECTED';
  reviewStatus: 'AUTO_APPROVED' | 'NEEDS_REVIEW' | 'MANUALLY_EDITED' | 'RESTORED_ORIGINAL';
  quality?: QualityAssessment;
  manualEdits: MaskStrokeEdit[];
}

export interface PageProcessResult {
  pageIndex: number;
  fileName: string;
  width: number;
  height: number;
  format: string;
  fileSizeBytes: number;
  stripWindowCount: number;
  processingTimeMs: number;
  peakMemoryMb: number;
  originalUrl: string;
  cleanedUrl: string;
  regions: DetectedRegion[];
  status: 'PENDING' | 'DETECTED' | 'CLEANED' | 'NEEDS_REVIEW' | 'ERROR';
  untouchedPixelsBitIdentical: boolean;
  errorMessage?: string;
}

export interface CleaningSettings {
  enableBubble: boolean;
  enableOT: boolean;
  enableST: boolean;
  enableSFX: boolean;
  bubbleCleanMode: BubbleCleanMode;
  defaultPaddingPx: number;
  defaultFeatherPx: number;
  engineRouting: 'AUTO_LADDER' | 'FORCE_SMART_FILL' | 'FORCE_LAMA_PATCH' | 'ALLOW_CLOUD_AI';
  sourceLanguages: ('KO' | 'JA' | 'ZH' | 'EN')[];
  strictQualityGate: boolean;
  preserveColoredBubbles: boolean;
  maxStripWindowHeight: number;
  cloudGpuConsent: boolean;
}

export interface HistoryEntry {
  seq: number;
  timestamp: string;
  pageIndex: number;
  regionId?: string;
  label: string;
  previousRegionsSnapshot: DetectedRegion[];
}

export interface ChapterJob {
  id: string;
  title: string;
  sourceType: 'GOOGLE_DRIVE_CHAPTER' | 'ZIP_UPLOAD' | 'IMAGE_BATCH';
  driveFolderId?: string;
  status: 'IDLE' | 'DETECTING' | 'CLEANING' | 'COMPLETED' | 'CANCELLED' | 'FAILED';
  createdAt: string;
  completedAt?: string;
  totalPages: number;
  processedPages: number;
  totalRegions: number;
  cleanedRegions: number;
  reviewFlaggedRegions: number;
  totalProcessingTimeMs: number;
  avgTimePerPageMs: number;
  peakMemoryMb: number;
  settings: CleaningSettings;
  pages: PageProcessResult[];
  undoStack: HistoryEntry[];
  redoStack: HistoryEntry[];
  zipDownloadUrl?: string;
}

export interface BotMessage {
  id: string;
  sender: 'USER' | 'BEAR_BOT';
  command?: string;
  timestamp: string;
  text: string;
  chapterId?: string;
  pageIndex?: number;
  embed?: {
    title: string;
    status: string;
    fields: { label: string; value: string }[];
    actions?: { label: string; command: string; variant?: 'primary' | 'secondary' | 'danger' }[];
    previewOriginalUrl?: string;
    previewCleanedUrl?: string;
    zipUrl?: string;
  };
}
