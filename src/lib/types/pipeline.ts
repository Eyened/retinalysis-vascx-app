export type StageName = 'preprocess' | 'inference' | 'features';
export type WorkerStatus = 'idle' | 'loading' | 'ready' | 'running' | 'error';
export type OrtBackend = 'webgpu' | 'wasm';

export interface ProgressMessage {
  type: 'progress';
  stage: StageName | 'init';
  message: string;
  progress?: number;
  phase?: 'init' | 'run';
  substage?: string;
  modelLabel?: string;
  bytesLoaded?: number;
  bytesTotal?: number;
  downloadProgress?: number;
  backend?: OrtBackend;
  adapterName?: string;
}

export interface PipelineError {
  message: string;
  detail?: string;
  recoverable?: boolean;
}

export interface ImagePayload {
  bytes: Uint8Array;
  name: string;
  mimeType?: string;
}

export interface BoundsPayload {
  hw: [number, number];
  center: [number, number];
  radius: number;
  lines: Record<string, [number, number][]>;
  min_y?: number;
  max_y?: number;
  min_x?: number;
  max_x?: number;
}

export interface PreprocessResult {
  id: string;
  originalName: string;
  rgbPng: Uint8Array;
  boundsPng: Uint8Array;
  bounds: BoundsPayload;
  width: number;
  height: number;
  durationMs: number;
}

export interface InferenceInitResult {
  backend: OrtBackend;
  adapterName?: string;
}

export interface InferenceResult {
  id: string;
  avMask: Uint8Array;
  vesselsMask: Uint8Array;
  discMask: Uint8Array;
  width: number;
  height: number;
  avMaskPng?: Uint8Array;
  vesselsMaskPng?: Uint8Array;
  discMaskPng?: Uint8Array;
  overlayPng?: Uint8Array;
  fovea: [number, number];
  backend: OrtBackend;
  durationMs: number;
}

export interface BiomarkerRow {
  key: string;
  name: string;
  value: string | number | null;
}

export interface FeatureSetSummary {
  name: string;
  displayName: string;
  description: string;
  biomarkers: Array<{ key: string; name: string }>;
}

export interface FeatureExtractionResult {
  id: string;
  featureSet: string;
  biomarkers: BiomarkerRow[];
  durationMs: number;
}

export interface WorkerRequestMap {
  init: { type: 'init' };
  preprocessImage: { type: 'preprocessImage'; image: ImagePayload };
  runInference: { type: 'runInference'; id: string; rgbPng: Uint8Array; includePng?: boolean; includeOverlay?: boolean };
  extractFeatures: {
    type: 'extractFeatures';
    id: string;
    rgbPng: Uint8Array;
    avMask: Uint8Array;
    vesselsMask: Uint8Array;
    discMask: Uint8Array;
    width: number;
    height: number;
    fovea: [number, number];
    bounds: BoundsPayload;
    featureSet: string;
  };
  renderFeaturePlot: { type: 'renderFeaturePlot'; key: string };
  disposeCurrentRetina: { type: 'disposeCurrentRetina' };
  loadFeatureSets: { type: 'loadFeatureSets' };
}

export type WorkerRequest = WorkerRequestMap[keyof WorkerRequestMap] & { requestId: string };

export type WorkerResponse =
  | ProgressMessage
  | { type: 'ready'; requestId: string; payload?: unknown }
  | { type: 'result'; requestId: string; payload: unknown }
  | { type: 'error'; requestId: string; error: PipelineError };

export interface ModelManifest {
  version: number;
  models: Record<
    'av' | 'vessels' | 'disc' | 'fovea',
    {
      label: string;
      url: string;
      kind: 'segmentation' | 'heatmap';
      classes: number;
      config?: Partial<InferenceConfig>;
    }
  >;
  defaults: InferenceConfig;
  notes?: string[];
}

export interface InferenceConfig {
  inputSize: number;
  patchSize: [number, number];
  overlap: number;
  blend: 'constant' | 'gaussian';
  normalize: false | 'imagenet' | 'diffusion';
  ttaFlips: number[][];
}

export interface BatchRecord {
  id: string;
  name: string;
  relativePath: string;
  status: 'pending' | 'running' | 'done' | 'error';
  durationMs?: number;
  error?: string;
}
