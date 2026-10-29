import { writable } from 'svelte/store';
import type { BiomarkerRow, FeatureSetSummary, InferenceResult, PreprocessResult, StageName, WorkerStatus } from '$lib/types/pipeline';

export interface StageState {
  status: WorkerStatus;
  message: string;
  progress?: number;
  durationMs?: number;
  error?: string;
}

export interface SingleImageState {
  activeStage?: StageName;
  uploadedUrl?: string;
  uploadedName?: string;
  preprocess?: PreprocessResult;
  inference?: InferenceResult;
  biomarkers: BiomarkerRow[];
  featureSets: FeatureSetSummary[];
  selectedFeatureSet: string;
  stage3Ready: boolean;
  plotUrl?: string;
}

export const stages = writable<Record<StageName | 'init', StageState>>({
  init: { status: 'idle', message: 'Waiting to initialise' },
  preprocess: { status: 'idle', message: 'Waiting for image' },
  inference: { status: 'idle', message: 'Waiting for preprocessing' },
  features: { status: 'idle', message: 'Waiting for inference' }
});

export const singleImage = writable<SingleImageState>({
  biomarkers: [],
  featureSets: [],
  selectedFeatureSet: '',
  stage3Ready: false
});

export function setStage(stage: StageName | 'init', patch: Partial<StageState>) {
  stages.update((state) => ({
    ...state,
    [stage]: { ...state[stage], ...patch }
  }));
}
