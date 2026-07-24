<script lang="ts">
  import { onMount } from 'svelte';
  import {
    AlertTriangle,
    CheckCircle2,
    Download,
    Eye,
    FolderOpen,
    Github,
    ImagePlus,
    Info,
    Loader2,
    Play,
    Square,
    Trash2,
    UploadCloud,
    X
  } from 'lucide-svelte';
  import PreprocessWorker from '$lib/workers/preprocess.worker?worker';
  import InferenceWorker from '$lib/workers/inference.worker?worker';
  import FeatureWorker from '$lib/workers/feature.worker?worker';
  import { PipelineWorkerClient } from '$lib/workers/client';
  import type {
    BatchRecord,
    BiomarkerRow,
    FeatureExtractionResult,
    FeatureSetSummary,
    InferenceInitResult,
    InferenceResult,
    OrtBackend,
    PreprocessResult,
    ProgressMessage
  } from '$lib/types/pipeline';
  import { biomarkerRowsToCsv, bytesToArrayBuffer, bytesToObjectUrl, downloadBytes, downloadUrlAsPng, fileToBytes, isValidImageName, VALID_IMAGE_EXTENSIONS } from '$lib/utils/files';
  import { clearBatchCache, loadBatchCache, saveBatchMeta, saveBatchResult, sourceHashForSignature, sourceSignature, type BatchResultRecord } from '$lib/stores/batch';
  import ImageCard from '$lib/components/ImageCard.svelte';

  type Screen = 'single' | 'batch';
  type WorkerName = 'preprocess' | 'inference' | 'features';

  interface WorkerUiState {
    status: 'idle' | 'loading' | 'ready' | 'error';
    message: string;
    progress?: number;
  }

  interface RunProgressState {
    active: boolean;
    stage: string;
    substage?: string;
    progress?: number;
    fileName?: string;
  }

  interface ModelDownloadState {
    label: string;
    progress?: number;
    bytesLoaded?: number;
    bytesTotal?: number;
  }

  let activeScreen: Screen = 'single';
  let preprocessClient: PipelineWorkerClient;
  let inferenceClient: PipelineWorkerClient;
  let featureClient: PipelineWorkerClient;

  let workers: Record<WorkerName, WorkerUiState> = {
    preprocess: { status: 'idle', message: 'Waiting' },
    inference: { status: 'idle', message: 'Waiting' },
    features: { status: 'idle', message: 'Waiting' }
  };

  let featureSets: FeatureSetSummary[] = [];
  let defaultFeatureSet = 'Full V3';
  let singleFeatureSet = 'Full V3';
  let batchFeatureSet = 'Full V3';
  let uploadError = '';
  let processing = false;
  let batchRunning = false;
  let stopRequested = false;
  let stopping = false;
  let uploadedUrl = '';
  let uploadedName = '';
  let preprocessResult: PreprocessResult | undefined;
  let inferenceResult: InferenceResult | undefined;
  let biomarkers: BiomarkerRow[] = [];
  let boundsUrl = '';
  let rgbUrl = '';
  let overlayUrl = '';
  let modalImage: { title: string; url: string; filename: string; bytes?: Uint8Array } | undefined;
  let plotModal: { title: string; url: string } | undefined;
  let runProgress: RunProgressState = { active: false, stage: '' };
  let currentCsvDownloaded = true;

  let sourceFiles: File[] = [];
  let batchRecords: BatchRecord[] = [];
  let batchResults: BatchResultRecord[] = [];
  let batchDownloaded = true;
  let currentBatchName = '';
  let sourceName = '';
  let segmentationDirectory: FileSystemDirectoryHandle | undefined;
  let segmentationFolderName = '';
  let batchListOpen = false;
  let restoredBatchCache = false;
  let batchCachedFeatureSet = '';
  let inferenceBackend: OrtBackend | undefined;
  let inferenceAdapterName = '';
  let modelDownload: ModelDownloadState | undefined;
  let attributionHtml = '';

  $: backendSummaryText = inferenceBackend ? (inferenceAdapterName ? `${inferenceBackend.toUpperCase()} · ${inferenceAdapterName}` : inferenceBackend.toUpperCase()) : '';
  $: appReady = workers.preprocess.status === 'ready' && workers.features.status === 'ready';
  $: inferenceReady = workers.inference.status === 'ready';
  $: runActive = processing || batchRunning;
  $: pipelineReady = appReady && inferenceReady && !runActive && !stopping;
  $: canPrepareSingle = pipelineReady;
  $: canRunSingleVascx = pipelineReady && Boolean(preprocessResult && inferenceResult);
  $: canRunBatch = pipelineReady && sourceFiles.length > 0;
  $: batchResumeAvailable = restoredBatchCache && batchResults.length > 0 && !batchRunning && batchCachedFeatureSet === batchFeatureSet;
  $: singleFeatureSetInfo = featureSets.find((set) => set.name === singleFeatureSet);
  $: batchFeatureSetInfo = featureSets.find((set) => set.name === batchFeatureSet);
  $: canStopSingle = processing || (stopping && !batchRunning);
  $: canStopBatch = batchRunning || (stopping && !processing);

  onMount(() => {
    preprocessClient = new PipelineWorkerClient(new PreprocessWorker(), handleProgress);
    inferenceClient = new PipelineWorkerClient(new InferenceWorker(), handleProgress);
    featureClient = new PipelineWorkerClient(new FeatureWorker(), handleProgress);
    void initialiseWorkers().then(restoreBatchCache);
    void loadAttribution();

    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (runActive || stopping || (!batchDownloaded && batchResults.length > 0)) {
        event.preventDefault();
        event.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      window.removeEventListener('beforeunload', beforeUnload);
      preprocessClient?.terminate();
      inferenceClient?.terminate();
      featureClient?.terminate();
    };
  });

  async function initialiseWorkers() {
    setWorker('preprocess', { status: 'loading', message: 'Loading Pyodide preprocessing worker', progress: 0 });
    setWorker('inference', { status: 'loading', message: 'Loading AI models', progress: 0 });
    setWorker('features', { status: 'loading', message: 'Loading biomarker extraction worker', progress: 0 });

    const results = await Promise.allSettled([
      preprocessClient.request({ type: 'init' }),
      inferenceClient.request({ type: 'init' }),
      featureClient.request({ type: 'init' })
    ]);

    if (results[0].status === 'fulfilled') setWorker('preprocess', { status: 'ready', message: 'Preprocessing ready', progress: 1 });
    else setWorker('preprocess', { status: 'error', message: results[0].reason.message });

    if (results[1].status === 'fulfilled') {
      const payload = results[1].value as InferenceInitResult | undefined;
      inferenceBackend = payload?.backend;
      inferenceAdapterName = payload?.adapterName ?? '';
      defaultFeatureSet = payload?.defaultFeatureSet ?? defaultFeatureSet;
      console.info('[VascX] ONNX Runtime backend ready:', inferenceBackend ?? 'unknown', inferenceAdapterName);
      setWorker('inference', { status: 'ready', message: `AI models ready${backendSummaryText ? ` (${backendSummaryText})` : ''}`, progress: 1 });
    } else setWorker('inference', { status: 'error', message: results[1].reason.message });

    if (results[2].status === 'fulfilled') {
      setWorker('features', { status: 'ready', message: 'Biomarker extraction ready', progress: 1 });
      featureSets = (await featureClient.request({ type: 'loadFeatureSets' })) as FeatureSetSummary[];
      const resolvedDefault = resolveFeatureSetName(featureSets, defaultFeatureSet, singleFeatureSet) ?? singleFeatureSet;
      singleFeatureSet = resolveFeatureSetName(featureSets, singleFeatureSet, resolvedDefault) ?? resolvedDefault;
      batchFeatureSet = resolveFeatureSetName(featureSets, batchFeatureSet, resolvedDefault) ?? resolvedDefault;
    } else {
      setWorker('features', { status: 'error', message: results[2].reason.message });
    }
  }

  function handleProgress(message: ProgressMessage) {
    if (message.stage === 'inference' && message.backend) {
      inferenceBackend = message.backend;
      inferenceAdapterName = message.adapterName ?? '';
    }
    if (message.stage === 'preprocess' || message.stage === 'inference' || message.stage === 'features') {
      if (message.phase === 'init' || workers[message.stage].status !== 'ready') {
        setWorker(message.stage, { status: 'loading', message: message.message, progress: message.progress });
      }
      updateModelDownload(message);
      if (message.phase === 'run' || runProgress.active) {
        updateRunProgress(message);
      }
    }
  }

  function updateModelDownload(message: ProgressMessage) {
    if (!message.modelLabel) return;
    modelDownload = {
      label: message.modelLabel,
      progress: message.downloadProgress,
      bytesLoaded: message.bytesLoaded,
      bytesTotal: message.bytesTotal
    };
    if (message.downloadProgress === 1) {
      const label = message.modelLabel;
      setTimeout(() => {
        if (modelDownload?.label === label && modelDownload.progress === 1) modelDownload = undefined;
      }, 600);
    }
  }

  function updateRunProgress(message: ProgressMessage) {
    runProgress = {
      active: true,
      stage: stageLabel(message.stage),
      substage: message.substage ?? message.message,
      progress: message.progress,
      fileName: currentBatchName || uploadedName || runProgress.fileName
    };
  }

  function setWorker(name: WorkerName, patch: Partial<WorkerUiState>) {
    workers = { ...workers, [name]: { ...workers[name], ...patch } };
  }

  function stageLabel(stage: ProgressMessage['stage']) {
    if (stage === 'preprocess') return 'Preprocessing';
    if (stage === 'inference') return 'AI models';
    if (stage === 'features') return 'Biomarker extraction';
    return 'Initialisation';
  }

  function setRunProgress(stage: string, substage: string, progress?: number, fileName?: string) {
    runProgress = { active: true, stage, substage, progress, fileName: (fileName ?? currentBatchName) || uploadedName };
  }

  function clearRunProgress() {
    runProgress = { active: false, stage: '' };
  }

  function resolveFeatureSetName(sets: FeatureSetSummary[], preferred: string, fallback: string) {
    const candidates = [preferred, fallback].map(normalizeFeatureSetName).filter(Boolean);
    for (const candidate of candidates) {
      const match = sets.find((set) => normalizeFeatureSetName(set.name) === candidate || normalizeFeatureSetName(set.displayName) === candidate);
      if (match) return match.name;
    }
    return sets[0]?.name;
  }

  function normalizeFeatureSetName(value: string) {
    return value.trim().toLowerCase().replace(/[ _-]+/g, '');
  }

  function formatBytes(value?: number) {
    if (value === undefined) return '';
    if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
    if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
    return `${value} B`;
  }

  function workerDisplayName(name: string) {
    if (name === 'preprocess') return 'Preprocessing';
    if (name === 'inference') return 'AI models';
    if (name === 'features') return 'Biomarker extraction';
    return name;
  }

  async function loadAttribution() {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}attribution.md`);
      if (!response.ok) throw new Error(response.statusText);
      attributionHtml = markdownToHtml(await response.text());
    } catch {
      attributionHtml = markdownToHtml(DEFAULT_ATTRIBUTION_MD);
    }
  }

  function markdownToHtml(markdown: string) {
    return markdown
      .trim()
      .split(/\n{2,}/)
      .map((block) => `<p>${inlineMarkdown(block).replace(/\n/g, '<br>')}</p>`)
      .join('');
  }

  function inlineMarkdown(value: string) {
    return escapeHtml(value).replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  }

  function escapeHtml(value: string) {
    return value.replace(/[&<>"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[char] ?? char);
  }

  class PipelineStoppedError extends Error {
    constructor() {
      super('Pipeline stopped');
      this.name = 'PipelineStoppedError';
    }
  }

  function isPipelineStoppedError(error: unknown): boolean {
    return error instanceof PipelineStoppedError || (error instanceof Error && error.name === 'PipelineStoppedError');
  }

  function requestStop() {
    if (!runActive && !stopping) return;
    stopRequested = true;
    stopping = true;
    if (runProgress.active) {
      runProgress = { ...runProgress, substage: 'Stopping after the current step…' };
    }
  }

  function clearStopFlags() {
    stopRequested = false;
    stopping = false;
  }

  function assertNotStopped() {
    if (stopRequested) throw new PipelineStoppedError();
  }

  async function handleSingleFile(file: File) {
    if (!file || !isValidImageName(file.name)) {
      uploadError = `Unsupported image format. Use ${VALID_IMAGE_EXTENSIONS.join(', ')}.`;
      return;
    }
    uploadError = '';
    clearSingleOutputs();
    uploadedName = file.name;
    uploadedUrl = URL.createObjectURL(file);
    try {
      processing = true;
      clearStopFlags();
      const prepared = await prepareImageArtifacts(file, { keepArtifacts: true });
      assertNotStopped();
      preprocessResult = prepared.preprocess;
      inferenceResult = prepared.inference;
      boundsUrl = bytesToObjectUrl(prepared.preprocess.boundsPng);
      rgbUrl = bytesToObjectUrl(prepared.preprocess.rgbPng);
      overlayUrl = bytesToObjectUrl(prepared.inference.overlayPng!);
    } catch (error) {
      if (!isPipelineStoppedError(error)) {
        uploadError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      processing = false;
      clearStopFlags();
      clearRunProgress();
      markReadyAfterRun();
    }
  }

  async function runSingleVascx() {
    if (!preprocessResult || !inferenceResult) {
      uploadError = 'Drop a fundus image first, then run VascX.';
      return;
    }
    uploadError = '';
    try {
      processing = true;
      clearStopFlags();
      const features = await extractFeaturesFromArtifacts(preprocessResult, inferenceResult, {
        featureSet: singleFeatureSet,
        keepRetina: true,
        transferMasks: false
      });
      assertNotStopped();
      biomarkers = features.biomarkers;
      currentCsvDownloaded = false;
    } catch (error) {
      if (!isPipelineStoppedError(error)) {
        uploadError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      processing = false;
      clearStopFlags();
      clearRunProgress();
      markReadyAfterRun();
    }
  }

  type PrepareOptions = {
    keepArtifacts: boolean;
    segmentationDirectory?: FileSystemDirectoryHandle;
  };
  type PreparedArtifacts = { preprocess: PreprocessResult; inference: InferenceResult };
  type PipelineRunOptions = PrepareOptions & {
    featureSet: string;
    keepRetina: boolean;
  };
  type FullPipelineResult = PreparedArtifacts & { features: FeatureExtractionResult };
  type BatchPipelineResult = { id: string; features: FeatureExtractionResult };

  async function prepareImageArtifacts(file: File, options: PrepareOptions): Promise<PreparedArtifacts> {
    const fileName = file.webkitRelativePath || file.name;
    setRunProgress('Preprocessing', 'Preparing image', 0, fileName);
    await featureClient.request({ type: 'disposeCurrentRetina' });
    assertNotStopped();

    const bytes = await fileToBytes(file);
    assertNotStopped();
    setRunProgress('Preprocessing', 'Detecting fundus bounds', 0.1, fileName);
    const preprocess = (await preprocessClient.request(
      { type: 'preprocessImage', image: { bytes, name: file.name, mimeType: file.type } },
      transferList(bytes)
    )) as PreprocessResult;
    assertNotStopped();

    setRunProgress('AI models', 'Preparing AI model inputs', 0, fileName);
    const inference = (await inferenceClient.request({
      type: 'runInference',
      id: preprocess.id,
      rgbPng: preprocess.rgbPng,
      includePng: options.keepArtifacts || Boolean(options.segmentationDirectory),
      includeOverlay: options.keepArtifacts
    })) as InferenceResult;
    assertNotStopped();

    if (options.segmentationDirectory) {
      await writeSegmentationOutputs(options.segmentationDirectory, preprocess.id, inference, preprocess.bounds);
      assertNotStopped();
    }

    return { preprocess, inference };
  }

  async function extractFeaturesFromArtifacts(
    preprocess: PreprocessResult,
    inference: InferenceResult,
    options: { featureSet: string; keepRetina: boolean; transferMasks: boolean }
  ): Promise<FeatureExtractionResult> {
    setRunProgress('Biomarker extraction', 'Preparing biomarker analysis', 0, preprocess.id);
    await featureClient.request({ type: 'disposeCurrentRetina' });
    assertNotStopped();
    const featureRequest = {
      type: 'extractFeatures' as const,
      id: preprocess.id,
      rgbPng: preprocess.rgbPng,
      avMask: inference.avMask,
      vesselsMask: inference.vesselsMask,
      discMask: inference.discMask,
      width: inference.width,
      height: inference.height,
      fovea: inference.fovea,
      bounds: preprocess.bounds,
      featureSet: options.featureSet
    };
    try {
      const features = (await featureClient.request(
        featureRequest,
        options.transferMasks ? transferList(preprocess.rgbPng, inference.avMask, inference.vesselsMask, inference.discMask) : []
      )) as FeatureExtractionResult;
      assertNotStopped();
      return features;
    } finally {
      if (!options.keepRetina) {
        await featureClient.request({ type: 'disposeCurrentRetina' });
      }
    }
  }

  function runPipelineForFile(file: File, options: PipelineRunOptions & { keepArtifacts: true }): Promise<FullPipelineResult>;
  function runPipelineForFile(file: File, options: PipelineRunOptions & { keepArtifacts: false }): Promise<BatchPipelineResult>;
  async function runPipelineForFile(file: File, options: PipelineRunOptions): Promise<FullPipelineResult | BatchPipelineResult> {
    const prepared = await prepareImageArtifacts(file, options);
    const features = await extractFeaturesFromArtifacts(prepared.preprocess, prepared.inference, {
      featureSet: options.featureSet,
      keepRetina: options.keepRetina,
      transferMasks: !options.keepArtifacts
    });
    if (options.keepArtifacts) {
      return { ...prepared, features };
    }
    return { id: prepared.preprocess.id, features };
  }

  function transferList(...arrays: Uint8Array[]): Transferable[] {
    return arrays.map((array) => array.buffer).filter((buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer);
  }

  function markReadyAfterRun() {
    if (workers.preprocess.status !== 'error') setWorker('preprocess', { status: 'ready', message: 'Preprocessing ready', progress: 1 });
    if (workers.inference.status !== 'error') setWorker('inference', { status: 'ready', message: `AI models ready${backendSummaryText ? ` (${backendSummaryText})` : ''}`, progress: 1 });
    if (workers.features.status !== 'error') setWorker('features', { status: 'ready', message: 'Biomarker extraction ready', progress: 1 });
  }

  async function renderPlot(row: BiomarkerRow) {
    plotModal = { title: row.name, url: '' };
    const url = (await featureClient.request({ type: 'renderFeaturePlot', key: row.key })) as string;
    plotModal = { title: row.name, url };
    markReadyAfterRun();
  }

  function clearSingleOutputs() {
    for (const url of [boundsUrl, rgbUrl, overlayUrl]) {
      if (url) URL.revokeObjectURL(url);
    }
    preprocessResult = undefined;
    inferenceResult = undefined;
    biomarkers = [];
    boundsUrl = '';
    rgbUrl = '';
    overlayUrl = '';
    modalImage = undefined;
    plotModal = undefined;
  }

  async function onDrop(event: DragEvent) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) await handleSingleFile(file);
  }

  async function onFileInput(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (file) await handleSingleFile(file);
    input.value = '';
  }

  async function chooseSourceFolder() {
    const picker = window.showDirectoryPicker;
    if (picker) {
      const directory = await picker({ mode: 'read' });
      sourceName = directory.name;
      sourceFiles = await filesFromDirectory(directory);
      await applySourceFiles(sourceFiles);
      return;
    }
    document.getElementById('batch-file-input')?.click();
  }

  async function chooseSegmentationFolder() {
    const picker = window.showDirectoryPicker;
    if (!picker) {
      alert('Writing segmentations requires a browser with the File System Access API.');
      return;
    }
    segmentationDirectory = await picker({ mode: 'readwrite' });
    segmentationFolderName = segmentationDirectory.name;
  }

  async function filesFromDirectory(directory: FileSystemDirectoryHandle, prefix = ''): Promise<File[]> {
    const files: File[] = [];
    for await (const [name, handle] of directory.entries()) {
      if (handle.kind === 'file') {
        const file = await handle.getFile();
        Object.defineProperty(file, 'webkitRelativePath', { value: `${prefix}${name}` });
        if (isValidImageName(file.name)) files.push(file);
      } else {
        files.push(...(await filesFromDirectory(handle, `${prefix}${name}/`)));
      }
    }
    return files;
  }

  async function onBatchFileInput(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    sourceFiles = Array.from(input.files ?? []).filter((file) => isValidImageName(file.name));
    sourceName = input.files?.[0]?.webkitRelativePath?.split('/')[0] || 'Selected files';
    await applySourceFiles(sourceFiles);
    input.value = '';
  }

  async function applySourceFiles(files: File[]) {
    const signature = sourceSignature(files);
    const cache = await loadBatchCache();
    if (cache?.sourceSignature === signature) {
      batchRecords = cache.records;
      batchResults = cache.results;
      batchDownloaded = cache.downloaded;
      batchCachedFeatureSet = cache.featureSet ?? '';
      if (cache.featureSet) {
        batchFeatureSet = resolveFeatureSetName(featureSets, cache.featureSet, batchFeatureSet) ?? cache.featureSet;
      }
      restoredBatchCache = cache.results.length > 0 || cache.records.some((record) => record.status === 'done');
      return;
    }

    if (cache) {
      await clearBatchCache();
    }

    batchRecords = files.map((file) => ({
      id: file.name.replace(/\.[^.]+$/, ''),
      name: file.name,
      relativePath: file.webkitRelativePath || file.name,
      status: 'pending'
    }));
    batchResults = [];
    batchDownloaded = true;
    batchCachedFeatureSet = '';
    restoredBatchCache = false;
  }

  async function restoreBatchCache() {
    const cache = await loadBatchCache();
    if (!cache) return;
    batchRecords = cache.records;
    batchResults = cache.results;
    batchDownloaded = cache.downloaded;
    batchCachedFeatureSet = cache.featureSet ?? '';
    if (cache.featureSet) {
      batchFeatureSet = resolveFeatureSetName(featureSets, cache.featureSet, batchFeatureSet) ?? cache.featureSet;
    }
    restoredBatchCache = cache.results.length > 0 || cache.records.some((record) => record.status === 'done');
  }

  async function resetBatchOutputsForFeatureSetChange() {
    const signature = sourceSignature(sourceFiles);
    const sourceHash = sourceHashForSignature(signature);
    await clearBatchCache();
    batchRecords = sourceFiles.map((file) => ({
      id: file.name.replace(/\.[^.]+$/, ''),
      name: file.name,
      relativePath: file.webkitRelativePath || file.name,
      status: 'pending'
    }));
    batchResults = [];
    batchDownloaded = true;
    batchCachedFeatureSet = '';
    restoredBatchCache = false;
    await saveBatchMeta({
      sourceSignature: signature,
      sourceHash,
      featureSet: batchFeatureSet,
      createdAt: new Date().toISOString(),
      downloaded: true,
      records: batchRecords
    });
  }

  async function runBatch() {
    if (!sourceFiles.length) return;
    const featureSetChanged = batchResults.length > 0 && batchCachedFeatureSet !== batchFeatureSet;
    if (featureSetChanged) {
      if (!confirm('The selected biomarker set differs from the cached batch results. Previous outputs will be cleared and all images will be reprocessed. Continue?')) {
        return;
      }
      await resetBatchOutputsForFeatureSetChange();
    } else if (!confirm(`Process ${sourceFiles.length} images one at a time with ${batchFeatureSet}?`)) {
      return;
    }

    batchRunning = true;
    clearStopFlags();
    batchDownloaded = false;
    const signature = sourceSignature(sourceFiles);
    const sourceHash = sourceHashForSignature(signature);
    const createdAt = new Date().toISOString();
    const recordIndexByPath = new Map(batchRecords.map((record, index) => [record.relativePath, index]));

    await saveBatchMeta({
      sourceSignature: signature,
      sourceHash,
      featureSet: batchFeatureSet,
      createdAt,
      downloaded: batchDownloaded,
      records: batchRecords
    });

    for (const file of sourceFiles) {
      if (stopRequested) break;
      const relativePath = file.webkitRelativePath || file.name;
      const index = recordIndexByPath.get(relativePath);
      if (index === undefined || batchRecords[index].status === 'done') continue;
      currentBatchName = relativePath;
      batchRecords[index] = { ...batchRecords[index], status: 'running', error: undefined };
      batchRecords = batchRecords;
      const started = performance.now();
      try {
        const result = await runPipelineForFile(file, {
          keepRetina: false,
          keepArtifacts: false,
          featureSet: batchFeatureSet,
          segmentationDirectory
        });
        const output: BatchResultRecord = { id: result.id, relativePath, biomarkers: result.features.biomarkers };
        batchRecords[index] = { ...batchRecords[index], status: 'done', durationMs: performance.now() - started };
        batchResults.push(output);
        batchResults = batchResults;
        await saveBatchResult(sourceHash, output);
      } catch (error) {
        if (isPipelineStoppedError(error)) {
          batchRecords[index] = { ...batchRecords[index], status: 'pending', error: undefined };
          batchRecords = batchRecords;
          break;
        }
        batchRecords[index] = { ...batchRecords[index], status: 'error', durationMs: performance.now() - started, error: error instanceof Error ? error.message : String(error) };
      }
      batchRecords = batchRecords;
      await saveBatchMeta({
        sourceSignature: signature,
        sourceHash,
        featureSet: batchFeatureSet,
        createdAt,
        downloaded: batchDownloaded,
        records: batchRecords
      });
      if (stopRequested) break;
    }

    currentBatchName = '';
    batchRunning = false;
    batchCachedFeatureSet = batchFeatureSet;
    restoredBatchCache = batchResults.length > 0;
    clearStopFlags();
    clearRunProgress();
    await featureClient.request({ type: 'disposeCurrentRetina' });
    await saveBatchMeta({
      sourceSignature: signature,
      sourceHash,
      featureSet: batchFeatureSet,
      createdAt,
      downloaded: batchDownloaded,
      records: batchRecords
    });
    markReadyAfterRun();
  }

  async function writeSegmentationOutputs(directory: FileSystemDirectoryHandle, id: string, inference: InferenceResult, bounds: unknown) {
    if (!inference.avMaskPng || !inference.vesselsMaskPng || !inference.discMaskPng) {
      throw new Error('Segmentation PNG outputs were not rendered for this inference run.');
    }
    await writeFile(directory, `${id}_av.png`, inference.avMaskPng);
    await writeFile(directory, `${id}_vessels.png`, inference.vesselsMaskPng);
    await writeFile(directory, `${id}_disc.png`, inference.discMaskPng);
    await writeFile(directory, `${id}_fovea.json`, new TextEncoder().encode(JSON.stringify({ fovea: inference.fovea, bounds }, null, 2)), 'application/json');
  }

  async function writeFile(directory: FileSystemDirectoryHandle, name: string, bytes: Uint8Array, type = 'image/png') {
    const handle = await directory.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    await writable.write(new Blob([bytesToArrayBuffer(bytes)], { type }));
    await writable.close();
  }

  async function downloadBatchCsv() {
    const csv = biomarkerRowsToCsv(batchResults);
    downloadBytes(new TextEncoder().encode(csv), 'vascx-biomarkers.csv', 'text/csv');
    batchDownloaded = true;
    const cache = await loadBatchCache();
    if (cache) {
      const { results: _results, ...meta } = cache;
      await saveBatchMeta({ ...meta, downloaded: true });
    }
  }

  async function clearBatchResults() {
    await clearBatchCache();
    batchRecords = sourceFiles.map((file) => ({ id: file.name.replace(/\.[^.]+$/, ''), name: file.name, relativePath: file.webkitRelativePath || file.name, status: 'pending' }));
    batchResults = [];
    batchDownloaded = true;
    batchCachedFeatureSet = '';
    restoredBatchCache = false;
  }

  function downloadSingleCsv() {
    if (!preprocessResult) return;
    const csv = biomarkerRowsToCsv([{ id: preprocessResult.id, biomarkers }]);
    downloadBytes(new TextEncoder().encode(csv), `${preprocessResult.id}_biomarkers.csv`, 'text/csv');
    currentCsvDownloaded = true;
  }


  function closeImageModalFromBackdrop(event: MouseEvent) {
    if (event.currentTarget === event.target) modalImage = undefined;
  }

  function closePlotModalFromBackdrop(event: MouseEvent) {
    if (event.currentTarget === event.target) plotModal = undefined;
  }

  function pngFilename(name: string): string {
    const base = name.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_').replace(/^_|_$/g, '') || 'image';
    return `${base}.png`;
  }

  async function downloadModalImage() {
    if (!modalImage) return;
    const filename = pngFilename(modalImage.filename || modalImage.title);
    if (modalImage.bytes) {
      downloadBytes(modalImage.bytes, filename);
      return;
    }
    await downloadUrlAsPng(modalImage.url, filename);
  }

  async function downloadPlotImage() {
    if (!plotModal?.url) return;
    await downloadUrlAsPng(plotModal.url, pngFilename(plotModal.title));
  }

  async function switchScreen(screen: Screen) {
    if (screen === activeScreen) return;
    if (runActive || stopping) {
      alert('Wait for the current run to finish or stop before switching screens.');
      return;
    }
    if (activeScreen === 'batch' && !batchDownloaded && batchResults.length > 0) {
      if (!confirm('Biomarker results may be lost if you leave before downloading the CSV. Continue?')) return;
    }
    activeScreen = screen;
    plotModal = undefined;
    await featureClient.request({ type: 'disposeCurrentRetina' });
    markReadyAfterRun();
  }

  const DEFAULT_ATTRIBUTION_MD = `VascX was developed at the Eyened Reading Center (Rotterdam) as open source AI / software to support oculomics and ophthalmic clinical research. Our Python pipeline is available at [eyened/retinalysis-vascx](https://github.com/eyened/retinalysis-vascx)

If VascX is helpful to your research, please consider citing our manuscripts:
[retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers](https://arxiv.org/abs/2602.08580)
[VascX Models: Model Ensembles for Retinal Vascular Analysis from Color Fundus Images](https://arxiv.org/abs/2409.16016)`;
</script>

<svelte:head>
  <title>Retinalysis VascX</title>
</svelte:head>

<div class="app-shell">
  <header class="topbar">
    <div>
      <p class="eyebrow">Retinalysis</p>
      <h1>VascX Browser</h1>
    </div>
    <div class="topbar-actions">
      {#each Object.entries(workers) as [name, worker]}
        <span class:ready={worker.status === 'ready'} class:error={worker.status === 'error'} class="status-pill">
          {#if worker.status === 'loading'}<Loader2 size={14} class="spin" />{:else if worker.status === 'ready'}<CheckCircle2 size={14} />{:else if worker.status === 'error'}<AlertTriangle size={14} />{/if}
          {workerDisplayName(name)}: {worker.status}
        </span>
      {/each}
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-nav">
        <button class:active={activeScreen === 'single'} disabled={runActive || stopping} on:click={() => switchScreen('single')}><ImagePlus size={18} /> Single image</button>
        <button class:active={activeScreen === 'batch'} disabled={runActive || stopping} on:click={() => switchScreen('batch')}><FolderOpen size={18} /> Batch inference</button>
      </div>
      <a class="sidebar-github" href="https://github.com/eyened/retinalysis-vascx" target="_blank" rel="noreferrer"><Github size={18} /> eyened/retinalysis-vascx</a>
    </aside>

    <main class="content">
      <div class="alert info-notice">
        <Info size={18} />
        <p>This app extracts retinal vascular biomarkers from color fundus images using the <a href="https://github.com/eyened/retinalysis-vascx" target="_blank">VascX algorithm</a>. Images and biomarkers are not uploaded to the internet. Instead, algorithms and models are downloaded to your browser. <b>Please wait for the pipeline to be ready and then follow the steps below.</b> For more information please contact <a href="mailto:j.vargasquiros@erasmusmc.nl">j.vargasquiros@erasmusmc.nl</a></p>
      </div>

      <section class:compact={appReady && inferenceReady} class="panel setup-panel">
        <div>
          <p class="eyebrow">Initialisation</p>
          <h2>{appReady && inferenceReady ? 'Pipeline ready' : 'Loading browser pipeline'}</h2>
          <div class="backend-summary">
            <span>ONNX backend</span>
            <strong class:warn={inferenceBackend === 'wasm'}>{backendSummaryText || 'Detecting'}</strong>
          </div>
        </div>
        <div class="worker-list">
          {#each Object.entries(workers) as [name, worker]}
            <div class="worker-row">
              <span>{workerDisplayName(name)}</span>
              <progress value={worker.progress ?? 0} max="1"></progress>
              <small>{worker.message}</small>
              {#if name === 'inference' && modelDownload}
                <div class="download-progress">
                  <div>
                    <span>Downloading {modelDownload.label}</span>
                    <small>{formatBytes(modelDownload.bytesLoaded)}{modelDownload.bytesTotal ? ` / ${formatBytes(modelDownload.bytesTotal)}` : ''}</small>
                  </div>
                  <progress value={modelDownload.progress ?? 0} max="1"></progress>
                </div>
              {/if}
            </div>
          {/each}
        </div>
        {#if inferenceBackend === 'wasm'}
          <div class="alert gpu-warning"><AlertTriangle size={18} /> No useable GPU acceleration was detected. The AI models will run on the CPU. This may make the pipeline slow. We recommend running on a computer with a dedicated GPU or Apple M chip. If your machine does have a GPU, make sure that you are using an up-to-date browser (Chrome or Edge are recommended) and that the system's GPU drivers are up to date.</div>
        {/if}
      </section>

      {#if activeScreen === 'single'}
        <section class="screen single-screen">
          <section class="step-section">
            <h2 class="step-title">1. Drop a fundus image here</h2>
            <div class="upload-zone" class:disabled={!canPrepareSingle} role="region" aria-label="Image upload drop zone" on:drop={onDrop} on:dragover|preventDefault>
              <UploadCloud size={34} />
              <strong>Drop a fundus image here</strong>
              <span>{VALID_IMAGE_EXTENSIONS.join(', ')}</span>
              <label class="button primary">
                <input type="file" accept="image/*" on:change={onFileInput} disabled={!canPrepareSingle} />
                Select image
              </label>
              <small class="privacy-note">The image will not be uploaded to the internet.</small>
            </div>
            {#if uploadError}
              <div class="alert"><AlertTriangle size={18} /> {uploadError} Please try another image or report the error with the browser console details.</div>
            {/if}
            {#if processing && runProgress.active && runProgress.stage !== 'Biomarker extraction'}
              <div class="progress-card run-progress-card">
                <Loader2 class="spin" />
                <div class="run-progress-copy">
                  <strong>{runProgress.stage}</strong>
                  {#if runProgress.fileName}<span>{runProgress.fileName}</span>{/if}
                  {#if runProgress.substage}<span class="substage"><Loader2 class="spin" size={14} /> {runProgress.substage}</span>{/if}
                  {#if runProgress.progress !== undefined}<progress value={runProgress.progress} max="1"></progress>{/if}
                </div>
              </div>
            {/if}
            <div class="image-grid">
              <ImageCard title="Uploaded image" url={uploadedUrl} filename={uploadedName || 'uploaded.png'} onOpen={() => (modalImage = { title: 'Uploaded image', url: uploadedUrl, filename: uploadedName || 'uploaded.png' })} />
              <ImageCard title="Bounds" url={boundsUrl} filename={`${preprocessResult?.id ?? 'image'}_bounds.png`} bytes={preprocessResult?.boundsPng} onOpen={() => (modalImage = { title: 'Bounds', url: boundsUrl, filename: `${preprocessResult?.id ?? 'image'}_bounds.png`, bytes: preprocessResult?.boundsPng })} />
              <ImageCard title="AI model overlay" url={overlayUrl} filename={`${preprocessResult?.id ?? 'image'}_overlay.png`} bytes={inferenceResult?.overlayPng} onOpen={() => (modalImage = { title: 'AI model overlay', url: overlayUrl, filename: `${preprocessResult?.id ?? 'image'}_overlay.png`, bytes: inferenceResult?.overlayPng })} />
            </div>
          </section>

          <section class="step-section">
            <h2 class="step-title">2. Select a biomarker set and run VascX</h2>
            <div class="step-body panel feature-set-panel">
              <div class="feature-set-controls">
                <label class="field-label">Biomarker set
                  <select bind:value={singleFeatureSet} disabled={processing || stopping}>
                    {#each featureSets as set}
                      <option value={set.name}>{set.displayName}</option>
                    {/each}
                  </select>
                </label>
                {#if canStopSingle}
                  <button class="button danger" disabled={stopping} on:click={requestStop}>
                    {#if stopping}<Loader2 class="spin" size={16} /> Stopping{:else}<Square size={16} /> Stop{/if}
                  </button>
                {:else}
                  <button class="button primary" disabled={!canRunSingleVascx} on:click={runSingleVascx}><Play size={16} /> Run VascX</button>
                {/if}
              </div>
              {#if singleFeatureSetInfo?.description}
                <p class="feature-set-description">{singleFeatureSetInfo.description}</p>
              {/if}
              {#if processing && runProgress.active && runProgress.stage === 'Biomarker extraction'}
                <div class="progress-card run-progress-card">
                  <Loader2 class="spin" />
                  <div class="run-progress-copy">
                    <strong>{runProgress.stage}</strong>
                    {#if runProgress.fileName}<span>{runProgress.fileName}</span>{/if}
                    {#if runProgress.substage}<span class="substage"><Loader2 class="spin" size={14} /> {runProgress.substage}</span>{/if}
                    {#if runProgress.progress !== undefined}<progress value={runProgress.progress} max="1"></progress>{/if}
                  </div>
                </div>
              {/if}
            </div>
          </section>

          <section class="step-section">
            <h2 class="step-title">3. Output Biomarkers</h2>
            <div class="panel biomarkers-panel">
              <div class="panel-header">
                <div>
                  <p>{biomarkers.length ? `${biomarkers.length} biomarkers from ${singleFeatureSet}` : 'Select a biomarker set and run VascX to calculate biomarkers.'}</p>
                </div>
                <button class="button" disabled={!biomarkers.length} on:click={downloadSingleCsv}><Download size={16} /> CSV</button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Name</th><th>Value</th><th>Plot</th></tr></thead>
                  <tbody>
                    {#each biomarkers as row}
                      <tr>
                        <td>{row.name}</td>
                        <td>{row.value ?? 'N/A'}</td>
                        <td><button class="icon-button" on:click={() => renderPlot(row)} title="Open biomarker visualisation"><Eye size={16} /></button></td>
                      </tr>
                    {:else}
                      <tr><td colspan="3" class="empty">No biomarkers yet.</td></tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <section class="panel about-panel">
            <h3>About VascX</h3>
            <div class="about-copy">{@html attributionHtml}</div>
          </section>
        </section>
      {:else}
        <section class="screen batch-screen">
          <section class="step-section">
            <h2 class="step-title">1. Select input and output folders</h2>
            <div class="folder-row">
              <section class="folder-box">
                <div class="folder-title"><FolderOpen size={22} /><h3>Source folder</h3></div>
                <p>Deep search for {VALID_IMAGE_EXTENSIONS.join(', ')}.</p>
                <button class="button primary" on:click={chooseSourceFolder}><FolderOpen size={16} /> Select source</button>
                <small class="privacy-note">Images will not be uploaded to the internet.</small>
                <input id="batch-file-input" class="hidden" type="file" multiple webkitdirectory on:change={onBatchFileInput} />
                {#if sourceName}<small>Currently selected folder: {sourceName}</small>{/if}
              </section>
              <section class="folder-box">
                <div class="folder-title"><FolderOpen size={22} /><h3>Segmentations folder</h3></div>
                <p>Optional empty folder. If selected, the app will write intermediate segmentation masks, fovea, and bounds files to this folder. Select only if these files need to be exported.</p>
                <button class="button" on:click={chooseSegmentationFolder}><FolderOpen size={16} /> Select output</button>
                {#if segmentationFolderName}<small>{segmentationFolderName}</small>{/if}
              </section>
            </div>
            {#if batchRecords.length}
              <div class="panel batch-run-panel">
                <div class="panel-header">
                  <div>
                    <h3>{batchRecords.length} images found</h3>
                    <p>{batchRecords.slice(0, 3).map((record) => record.relativePath).join(', ')}</p>
                  </div>
                </div>
                <details bind:open={batchListOpen}>
                  <summary>Complete image list</summary>
                  {#if batchListOpen}
                    <div class="table-wrap compact">
                      <table>
                        <thead><tr><th>Image</th><th>Status</th><th>Time</th></tr></thead>
                        <tbody>
                          {#each batchRecords as record}
                            <tr>
                              <td>{record.relativePath}</td>
                              <td>{record.status}{record.error ? `: ${record.error}` : ''}</td>
                              <td>{record.durationMs ? `${(record.durationMs / 1000).toFixed(1)}s` : ''}</td>
                            </tr>
                          {/each}
                        </tbody>
                      </table>
                    </div>
                  {/if}
                </details>
              </div>
            {/if}
          </section>

          <section class="step-section">
            <h2 class="step-title">2. Select a biomarker set and run VascX</h2>
            <div class="step-body panel feature-set-panel">
              <div class="feature-set-controls">
                <label class="field-label">Biomarker set
                  <select bind:value={batchFeatureSet} disabled={batchRunning || stopping}>
                    {#each featureSets as set}
                      <option value={set.name}>{set.displayName}</option>
                    {/each}
                  </select>
                </label>
                {#if canStopBatch}
                  <button class="button danger" disabled={stopping} on:click={requestStop}>
                    {#if stopping}<Loader2 class="spin" size={16} /> Stopping{:else}<Square size={16} /> Stop{/if}
                  </button>
                {:else}
                  <button class="button primary" disabled={!canRunBatch} on:click={runBatch}><Play size={16} /> Run VascX</button>
                {/if}
              </div>
              {#if batchFeatureSetInfo?.description}
                <p class="feature-set-description">{batchFeatureSetInfo.description}</p>
              {/if}
              {#if batchCachedFeatureSet && batchCachedFeatureSet !== batchFeatureSet && batchResults.length}
                <div class="alert resume-alert"><AlertTriangle size={18} /> Cached batch results were computed with <code>{batchCachedFeatureSet}</code>. Running VascX with <code>{batchFeatureSet}</code> will clear those outputs and reprocess all images.</div>
              {/if}
            </div>
          </section>

          <section class="step-section">
            <h2 class="step-title">3. Output Biomarkers</h2>
            {#if batchRecords.length}
              <div class="panel batch-run-panel">
                {#if batchResumeAvailable}
                  <div class="alert resume-alert"><AlertTriangle size={18} /> Partial results from a previous run with <code>{batchCachedFeatureSet}</code> are available. Download CSV saves the current cached outputs. Re-select the same source folder to continue only the remaining images. Clear results deletes the cached outputs and starts this folder from scratch.</div>
                {/if}
                <div class="panel-header">
                  <div>
                    <h3>{batchResults.length ? `${batchResults.length} / ${batchRecords.length} completed` : 'No biomarkers yet'}</h3>
                    <p>{batchCachedFeatureSet ? `Biomarker set: ${batchCachedFeatureSet}` : 'Run VascX to extract biomarkers for the selected images.'}</p>
                  </div>
                  <div class="button-row">
                    <button class="button" disabled={!batchResults.length} on:click={downloadBatchCsv}><Download size={16} /> CSV</button>
                    <button class="button danger" on:click={clearBatchResults}><Trash2 size={16} /> Clear results</button>
                  </div>
                </div>
                <progress value={batchRecords.filter((record) => record.status === 'done').length} max={batchRecords.length}></progress>
                {#if currentBatchName}<p class="current-file"><Loader2 class="spin" size={16} /> {currentBatchName}</p>{/if}
                {#if batchRunning && runProgress.active}
                  <div class="progress-card run-progress-card batch-current-progress">
                    <Loader2 class="spin" />
                    <div class="run-progress-copy">
                      <strong>{runProgress.stage}</strong>
                      {#if runProgress.fileName}<span>{runProgress.fileName}</span>{/if}
                      {#if runProgress.substage}<span class="substage"><Loader2 class="spin" size={14} /> {runProgress.substage}</span>{/if}
                      {#if runProgress.progress !== undefined}<progress value={runProgress.progress} max="1"></progress>{/if}
                    </div>
                  </div>
                {/if}
              </div>
            {:else}
              <div class="panel step-placeholder">
                <p>Select a source folder to list images and run batch biomarker extraction.</p>
              </div>
            {/if}
          </section>

          <section class="panel about-panel">
            <h3>About VascX</h3>
            <div class="about-copy">{@html attributionHtml}</div>
          </section>
        </section>
      {/if}
    </main>
  </div>
</div>

{#if modalImage}
  <div class="modal-backdrop" role="presentation" on:click={closeImageModalFromBackdrop}>
    <div class="modal" role="dialog" aria-modal="true" aria-label={modalImage.title}>
      <button class="icon-button close" on:click={() => (modalImage = undefined)}><X size={18} /></button>
      <h3>{modalImage.title}</h3>
      <div class="modal-image-wrap">
        <img src={modalImage.url} alt={modalImage.title} />
        <button class="icon-button modal-download" title="Download PNG" on:click={downloadModalImage}><Download size={18} /></button>
      </div>
    </div>
  </div>
{/if}

{#if plotModal}
  <div class="modal-backdrop" role="presentation" on:click={closePlotModalFromBackdrop}>
    <div class="modal" role="dialog" aria-modal="true" aria-label={plotModal.title}>
      <button class="icon-button close" on:click={() => (plotModal = undefined)}><X size={18} /></button>
      <h3>{plotModal.title}</h3>
      {#if plotModal.url}
        <div class="modal-image-wrap">
          <img class="plot-image" src={plotModal.url} alt={plotModal.title} />
          <button class="icon-button modal-download" title="Download PNG" on:click={downloadPlotImage}><Download size={18} /></button>
        </div>
      {:else}
        <div class="progress-card"><Loader2 class="spin" /> Rendering plot</div>
      {/if}
    </div>
  </div>
{/if}
