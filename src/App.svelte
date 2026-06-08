<script lang="ts">
  import { onMount } from 'svelte';
  import {
    AlertTriangle,
    CheckCircle2,
    Download,
    Eye,
    FolderOpen,
    ImagePlus,
    Loader2,
    Play,
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
    InferenceResult,
    OrtBackend,
    PreprocessResult,
    ProgressMessage
  } from '$lib/types/pipeline';
  import { biomarkerRowsToCsv, bytesToArrayBuffer, bytesToObjectUrl, downloadBytes, fileToBytes, isValidImageName, VALID_IMAGE_EXTENSIONS } from '$lib/utils/files';
  import { clearBatchCache, loadBatchCache, saveBatchMeta, saveBatchResult, sourceHashForSignature, sourceSignature, type BatchResultRecord } from '$lib/stores/batch';
  import ImageCard from '$lib/components/ImageCard.svelte';

  type Screen = 'single' | 'batch';
  type WorkerName = 'preprocess' | 'inference' | 'features';

  interface WorkerUiState {
    status: 'idle' | 'loading' | 'ready' | 'error';
    message: string;
    progress?: number;
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
  let selectedFeatureSet = 'full_v2';
  let uploadError = '';
  let processing = false;
  let uploadedUrl = '';
  let uploadedName = '';
  let preprocessResult: PreprocessResult | undefined;
  let inferenceResult: InferenceResult | undefined;
  let biomarkers: BiomarkerRow[] = [];
  let boundsUrl = '';
  let rgbUrl = '';
  let overlayUrl = '';
  let modalImage: { title: string; url: string; filename: string } | undefined;
  let plotModal: { title: string; url: string } | undefined;
  let stageMessages: string[] = [];
  let currentCsvDownloaded = true;

  let sourceFiles: File[] = [];
  let batchRecords: BatchRecord[] = [];
  let batchResults: BatchResultRecord[] = [];
  let batchRunning = false;
  let batchDownloaded = true;
  let currentBatchName = '';
  let sourceName = '';
  let segmentationDirectory: FileSystemDirectoryHandle | undefined;
  let segmentationFolderName = '';
  let batchListOpen = false;
  let inferenceBackend: OrtBackend | undefined;

  $: appReady = workers.preprocess.status === 'ready' && workers.features.status === 'ready';
  $: inferenceReady = workers.inference.status === 'ready';
  $: canRunSingle = appReady && inferenceReady && !processing;
  $: canRunBatch = canRunSingle && sourceFiles.length > 0 && !batchRunning;

  onMount(() => {
    preprocessClient = new PipelineWorkerClient(new PreprocessWorker(), handleProgress);
    inferenceClient = new PipelineWorkerClient(new InferenceWorker(), handleProgress);
    featureClient = new PipelineWorkerClient(new FeatureWorker(), handleProgress);
    void initialiseWorkers().then(restoreBatchCache);

    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (batchRunning || (!batchDownloaded && batchResults.length > 0)) {
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
    setWorker('inference', { status: 'loading', message: 'Loading ONNX models', progress: 0 });
    setWorker('features', { status: 'loading', message: 'Loading Pyodide feature worker', progress: 0 });

    const results = await Promise.allSettled([
      preprocessClient.request({ type: 'init' }),
      inferenceClient.request({ type: 'init' }),
      featureClient.request({ type: 'init' })
    ]);

    if (results[0].status === 'fulfilled') setWorker('preprocess', { status: 'ready', message: 'Preprocessing ready', progress: 1 });
    else setWorker('preprocess', { status: 'error', message: results[0].reason.message });

    if (results[1].status === 'fulfilled') {
      const payload = results[1].value as { backend?: OrtBackend } | undefined;
      inferenceBackend = payload?.backend;
      console.info('[VascX] ONNX Runtime backend ready:', inferenceBackend ?? 'unknown');
      setWorker('inference', { status: 'ready', message: `Inference ready${inferenceBackend ? ` (${inferenceBackend.toUpperCase()})` : ''}`, progress: 1 });
    } else setWorker('inference', { status: 'error', message: results[1].reason.message });

    if (results[2].status === 'fulfilled') {
      setWorker('features', { status: 'ready', message: 'Feature extraction ready', progress: 1 });
      featureSets = (await featureClient.request({ type: 'loadFeatureSets' })) as FeatureSetSummary[];
      if (!featureSets.some((set) => set.name === selectedFeatureSet)) {
        selectedFeatureSet = featureSets[0]?.name ?? selectedFeatureSet;
      }
    } else {
      setWorker('features', { status: 'error', message: results[2].reason.message });
    }
  }

  function handleProgress(message: ProgressMessage) {
    if (message.stage === 'preprocess' || message.stage === 'inference' || message.stage === 'features') {
      setWorker(message.stage, { status: 'loading', message: message.message, progress: message.progress });
    }
  }

  function setWorker(name: WorkerName, patch: Partial<WorkerUiState>) {
    workers = { ...workers, [name]: { ...workers[name], ...patch } };
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
      const result = await runPipelineForFile(file, { keepRetina: true, keepArtifacts: true });
      preprocessResult = result.preprocess;
      inferenceResult = result.inference;
      biomarkers = result.features.biomarkers;
      currentCsvDownloaded = false;
      boundsUrl = bytesToObjectUrl(result.preprocess.boundsPng);
      rgbUrl = bytesToObjectUrl(result.preprocess.rgbPng);
      overlayUrl = bytesToObjectUrl(result.inference.overlayPng!);
    } catch (error) {
      uploadError = error instanceof Error ? error.message : String(error);
    } finally {
      processing = false;
      markReadyAfterRun();
    }
  }

  type PipelineRunOptions = {
    keepRetina: boolean;
    keepArtifacts: boolean;
    segmentationDirectory?: FileSystemDirectoryHandle;
  };
  type FullPipelineResult = { preprocess: PreprocessResult; inference: InferenceResult; features: FeatureExtractionResult };
  type BatchPipelineResult = { id: string; features: FeatureExtractionResult };

  function runPipelineForFile(file: File, options: PipelineRunOptions & { keepArtifacts: true }): Promise<FullPipelineResult>;
  function runPipelineForFile(file: File, options: PipelineRunOptions & { keepArtifacts: false }): Promise<BatchPipelineResult>;
  async function runPipelineForFile(file: File, options: PipelineRunOptions): Promise<FullPipelineResult | BatchPipelineResult> {
    stageMessages = [];
    await featureClient.request({ type: 'disposeCurrentRetina' });

    try {
      const bytes = await fileToBytes(file);

      stageMessages = ['Preprocessing image'];
      const preprocess = (await preprocessClient.request(
        { type: 'preprocessImage', image: { bytes, name: file.name, mimeType: file.type } },
        transferList(bytes)
      )) as PreprocessResult;

      stageMessages = ['Running ONNX inference'];
      const inference = (await inferenceClient.request({
        type: 'runInference',
        id: preprocess.id,
        rgbPng: preprocess.rgbPng,
        includePng: options.keepArtifacts || Boolean(options.segmentationDirectory),
        includeOverlay: options.keepArtifacts
      })) as InferenceResult;

      if (options.segmentationDirectory) {
        await writeSegmentationOutputs(options.segmentationDirectory, preprocess.id, inference, preprocess.bounds);
      }

      stageMessages = ['Extracting biomarkers'];
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
        featureSet: selectedFeatureSet
      };
      const features = (await featureClient.request(
        featureRequest,
        options.keepArtifacts ? [] : transferList(preprocess.rgbPng, inference.avMask, inference.vesselsMask, inference.discMask)
      )) as FeatureExtractionResult;

      if (options.keepArtifacts) {
        return { preprocess, inference, features };
      }

      return { id: preprocess.id, features };
    } finally {
      if (!options.keepRetina) {
        await featureClient.request({ type: 'disposeCurrentRetina' });
      }
    }
  }

  function transferList(...arrays: Uint8Array[]): Transferable[] {
    return arrays.map((array) => array.buffer).filter((buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer);
  }

  function markReadyAfterRun() {
    if (workers.preprocess.status !== 'error') setWorker('preprocess', { status: 'ready', message: 'Preprocessing ready', progress: 1 });
    if (workers.inference.status !== 'error') setWorker('inference', { status: 'ready', message: `Inference ready${inferenceBackend ? ` (${inferenceBackend.toUpperCase()})` : ''}`, progress: 1 });
    if (workers.features.status !== 'error') setWorker('features', { status: 'ready', message: 'Feature extraction ready', progress: 1 });
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
  }

  async function restoreBatchCache() {
    const cache = await loadBatchCache();
    if (!cache) return;
    batchRecords = cache.records;
    batchResults = cache.results;
    batchDownloaded = cache.downloaded;
  }

  async function runBatch() {
    if (!confirm(`Process ${sourceFiles.length} images one at a time?`)) return;
    batchRunning = true;
    batchDownloaded = false;
    const signature = sourceSignature(sourceFiles);
    const sourceHash = sourceHashForSignature(signature);
    const createdAt = new Date().toISOString();
    const recordIndexByPath = new Map(batchRecords.map((record, index) => [record.relativePath, index]));

    await saveBatchMeta({ sourceSignature: signature, sourceHash, createdAt, downloaded: batchDownloaded, records: batchRecords });

    for (const file of sourceFiles) {
      const relativePath = file.webkitRelativePath || file.name;
      const index = recordIndexByPath.get(relativePath);
      if (index === undefined || batchRecords[index].status === 'done') continue;
      currentBatchName = relativePath;
      batchRecords[index] = { ...batchRecords[index], status: 'running', error: undefined };
      batchRecords = batchRecords;
      const started = performance.now();
      try {
        const result = await runPipelineForFile(file, { keepRetina: false, keepArtifacts: false, segmentationDirectory });
        const output: BatchResultRecord = { id: result.id, relativePath, biomarkers: result.features.biomarkers };
        batchRecords[index] = { ...batchRecords[index], status: 'done', durationMs: performance.now() - started };
        batchResults.push(output);
        batchResults = batchResults;
        await saveBatchResult(sourceHash, output);
      } catch (error) {
        batchRecords[index] = { ...batchRecords[index], status: 'error', durationMs: performance.now() - started, error: error instanceof Error ? error.message : String(error) };
      }
      batchRecords = batchRecords;
      await saveBatchMeta({ sourceSignature: signature, sourceHash, createdAt, downloaded: batchDownloaded, records: batchRecords });
    }

    currentBatchName = '';
    batchRunning = false;
    await featureClient.request({ type: 'disposeCurrentRetina' });
    await saveBatchMeta({ sourceSignature: signature, sourceHash, createdAt, downloaded: batchDownloaded, records: batchRecords });
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

  function switchScreen(screen: Screen) {
    if (activeScreen === 'batch' && (batchRunning || (!batchDownloaded && batchResults.length > 0))) {
      if (!confirm('Batch data may be lost if you leave before downloading the CSV. Continue?')) return;
    }
    activeScreen = screen;
  }
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
          {name}: {worker.status}
        </span>
      {/each}
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <button class:active={activeScreen === 'single'} on:click={() => switchScreen('single')}><ImagePlus size={18} /> Single image</button>
      <button class:active={activeScreen === 'batch'} on:click={() => switchScreen('batch')}><FolderOpen size={18} /> Batch inference</button>
    </aside>

    <main class="content">
      <section class:compact={appReady && inferenceReady} class="panel setup-panel">
        <div>
          <p class="eyebrow">Initialisation</p>
          <h2>{appReady && inferenceReady ? 'Pipeline ready' : 'Loading browser pipeline'}</h2>
          <div class="backend-summary">
            <span>ONNX backend</span>
            <strong class:warn={inferenceBackend === 'wasm'}>{inferenceBackend ? inferenceBackend.toUpperCase() : 'Detecting'}</strong>
          </div>
        </div>
        <div class="worker-list">
          {#each Object.entries(workers) as [name, worker]}
            <div class="worker-row">
              <span>{name}</span>
              <progress value={worker.progress ?? 0} max="1"></progress>
              <small>{worker.message}</small>
            </div>
          {/each}
        </div>
      </section>

      {#if activeScreen === 'single'}
        <section class="screen single-screen">
          <div class="screen-header">
            <div>
              <p class="eyebrow">Single-image inference</p>
              <h2>Extract retinal vascular parameters</h2>
              <p>Drop one fundus image, choose a VascX feature set, and run all stages in browser memory.</p>
            </div>
            <label class="field-label">Feature set
              <select bind:value={selectedFeatureSet} disabled={processing}>
                {#each featureSets as set}
                  <option value={set.name}>{set.displayName}</option>
                {/each}
              </select>
            </label>
          </div>

          <div class="upload-zone" class:disabled={!canRunSingle} role="region" aria-label="Image upload drop zone" on:drop={onDrop} on:dragover|preventDefault>
            <UploadCloud size={34} />
            <strong>Drop a fundus image here</strong>
            <span>{VALID_IMAGE_EXTENSIONS.join(', ')}</span>
            <label class="button primary">
              <input type="file" accept="image/*" on:change={onFileInput} disabled={!canRunSingle} />
              Select image
            </label>
          </div>

          {#if uploadError}
            <div class="alert"><AlertTriangle size={18} /> {uploadError} Please try another image or report the error with the browser console details.</div>
          {/if}

          {#if processing}
            <div class="progress-card"><Loader2 class="spin" /> {stageMessages[0] ?? 'Processing image'}</div>
          {/if}

          <div class="image-grid">
            <ImageCard title="Uploaded image" url={uploadedUrl} filename={uploadedName || 'uploaded.png'} onOpen={() => (modalImage = { title: 'Uploaded image', url: uploadedUrl, filename: uploadedName || 'uploaded.png' })} />
            <ImageCard title="Bounds" url={boundsUrl} filename={`${preprocessResult?.id ?? 'image'}_bounds.png`} bytes={preprocessResult?.boundsPng} onOpen={() => (modalImage = { title: 'Bounds', url: boundsUrl, filename: `${preprocessResult?.id ?? 'image'}_bounds.png` })} />
            <ImageCard title="Inference overlay" url={overlayUrl} filename={`${preprocessResult?.id ?? 'image'}_overlay.png`} bytes={inferenceResult?.overlayPng} onOpen={() => (modalImage = { title: 'Inference overlay', url: overlayUrl, filename: `${preprocessResult?.id ?? 'image'}_overlay.png` })} />
          </div>

          <section class="panel biomarkers-panel">
            <div class="panel-header">
              <div>
                <h3>Biomarkers</h3>
                <p>{biomarkers.length ? `${biomarkers.length} values from ${selectedFeatureSet}` : 'Run an image to calculate biomarkers.'}</p>
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
          </section>
        </section>
      {:else}
        <section class="screen batch-screen">
          <div class="screen-header">
            <div>
              <p class="eyebrow">Batch inference</p>
              <h2>Process an authorised image folder</h2>
              <p>Images are processed one at a time. Results are cached locally until cleared.</p>
            </div>
          </div>

          <div class="folder-row">
            <section class="folder-box">
              <FolderOpen size={22} />
              <h3>Source folder</h3>
              <p>Deep search for {VALID_IMAGE_EXTENSIONS.join(', ')}.</p>
              <button class="button primary" on:click={chooseSourceFolder}><FolderOpen size={16} /> Select source</button>
              <input id="batch-file-input" class="hidden" type="file" multiple webkitdirectory on:change={onBatchFileInput} />
              <small>{sourceName || 'No source selected'}</small>
            </section>
            <section class="folder-box">
              <FolderOpen size={22} />
              <h3>Segmentations folder</h3>
              <p>Optional empty folder for masks, fovea, and bounds sidecars.</p>
              <button class="button" on:click={chooseSegmentationFolder}><FolderOpen size={16} /> Select output</button>
              <small>{segmentationFolderName || 'Intermediate files stay in memory'}</small>
            </section>
          </div>

          {#if batchRecords.length}
            <section class="panel">
              <div class="panel-header">
                <div>
                  <h3>{batchRecords.length} images found</h3>
                  <p>{batchRecords.slice(0, 3).map((record) => record.relativePath).join(', ')}</p>
                </div>
                <div class="button-row">
                  <button class="button primary" disabled={!canRunBatch} on:click={runBatch}><Play size={16} /> Run</button>
                  <button class="button" disabled={!batchResults.length} on:click={downloadBatchCsv}><Download size={16} /> CSV</button>
                  <button class="button danger" on:click={clearBatchResults}><Trash2 size={16} /> Clear results</button>
                </div>
              </div>
              <progress value={batchRecords.filter((record) => record.status === 'done').length} max={batchRecords.length}></progress>
              {#if currentBatchName}<p class="current-file"><Loader2 class="spin" size={16} /> {currentBatchName}</p>{/if}
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
            </section>
          {/if}
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
      <img src={modalImage.url} alt={modalImage.title} />
    </div>
  </div>
{/if}

{#if plotModal}
  <div class="modal-backdrop" role="presentation" on:click={closePlotModalFromBackdrop}>
    <div class="modal" role="dialog" aria-modal="true" aria-label={plotModal.title}>
      <button class="icon-button close" on:click={() => (plotModal = undefined)}><X size={18} /></button>
      <h3>{plotModal.title}</h3>
      {#if plotModal.url}
        <img class="plot-image" src={plotModal.url} alt={plotModal.title} />
      {:else}
        <div class="progress-card"><Loader2 class="spin" /> Rendering plot</div>
      {/if}
    </div>
  </div>
{/if}
