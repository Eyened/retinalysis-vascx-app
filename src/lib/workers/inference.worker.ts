import * as ort from 'onnxruntime-web/webgpu';
import type { InferenceConfig, InferenceResult, ModelManifest, OrtBackend, WorkerRequest, WorkerResponse } from '$lib/types/pipeline';
import {
  createOverlayPng,
  decodeImageBytes,
  extractTile,
  flipTensorNchw,
  heatmapArgmax,
  imageToNormalizedTensor,
  makeBlendWeights,
  makeSlidingWindowTiles,
  maskToPng,
  softmaxArgmax,
  type TensorImage
} from '$lib/utils/image';

let manifest: ModelManifest | undefined;
let backend: OrtBackend | undefined;
let sessions: Record<'av' | 'vessels' | 'disc' | 'fovea', ort.InferenceSession> | undefined;

const MODEL_CACHE_PREFIX = 'vascx-onnx-models';

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      await init(message.requestId);
    } else if (message.type === 'runInference') {
      const result = await runInference(message.id, message.rgbPng, message.includePng ?? true, message.includeOverlay ?? true);
      post({ type: 'result', requestId: message.requestId, payload: result }, compactTransferList([
        result.avMask.buffer,
        result.vesselsMask.buffer,
        result.discMask.buffer,
        result.avMaskPng?.buffer,
        result.vesselsMaskPng?.buffer,
        result.discMaskPng?.buffer,
        result.overlayPng?.buffer
      ]));
    }
  } catch (error) {
    post({
      type: 'error',
      requestId: message.requestId,
      error: { message: 'Inference worker failed', detail: error instanceof Error ? error.message : String(error), recoverable: true }
    });
  }
};

async function init(requestId: string) {
  progress('Loading model manifest', 0.05);
  manifest = await fetchManifest();
  await cleanupOldModelCaches();
  await pruneCurrentModelCache();
  backend = await resolveBackend();
  progress(`Using ONNX Runtime ${backend.toUpperCase()} backend`, 0.15);

  const executionProviders = backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  console.info('[VascX inference] ONNX Runtime backend selected:', backend);
  console.info('[VascX inference] Session execution providers:', executionProviders.join(', '));
  sessions = {} as Record<'av' | 'vessels' | 'disc' | 'fovea', ort.InferenceSession>;
  const modelKeys = ['av', 'vessels', 'disc', 'fovea'] as const;
  for (let index = 0; index < modelKeys.length; index += 1) {
    const key = modelKeys[index];
    progress(`Loading ${manifest.models[key].label} model`, 0.15 + (index / modelKeys.length) * 0.8);
    const modelUrl = manifest.models[key].url;
    assertPublicModelUrl(modelUrl, manifest.models[key].label);
    const modelBytes = await fetchOnnxModelBytes(modelUrl, manifest.models[key].label);
    sessions[key] = await ort.InferenceSession.create(modelBytes, {
      executionProviders,
      graphOptimizationLevel: 'all'
    });
    console.info(`[VascX inference] ${manifest.models[key].label} session ready`, {
      backend,
      executionProviders,
      inputs: sessions[key].inputNames,
      outputs: sessions[key].outputNames
    });
  }
  progress('Inference models ready', 1);
  post({ type: 'ready', requestId, payload: { backend } });
}

async function fetchManifest(): Promise<ModelManifest> {
  const response = await fetch(new URL('models/manifest.json', new URL(import.meta.env.BASE_URL, self.location.origin)));
  if (!response.ok) {
    throw new Error(`Could not load model manifest: ${response.statusText}`);
  }
  return (await response.json()) as ModelManifest;
}

async function fetchOnnxModelBytes(url: string, label: string): Promise<Uint8Array> {
  const request = new Request(url);
  const cache = await openModelCache();
  const cached = cache ? await cache.match(request) : undefined;

  if (cached) {
    return await readAndValidateOnnxResponse(cached, url, label);
  }

  const response = await fetch(request);
  const bytes = await readAndValidateOnnxResponse(response, url, label);

  if (cache) {
    await cache.put(
      request,
      new Response(toArrayBuffer(bytes), {
        headers: {
          'content-type': 'application/octet-stream',
          'x-vascx-model-cache-version': String(manifest?.version ?? 1)
        }
      })
    );
  }

  return bytes;
}

async function readAndValidateOnnxResponse(response: Response, url: string, label: string): Promise<Uint8Array> {
  if (!response.ok) {
    throw new Error(`${label} model could not be fetched from ${url}: HTTP ${response.status}`);
  }

  const contentType = response.headers.get('content-type') ?? '';
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const prefix = new TextDecoder().decode(bytes.slice(0, Math.min(bytes.length, 128))).toLowerCase();

  if (contentType.includes('text/html') || prefix.includes('<!doctype html') || prefix.includes('<html')) {
    throw new Error(`${label} model URL returned HTML instead of ONNX bytes: ${url}. Check that manifest.json points to a public direct ONNX download URL.`);
  }

  if (prefix.startsWith('version https://git-lfs.github.com/spec')) {
    throw new Error(`${label} model is a Git LFS pointer, not the ONNX binary: ${url}. Use a public URL that resolves to the real ONNX bytes.`);
  }

  if (bytes.byteLength < 1024) {
    throw new Error(`${label} model is unexpectedly small (${bytes.byteLength} bytes): ${url}`);
  }

  return bytes;
}

async function openModelCache(): Promise<Cache | undefined> {
  if (!('caches' in self)) {
    return undefined;
  }

  try {
    return await caches.open(currentModelCacheName());
  } catch {
    return undefined;
  }
}

async function cleanupOldModelCaches(): Promise<void> {
  if (!('caches' in self)) {
    return;
  }

  const current = currentModelCacheName();
  try {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) => key.startsWith(MODEL_CACHE_PREFIX) && key !== current).map((key) => caches.delete(key)));
  } catch {
    // Cache cleanup is opportunistic; model loading should still work without it.
  }
}

function currentModelCacheName(): string {
  return `${MODEL_CACHE_PREFIX}-v${manifest?.version ?? 1}`;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}


async function pruneCurrentModelCache(): Promise<void> {
  if (!manifest) return;
  const cache = await openModelCache();
  if (!cache) return;
  const expectedUrls = new Set(Object.values(manifest.models).map((model) => model.url));
  try {
    const requests = await cache.keys();
    await Promise.all(requests.filter((request) => !expectedUrls.has(request.url)).map((request) => cache.delete(request)));
  } catch {
    // Stale model cache entries are harmless; pruning is best effort.
  }
}

function assertPublicModelUrl(url: string, label: string) {
  if (!/^https:\/\//.test(url)) {
    throw new Error(`${label} model URL must be a public HTTPS URL in models/manifest.json. Got: ${url}`);
  }
}

async function resolveBackend(): Promise<OrtBackend> {
  const gpu = navigator.gpu;
  if (!gpu) {
    console.warn('[VascX inference] navigator.gpu is unavailable; falling back to ONNX Runtime WASM.');
    return 'wasm';
  }

  console.info('[VascX inference] navigator.gpu is available; requesting a high-performance adapter.');
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      console.info('[VascX inference] WebGPU adapter acquired; using ONNX Runtime WebGPU.');
      return 'webgpu';
    }
    console.warn('[VascX inference] WebGPU adapter request returned null; falling back to ONNX Runtime WASM.');
  } catch (error) {
    console.warn('[VascX inference] WebGPU adapter request failed; falling back to ONNX Runtime WASM.', error);
  }
  return 'wasm';
}

async function runInference(id: string, rgbPng: Uint8Array, includePng: boolean, includeOverlay: boolean): Promise<InferenceResult> {
  if (!manifest || !sessions || !backend) {
    throw new Error('Inference worker has not been initialised.');
  }
  const started = performance.now();
  console.info('[VascX inference] Starting inference run', { id, backend });
  progress('Decoding preprocessed image', 0.05);
  const decoded = await decodeImageBytes(rgbPng);
  const tensor = imageToNormalizedTensor(decoded, manifest.defaults.normalize);

  progress('Segmenting artery / vein mask', 0.18);
  const avLogits = await runSlidingWindow(sessions.av, tensor, mergeConfig(manifest, 'av'), manifest.models.av.classes);
  const avMask = softmaxArgmax(avLogits, manifest.models.av.classes, tensor.width, tensor.height);

  progress('Segmenting vessel mask', 0.42);
  const vesselsLogits = await runSlidingWindow(sessions.vessels, tensor, mergeConfig(manifest, 'vessels'), manifest.models.vessels.classes);
  const vesselsMask = softmaxArgmax(vesselsLogits, manifest.models.vessels.classes, tensor.width, tensor.height);

  progress('Segmenting optic disc', 0.65);
  const discLogits = await runResizedWholeImage(sessions.disc, tensor, mergeConfig(manifest, 'disc'), manifest.models.disc.classes);
  const discMask = softmaxArgmax(discLogits, manifest.models.disc.classes, tensor.width, tensor.height);

  progress('Detecting fovea', 0.82);
  const foveaHeatmap = await runResizedWholeImage(sessions.fovea, tensor, mergeConfig(manifest, 'fovea'), manifest.models.fovea.classes);
  const fovea = heatmapArgmax(foveaHeatmap, tensor.width, tensor.height);

  progress(includePng ? 'Rendering inference outputs' : 'Finalising inference outputs', 0.92);
  const [avMaskPng, vesselsMaskPng, discMaskPng, overlayPng] = includePng
    ? await Promise.all([
        maskToPng(avMask, tensor.width, tensor.height, [[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]),
        maskToPng(vesselsMask, tensor.width, tensor.height),
        maskToPng(discMask, tensor.width, tensor.height),
        includeOverlay ? createOverlayPng(rgbPng, avMask, discMask, fovea) : Promise.resolve(undefined)
      ])
    : [undefined, undefined, undefined, undefined];

  const durationMs = performance.now() - started;
  progress('Inference complete', 1);
  console.info('[VascX inference] Inference run complete', { id, backend, durationMs });
  return {
    id,
    avMask,
    vesselsMask,
    discMask,
    width: tensor.width,
    height: tensor.height,
    avMaskPng,
    vesselsMaskPng,
    discMaskPng,
    overlayPng,
    fovea,
    backend,
    durationMs
  };
}

function mergeConfig(currentManifest: ModelManifest, key: keyof ModelManifest['models']): InferenceConfig {
  return { ...currentManifest.defaults, ...currentManifest.models[key].config };
}

async function runSlidingWindow(session: ort.InferenceSession, image: TensorImage, config: InferenceConfig, classes: number): Promise<Float32Array> {
  const basePrediction = await runSlidingWindowOnce(session, image, config, classes);
  const predictions = [basePrediction];

  for (const axes of config.ttaFlips ?? []) {
    const flipped = { ...image, data: flipTensorNchw(image.data, image.channels, image.width, image.height, axes) };
    const prediction = await runSlidingWindowOnce(session, flipped, config, classes);
    predictions.push(flipLogitsNhwc(prediction, classes, image.width, image.height, axes));
  }

  const averaged = new Float32Array(basePrediction.length);
  for (const prediction of predictions) {
    for (let index = 0; index < averaged.length; index += 1) {
      averaged[index] += prediction[index] / predictions.length;
    }
  }
  return averaged;
}

async function runSlidingWindowOnce(session: ort.InferenceSession, image: TensorImage, config: InferenceConfig, classes: number): Promise<Float32Array> {
  const output = new Float32Array(image.width * image.height * classes);
  const weightSum = new Float32Array(image.width * image.height);
  const tiles = makeSlidingWindowTiles(image.width, image.height, config.patchSize, config.overlap);
  const weights = makeBlendWeights(config.patchSize[1], config.patchSize[0], config.blend);

  if (tiles.length === 0) {
    throw new Error('Sliding-window inference produced no tiles.');
  }

  const tileWidth = tiles[0].width;
  const tileHeight = tiles[0].height;
  const tilePlane = image.channels * tileWidth * tileHeight;
  const input = new Float32Array(tiles.length * tilePlane);

  for (let batchIndex = 0; batchIndex < tiles.length; batchIndex += 1) {
    input.set(extractTile(image, tiles[batchIndex]), batchIndex * tilePlane);
  }

  const tensor = new ort.Tensor('float32', input, [tiles.length, image.channels, tileHeight, tileWidth]);
  const result = await session.run({ image: tensor });
  const prediction = result.prediction ?? Object.values(result)[0];
  if (!prediction || !(prediction.data instanceof Float32Array)) {
    throw new Error('ONNX model did not return a Float32 prediction tensor.');
  }

  const tileLogits = decodeOnnxOutputBatches(prediction.data, prediction.dims, classes, tileWidth, tileHeight, tiles.length);
  for (let batchIndex = 0; batchIndex < tiles.length; batchIndex += 1) {
    const tile = tiles[batchIndex];
    blendTile(output, weightSum, tileLogits[batchIndex], weights, tile.x, tile.y, tile.width, tile.height, image.width, classes);
  }

  for (let pixel = 0; pixel < weightSum.length; pixel += 1) {
    const weight = weightSum[pixel] || 1;
    for (let klass = 0; klass < classes; klass += 1) {
      output[pixel * classes + klass] /= weight;
    }
  }
  return output;
}

async function runResizedWholeImage(session: ort.InferenceSession, image: TensorImage, config: InferenceConfig, classes: number): Promise<Float32Array> {
  const [targetHeight, targetWidth] = config.patchSize;
  const resized = resizeTensorNchw(image, targetWidth, targetHeight);
  const logits = await runWholeImage(session, resized, classes);
  if (resized.width === image.width && resized.height === image.height) {
    return logits;
  }
  return resizeLogitsNhwc(logits, classes, resized.width, resized.height, image.width, image.height);
}

async function runWholeImage(session: ort.InferenceSession, image: TensorImage, classes: number): Promise<Float32Array> {
  const tensor = new ort.Tensor('float32', image.data, [1, image.channels, image.height, image.width]);
  const result = await session.run({ image: tensor });
  const prediction = result.prediction ?? Object.values(result)[0];
  if (!prediction || !(prediction.data instanceof Float32Array)) {
    throw new Error('ONNX model did not return a Float32 prediction tensor.');
  }
  return decodeOnnxOutputBatches(prediction.data, prediction.dims, classes, image.width, image.height, 1)[0];
}

function decodeOnnxOutputBatches(data: Float32Array, dims: readonly number[], classes: number, width: number, height: number, expectedBatch: number): Float32Array[] {
  const outputs = Array.from({ length: expectedBatch }, () => new Float32Array(width * height * classes));
  const rank = dims.length;

  if (rank === 5) {
    const [models, batch, outputClasses, outputHeight, outputWidth] = dims;
    assertOutputShape(batch, outputClasses, outputWidth, outputHeight, classes, width, height, expectedBatch);
    for (let model = 0; model < models; model += 1) {
      for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
        const output = outputs[batchIndex];
        for (let klass = 0; klass < classes; klass += 1) {
          for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
              const source = (((((model * batch + batchIndex) * classes + klass) * height + y) * width) + x);
              output[(y * width + x) * classes + klass] += data[source] / models;
            }
          }
        }
      }
    }
    return outputs;
  }

  if (rank === 4) {
    const [batch, outputClasses, outputHeight, outputWidth] = dims;
    assertOutputShape(batch, outputClasses, outputWidth, outputHeight, classes, width, height, expectedBatch);
    for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
      const output = outputs[batchIndex];
      for (let klass = 0; klass < classes; klass += 1) {
        for (let y = 0; y < height; y += 1) {
          for (let x = 0; x < width; x += 1) {
            const source = ((((batchIndex * classes + klass) * height + y) * width) + x);
            output[(y * width + x) * classes + klass] = data[source];
          }
        }
      }
    }
    return outputs;
  }

  throw new Error(`Unsupported ONNX output shape: ${dims.join('x')}`);
}

function assertOutputShape(
  batch: number,
  outputClasses: number,
  outputWidth: number,
  outputHeight: number,
  classes: number,
  width: number,
  height: number,
  expectedBatch: number
) {
  if (batch !== expectedBatch || outputClasses !== classes || outputWidth !== width || outputHeight !== height) {
    throw new Error(`Unexpected ONNX output shape. Got batch=${batch}, classes=${outputClasses}, size=${outputWidth}x${outputHeight}. Expected batch=${expectedBatch}, classes=${classes}, size=${width}x${height}.`);
  }
}

function resizeTensorNchw(image: TensorImage, width: number, height: number): TensorImage {
  if (image.width === width && image.height === height) {
    return image;
  }

  const output = new Float32Array(image.channels * width * height);
  for (let channel = 0; channel < image.channels; channel += 1) {
    resizePlaneBilinear(
      image.data.subarray(channel * image.width * image.height, (channel + 1) * image.width * image.height),
      output.subarray(channel * width * height, (channel + 1) * width * height),
      image.width,
      image.height,
      width,
      height
    );
  }
  return { ...image, width, height, data: output };
}

function resizeLogitsNhwc(data: Float32Array, classes: number, sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number): Float32Array {
  const output = new Float32Array(targetWidth * targetHeight * classes);
  const sourcePlane = sourceWidth * sourceHeight;
  const targetPlane = targetWidth * targetHeight;

  for (let klass = 0; klass < classes; klass += 1) {
    const source = new Float32Array(sourcePlane);
    const target = new Float32Array(targetPlane);
    for (let pixel = 0; pixel < sourcePlane; pixel += 1) {
      source[pixel] = data[pixel * classes + klass];
    }
    resizePlaneBilinear(source, target, sourceWidth, sourceHeight, targetWidth, targetHeight);
    for (let pixel = 0; pixel < targetPlane; pixel += 1) {
      output[pixel * classes + klass] = target[pixel];
    }
  }

  return output;
}

function resizePlaneBilinear(source: Float32Array, target: Float32Array, sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number) {
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.max(0, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const wy = sourceY - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.max(0, (x + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const wx = sourceX - x0;
      const top = source[y0 * sourceWidth + x0] * (1 - wx) + source[y0 * sourceWidth + x1] * wx;
      const bottom = source[y1 * sourceWidth + x0] * (1 - wx) + source[y1 * sourceWidth + x1] * wx;
      target[y * targetWidth + x] = top * (1 - wy) + bottom * wy;
    }
  }
}

function blendTile(
  output: Float32Array,
  weightSum: Float32Array,
  tileLogits: Float32Array,
  weights: Float32Array,
  tileX: number,
  tileY: number,
  tileWidth: number,
  tileHeight: number,
  imageWidth: number,
  classes: number
) {
  for (let y = 0; y < tileHeight; y += 1) {
    for (let x = 0; x < tileWidth; x += 1) {
      const imagePixel = (tileY + y) * imageWidth + tileX + x;
      const tilePixel = y * tileWidth + x;
      const weight = weights[tilePixel];
      weightSum[imagePixel] += weight;
      for (let klass = 0; klass < classes; klass += 1) {
        output[imagePixel * classes + klass] += tileLogits[tilePixel * classes + klass] * weight;
      }
    }
  }
}

function flipLogitsNhwc(data: Float32Array, classes: number, width: number, height: number, axes: number[]): Float32Array {
  const output = new Float32Array(data.length);
  const flipY = axes.includes(2);
  const flipX = axes.includes(3);
  for (let y = 0; y < height; y += 1) {
    const sourceY = flipY ? height - 1 - y : y;
    for (let x = 0; x < width; x += 1) {
      const sourceX = flipX ? width - 1 - x : x;
      for (let klass = 0; klass < classes; klass += 1) {
        output[(y * width + x) * classes + klass] = data[(sourceY * width + sourceX) * classes + klass];
      }
    }
  }
  return output;
}

function compactTransferList(buffers: Array<ArrayBufferLike | undefined>): Transferable[] {
  return buffers.filter((buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer);
}

function progress(message: string, progress?: number) {
  post({ type: 'progress', stage: 'inference', message, progress });
}

function post(message: WorkerResponse, transfer: Transferable[] = []) {
  (self as unknown as { postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void }).postMessage(message, transfer);
}
