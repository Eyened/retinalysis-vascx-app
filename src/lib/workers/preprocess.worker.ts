import type { PreprocessResult, WorkerRequest, WorkerResponse } from '$lib/types/pipeline';

const PYODIDE_VERSION = '0.27.7';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

interface PyodideRuntime {
  FS: { writeFile(path: string, data: Uint8Array): void; unlink(path: string): void };
  loadPackage(packages: string[] | string): Promise<void>;
  runPythonAsync<T = unknown>(code: string): Promise<T>;
  globals: { set(name: string, value: unknown): void };
}

let pyodide: PyodideRuntime | undefined;
let ready = false;

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      await init(message.requestId);
    } else if (message.type === 'preprocessImage') {
      const result = await preprocessImage(message.image.bytes, message.image.name);
      post({ type: 'result', requestId: message.requestId, payload: result }, [result.rgbPng.buffer, result.boundsPng.buffer]);
    }
  } catch (error) {
    post({
      type: 'error',
      requestId: message.requestId,
      error: { message: 'Preprocessing failed', detail: error instanceof Error ? error.message : String(error), recoverable: true }
    });
  }
};

async function loadPyodideRuntime(): Promise<PyodideRuntime> {
  const module = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`);
  return (await module.loadPyodide({ indexURL: PYODIDE_URL })) as unknown as PyodideRuntime;
}

async function init(requestId: string) {
  if (ready) {
    post({ type: 'ready', requestId });
    return;
  }
  progress('Loading Pyodide runtime', 0.05);
  pyodide = await loadPyodideRuntime();

  progress('Loading Pyodide scientific packages', 0.25);
  await pyodide.loadPackage(['micropip', 'numpy', 'pillow', 'scipy', 'opencv-python', 'scikit-image', 'scikit-learn', 'matplotlib', 'pandas', 'joblib']);

  progress('Installing retinalysis-fundusprep', 0.65);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install('retinalysis-fundusprep')
`);

  progress('Preprocessing worker ready', 1);
  ready = true;
  post({ type: 'ready', requestId });
}

async function preprocessImage(bytes: Uint8Array, name: string): Promise<PreprocessResult> {
  if (!pyodide || !ready) {
    throw new Error('Preprocessing worker has not been initialised.');
  }
  const started = performance.now();
  const suffix = extensionForName(name);
  const inputPath = `/tmp/vascx_input${suffix}`;
  pyodide.FS.writeFile(inputPath, bytes);
  pyodide.globals.set('INPUT_PATH', inputPath);
  pyodide.globals.set('IMAGE_ID', stemForName(name));
  pyodide.globals.set('ORIGINAL_NAME', name);

  progress('Detecting fundus bounds', 0.25);
  const proxy = await pyodide.runPythonAsync(`
from io import BytesIO
import numpy as np
from PIL import Image
from pyodide.ffi import to_js
import js
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.preprocessor import FundusPreprocessor
from rtnls_fundusprep.utils import open_image

image = open_image(INPUT_PATH)
if image.ndim == 2:
    image = np.stack([image, image, image], axis=-1)
if image.shape[-1] == 4:
    image = image[..., :3]

bounds = get_cfi_bounds(image)
bounds_image = bounds.make_bounds_image()
preprocessor = FundusPreprocessor(square_size=1024, contrast_enhance=False)
prep = preprocessor(image, None)
rgb = prep['image'].astype(np.uint8)

rgb_buffer = BytesIO()
Image.fromarray(rgb).save(rgb_buffer, format='PNG')
bounds_buffer = BytesIO()
Image.fromarray(bounds_image.astype(np.uint8)).save(bounds_buffer, format='PNG')

result = {
    'id': IMAGE_ID,
    'originalName': ORIGINAL_NAME,
    'rgbPng': rgb_buffer.getvalue(),
    'boundsPng': bounds_buffer.getvalue(),
    'bounds': bounds.to_dict_all(),
    'width': int(rgb.shape[1]),
    'height': int(rgb.shape[0]),
}
to_js(result, dict_converter=js.Object.fromEntries)
`);

  try {
    const result = toPlainObject(proxy) as Omit<PreprocessResult, 'durationMs'>;
    return {
      ...result,
      rgbPng: toBytes(result.rgbPng),
      boundsPng: toBytes(result.boundsPng),
      durationMs: performance.now() - started
    };
  } finally {
    destroyProxy(proxy);
    try {
      pyodide.FS.unlink(inputPath);
    } catch {
      // Ignore cleanup failures in the worker filesystem.
    }
  }
}

function toPlainObject(value: unknown): unknown {
  if (value && typeof value === 'object' && 'toJs' in value && typeof (value as { toJs: () => unknown }).toJs === 'function') {
    return (value as { toJs: () => unknown }).toJs();
  }
  return value;
}

function toBytes(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  if (Array.isArray(value)) {
    return new Uint8Array(value);
  }
  throw new Error('Expected Pyodide to return image bytes.');
}

function destroyProxy(value: unknown) {
  if (value && typeof value === 'object' && 'destroy' in value && typeof (value as { destroy: () => void }).destroy === 'function') {
    (value as { destroy: () => void }).destroy();
  }
}

function extensionForName(name: string): string {
  const match = /\.[A-Za-z0-9]+$/.exec(name);
  return match ? match[0].toLowerCase() : '.png';
}

function stemForName(name: string): string {
  return name.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_.-]+/g, '_') || 'image';
}

function progress(message: string, progress?: number) {
  post({ type: 'progress', stage: 'preprocess', message, progress });
}

function post(message: WorkerResponse, transfer: Transferable[] = []) {
  (self as unknown as { postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void }).postMessage(message, transfer);
}
