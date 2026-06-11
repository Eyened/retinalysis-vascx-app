import type { FeatureExtractionResult, FeatureSetSummary, WorkerRequest, WorkerResponse } from '$lib/types/pipeline';

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
    } else if (message.type === 'loadFeatureSets') {
      const payload = await loadFeatureSets();
      post({ type: 'result', requestId: message.requestId, payload });
    } else if (message.type === 'extractFeatures') {
      const result = await extractFeatures(message);
      post({ type: 'result', requestId: message.requestId, payload: result });
    } else if (message.type === 'renderFeaturePlot') {
      const payload = await renderFeaturePlot(message.key);
      post({ type: 'result', requestId: message.requestId, payload });
    } else if (message.type === 'disposeCurrentRetina') {
      await disposeCurrentRetina();
      post({ type: 'result', requestId: message.requestId, payload: true });
    }
  } catch (error) {
    post({
      type: 'error',
      requestId: message.requestId,
      error: { message: 'Biomarker extraction failed', detail: error instanceof Error ? error.message : String(error), recoverable: true }
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

  progress('Loading Pyodide scientific packages', 0.2);
  await pyodide.loadPackage([
    'micropip',
    'numpy',
    'pillow',
    'scipy',
    'opencv-python',
    'scikit-image',
    'scikit-learn',
    'matplotlib',
    'pandas',
    'networkx'
  ]);

  progress('Installing retinalysis-vascx', 0.55);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(['retinalysis-vascx', 'retinalysis-fundusprep'])
`);

  progress('Preparing browser biomarker helpers', 0.8);
  await pyodide.runPythonAsync(BROWSER_HELPERS_PY);

  progress('Biomarker extraction worker ready', 1);
  ready = true;
  post({ type: 'ready', requestId });
}

async function loadFeatureSets(): Promise<FeatureSetSummary[]> {
  assertReady();
  const proxy = await pyodide!.runPythonAsync(`
from pyodide.ffi import to_js
import js
feature_set_manifest = _vascx_browser_helpers['feature_set_manifest']
to_js(feature_set_manifest(), dict_converter=js.Object.fromEntries)
`);
  try {
    return toPlainObject(proxy) as FeatureSetSummary[];
  } finally {
    destroyProxy(proxy);
  }
}

async function extractFeatures(message: Extract<WorkerRequest, { type: 'extractFeatures' }>): Promise<FeatureExtractionResult> {
  assertReady();
  const started = performance.now();
  const paths = {
    rgb: `/tmp/${message.id}_rgb.png`,
    av: `/tmp/${message.id}_av.raw`,
    vessels: `/tmp/${message.id}_vessels.raw`,
    disc: `/tmp/${message.id}_disc.raw`
  };
  pyodide!.FS.writeFile(paths.rgb, message.rgbPng);
  pyodide!.FS.writeFile(paths.av, message.avMask);
  pyodide!.FS.writeFile(paths.vessels, message.vesselsMask);
  pyodide!.FS.writeFile(paths.disc, message.discMask);
  pyodide!.globals.set('RGB_PATH', paths.rgb);
  pyodide!.globals.set('AV_PATH', paths.av);
  pyodide!.globals.set('VESSELS_PATH', paths.vessels);
  pyodide!.globals.set('DISC_PATH', paths.disc);
  pyodide!.globals.set('MASK_WIDTH', message.width);
  pyodide!.globals.set('MASK_HEIGHT', message.height);
  pyodide!.globals.set('FOVEA_X', message.fovea[0]);
  pyodide!.globals.set('FOVEA_Y', message.fovea[1]);
  pyodide!.globals.set('BOUNDS_JSON', JSON.stringify(message.bounds));
  pyodide!.globals.set('IMAGE_ID', message.id);
  pyodide!.globals.set('FEATURE_SET', message.featureSet);

  progress('Building VascX Retina', 0.2);
  const proxy = await pyodide!.runPythonAsync(`
from pyodide.ffi import to_js
import js
import json
import numpy as np
from PIL import Image

retina_from_arrays = _vascx_browser_helpers['retina_from_arrays']
compute_features_for_browser = _vascx_browser_helpers['compute_features_for_browser']

fundus = np.asarray(Image.open(RGB_PATH).convert('RGB'))
mask_shape = (int(MASK_HEIGHT), int(MASK_WIDTH))
av = np.frombuffer(open(AV_PATH, 'rb').read(), dtype=np.uint8).reshape(mask_shape)
vessels = np.frombuffer(open(VESSELS_PATH, 'rb').read(), dtype=np.uint8).reshape(mask_shape)
disc = np.frombuffer(open(DISC_PATH, 'rb').read(), dtype=np.uint8).reshape(mask_shape)
bounds = json.loads(BOUNDS_JSON)
_current_retina = retina_from_arrays(
    fundus=fundus,
    av_mask=av,
    vessels_mask=vessels,
    disc_mask=disc,
    fovea_location=(float(FOVEA_X), float(FOVEA_Y)),
    bounds=bounds,
    id=IMAGE_ID,
)
_current_feature_set = FEATURE_SET
rows = compute_features_for_browser(_current_retina, FEATURE_SET)
globals()['_current_retina'] = _current_retina
globals()['_current_feature_set'] = _current_feature_set
to_js({'biomarkers': rows}, dict_converter=js.Object.fromEntries)
`);

  try {
    const result = toPlainObject(proxy) as { biomarkers: FeatureExtractionResult['biomarkers'] };
    return { id: message.id, featureSet: message.featureSet, biomarkers: result.biomarkers, durationMs: performance.now() - started };
  } finally {
    destroyProxy(proxy);
    await cleanupExtractionGlobals();
    for (const path of Object.values(paths)) {
      try {
        pyodide!.FS.unlink(path);
      } catch {
        // Ignore cleanup failures in the worker filesystem.
      }
    }
  }
}

async function cleanupExtractionGlobals() {
  await pyodide!.runPythonAsync(`
import gc
for _name in [
    'fundus', 'av', 'vessels', 'disc', 'bounds', 'rows', 'mask_shape',
    'retina_from_arrays', 'compute_features_for_browser'
]:
    globals().pop(_name, None)
globals().pop('_name', None)
gc.collect()
`);
}

async function renderFeaturePlot(key: string): Promise<string> {
  assertReady();
  pyodide!.globals.set('FEATURE_KEY', key);
  progress('Rendering biomarker plot', 0.5);
  return await pyodide!.runPythonAsync(`
plot_feature_data_uri = _vascx_browser_helpers['plot_feature_data_uri']
plot_feature_data_uri(_current_retina, _current_feature_set, FEATURE_KEY)
`);
}

async function disposeCurrentRetina() {
  assertReady();
  await pyodide!.runPythonAsync(`
import gc
globals()['_current_retina'] = None
globals()['_current_feature_set'] = None
gc.collect()
`);
}

function assertReady() {
  if (!pyodide || !ready) {
    throw new Error('Biomarker extraction worker has not been initialised.');
  }
}

function toPlainObject(value: unknown): unknown {
  if (value && typeof value === 'object' && 'toJs' in value && typeof (value as { toJs: () => unknown }).toJs === 'function') {
    return (value as { toJs: () => unknown }).toJs();
  }
  return value;
}

function destroyProxy(value: unknown) {
  if (value && typeof value === 'object' && 'destroy' in value && typeof (value as { destroy: () => void }).destroy === 'function') {
    (value as { destroy: () => void }).destroy();
  }
}

function progress(message: string, progress?: number) {
  post({ type: 'progress', stage: 'features', message, progress });
}

function post(message: WorkerResponse) {
  (self as unknown as { postMessage: (message: WorkerResponse) => void }).postMessage(message);
}

const BROWSER_HELPERS_PY = String.raw`
from __future__ import annotations

from typing import Any, Dict
import base64
import io
import json

import matplotlib
matplotlib.use('Agg', force=True)
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

from vascx.fundus.feature_sets import fs_bergmann, fs_full, fs_full_v2, fs_full_v3, fs_od_centered, fs_sparsity  # noqa: F401
from rtnls_enface.bounds import make_roi_mask_from_bounds
from vascx.fundus.features.base import LayerFeature, RetinaFeature, VesselsLayerFeature
from vascx.fundus.layer import VesselTreeLayer
from vascx.fundus.retina import Retina
from vascx.fundus.vessels_layer import FundusVesselsLayer
from vascx.shared.features import FeatureSet


def _as_uint8_mask(mask):
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def _largest_binary_component(mask):
    binary = np.asarray(mask) > 0
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.uint8)

    labels, num_features = ndi.label(binary)
    if num_features <= 1:
        return binary.astype(np.uint8)

    sizes = ndi.sum(binary, labels, index=np.arange(1, num_features + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return (labels == largest_label).astype(np.uint8)


def _normalize_disc_mask(disc_mask):
    return _largest_binary_component(_as_uint8_mask(disc_mask))


def _normalize_retina_disc(retina):
    if retina.disc is not None and retina.disc.mask is not None:
        retina.disc.mask = (np.asarray(retina.disc.mask) > 0).astype(np.uint8)
    return retina


def _av_layers_from_label_mask(av_mask):
    av = _as_uint8_mask(av_mask)
    return {
        'arteries': np.logical_or(av == 1, av == 3),
        'veins': np.logical_or(av == 2, av == 3),
    }


def retina_from_arrays(fundus, av_mask, vessels_mask, disc_mask, fovea_location, bounds=None, roi_mask=None, id=None, mm_per_pixel=None, scaling_factor=None):
    fundus_array = np.asarray(fundus).astype(np.uint8)
    if roi_mask is None and bounds is not None:
        roi_mask = make_roi_mask_from_bounds(bounds, target_diameter=int(fundus_array.shape[0]))
        bounds = None

    av_layers = _av_layers_from_label_mask(av_mask)
    layers = {
        'arteries': VesselTreeLayer('arteries', av_layers['arteries'], color=(1, 0, 0)),
        'veins': VesselTreeLayer('veins', av_layers['veins'], color=(0, 0, 1)),
        'vessels': FundusVesselsLayer('vessels', _as_uint8_mask(vessels_mask) > 0),
    }
    retina = Retina(
        disc_path_or_mask=_normalize_disc_mask(disc_mask),
        fundus_path_or_mask=fundus_array,
        layers=layers,
        fovea_location=fovea_location,
        bounds=bounds,
        roi_mask=roi_mask,
        mm_per_pixel=mm_per_pixel if mm_per_pixel is not None else scaling_factor,
        id=id,
    )
    return _normalize_retina_disc(retina)


def _feature_set(name):
    feature_set = FeatureSet.get_by_name(name)
    if feature_set is None:
        raise ValueError(f"Unknown biomarker set '{name}'")
    return feature_set


def feature_set_manifest():
    manifest = []
    for name, feature_set in sorted(FeatureSet._registry.items()):
        biomarkers = []
        for feature in feature_set:
            targets = []
            if isinstance(feature, RetinaFeature):
                targets = ['retina']
            elif isinstance(feature, LayerFeature):
                targets = ['arteries', 'veins']
            elif isinstance(feature, VesselsLayerFeature):
                targets = ['vessels']
            for target_name in targets:
                biomarkers.append({'key': feature.canonical_name(layer_name=target_name), 'name': feature.display_name(layer_name=target_name)})
        manifest.append({'name': name, 'displayName': name.replace('_', ' ').title(), 'biomarkers': biomarkers})
    return manifest


def compute_features_for_browser(retina, feature_set_name):
    feature_set = _feature_set(feature_set_name)
    values = retina.calc_features(feature_set, raise_on_error=False)
    display_names = Retina.make_feature_display_names(feature_set)
    rows = []
    for key, value in values.items():
        if hasattr(value, 'item'):
            value = value.item()
        rows.append({'key': key, 'name': display_names.get(key, key), 'value': value})
    return rows


def _feature_targets(retina, feature_set):
    for feature in feature_set:
        if isinstance(feature, RetinaFeature):
            yield feature.canonical_name(layer_name='retina'), feature, retina
        elif isinstance(feature, LayerFeature):
            for layer in (retina.arteries, retina.veins):
                yield feature.canonical_name(layer_name=layer.name), feature, layer
        elif isinstance(feature, VesselsLayerFeature):
            yield feature.canonical_name(layer_name=retina.vessels.name), feature, retina.vessels


def plot_feature_data_uri(retina, feature_set_name, canonical_name):
    feature_set = _feature_set(feature_set_name)
    for key, feature, target in _feature_targets(retina, feature_set):
        if key == canonical_name:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=128)
            try:
                if isinstance(feature, RetinaFeature):
                    feature.plot(ax=ax, retina=target)
                else:
                    feature.plot(ax=ax, layer=target)
                ax.set_axis_off()
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=128, bbox_inches=None, pad_inches=0.0)
                encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
                return f'data:image/png;base64,{encoded}'
            finally:
                plt.close(fig)
    raise KeyError(f"Biomarker '{canonical_name}' is not part of biomarker set '{feature_set_name}'")

_vascx_browser_helpers = {
    'retina_from_arrays': retina_from_arrays,
    'feature_set_manifest': feature_set_manifest,
    'compute_features_for_browser': compute_features_for_browser,
    'plot_feature_data_uri': plot_feature_data_uri,
}
`;
