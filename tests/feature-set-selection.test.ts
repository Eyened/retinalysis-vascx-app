import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import App from '../src/App.svelte';
import config from '../public/config.json';
import { clearBatchCache, loadBatchCache, type BatchCache } from '../src/lib/stores/batch';
import { downloadBytes } from '../src/lib/utils/files';

const { request } = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock('$lib/workers/preprocess.worker?worker', () => ({ default: class {} }));
vi.mock('$lib/workers/inference.worker?worker', () => ({ default: class {} }));
vi.mock('$lib/workers/feature.worker?worker', () => ({ default: class {} }));
vi.mock('$lib/workers/client', () => ({
  PipelineWorkerClient: class {
    request = request;
    terminate = vi.fn();
  }
}));
vi.mock('$lib/stores/batch', async (importOriginal) => ({
  ...await importOriginal<typeof import('../src/lib/stores/batch')>(),
  loadBatchCache: vi.fn(),
  clearBatchCache: vi.fn(),
  saveBatchMeta: vi.fn()
}));
vi.mock('$lib/utils/files', async (importOriginal) => ({
  ...await importOriginal<typeof import('../src/lib/utils/files')>(),
  downloadBytes: vi.fn()
}));

function serveConfig(value: unknown = config) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => new Response(
    url.endsWith('config.json') ? JSON.stringify(value) : ''
  )));
}

function selectedOptions() {
  const select = screen.getByRole('combobox', { name: 'Biomarker set' }) as HTMLSelectElement;
  return { value: select.value, options: Array.from(select.options, (option) => option.value) };
}

function cacheFor(featureSet: string): BatchCache {
  return {
    featureSet, sourceSignature: 'saved', createdAt: '2026-10-24T00:00:00Z', downloaded: true,
    records: [{ id: 'image', name: 'image.png', relativePath: 'image.png', status: 'done' }],
    results: [{ id: 'image', biomarkers: [{ key: 'legacy', name: 'Legacy', value: 42 }] }]
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(loadBatchCache).mockResolvedValue(undefined);
  request.mockImplementation(async (message: { type: string }) => {
    if (message.type === 'loadFeatureSets') {
      return ['full_v3', 'macula_centered', 'od_centered_narrow', 'od_centered'].map((name) => ({
        name, displayName: name.replaceAll('_', ' '), description: `${name} description`, biomarkers: []
      }));
    }
    if (message.type === 'init') return { backend: 'webgpu' };
  });
  serveConfig();
});

afterEach(() => vi.unstubAllGlobals());

describe('app biomarker selectors', () => {
  it('shows the configured choices and default on both screens', async () => {
    render(App);
    await screen.findByText('Pipeline ready');
    expect(selectedOptions()).toEqual({ value: 'od_centered', options: config.featureSets });
    expect(screen.getByText('od_centered description')).toBeTruthy();

    await fireEvent.click(screen.getByRole('button', { name: 'Batch inference' }));
    expect(selectedOptions()).toEqual({ value: 'od_centered', options: config.featureSets });
  });

  it('honors changes to config order and default instead of model defaults', async () => {
    const changed = { featureSets: ['od_centered_narrow', 'macula_centered'], defaultFeatureSet: 'macula_centered' };
    serveConfig(changed);
    render(App);
    await screen.findByText('Pipeline ready');
    expect(selectedOptions()).toEqual({ value: 'macula_centered', options: changed.featureSets });
    await fireEvent.click(screen.getByRole('button', { name: 'Batch inference' }));
    expect(selectedOptions()).toEqual({ value: 'macula_centered', options: changed.featureSets });
  });

  it('restores an allowed cached selection', async () => {
    vi.mocked(loadBatchCache).mockResolvedValue(cacheFor('macula_centered'));
    render(App);
    await screen.findByText('Pipeline ready');
    await fireEvent.click(screen.getByRole('button', { name: 'Batch inference' }));
    await waitFor(() => expect(selectedOptions().value).toBe('macula_centered'));
    expect(screen.getByText(/Partial results from a previous run/)).toBeTruthy();
  });

  it('keeps excluded cached results labeled and downloadable without adding their set', async () => {
    vi.mocked(loadBatchCache).mockResolvedValue(cacheFor('Full V3'));
    render(App);
    await screen.findByText('Pipeline ready');
    await fireEvent.click(screen.getByRole('button', { name: 'Batch inference' }));
    await screen.findByText('Biomarker set: Full V3');
    expect(selectedOptions()).toEqual({ value: 'od_centered', options: config.featureSets });
    expect(screen.queryByText(/Partial results from a previous run/)).toBeNull();
    expect(clearBatchCache).not.toHaveBeenCalled();
    const csv = screen.getByRole('button', { name: 'CSV' }) as HTMLButtonElement;
    expect(csv.disabled).toBe(false);
    await fireEvent.click(csv);
    expect(downloadBytes).toHaveBeenCalled();
    const bytes = vi.mocked(downloadBytes).mock.calls[0][0];
    expect(new TextDecoder().decode(bytes)).toContain('image,42');
  });

  it('displays configuration errors and keeps both selectors empty', async () => {
    serveConfig({ featureSets: ['unknown'], defaultFeatureSet: 'unknown' });
    render(App);
    await screen.findByText(/unknown biomarker set "unknown"/);
    expect(selectedOptions().options).toEqual([]);
    expect((screen.getByRole('combobox') as HTMLSelectElement).disabled).toBe(true);
    expect(screen.queryByText('Pipeline ready')).toBeNull();
    await fireEvent.click(screen.getByRole('button', { name: 'Batch inference' }));
    expect((screen.getByRole('combobox') as HTMLSelectElement).disabled).toBe(true);
    expect(screen.queryByText('Pipeline ready')).toBeNull();
  });
});
