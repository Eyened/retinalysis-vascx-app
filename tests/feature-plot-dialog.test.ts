import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import App from '../src/App.svelte';

const { request, plotRequest } = vi.hoisted(() => ({ request: vi.fn(), plotRequest: vi.fn() }));
vi.mock('$lib/workers/preprocess.worker?worker', () => ({ default: class {} }));
vi.mock('$lib/workers/inference.worker?worker', () => ({ default: class {} }));
vi.mock('$lib/workers/feature.worker?worker', () => ({ default: class {} }));
vi.mock('$lib/workers/client', () => ({
  PipelineWorkerClient: class { request = request; terminate = vi.fn(); }
}));
vi.mock('$lib/stores/batch', async (original) => ({
  ...await original<typeof import('../src/lib/stores/batch')>(),
  loadBatchCache: vi.fn().mockResolvedValue(undefined)
}));
vi.mock('$lib/utils/files', async (original) => ({
  ...await original<typeof import('../src/lib/utils/files')>(),
  fileToBytes: vi.fn().mockResolvedValue(new Uint8Array([0])),
  bytesToObjectUrl: vi.fn().mockReturnValue('blob:preview')
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL() { return 'blob:preview'; }
    static revokeObjectURL() {}
  });
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
    featureSets: ['od_centered'], defaultFeatureSet: 'od_centered'
  }))));
  request.mockImplementation(async (message: { type: string }) => {
    switch (message.type) {
      case 'init': return { backend: 'webgpu' };
      case 'loadFeatureSets': return [{ name: 'od_centered', displayName: 'OD centered', biomarkers: [] }];
      case 'preprocessImage': return { id: 'sample', rgbPng: new Uint8Array([0]), boundsPng: new Uint8Array([0]) };
      case 'runInference': return { width: 1, height: 1, overlayPng: new Uint8Array([0]), fovea: [0, 0] };
      case 'extractFeatures': return { biomarkers: [{ key: 'background_lab_l', name: 'Background lightness', value: 50 }] };
      case 'renderFeaturePlot': return plotRequest();
    }
  });
});
afterEach(() => vi.unstubAllGlobals());

async function openPlot() {
  const { container } = render(App);
  await screen.findByText('Pipeline ready');
  await fireEvent.change(container.querySelector('input[type="file"]')!, {
    target: { files: [new File(['image'], 'sample.png', { type: 'image/png' })] }
  });
  const run = await screen.findByRole('button', { name: 'Run VascX' });
  await waitFor(() => expect((run as HTMLButtonElement).disabled).toBe(false));
  await fireEvent.click(run);
  await fireEvent.click(await screen.findByTitle('Open biomarker visualisation'));
}

it('shows the returned plot for a component output', async () => {
  plotRequest.mockResolvedValue('data:image/png;base64,cGxvdA==');
  await openPlot();
  const image = await screen.findByAltText('Background lightness');
  expect(image.getAttribute('src')).toBe('data:image/png;base64,cGxvdA==');
  expect(request).toHaveBeenCalledWith({ type: 'renderFeaturePlot', key: 'background_lab_l' });
});

it('shows a plotting error instead of leaving the loading indicator', async () => {
  plotRequest.mockRejectedValue(new Error('Plot unavailable'));
  await openPlot();
  expect((await screen.findByRole('alert')).textContent).toContain('Plot unavailable');
  expect(screen.queryByText('Rendering plot')).toBeNull();
});

it('does not reopen a closed dialog when rendering finishes', async () => {
  let finish!: (value: string) => void;
  plotRequest.mockReturnValue(new Promise<string>((resolve) => { finish = resolve; }));
  await openPlot();
  const dialog = await screen.findByRole('dialog');
  await fireEvent.click(dialog.querySelector('button.close')!);
  finish('data:image/png;base64,cGxvdA==');
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
});
