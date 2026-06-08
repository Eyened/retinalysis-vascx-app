import { describe, expect, it } from 'vitest';
import { biomarkerRowsToCsv } from '../src/lib/utils/files';
import { flipTensorNchw, heatmapArgmax, makeBlendWeights, makeSlidingWindowTiles, softmaxArgmax } from '../src/lib/utils/image';

describe('image inference utilities', () => {
  it('creates sliding-window tiles that include the image edge', () => {
    const tiles = makeSlidingWindowTiles(1024, 1024, [512, 512], 0.5);
    expect(tiles[0]).toEqual({ x: 0, y: 0, width: 512, height: 512 });
    expect(tiles.at(-1)).toEqual({ x: 512, y: 512, width: 512, height: 512 });
    expect(tiles).toHaveLength(9);
  });

  it('uses non-zero gaussian blend weights', () => {
    const weights = makeBlendWeights(4, 4, 'gaussian');
    expect(Math.min(...weights)).toBeGreaterThan(0);
    expect(weights[5]).toBeGreaterThan(weights[0]);
  });

  it('flips NCHW tensors over spatial axes', () => {
    const tensor = new Float32Array([1, 2, 3, 4]);
    const flipped = flipTensorNchw(tensor, 1, 2, 2, [2, 3]);
    expect(Array.from(flipped)).toEqual([4, 3, 2, 1]);
  });

  it('selects class argmax per pixel', () => {
    const logits = new Float32Array([0.1, 0.9, 0.8, 0.2]);
    expect(Array.from(softmaxArgmax(logits, 2, 2, 1))).toEqual([1, 0]);
  });

  it('extracts fovea position at heatmap maximum with half-pixel offset', () => {
    const heatmap = new Float32Array([0, 1, 0, 5, 2, 1]);
    expect(heatmapArgmax(heatmap, 3, 2)).toEqual([0.5, 1.5]);
  });
});

describe('CSV export', () => {
  it('escapes biomarker values', () => {
    const csv = biomarkerRowsToCsv([
      { id: 'sample', biomarkers: [{ key: 'a', name: 'A', value: 'value, with comma' }] }
    ]);
    expect(csv).toBe('id,a\nsample,"value, with comma"');
  });
});
