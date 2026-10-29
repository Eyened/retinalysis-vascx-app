import { afterEach, describe, expect, it, vi } from 'vitest';
import config from '../public/config.json';
import { configureFeatureSets, loadAppConfig, parseAppConfig, resolveFeatureSetName } from '../src/lib/config';
import type { FeatureSetSummary } from '../src/lib/types/pipeline';

const available: FeatureSetSummary[] = [
  'full_v3', 'od_centered_narrow', 'macula_centered', 'od_centered'
].map((name) => ({ name, displayName: name.replaceAll('_', ' '), description: name, biomarkers: [] }));

afterEach(() => vi.unstubAllGlobals());

describe('biomarker set configuration', () => {
  it('uses only configured sets, in configuration order, with their original metadata', () => {
    const configured = configureFeatureSets(parseAppConfig(config), available);
    expect(configured.defaultFeatureSet).toBe('od_centered');
    expect(configured.featureSets.map((set) => set.name)).toEqual([
      'od_centered', 'macula_centered', 'od_centered_narrow'
    ]);
    expect(configured.featureSets[0]).toBe(available[3]);
  });

  it('supports package versions that use display names as registered names', () => {
    const namedSets = available.map((set) => ({ ...set, name: set.displayName }));
    expect(configureFeatureSets(parseAppConfig(config), namedSets).defaultFeatureSet).toBe('od centered');
  });

  it.each([
    null, [], {},
    { featureSets: [], defaultFeatureSet: 'od_centered' },
    { featureSets: [''], defaultFeatureSet: '' },
    { featureSets: [42], defaultFeatureSet: 'od_centered' },
    { featureSets: ['od_centered', 'OD Centered'], defaultFeatureSet: 'od_centered' },
    { featureSets: ['od_centered'], defaultFeatureSet: 'macula_centered' }
  ])('rejects invalid configuration: %j', (value) => {
    expect(() => parseAppConfig(value)).toThrow('config.json');
  });

  it('reports a configured set missing from the loaded package', () => {
    expect(() => configureFeatureSets(parseAppConfig(config), available.slice(1, 3)))
      .toThrow('unknown biomarker set "od_centered"');
  });

  it('restores allowed selections and falls back for excluded selections', () => {
    const { featureSets, defaultFeatureSet } = configureFeatureSets(parseAppConfig(config), available);
    expect(resolveFeatureSetName(featureSets, 'Macula Centered', defaultFeatureSet)).toBe('macula_centered');
    expect(resolveFeatureSetName(featureSets, 'Full V3', defaultFeatureSet)).toBe('od_centered');
    expect(resolveFeatureSetName([], 'Full V3', '')).toBeUndefined();
  });

  it('loads config.json under the app base path', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(config)));
    vi.stubGlobal('fetch', fetch);
    expect(await loadAppConfig()).toEqual(config);
    expect(fetch).toHaveBeenCalledWith(`${import.meta.env.BASE_URL}config.json`);
  });

  it('reports a missing configuration instead of exposing every set', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 404 })));
    await expect(loadAppConfig()).rejects.toThrow('Could not load config.json: 404');
  });
});
