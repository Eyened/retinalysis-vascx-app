import type { FeatureSetSummary } from '$lib/types/pipeline';

// Edit public/config.json to choose the available sets, their order, and the default.
export interface AppConfig {
  defaultFeatureSet: string;
  featureSets: string[];
}

export async function loadAppConfig(): Promise<AppConfig> {
  const response = await fetch(`${import.meta.env.BASE_URL}config.json`);
  if (!response.ok) {
    throw new Error(`Could not load config.json: ${response.status} ${response.statusText}`);
  }
  return parseAppConfig(await response.json());
}

export function parseAppConfig(value: unknown): AppConfig {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('config.json must contain an object.');
  }
  const { featureSets, defaultFeatureSet } = value as Partial<AppConfig>;
  if (!Array.isArray(featureSets) || !featureSets.length ||
      !featureSets.every((name) => typeof name === 'string' && name.trim())) {
    throw new Error('config.json: featureSets must be a nonempty list of biomarker set names.');
  }
  const names = featureSets.map(normalizeFeatureSetName);
  if (new Set(names).size !== names.length) {
    throw new Error('config.json: featureSets must not contain duplicate names.');
  }
  if (typeof defaultFeatureSet !== 'string' || !names.includes(normalizeFeatureSetName(defaultFeatureSet))) {
    throw new Error('config.json: defaultFeatureSet must name a set in featureSets.');
  }
  return { featureSets, defaultFeatureSet };
}

export function configureFeatureSets(config: AppConfig, available: FeatureSetSummary[]) {
  const featureSets = config.featureSets.map((name) => {
    const match = findFeatureSet(available, name);
    if (!match) {
      throw new Error(`config.json: unknown biomarker set "${name}" in the loaded VascX package.`);
    }
    return match;
  });
  const defaultFeatureSet = findFeatureSet(featureSets, config.defaultFeatureSet)?.name;
  if (!defaultFeatureSet) {
    throw new Error('config.json: the default biomarker set is unavailable.');
  }
  return { featureSets, defaultFeatureSet };
}

export function resolveFeatureSetName(sets: FeatureSetSummary[], preferred: string, fallback: string) {
  return findFeatureSet(sets, preferred)?.name ?? findFeatureSet(sets, fallback)?.name ?? sets[0]?.name;
}

function findFeatureSet(sets: FeatureSetSummary[], name: string) {
  const normalized = normalizeFeatureSetName(name);
  return sets.find((set) => set.name === name) ?? sets.find((set) =>
    normalizeFeatureSetName(set.name) === normalized || normalizeFeatureSetName(set.displayName) === normalized
  );
}

function normalizeFeatureSetName(value: string) {
  return value.trim().toLowerCase().replace(/[ _-]+/g, '');
}
