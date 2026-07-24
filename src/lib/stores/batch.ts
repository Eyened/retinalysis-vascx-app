import { get, set, del, keys } from 'idb-keyval';
import { writable } from 'svelte/store';
import type { BatchRecord, BiomarkerRow } from '$lib/types/pipeline';

const LEGACY_CACHE_KEY = 'vascx-batch-cache-v1';
const CACHE_META_KEY = 'vascx-batch-cache-v2:meta';
const RESULT_PREFIX = 'vascx-batch-cache-v2:result:';

export interface BatchResultRecord {
  id: string;
  relativePath?: string;
  biomarkers: BiomarkerRow[];
}

export interface BatchCache {
  sourceSignature: string;
  sourceHash?: string;
  featureSet?: string;
  createdAt: string;
  downloaded: boolean;
  records: BatchRecord[];
  results: BatchResultRecord[];
}

export interface BatchState {
  sourceName?: string;
  segmentationFolderName?: string;
  files: File[];
  records: BatchRecord[];
  results: BatchResultRecord[];
  running: boolean;
  downloaded: boolean;
  currentName?: string;
}

export const batchState = writable<BatchState>({
  files: [],
  records: [],
  results: [],
  running: false,
  downloaded: true
});

export async function loadBatchCache(): Promise<BatchCache | undefined> {
  const meta = (await get(CACHE_META_KEY)) as Omit<BatchCache, 'results'> | undefined;
  if (meta) {
    const sourceHash = meta.sourceHash ?? hashString(meta.sourceSignature);
    const resultKeys = (await keys()).filter((key): key is string => typeof key === 'string' && key.startsWith(resultPrefix(sourceHash)));
    const results = (await Promise.all(resultKeys.map((key) => get(key)))).filter(Boolean) as BatchResultRecord[];
    results.sort((a, b) => (a.relativePath ?? a.id).localeCompare(b.relativePath ?? b.id));
    return { ...meta, sourceHash, results };
  }

  const legacy = (await get(LEGACY_CACHE_KEY)) as BatchCache | undefined;
  if (!legacy) return undefined;
  const sourceHash = hashString(legacy.sourceSignature);
  await saveBatchCache({ ...legacy, sourceHash });
  await del(LEGACY_CACHE_KEY);
  return { ...legacy, sourceHash };
}

export async function saveBatchCache(cache: BatchCache): Promise<void> {
  const sourceHash = cache.sourceHash ?? hashString(cache.sourceSignature);
  await clearResultKeys(sourceHash);
  await Promise.all(cache.results.map((result) => saveBatchResult(sourceHash, result)));
  const { results: _results, ...meta } = cache;
  await saveBatchMeta({ ...meta, sourceHash });
}

export async function saveBatchMeta(cache: Omit<BatchCache, 'results'>): Promise<void> {
  const sourceHash = cache.sourceHash ?? hashString(cache.sourceSignature);
  await set(CACHE_META_KEY, { ...cache, sourceHash });
}

export async function saveBatchResult(sourceHash: string, result: BatchResultRecord): Promise<void> {
  await set(resultKey(sourceHash, result.relativePath ?? result.id), result);
}

export async function clearBatchCache(): Promise<void> {
  const allKeys = await keys();
  await Promise.all(
    allKeys
      .filter((key): key is string => typeof key === 'string' && (key === CACHE_META_KEY || key === LEGACY_CACHE_KEY || key.startsWith(RESULT_PREFIX)))
      .map((key) => del(key))
  );
}

export function sourceSignature(files: File[]): string {
  return files
    .map((file) => `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`)
    .sort()
    .join('|');
}

export function sourceHashForSignature(signature: string): string {
  return hashString(signature);
}

async function clearResultKeys(sourceHash: string): Promise<void> {
  const prefix = resultPrefix(sourceHash);
  const allKeys = await keys();
  await Promise.all(allKeys.filter((key): key is string => typeof key === 'string' && key.startsWith(prefix)).map((key) => del(key)));
}

function resultPrefix(sourceHash: string): string {
  return `${RESULT_PREFIX}${sourceHash}:`;
}

function resultKey(sourceHash: string, id: string): string {
  return `${resultPrefix(sourceHash)}${encodeURIComponent(id)}`;
}

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}
