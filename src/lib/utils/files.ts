import type { BiomarkerRow } from '$lib/types/pipeline';

export const VALID_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'];

export function isValidImageName(name: string): boolean {
  const lower = name.toLowerCase();
  return VALID_IMAGE_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

export async function fileToBytes(file: File): Promise<Uint8Array> {
  return new Uint8Array(await file.arrayBuffer());
}

export function bytesToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

export function bytesToObjectUrl(bytes: Uint8Array, mimeType = 'image/png'): string {
  return URL.createObjectURL(new Blob([bytesToArrayBuffer(bytes)], { type: mimeType }));
}

export function downloadBytes(bytes: Uint8Array, filename: string, mimeType = 'image/png') {
  const url = bytesToObjectUrl(bytes, mimeType);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function biomarkerRowsToCsv(rows: Array<{ id?: string; biomarkers: BiomarkerRow[] }>): string {
  const columns = Array.from(
    new Set(rows.flatMap((row) => row.biomarkers.map((biomarker) => biomarker.name)))
  );
  const header = ['id', ...columns];
  const lines = [header.map(csvEscape).join(',')];

  for (const row of rows) {
    const values = new Map(row.biomarkers.map((biomarker) => [biomarker.name, biomarker.value]));
    lines.push([row.id ?? '', ...columns.map((column) => values.get(column) ?? '')].map(csvEscape).join(','));
  }

  return lines.join('\n');
}

function csvEscape(value: unknown): string {
  const text = value == null ? '' : String(value);
  if (!/[",\n]/.test(text)) {
    return text;
  }
  return `"${text.replace(/"/g, '""')}"`;
}
