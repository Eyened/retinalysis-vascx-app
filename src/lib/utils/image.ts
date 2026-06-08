import type { InferenceConfig } from '$lib/types/pipeline';
import { bytesToArrayBuffer } from '$lib/utils/files';

export interface DecodedImage {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

export interface TensorImage {
  width: number;
  height: number;
  channels: number;
  data: Float32Array;
}

export interface Tile {
  x: number;
  y: number;
  width: number;
  height: number;
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export async function decodeImageBytes(bytes: Uint8Array): Promise<DecodedImage> {
  const blob = new Blob([bytesToArrayBuffer(bytes)]);
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Could not create image canvas context.');
  }
  context.drawImage(bitmap, 0, 0);
  const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return { width: image.width, height: image.height, rgba: image.data };
}

export function imageToNormalizedTensor(image: DecodedImage, normalize: InferenceConfig['normalize']): TensorImage {
  const channels = 3;
  const data = new Float32Array(channels * image.width * image.height);
  const pixelCount = image.width * image.height;

  for (let index = 0; index < pixelCount; index += 1) {
    const rgbaIndex = index * 4;
    const rgb = [
      image.rgba[rgbaIndex] / 255,
      image.rgba[rgbaIndex + 1] / 255,
      image.rgba[rgbaIndex + 2] / 255
    ];

    for (let channel = 0; channel < channels; channel += 1) {
      let value = rgb[channel];
      if (normalize === 'imagenet') {
        value = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
      } else if (normalize === 'diffusion') {
        value = (value - 0.5) / 0.5;
      }
      data[channel * pixelCount + index] = value;
    }
  }

  return { width: image.width, height: image.height, channels, data };
}

export function makeSlidingWindowTiles(width: number, height: number, patch: [number, number], overlap: number): Tile[] {
  const [patchHeight, patchWidth] = patch;
  const strideX = Math.max(1, Math.floor(patchWidth * (1 - overlap)));
  const strideY = Math.max(1, Math.floor(patchHeight * (1 - overlap)));
  const xs = axisStarts(width, patchWidth, strideX);
  const ys = axisStarts(height, patchHeight, strideY);
  return ys.flatMap((y) => xs.map((x) => ({ x, y, width: patchWidth, height: patchHeight })));
}

function axisStarts(size: number, patch: number, stride: number): number[] {
  if (size <= patch) {
    return [0];
  }
  const starts: number[] = [];
  for (let value = 0; value < size - patch; value += stride) {
    starts.push(value);
  }
  starts.push(size - patch);
  return Array.from(new Set(starts));
}

export function extractTile(tensor: TensorImage, tile: Tile): Float32Array {
  const output = new Float32Array(tensor.channels * tile.width * tile.height);
  const sourcePlane = tensor.width * tensor.height;
  const tilePlane = tile.width * tile.height;

  for (let channel = 0; channel < tensor.channels; channel += 1) {
    for (let row = 0; row < tile.height; row += 1) {
      const sourceStart = channel * sourcePlane + (tile.y + row) * tensor.width + tile.x;
      const destStart = channel * tilePlane + row * tile.width;
      output.set(tensor.data.subarray(sourceStart, sourceStart + tile.width), destStart);
    }
  }

  return output;
}

export function makeBlendWeights(width: number, height: number, mode: InferenceConfig['blend']): Float32Array {
  const weights = new Float32Array(width * height);
  if (mode === 'constant') {
    weights.fill(1);
    return weights;
  }

  const sigmaX = width / 8;
  const sigmaY = height / 8;
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = (x - cx) / sigmaX;
      const dy = (y - cy) / sigmaY;
      weights[y * width + x] = Math.max(Math.exp(-0.5 * (dx * dx + dy * dy)), 1e-3);
    }
  }
  return weights;
}

export function softmaxArgmax(logits: Float32Array, classes: number, width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (let index = 0; index < width * height; index += 1) {
    let bestClass = 0;
    let bestValue = -Infinity;
    for (let klass = 0; klass < classes; klass += 1) {
      const value = logits[index * classes + klass];
      if (value > bestValue) {
        bestClass = klass;
        bestValue = value;
      }
    }
    mask[index] = bestClass;
  }
  return mask;
}

export function heatmapArgmax(heatmap: Float32Array, width: number, height: number): [number, number] {
  let maxIndex = 0;
  let maxValue = -Infinity;
  for (let index = 0; index < heatmap.length; index += 1) {
    if (heatmap[index] > maxValue) {
      maxValue = heatmap[index];
      maxIndex = index;
    }
  }
  const y = Math.floor(maxIndex / width);
  const x = maxIndex % width;
  return [x + 0.5, y + 0.5];
}

export function flipTensorNchw(data: Float32Array, channels: number, width: number, height: number, axes: number[]): Float32Array {
  const output = new Float32Array(data.length);
  const flipY = axes.includes(2);
  const flipX = axes.includes(3);
  const plane = width * height;

  for (let channel = 0; channel < channels; channel += 1) {
    for (let y = 0; y < height; y += 1) {
      const sourceY = flipY ? height - 1 - y : y;
      for (let x = 0; x < width; x += 1) {
        const sourceX = flipX ? width - 1 - x : x;
        output[channel * plane + y * width + x] = data[channel * plane + sourceY * width + sourceX];
      }
    }
  }

  return output;
}

export async function maskToPng(mask: Uint8Array, width: number, height: number, palette?: Array<[number, number, number]>): Promise<Uint8Array> {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Could not create mask canvas context.');
  }
  const imageData = new ImageData(width, height);
  for (let index = 0; index < mask.length; index += 1) {
    const color = palette?.[mask[index]] ?? [mask[index] > 0 ? 255 : 0, mask[index] > 0 ? 255 : 0, mask[index] > 0 ? 255 : 0];
    const offset = index * 4;
    imageData.data[offset] = color[0];
    imageData.data[offset + 1] = color[1];
    imageData.data[offset + 2] = color[2];
    imageData.data[offset + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return new Uint8Array(await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer());
}

export async function createOverlayPng(rgbPng: Uint8Array, avMask: Uint8Array, discMask: Uint8Array, fovea: [number, number]): Promise<Uint8Array> {
  const image = await decodeImageBytes(rgbPng);
  const canvas = new OffscreenCanvas(image.width, image.height);
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Could not create overlay canvas context.');
  }
  const data = new ImageData(new Uint8ClampedArray(image.rgba), image.width, image.height);
  for (let index = 0; index < avMask.length; index += 1) {
    const offset = index * 4;
    if (avMask[index] === 1 || avMask[index] === 3) {
      data.data[offset] = 255;
      data.data[offset + 1] = 0;
      data.data[offset + 2] = 0;
    }
    if (avMask[index] === 2 || avMask[index] === 3) {
      data.data[offset] = 0;
      data.data[offset + 1] = 0;
      data.data[offset + 2] = 255;
    }
    if (discMask[index] > 0) {
      data.data[offset] = 255;
      data.data[offset + 1] = 255;
      data.data[offset + 2] = 255;
    }
  }
  context.putImageData(data, 0, 0);
  context.strokeStyle = '#facc15';
  context.lineWidth = 2;
  const size = Math.max(8, Math.round(Math.min(image.width, image.height) / 50));
  context.beginPath();
  context.moveTo(fovea[0] - size, fovea[1] - size);
  context.lineTo(fovea[0] + size, fovea[1] + size);
  context.moveTo(fovea[0] - size, fovea[1] + size);
  context.lineTo(fovea[0] + size, fovea[1] - size);
  context.stroke();
  return new Uint8Array(await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer());
}
