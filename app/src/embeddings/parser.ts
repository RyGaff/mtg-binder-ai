import { File, Paths } from 'expo-file-system';

// Lazy getters — File instances require native module, can't be created at module level in tests
export const getEmbeddingsFile = () => new File(Paths.document, 'embeddings.bin');
export const getVersionFile = () => new File(Paths.document, 'embeddings_version.txt');
export const getImageEmbeddingsFile = () => new File(Paths.document, 'embeddings_image.bin');
export const getImageVersionFile = () => new File(Paths.document, 'embeddings_image_version.txt');

export type EmbeddingIndex = {
  version: 1 | 2;
  dim: number;
  modelHash: number;
  size: number;
  ids: string[];
  vectors: Float32Array;
  idIndex: Map<string, number>;
  byName: Map<string, string>;
};

const MAGIC_MTGE = 0x4D544745;

let cachedText: EmbeddingIndex | null = null;
let cachedImage: EmbeddingIndex | null = null;
let pendingText: Promise<EmbeddingIndex> | null = null;
let pendingImage: Promise<EmbeddingIndex> | null = null;

export function clearEmbeddingCache(): void {
  cachedText = null; pendingText = null;
  cachedImage = null; pendingImage = null;
}

export async function getEmbeddingMap(): Promise<EmbeddingIndex> {
  if (cachedText) return cachedText;
  if (pendingText) return pendingText;
  pendingText = loadFile(getEmbeddingsFile())
    .then((i) => { cachedText = i; pendingText = null; return i; })
    .catch((e) => { pendingText = null; throw e; });
  return pendingText;
}

export async function getImageEmbeddingMap(): Promise<EmbeddingIndex> {
  if (cachedImage) return cachedImage;
  if (pendingImage) return pendingImage;
  pendingImage = loadFile(getImageEmbeddingsFile())
    .then((i) => { cachedImage = i; pendingImage = null; return i; })
    .catch((e) => { pendingImage = null; throw e; });
  return pendingImage;
}

async function loadFile(file: File): Promise<EmbeddingIndex> {
  return parseEmbeddingBuffer(await file.arrayBuffer());
}

/**
 * v2 (image): magic 'MTGE' | version=2 | N | D | model_hash, then N × [36B id][D × f32]
 * v1 (text, legacy): N | D, then N × [36B id][64B name][D × f32]
 */
export function parseEmbeddingBuffer(buffer: ArrayBuffer): EmbeddingIndex {
  if (buffer.byteLength < 4) {
    throw new RangeError(`Embedding buffer too small: ${buffer.byteLength} bytes`);
  }
  const view = new DataView(buffer);
  return view.getUint32(0, false) === MAGIC_MTGE ? parseV2(buffer, view) : parseV1(buffer, view);
}

function readId(buffer: ArrayBuffer, base: number, len: number): string {
  return String.fromCharCode(...new Uint8Array(buffer, base, len)).replace(/\0/g, '');
}

function parseV2(buffer: ArrayBuffer, view: DataView): EmbeddingIndex {
  if (buffer.byteLength < 20) {
    throw new RangeError(`v2 buffer too small for header: ${buffer.byteLength} bytes`);
  }
  const version = view.getUint32(4, true);
  if (version !== 2) throw new RangeError(`Unsupported v2 sub-version: ${version}`);
  const n = view.getUint32(8, true);
  const dim = view.getUint32(12, true);
  const modelHash = view.getUint32(16, true);
  const recordSize = 36 + dim * 4;

  const ids: string[] = new Array(n);
  const vectors = new Float32Array(n * dim);
  const idIndex = new Map<string, number>();

  for (let i = 0; i < n; i++) {
    const base = 20 + i * recordSize;
    const id = readId(buffer, base, 36);
    normalizeInto(new Float32Array(buffer, base + 36, dim), vectors, i * dim);
    ids[i] = id;
    idIndex.set(id, i);
  }

  return { version: 2, dim, modelHash, size: n, ids, vectors, idIndex, byName: new Map() };
}

function parseV1(buffer: ArrayBuffer, view: DataView): EmbeddingIndex {
  const n = view.getUint32(0, true);
  const dim = view.getUint32(4, true);
  const recordSize = 36 + 64 + dim * 4;

  const ids: string[] = new Array(n);
  const vectors = new Float32Array(n * dim);
  const idIndex = new Map<string, number>();
  const byName = new Map<string, string>();

  for (let i = 0; i < n; i++) {
    const base = 8 + i * recordSize;
    const id = readId(buffer, base, 36);
    const name = readId(buffer, base + 36, 64);
    normalizeInto(new Float32Array(buffer, base + 100, dim), vectors, i * dim);
    ids[i] = id;
    idIndex.set(id, i);
    if (name) byName.set(name, id);
  }

  return { version: 1, dim, modelHash: 0, size: n, ids, vectors, idIndex, byName };
}

export function normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = norm > 0 ? v[i] / norm : 0;
  return out;
}

function normalizeInto(src: Float32Array, dst: Float32Array, offset: number): void {
  const n = src.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += src[i] * src[i];
  const norm = Math.sqrt(sum);
  if (norm > 0) {
    for (let i = 0; i < n; i++) dst[offset + i] = src[i] / norm;
  } else {
    for (let i = 0; i < n; i++) dst[offset + i] = 0;
  }
}
