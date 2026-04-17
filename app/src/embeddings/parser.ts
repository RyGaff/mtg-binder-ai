import { File, Paths } from 'expo-file-system';
import { Buffer } from 'buffer';

// Lazy getters — File instances require native module, can't be created at module level in tests
export const getEmbeddingsFile = () => new File(Paths.document, 'embeddings.bin');
export const getVersionFile = () => new File(Paths.document, 'embeddings_version.txt');

export type EmbeddingMap = Map<string, Float32Array>;

/**
 * Both lookup indexes built from the binary file:
 * - byId: scryfall_id → normalized vector
 * - byName: card name → scryfall_id (for resolving a different printing of the same card)
 */
export type EmbeddingIndex = {
  byId: EmbeddingMap;
  byName: Map<string, string>;
};

let cachedIndex: EmbeddingIndex | null = null;
let pendingLoad: Promise<EmbeddingIndex> | null = null;

/** Clear the in-memory cache — call after downloading a new file. */
export function clearEmbeddingCache(): void {
  cachedIndex = null;
  pendingLoad = null;
}

/** Return the parsed embedding index, loading from disk on first call. */
export async function getEmbeddingMap(): Promise<EmbeddingIndex> {
  if (cachedIndex !== null) return cachedIndex;
  if (pendingLoad !== null) return pendingLoad;
  pendingLoad = loadEmbeddingFile().then(index => {
    cachedIndex = index;
    pendingLoad = null;
    return index;
  }).catch(err => {
    pendingLoad = null; // allow retry on next call
    throw err;
  });
  return pendingLoad;
}

async function loadEmbeddingFile(): Promise<EmbeddingIndex> {
  const file = getEmbeddingsFile();
  console.log('[embeddings] reading from:', file.uri);
  const arrayBuffer = await file.arrayBuffer();
  console.log('[embeddings] read', arrayBuffer.byteLength, 'bytes');
  const index = parseEmbeddingBuffer(arrayBuffer);
  console.log('[embeddings] parsed — byId:', index.byId.size, 'byName:', index.byName.size);
  return index;
}

/**
 * Parse a binary embedding buffer into both lookup indexes.
 *
 * Format: [uint32 N][uint32 D][N × (36-byte scryfall_id + 64-byte name + D × float32)]
 * All vectors are L2-normalized at parse time so similarity = dot product.
 * Exported for testing.
 */
export function parseEmbeddingBuffer(buffer: ArrayBuffer): EmbeddingIndex {
  const view = new DataView(buffer);
  const n = view.getUint32(0, true);
  const d = view.getUint32(4, true);
  const recordSize = 36 + 64 + d * 4;
  const byId: EmbeddingMap = new Map();
  const byName: Map<string, string> = new Map();

  for (let i = 0; i < n; i++) {
    const base = 8 + i * recordSize;

    const idBytes = new Uint8Array(buffer, base, 36);
    const id = String.fromCharCode(...idBytes).replace(/\0/g, '');

    const nameBytes = new Uint8Array(buffer, base + 36, 64);
    const name = String.fromCharCode(...nameBytes).replace(/\0/g, '');

    const raw = new Float32Array(buffer, base + 100, d);
    byId.set(id, normalize(raw));
    if (name) byName.set(name, id);
  }

  return { byId, byName };
}

/** L2-normalize a Float32Array into a new Float32Array. Does not mutate input. */
export function normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = norm > 0 ? v[i] / norm : 0;
  return out;
}
