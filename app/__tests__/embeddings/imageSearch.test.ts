jest.mock('../../src/embeddings/parser', () => ({
  getImageEmbeddingMap: jest.fn(),
}));

jest.mock('../../src/embeddings/imageEncoder', () => ({
  encodeCardImage: jest.fn(),
}));

import { findCardByImage } from '../../src/embeddings/imageSearch';
import { getImageEmbeddingMap } from '../../src/embeddings/parser';
import { encodeCardImage } from '../../src/embeddings/imageEncoder';

function vec(values: number[]): Float32Array {
  const a = new Float32Array(values.length);
  values.forEach((v, i) => a[i] = v);
  let s = 0; a.forEach(x => s += x * x);
  const n = Math.sqrt(s);
  if (n > 0) for (let i = 0; i < a.length; i++) a[i] /= n;
  return a;
}

function makeIndex(
  version: 1 | 2,
  dim: number,
  entries: Array<{ id: string; vec: Float32Array }>,
) {
  const n = entries.length;
  const ids: string[] = new Array(n);
  const vectors = new Float32Array(n * dim);
  const idIndex = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    ids[i] = entries[i].id;
    idIndex.set(entries[i].id, i);
    vectors.set(entries[i].vec, i * dim);
  }
  return { version, dim, modelHash: 0, size: n, ids, vectors, idIndex, byName: new Map() };
}

beforeEach(() => {
  jest.clearAllMocks();
});

it('returns null when the encoder returns null (not ready)', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(null);
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});

it('returns null when the embeddings file throws', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockRejectedValue(new Error('file not found'));
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});

it('returns null when the embeddings index is empty', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue(makeIndex(2, 4, []));
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});

it('returns null when version is not 2 (text embeddings loaded by mistake)', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue(
    makeIndex(1, 4, [{ id: 'x', vec: vec([1, 0, 0, 0]) }]),
  );
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});

it('returns the top-1 match with highest cosine similarity', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue(
    makeIndex(2, 4, [
      { id: 'card-a', vec: vec([0.95, 0.05, 0, 0]) },
      { id: 'card-b', vec: vec([0.5, 0.5, 0, 0]) },
      { id: 'card-c', vec: vec([0, 1, 0, 0]) },
    ]),
  );
  const result = await findCardByImage('file:///fake.jpg');
  expect(result?.scryfallId).toBe('card-a');
  expect(result?.score).toBeGreaterThan(0.99);
});

it('scores identical query and gallery vector at ~1.0', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue(
    makeIndex(2, 4, [{ id: 'same', vec: vec([1, 0, 0, 0]) }]),
  );
  const result = await findCardByImage('file:///fake.jpg');
  expect(result?.score).toBeCloseTo(1.0, 5);
  expect(result?.scryfallId).toBe('same');
});

it('returns a single-card match when the database has one entry', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue(
    makeIndex(2, 4, [{ id: 'only-card', vec: vec([1, 0, 0, 0]) }]),
  );
  const result = await findCardByImage('file:///fake.jpg');
  expect(result?.scryfallId).toBe('only-card');
});

it('returns null when encoder output dim != gallery dim (model mismatch)', async () => {
  const q = new Float32Array(256); q[0] = 1;
  (encodeCardImage as jest.Mock).mockResolvedValue(q);
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue(
    makeIndex(2, 4, [{ id: 'card-a', vec: vec([1, 0, 0, 0]) }]),
  );
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});
