jest.mock('../../src/embeddings/parser', () => ({
  getImageEmbeddingMap: jest.fn(),
}));

jest.mock('../../src/embeddings/imageEncoder', () => ({
  encodeCardImage: jest.fn(),
}));

import { findCardByImage } from '../../src/embeddings/imageSearch';
import { getImageEmbeddingMap } from '../../src/embeddings/parser';
import { encodeCardImage } from '../../src/embeddings/imageEncoder';

/** Build an L2-normalized Float32Array from values. */
function vec(values: number[]): Float32Array {
  const a = new Float32Array(values.length);
  values.forEach((v, i) => a[i] = v);
  let s = 0; a.forEach(x => s += x * x);
  const n = Math.sqrt(s);
  if (n > 0) for (let i = 0; i < a.length; i++) a[i] /= n;
  return a;
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

it('returns null when the embeddings map is empty', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
    version: 2, dim: 4, modelHash: 0, byId: new Map(), byName: new Map(),
  });
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});

it('returns null when version is not 2 (text embeddings loaded by mistake)', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
    version: 1, dim: 4, modelHash: 0,
    byId: new Map([['x', vec([1, 0, 0, 0])]]),
    byName: new Map(),
  });
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});

it('returns the top-1 match with highest cosine similarity', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
    version: 2, dim: 4, modelHash: 0,
    byId: new Map<string, Float32Array>([
      ['card-a', vec([0.95, 0.05, 0, 0])],   // very close to query
      ['card-b', vec([0.5, 0.5, 0, 0])],
      ['card-c', vec([0, 1, 0, 0])],
    ]),
    byName: new Map(),
  });
  const result = await findCardByImage('file:///fake.jpg');
  expect(result?.scryfallId).toBe('card-a');
  expect(result?.score).toBeGreaterThan(0.99);
});

it('scores identical query and gallery vector at ~1.0', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
    version: 2, dim: 4, modelHash: 0,
    byId: new Map<string, Float32Array>([['same', vec([1, 0, 0, 0])]]),
    byName: new Map(),
  });
  const result = await findCardByImage('file:///fake.jpg');
  expect(result?.score).toBeCloseTo(1.0, 5);
  expect(result?.scryfallId).toBe('same');
});

it('returns a single-card match when the database has one entry', async () => {
  (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
    version: 2, dim: 4, modelHash: 0,
    byId: new Map<string, Float32Array>([
      ['only-card', vec([1, 0, 0, 0])],
    ]),
    byName: new Map(),
  });
  const result = await findCardByImage('file:///fake.jpg');
  expect(result?.scryfallId).toBe('only-card');
});

it('returns null when encoder output dim ≠ gallery dim (model mismatch)', async () => {
  // Encoder returns 256-d vector but gallery is 4-d — unchecked, the dot
  // product loop would read undefined and produce NaN scores that slip
  // through MATCH_MIN. The guard must refuse to search.
  const q = new Float32Array(256); q[0] = 1;
  (encodeCardImage as jest.Mock).mockResolvedValue(q);
  (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
    version: 2, dim: 4, modelHash: 0,
    byId: new Map<string, Float32Array>([['card-a', vec([1, 0, 0, 0])]]),
    byName: new Map(),
  });
  const result = await findCardByImage('file:///fake.jpg');
  expect(result).toBeNull();
});
