import { similaritySearch } from '../../src/embeddings/similarity';
import type { EmbeddingIndex } from '../../src/embeddings/parser';

function makeIndex(entries: Array<{ id: string; vector: number[] }>): EmbeddingIndex {
  const dim = entries[0]?.vector.length ?? 0;
  const n = entries.length;
  const ids: string[] = new Array(n);
  const vectors = new Float32Array(n * dim);
  const idIndex = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const { id, vector } = entries[i];
    let sum = 0;
    for (const x of vector) sum += x * x;
    const norm = Math.sqrt(sum);
    for (let j = 0; j < dim; j++) {
      vectors[i * dim + j] = norm > 0 ? vector[j] / norm : 0;
    }
    ids[i] = id;
    idIndex.set(id, i);
  }
  return { version: 2, dim, modelHash: 0, size: n, ids, vectors, idIndex, byName: new Map() };
}

describe('similaritySearch', () => {
  it('returns empty array when target id is not in map', () => {
    const idx = makeIndex([{ id: 'card-a', vector: [1, 0, 0] }]);
    expect(similaritySearch('missing-id', idx, 10)).toEqual([]);
  });

  it('excludes the source card from results', () => {
    const idx = makeIndex([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [1, 0, 0] },
    ]);
    const results = similaritySearch('card-a', idx, 10);
    expect(results.map(r => r.scryfallId)).not.toContain('card-a');
  });

  it('assigns score ~1.0 to identical vectors', () => {
    const idx = makeIndex([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [1, 0, 0] },
    ]);
    const results = similaritySearch('card-a', idx, 10);
    expect(results[0].score).toBeCloseTo(1.0, 5);
  });

  it('assigns score ~0.0 to orthogonal vectors', () => {
    const idx = makeIndex([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [0, 1, 0] },
    ]);
    const results = similaritySearch('card-a', idx, 10);
    expect(results[0].score).toBeCloseTo(0.0, 5);
  });

  it('returns results sorted by descending score', () => {
    const idx = makeIndex([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [0.9, 0.1, 0] },
      { id: 'card-c', vector: [0, 1, 0] },
    ]);
    const results = similaritySearch('card-a', idx, 10);
    expect(results[0].scryfallId).toBe('card-b');
    expect(results[1].scryfallId).toBe('card-c');
  });

  it('limits results to topN', () => {
    const idx = makeIndex([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [1, 0, 0] },
      { id: 'card-c', vector: [1, 0, 0] },
      { id: 'card-d', vector: [1, 0, 0] },
    ]);
    const results = similaritySearch('card-a', idx, 2);
    expect(results.length).toBe(2);
  });

  it('returns all results when index has fewer than topN cards', () => {
    const idx = makeIndex([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [0, 1, 0] },
    ]);
    const results = similaritySearch('card-a', idx, 20);
    expect(results.length).toBe(1);
  });
});
