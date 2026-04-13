import { similaritySearch } from '../../src/embeddings/similarity';

function makeMap(entries: Array<{ id: string; vector: number[] }>): Map<string, Float32Array> {
  const map = new Map<string, Float32Array>();
  for (const { id, vector } of entries) {
    const v = new Float32Array(vector);
    let sum = 0;
    for (const x of v) sum += x * x;
    const norm = Math.sqrt(sum);
    const normalized = new Float32Array(vector.map(x => (norm > 0 ? x / norm : 0)));
    map.set(id, normalized);
  }
  return map;
}

describe('similaritySearch', () => {
  it('returns empty array when target id is not in map', () => {
    const map = makeMap([{ id: 'card-a', vector: [1, 0, 0] }]);
    expect(similaritySearch('missing-id', map, 10)).toEqual([]);
  });

  it('excludes the source card from results', () => {
    const map = makeMap([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [1, 0, 0] },
    ]);
    const results = similaritySearch('card-a', map, 10);
    expect(results.map(r => r.scryfallId)).not.toContain('card-a');
  });

  it('assigns score ≈ 1.0 to identical vectors', () => {
    const map = makeMap([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [1, 0, 0] },
    ]);
    const results = similaritySearch('card-a', map, 10);
    expect(results[0].score).toBeCloseTo(1.0, 5);
  });

  it('assigns score ≈ 0.0 to orthogonal vectors', () => {
    const map = makeMap([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [0, 1, 0] },
    ]);
    const results = similaritySearch('card-a', map, 10);
    expect(results[0].score).toBeCloseTo(0.0, 5);
  });

  it('returns results sorted by descending score', () => {
    const map = makeMap([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [0.9, 0.1, 0] },
      { id: 'card-c', vector: [0, 1, 0] },
    ]);
    const results = similaritySearch('card-a', map, 10);
    expect(results[0].scryfallId).toBe('card-b');
    expect(results[1].scryfallId).toBe('card-c');
  });

  it('limits results to topN', () => {
    const map = makeMap([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [1, 0, 0] },
      { id: 'card-c', vector: [1, 0, 0] },
      { id: 'card-d', vector: [1, 0, 0] },
    ]);
    const results = similaritySearch('card-a', map, 2);
    expect(results.length).toBe(2);
  });

  it('returns all results when map has fewer than topN cards', () => {
    const map = makeMap([
      { id: 'card-a', vector: [1, 0, 0] },
      { id: 'card-b', vector: [0, 1, 0] },
    ]);
    const results = similaritySearch('card-a', map, 20);
    expect(results.length).toBe(1);
  });
});
