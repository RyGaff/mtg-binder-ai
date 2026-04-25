import type { EmbeddingIndex } from './parser';

export type SimilarityResult = { scryfallId: string; score: number };

const CACHE_CAPACITY = 50;
const cache = new Map<string, SimilarityResult[]>();

export function clearSimilarityCache(): void {
  cache.clear();
}

export function similaritySearch(
  scryfallId: string,
  index: EmbeddingIndex,
  topN = 20,
): SimilarityResult[] {
  const cacheKey = `${scryfallId}:${topN}`;
  const hit = cache.get(cacheKey);
  if (hit) {
    cache.delete(cacheKey);
    cache.set(cacheKey, hit);
    return hit;
  }

  const targetRow = index.idIndex.get(scryfallId);
  if (targetRow === undefined) return [];

  const { ids, vectors, dim, size } = index;
  const targetOffset = targetRow * dim;
  const limit = dim - (dim % 4);

  const rows = new Int32Array(topN);
  const scores = new Float64Array(topN);
  let filled = 0;
  let minIdx = 0;
  let minScore = Infinity;

  function recomputeMin() {
    minIdx = 0;
    minScore = scores[0];
    for (let i = 1; i < topN; i++) {
      if (scores[i] < minScore) {
        minScore = scores[i];
        minIdx = i;
      }
    }
  }

  for (let r = 0; r < size; r++) {
    if (r === targetRow) continue;
    const offset = r * dim;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    let j = 0;
    for (; j < limit; j += 4) {
      s0 += vectors[targetOffset + j]     * vectors[offset + j];
      s1 += vectors[targetOffset + j + 1] * vectors[offset + j + 1];
      s2 += vectors[targetOffset + j + 2] * vectors[offset + j + 2];
      s3 += vectors[targetOffset + j + 3] * vectors[offset + j + 3];
    }
    let score = s0 + s1 + s2 + s3;
    for (; j < dim; j++) score += vectors[targetOffset + j] * vectors[offset + j];

    if (filled < topN) {
      rows[filled] = r;
      scores[filled] = score;
      filled++;
      if (filled === topN) recomputeMin();
      continue;
    }

    if (score <= minScore) continue;
    rows[minIdx] = r;
    scores[minIdx] = score;
    recomputeMin();
  }

  const results: SimilarityResult[] = new Array(filled);
  for (let i = 0; i < filled; i++) {
    results[i] = { scryfallId: ids[rows[i]], score: scores[i] };
  }
  results.sort((a, b) => b.score - a.score);

  if (cache.size >= CACHE_CAPACITY) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) cache.delete(oldestKey);
  }
  cache.set(cacheKey, results);

  return results;
}
