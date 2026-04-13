import type { EmbeddingMap } from './parser';

export type SimilarityResult = { scryfallId: string; score: number };

/**
 * Find the topN most similar cards to the given scryfallId using dot-product similarity.
 * Assumes all vectors in the map are L2-normalized (cosine similarity = dot product).
 */
export function similaritySearch(
  scryfallId: string,
  map: EmbeddingMap,
  topN = 20
): SimilarityResult[] {
  const target = map.get(scryfallId);
  if (!target) return [];

  const results: SimilarityResult[] = [];
  for (const [id, vector] of map) {
    if (id === scryfallId) continue;
    results.push({ scryfallId: id, score: dotProduct(target, vector) });
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topN);
}

function dotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
